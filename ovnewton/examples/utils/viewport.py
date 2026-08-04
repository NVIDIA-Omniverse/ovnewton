# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone live viewer for the ovnewton examples: a pyglet/OpenGL window
presenting CUDA RGBA8 frames via Warp's GL interop.

Follows Newton's ViewerRTX presentation path (newton/_src/viewer/viewer_rtx.py)
without the viewer framework around it: a CUDA ``vec4ub [H, W]`` frame (e.g. an
ovrtx LdrColor render var mapped to CUDA) is copied device-to-device into an
OpenGL texture via ``wp.GLTextureResource`` and drawn as a fullscreen triangle
(Y-flip in the fragment shader). No CPU readback.

This replaces the old ``GpuViewport`` ctypes wrapper around
``libov_viewport.so``, which required a source build of ovstage; presenting
frames needs only ``pyglet`` (installed with the ``examples`` extra) and
``warp``, so it works in any pip-installed layout. Import and use:

    from utils.viewport import GLViewport

    vp = GLViewport("my example", width=w, height=h, stage=stage)
    vp.show(frame)                    # per frame; False once the user closes it
    vp.update_camera(dt, ordinal)     # fly the camera, publish if it moved
    vp.close()

Passing ``stage`` also enables the interactive fly camera: the viewport finds
the scene's Camera prim, starts from the transform authored in USD, and
publishes its own moves back to ovstage. That path imports ovstage (a core
ovnewton dependency) lazily, so a viewport built without a stage still needs
nothing beyond pyglet and warp. ``FlyCamera`` holds the camera state and math
and is usable on its own.
"""

from __future__ import annotations

import ctypes
import math

import numpy as np

# Fly-camera tuning: speed/damping match Newton's ViewerGL/ViewerRTX.
CAMERA_SPEED = 4.0  # [m/s] WASD fly speed (LSHIFT: x4)
CAMERA_DAMP_TAU = 0.083  # [s] velocity smoothing time constant
LOOK_SENSITIVITY = 0.2  # [deg/pixel] left-drag look

_VS = b"""#version 330
out vec2 uv;
void main() {
    uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
\x00"""

_FS = b"""#version 330
uniform sampler2D tex;
in vec2 uv;
out vec4 fragColor;
void main() {
    fragColor = texture(tex, vec2(uv.x, 1.0 - uv.y));
}
\x00"""


class GLViewport:
    """pyglet window presenting CUDA RGBA8 frames via Warp GL interop.

    Pass ``stage`` to enable the interactive fly camera: the viewport finds the
    scene's Camera prim, starts from the transform authored in USD, and
    publishes its own moves back to ovstage (see :meth:`update_camera`).
    """

    def __init__(self, title: str = "ovnewton", width: int = 1280, height: int = 720,
                 vsync: bool = False, stage=None):
        import pyglet

        pyglet.options["debug_gl"] = False
        import warp as wp
        from pyglet import gl

        self._pyglet = pyglet
        self._gl = gl
        self._render_width = width
        self._render_height = height
        self._closed = False

        self._window = pyglet.window.Window(
            width=width, height=height, caption=title, resizable=True, vsync=vsync)

        # -- input state for camera controls (hooks for future use) ------
        self._keys_down: set[int] = set()
        self._look_dx = 0.0
        self._look_dy = 0.0

        @self._window.event
        def on_close():
            self._closed = True

        @self._window.event
        def on_key_press(symbol, modifiers):
            if symbol == pyglet.window.key.ESCAPE:
                self._closed = True
                return True  # suppress pyglet's default ESC handling
            self._keys_down.add(symbol)

        @self._window.event
        def on_key_release(symbol, modifiers):
            self._keys_down.discard(symbol)

        @self._window.event
        def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
            if buttons & pyglet.window.mouse.LEFT:
                self._look_dx += dx
                self._look_dy += dy

        self._window.switch_to()

        # Render-sized RGBA8 texture, shared with CUDA through Warp.
        tex_id = (gl.GLuint * 1)()
        gl.glGenTextures(1, tex_id)
        self._texture = tex_id[0]
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, width, height, 0,
                        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        self._tex_resource = wp.GLTextureResource(
            self._texture, gl.GL_TEXTURE_2D, flags=wp.TextureResourceFlags.WRITE_DISCARD)

        self._program = self._build_program()

        vao = (gl.GLuint * 1)()
        gl.glGenVertexArrays(1, vao)
        self._vao = vao[0]

        # Optional interactive camera. Imported here rather than at module
        # scope so the presentation path stays pyglet + warp only.
        self.camera = None
        self._camera_relay = None
        if stage is not None:
            from .transform_relay import TransformRelay, find_paths_by_type

            camera_paths = find_paths_by_type(stage, "Camera", ordinal=1)
            if camera_paths:
                self._camera_relay = TransformRelay(stage, camera_paths[:1])
                # Start from the pose the scene authored in USD.
                self.camera = FlyCamera.from_world_matrix(self._camera_relay.read(1)[0])
                print(f"camera: {camera_paths[0]} — WASD/arrows fly, Q/E down/up, "
                      "left-drag look, SHIFT boost")

    # -- GL helpers ------------------------------------------------------

    def _compile_shader(self, src: bytes, stype):
        gl = self._gl
        shader = gl.glCreateShader(stype)
        src_p = ctypes.c_char_p(src)
        src_pp = (ctypes.c_char_p * 1)(src_p)
        gl.glShaderSource(shader, 1,
                          ctypes.cast(src_pp, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))),
                          None)
        gl.glCompileShader(shader)
        status = (gl.GLint * 1)()
        gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS, status)
        if not status[0]:
            raise RuntimeError("viewport shader compilation failed")
        return shader

    def _build_program(self):
        gl = self._gl
        vs = self._compile_shader(_VS, gl.GL_VERTEX_SHADER)
        fs = self._compile_shader(_FS, gl.GL_FRAGMENT_SHADER)
        program = gl.glCreateProgram()
        gl.glAttachShader(program, vs)
        gl.glAttachShader(program, fs)
        gl.glLinkProgram(program)
        status = (gl.GLint * 1)()
        gl.glGetProgramiv(program, gl.GL_LINK_STATUS, status)
        if not status[0]:
            raise RuntimeError("viewport shader link failed")
        gl.glDeleteShader(vs)
        gl.glDeleteShader(fs)
        return program

    # -- API ---------------------------------------------------------------

    def show(self, pixels) -> bool:
        """Present one frame (a Warp ``vec4ub`` [H, W] CUDA array).

        Returns False once the user closed the window.
        """
        if self._closed:
            self.close()
            return False
        gl = self._gl

        self._window.switch_to()
        self._window.dispatch_events()
        if self._closed:
            # X button or ESC: tear the window down immediately rather than
            # leaving a frozen window while the caller keeps simulating.
            self.close()
            return False

        # CUDA -> GL texture (device-to-device).
        frame_tex = self._tex_resource.map()
        frame_tex.copy_from(pixels)
        self._tex_resource.unmap()

        # Letterbox viewport preserving the render aspect.
        fb_w, fb_h = self._window.get_framebuffer_size()
        render_aspect = self._render_width / max(self._render_height, 1)
        window_aspect = fb_w / max(fb_h, 1)
        if window_aspect >= render_aspect:
            vp_h = fb_h
            vp_w = int(fb_h * render_aspect)
            vp_x, vp_y = (fb_w - vp_w) // 2, 0
        else:
            vp_w = fb_w
            vp_h = int(fb_w / render_aspect)
            vp_x, vp_y = 0, (fb_h - vp_h) // 2

        gl.glViewport(0, 0, fb_w, fb_h)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glViewport(vp_x, vp_y, vp_w, vp_h)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture)
        gl.glUseProgram(self._program)
        gl.glBindVertexArray(self._vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        gl.glBindVertexArray(0)
        gl.glUseProgram(0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        self._window.flip()
        return True

    # -- camera input accessors ------------------------------------------

    def is_key_down(self, symbol: int) -> bool:
        return symbol in self._keys_down

    def take_look_delta(self) -> tuple[float, float]:
        """Return and reset accumulated left-drag deltas (pixels)."""
        d = (self._look_dx, self._look_dy)
        self._look_dx = 0.0
        self._look_dy = 0.0
        return d

    def update_camera(self, dt: float, ordinal: int) -> bool:
        """Fly the camera from this frame's input and publish it at ``ordinal``.

        Returns True when the camera moved (and was therefore written). A
        viewport built without a stage, or over a scene with no Camera prim,
        has no camera and always returns False.

        Call while ``ordinal`` is still open — i.e. before
        ``stage.advance_write_floor(ordinal)``.
        """
        if self.camera is None:
            return False
        if not self.camera.apply_input(self, dt):
            return False
        self._camera_relay.write(self.camera.world_matrix()[np.newaxis], ordinal)
        return True

    def close(self) -> None:
        if not self._closed:
            self._closed = True
        if self._camera_relay is not None:
            self._camera_relay.close()
            self._camera_relay = None
        if self._window is not None:
            self._window.close()
            self._window = None

    def __del__(self):  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass


class FlyCamera:
    """Fly camera: position + yaw/elevation (Z-up world, USD -Z-forward).

    ``elevation`` is degrees above the horizon (negative = looking down);
    ``yaw`` rotates about world Z, 0 = looking along +Y. Controls match
    Newton's ViewerGL/ViewerRTX: WASD/arrows fly in camera space, Q/E move
    down/up, left-drag looks around, LSHIFT boosts speed.
    """

    def __init__(self, position, elevation_deg: float, yaw_deg: float = 0.0):
        self.pos = np.array(position, dtype=np.float64)
        self.elevation = float(elevation_deg)
        self.yaw = float(yaw_deg)
        self._velocity = np.zeros(3, dtype=np.float64)

    @classmethod
    def from_world_matrix(cls, matrix) -> "FlyCamera":
        """Recover a camera from an authored USD camera world transform.

        Inverse of :meth:`world_matrix`, so a camera built this way starts
        exactly where the scene authored it.
        """
        mat = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
        front = -mat[2, :3]
        norm = np.linalg.norm(front)
        if norm > 1e-9:
            front = front / norm
        elevation = math.degrees(math.asin(max(-1.0, min(1.0, float(front[2])))))
        yaw = math.degrees(math.atan2(float(front[0]), float(front[1])))
        return cls(mat[3, :3], elevation, yaw)

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        el = math.radians(self.elevation)
        yaw = math.radians(self.yaw)
        front = np.array(
            [math.cos(el) * math.sin(yaw), math.cos(el) * math.cos(yaw), math.sin(el)])
        right = np.array([math.cos(yaw), -math.sin(yaw), 0.0])
        up = np.cross(right, front)
        return front, right, up

    def world_matrix(self) -> np.ndarray:
        """Row-major USD camera world transform (rows: right, up, -front, pos)."""
        front, right, up = self.basis()
        mat = np.eye(4, dtype=np.float64)
        mat[0, :3] = right
        mat[1, :3] = up
        mat[2, :3] = -front
        mat[3, :3] = self.pos
        return mat

    def apply_input(self, viewport: GLViewport, dt: float) -> bool:
        """Consume ``viewport`` input and move the camera over ``dt`` seconds.

        Returns True when the camera moved, i.e. when the caller needs to
        re-publish :meth:`world_matrix`.
        """
        import pyglet

        key = pyglet.window.key
        moved = False

        dx, dy = viewport.take_look_delta()
        if dx or dy:
            self.yaw += dx * LOOK_SENSITIVITY
            self.elevation = max(-89.0, min(89.0, self.elevation + dy * LOOK_SENSITIVITY))
            moved = True

        front, right, up = self.basis()
        desired = np.zeros(3, dtype=np.float64)
        if viewport.is_key_down(key.W) or viewport.is_key_down(key.UP):
            desired += front
        if viewport.is_key_down(key.S) or viewport.is_key_down(key.DOWN):
            desired -= front
        if viewport.is_key_down(key.D) or viewport.is_key_down(key.RIGHT):
            desired += right
        if viewport.is_key_down(key.A) or viewport.is_key_down(key.LEFT):
            desired -= right
        if viewport.is_key_down(key.E):
            desired += up
        if viewport.is_key_down(key.Q):
            desired -= up

        norm = np.linalg.norm(desired)
        if norm > 1e-6:
            speed = CAMERA_SPEED * (4.0 if viewport.is_key_down(key.LSHIFT) else 1.0)
            desired = desired / norm * speed
        else:
            desired[:] = 0.0

        # Damped velocity smoothing, so starts and stops are not instant.
        self._velocity += (desired - self._velocity) * (dt / max(1e-4, CAMERA_DAMP_TAU))
        if np.linalg.norm(self._velocity) > 1e-4:
            self.pos += self._velocity * dt
            moved = True
        return moved
