# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simulate a USD scene with ovstage and ovnewton.

The default scene is included with ovnewton. Rendering needs the ``examples``
extra and a CUDA device. ``--no-render`` also supports CPU-only machines.
"""

import argparse
import itertools
import os
import time

import numpy as np
import warp as wp

RENDER_PRODUCT = "/Render/Camera"
LDR_COLOR_PATH = f"{RENDER_PRODUCT}/LdrColor"

# ovrtx 0.4 does not observe CUDA transform writes. Mirror them through a host
# write until the minimum supported ovrtx version no longer needs this path.
USE_TRANSFORM_RELAY = True


class Example:
    """An ovstage scene simulated by Newton and optionally rendered by ovrtx."""

    def __init__(self, args):
        self.args = args
        self.frame_dt = 1.0 / 60.0
        self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.ordinal = 1
        self.next_timing_time = time.time()
        self.frame_timing = False
        self.viewport = None
        self.transform_relay = None
        self.renderer = None
        self.renderer_attached = False

        try:
            wp.set_device(args.device)
            self.renderer = self._create_renderer()

            # Newton and ovstage load OpenUSD. Import them only after ovrtx has
            # initialized its USD plugins.
            import newton
            import ovstage
            from ovstage import PopulationDomain, population

            import ovnewton

            scene = self._resolve_stage(args.stage)

            # ── 1. Populate the ovstage scene ─────────────────────────
            print(f"ovstage: loading scene {os.path.basename(scene)} ...")
            with wp.ScopedTimer("ovstage init", active=args.timing, synchronize=True):
                self.stage = ovstage.Stage("ovnewton-basic")
                population.open_usd(self.stage, scene, ordinal=1, domains=PopulationDomain.ALL)
                self.stage.advance_write_floor(ordinal=1).wait()

            # ── 2. Attach ovrtx to ovstage ─────────────────────────────
            if self.renderer is not None:
                with wp.ScopedTimer("ovrtx attach", active=args.timing, synchronize=True):
                    self.renderer.attach_ovstage(self.stage)
                    self.renderer_attached = True

            # ── 3. Build a Newton model from ovstage ───────────────────
            with wp.ScopedTimer("ovnewton init", active=args.timing, synchronize=True):
                self.binding = ovnewton.attach_ovstage(self.stage)
                self.model = self.binding.model
            if self.model.body_count == 0:
                raise RuntimeError("No rigid bodies were found in the scene")

            # The application owns the solver, state, control, and step loop.
            self.solver = newton.solvers.SolverXPBD(self.model, iterations=8)
            self.state_0 = self.model.state()
            self.state_1 = self.model.state()
            self.control = self.model.control()
            self.contacts = self.model.contacts()
            self.initial_body_q = self.state_0.body_q.numpy().copy()

            if self.renderer is not None and USE_TRANSFORM_RELAY:
                from .utils import TransformRelay

                self.transform_relay = TransformRelay(self.stage, list(self.model.body_label))
        except Exception:
            try:
                self.close()
            except Exception:
                pass
            raise

    def step(self, frame):
        """Simulate, publish, and optionally render one frame."""
        now = time.time()
        self.frame_timing = self.args.timing and now >= self.next_timing_time
        if self.frame_timing:
            self.next_timing_time = now + 1.0
            print(f"\nFrame {frame}")

        # ── 4. Advance the Newton simulation ──────────────────────────
        self.simulate()

        # ── 5. Publish the Newton state to ovstage ─────────────────────
        self.publish()

        # ── 6. Render the updated ovstage scene with ovrtx ─────────────
        return self.renderer is None or self.render(frame)

    def simulate(self):
        """Advance Newton by one simulation frame."""
        with wp.ScopedTimer("step", active=self.frame_timing, synchronize=True):
            for _ in range(self.sim_substeps):
                self.state_0.clear_forces()
                self.model.collide(self.state_0, self.contacts)
                self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
                self.state_0, self.state_1 = self.state_1, self.state_0

    def publish(self):
        """Publish the current Newton state to ovstage."""
        with wp.ScopedTimer("update to ovstage", active=self.frame_timing, synchronize=True):
            self.ordinal += 1
            self.binding.update_to_ovstage(self.state_0, ordinal=self.ordinal)
            if self.transform_relay is not None:
                # TODO: Replace this private pose buffer when ovrtx observes CUDA writes.
                self.transform_relay.write(self.binding._runtime.pose_out.numpy(), self.ordinal)
            if self.viewport is not None:
                self.viewport.update_camera(self.frame_dt, self.ordinal)
            self.stage.advance_write_floor(self.ordinal).wait()

    def render(self, frame):
        """Render and present or save one frame. Return False when the window closes."""
        with wp.ScopedTimer("render", active=self.frame_timing, synchronize=True):
            products = self.renderer.step(
                render_products={RENDER_PRODUCT},
                delta_time=self.frame_dt,
                ordinal=self.ordinal,
            )
            window_open = True
            for product in products.values():
                for output in product.frames:
                    ldr_color = output.render_vars.get(LDR_COLOR_PATH)
                    if ldr_color is None:
                        # TODO: Drop this fallback when ovrtx 0.4 is no longer supported.
                        ldr_color = output.render_vars.get("LdrColor")
                    if ldr_color is None:
                        continue
                    if not self.args.headless:
                        window_open = self._present(ldr_color) and window_open
                    if self.args.save_png:
                        self._save_png(ldr_color, frame)
        return window_open

    def _resolve_stage(self, name: str) -> str:
        """Resolve a ``--stage`` value to a USD file. An absolute path is used
        directly; a bare name is looked up in the assets directory. The extension
        is optional either way: tries the name as given, then ``.usda``, then
        ``.usd``."""
        import ovnewton.examples

        for candidate in (name, name + ".usda", name + ".usd"):
            path = candidate if os.path.isabs(candidate) else ovnewton.examples.get_asset(candidate)
            if os.path.isfile(path):
                return path
        where = os.path.dirname(name) if os.path.isabs(name) else ovnewton.examples.get_asset_directory()
        raise SystemExit(f"--stage: no '{name}' (.usd/.usda) found in {where}")

    def _create_renderer(self):
        if self.args.no_render:
            return None
        if not wp.is_cuda_available():
            raise RuntimeError("ovrtx requires a CUDA device")

        # ovrtx must initialize USD before ovstage is imported.
        from ovrtx import Renderer, RendererConfig

        with wp.ScopedTimer("ovrtx init", active=self.args.timing, synchronize=True):
            renderer = Renderer(config=RendererConfig(sync_mode=True))
        print(f"ovrtx: renderer v{renderer.version}")
        return renderer

    def _present(self, ldr_color):
        from ovrtx import Device

        from .utils import GLViewport

        with ldr_color.map(device=Device.CUDA) as render_var:
            pixels = wp.from_dlpack(render_var, dtype=wp.vec4ub)
            height, width = int(pixels.shape[0]), int(pixels.shape[1])
            if self.viewport is None:
                self.viewport = GLViewport(
                    "ovstage -> ovnewton -> ovrtx",
                    width=width,
                    height=height,
                    stage=self.stage,
                )
            window_open = self.viewport.show(pixels)
            render_var.unmap(stream=pixels.device.stream.cuda_stream)
            return window_open

    def _save_png(self, ldr_color, frame):
        from ovrtx import Device
        from PIL import Image

        with ldr_color.map(device=Device.CPU) as render_var:
            pixels = np.from_dlpack(render_var)
            Image.fromarray(pixels).save(os.path.join(self.args.output_dir, f"frame_{frame:03d}.png"))

    def test_final(self):
        """Check that the simulation produced a finite, updated state."""
        final_body_q = self.state_0.body_q.numpy()
        if not np.isfinite(final_body_q).all():
            raise RuntimeError("Newton produced an invalid body transform")
        if (
            self.args.no_render
            and self.args.stage == "scene_rigid_bodies.usda"
            and np.allclose(final_body_q, self.initial_body_q)
        ):
            raise RuntimeError("Newton did not advance the default scene")

    def run(self):
        """Run until the requested frame count or until the window closes."""
        fixed_length = self.args.headless or self.args.no_render
        frames = range(self.args.num_frames) if fixed_length else itertools.count()
        for frame in frames:
            if not self.step(frame):
                break
            if fixed_length and frame % 20 == 0:
                print(f"  frame {frame}/{self.args.num_frames}")

        if self.args.headless:
            print(f"\n{self.args.num_frames} frames rendered")
            if self.args.save_png:
                count = len([path for path in os.listdir(self.args.output_dir) if path.endswith(".png")])
                print(f"{count} PNGs saved to {self.args.output_dir}/")
        elif self.args.no_render:
            print(f"\n{self.args.num_frames} frames simulated on {wp.get_device()}")

    def close(self):
        """Release resources owned by the example."""
        if self.viewport is not None:
            self.viewport.close()
        if self.transform_relay is not None:
            self.transform_relay.close()
            self.transform_relay = None
        if self.renderer is not None:
            if self.renderer_attached:
                self.renderer.detach_ovstage()
                self.renderer_attached = False
            self.renderer.destroy()
            self.renderer = None

    @staticmethod
    def create_parser():
        """Create the command-line parser for this example."""
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("--device", default="cuda:0", help="Warp device used for the simulation")
        parser.add_argument(
            "--num-frames",
            type=int,
            default=120,
            help="Number of frames in headless or no-render mode",
        )
        parser.add_argument(
            "--stage",
            default="scene_rigid_bodies.usda",
            help="USD scene in examples/assets/ or an absolute path (extension optional)",
        )
        parser.add_argument(
            "--output-dir",
            default="output",
            help="Directory used by --save-png and --headless",
        )
        parser.add_argument("--save-png", action="store_true", help="Save rendered frames as PNG files")
        parser.add_argument(
            "--headless",
            action="store_true",
            help="Skip the viewport and save PNGs",
        )
        parser.add_argument(
            "--no-render",
            action="store_true",
            help="Run the simulation without ovrtx or image output",
        )
        parser.add_argument(
            "--timing",
            action="store_true",
            help="Print initialization and per-frame timings",
        )
        return parser


def main():
    parser = Example.create_parser()
    args = parser.parse_args()
    if args.num_frames < 1:
        parser.error("--num-frames must be at least 1")
    if args.no_render and (args.headless or args.save_png):
        parser.error("--no-render cannot be combined with --headless or --save-png")
    if args.headless:
        args.save_png = True
    if args.save_png:
        os.makedirs(args.output_dir, exist_ok=True)

    example = Example(args)
    try:
        example.run()
        example.test_final()
    finally:
        example.close()


if __name__ == "__main__":
    main()
