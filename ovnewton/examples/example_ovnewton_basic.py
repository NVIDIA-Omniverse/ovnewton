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
VBD_SOLVER_ITERATIONS = 40

VBD_CONTACT_MATERIAL_OVERRIDE_USDA = """#usda 1.0
(
    defaultPrim = "ContactMaterial"
)

def Material "ContactMaterial" (
    prepend apiSchemas = ["NewtonMaterialAPI"]
)
{
    float newton:contactStiffness = 100000
    float newton:contactDamping = 1000
}
"""


def _solver_spec(newton, solver_name):
    """Return the example specification for one Newton solver."""
    return {
        "xpbd": {
            "register_attributes": newton.solvers.SolverXPBD.register_custom_attributes,
            "finish_builder": None,
            "create_solver": lambda model: newton.solvers.SolverXPBD(model, iterations=8),
        },
        # Feed MuJoCo-Warp contacts from Newton's collision pipeline and
        # reserve enough constraints for the shipped rigid-body scene.
        "mujoco": {
            "register_attributes": newton.solvers.SolverMuJoCo.register_custom_attributes,
            "finish_builder": None,
            "create_solver": lambda model: newton.solvers.SolverMuJoCo(
                model,
                integrator="implicitfast",
                njmax=128,
                use_mujoco_contacts=False,
            ),
        },
        "vbd": {
            "register_attributes": newton.solvers.SolverVBD.register_custom_attributes,
            "finish_builder": lambda builder: builder.color(),
            "create_solver": lambda model: newton.solvers.SolverVBD(
                model,
                iterations=VBD_SOLVER_ITERATIONS,
            ),
        },
    }[solver_name]


class Example:
    """An ovstage scene simulated by Newton and optionally rendered by ovrtx."""

    def __init__(self, args):
        self.args = args
        solver_name = getattr(args, "solver", "xpbd")
        self.frame_dt = 1.0 / 60.0
        self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.ordinal = 1
        self.next_timing_time = time.time()
        self.frame_timing = False
        self.viewport = None
        self.renderer = None
        self.renderer_attached = False

        try:
            wp.set_device(args.device)

            # ── 0. Register USD schemas ───────────────────────────
            # Both libraries must publish their schema paths before ovstage's
            # first schema read, including reads performed while ovrtx starts.
            if not args.no_render:
                from ovrtx import register_schema_paths

                register_schema_paths()

            import ovnewton

            ovnewton.register_usd_schemas()
            self.renderer = self._create_renderer()

            # ── 1. Populate the ovstage scene ────────────────────────
            import ovstage
            from ovstage import PopulationDomain, population

            scene = self._resolve_stage(args.stage)
            apply_vbd_material_override = solver_name == "vbd" and args.stage in (
                "scene_rigid_bodies",
                "scene_rigid_bodies.usda",
            )

            print(f"ovstage: loading scene {os.path.basename(scene)} ...")
            with wp.ScopedTimer("ovstage init", active=args.timing, synchronize=True):
                self.stage = ovstage.Stage("ovnewton-basic")
                population.open_usd(self.stage, scene, ordinal=1, domains=PopulationDomain.ALL)
                if apply_vbd_material_override:
                    # Temporary VBD-only USD tuning until these values can be
                    # resolved from solver-specific PhysicsScene attributes.
                    population.add_usd_reference_from_string(
                        self.stage,
                        VBD_CONTACT_MATERIAL_OVERRIDE_USDA,
                        "/World/ContactMaterial",
                    )
                    self.ordinal = 2
                    population.apply_usd_changes(self.stage, ordinal=self.ordinal)
                self.stage.advance_write_floor(ordinal=self.ordinal).wait()

            # ── 2. Attach ovrtx to ovstage ─────────────────────────────
            if self.renderer is not None:
                with wp.ScopedTimer("ovrtx attach", active=args.timing, synchronize=True):
                    self.renderer.attach_ovstage(self.stage)
                    self.renderer_attached = True

            # ── 3. Build a Newton model from ovstage ───────────────────
            import newton

            solver_spec = _solver_spec(newton, solver_name)
            builder = newton.ModelBuilder()
            solver_spec["register_attributes"](builder)

            with wp.ScopedTimer("ovnewton init", active=args.timing, synchronize=True):
                import_result = ovnewton.add_ovstage(builder, self.stage, ordinal=self.ordinal)
                finish_builder = solver_spec["finish_builder"]
                if finish_builder is not None:
                    finish_builder(builder)
                self.model = builder.finalize(skip_validation_joints=import_result.has_orphan_joints)
            if self.model.body_count == 0:
                raise RuntimeError("No rigid bodies were found in the scene")

            # The application owns the solver, state, control, and step loop.
            self.solver = solver_spec["create_solver"](self.model)
            self.binding = ovnewton.attach_ovstage(self.stage, model=self.model, ordinal=self.ordinal)
            self.collision_pipeline = newton.CollisionPipeline(self.model)
            self.contacts = self.collision_pipeline.contacts()
            print(f"newton: using {solver_name} solver")
            self.state_0 = self.model.state()
            self.state_1 = self.model.state()
            self.control = self.model.control()

            # Needed by maximal-coordinate solvers; harmless for MuJoCo.
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
            self.initial_body_q = self.state_0.body_q.numpy().copy()

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
                self.collision_pipeline.collide(self.state_0, self.contacts)
                self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
                self.state_0, self.state_1 = self.state_1, self.state_0

    def publish(self):
        """Publish the current Newton state to ovstage."""
        with wp.ScopedTimer("update to ovstage", active=self.frame_timing, synchronize=True):
            self.ordinal += 1
            self.binding.update_to_ovstage(self.state_0, ordinal=self.ordinal)
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
            render_device_arg = getattr(self.args, "render_device", None)
            render_device = None if render_device_arg is None else str(render_device_arg)
            renderer = Renderer(
                config=RendererConfig(
                    sync_mode=True,
                    active_cuda_gpus=render_device,
                )
            )
        print(f"ovrtx: renderer v{renderer.version}")
        if render_device is not None:
            print(f"ovrtx: rendering on CUDA-visible device {render_device}")
        return renderer

    def _present(self, ldr_color):
        from ovrtx import Device

        from .utils import GLViewport

        with ldr_color.map(device=Device.CUDA) as render_var:
            pixels = wp.from_dlpack(render_var, dtype=wp.vec4ub)
            # Record the copy stream now; pixels keeps the buffer alive until
            # the view is released, so presentation can use it after unmap().
            render_var.unmap(stream=pixels.device.stream.cuda_stream)

        height, width = int(pixels.shape[0]), int(pixels.shape[1])
        # CUDA/OpenGL interop uses the rendered image's device, not the
        # simulation device, which may be CPU or a different GPU.
        with wp.ScopedDevice(pixels.device):
            if self.viewport is None:
                self.viewport = GLViewport(
                    "ovstage -> ovnewton -> ovrtx",
                    width=width,
                    height=height,
                    stage=self.stage,
                )
            return self.viewport.show(pixels)

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
            "--solver",
            choices=("xpbd", "mujoco", "vbd"),
            default="xpbd",
            help="Newton solver used to advance the scene",
        )
        parser.add_argument(
            "--render-device",
            type=int,
            default=None,
            help="CUDA-visible device index used by ovrtx; unset uses the ovrtx default",
        )
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
