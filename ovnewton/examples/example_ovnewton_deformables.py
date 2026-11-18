# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load canonical USD cloth, cables, and a volume through ovstage and simulate with VBD."""

import os

import numpy as np
import warp as wp

from .example_ovnewton_basic import Example as BasicExample


class Example(BasicExample):
    """Cloth over a rigid sphere, with cables on the left and a volume on the right."""

    sim_substeps = 10

    def _create_simulation(self):
        """Use VBD with cloth bending and soft-contact settings."""
        import newton

        import ovnewton

        builder = newton.ModelBuilder()
        builder.default_tri_kd = 0.1
        newton.solvers.SolverVBD.register_custom_attributes(builder)
        with wp.ScopedTimer("ovnewton init", active=self.args.timing, synchronize=True):
            result = ovnewton.add_ovstage(builder, self.stage, ordinal=self.ordinal)
            builder.color(include_bending=True)
            model = builder.finalize(skip_validation_joints=result.has_orphan_joints)
        if model.particle_count == 0:
            raise RuntimeError("No particle deformable was found in the scene")
        model.soft_contact_ke = 1.0e5
        model.soft_contact_kd = 1.0e2
        model.soft_contact_mu = 0.5
        return model, newton.solvers.SolverVBD(model, iterations=10)

    def test_final(self):
        """Check finite particle and cable state, and motion from the rest pose."""
        final_particle_q = self.state_0.particle_q.numpy()
        if not np.isfinite(final_particle_q).all():
            raise RuntimeError("Newton produced invalid particle positions")
        if not np.isfinite(self.state_0.body_q.numpy()).all():
            raise RuntimeError("Newton produced invalid body transforms")
        if np.allclose(final_particle_q, self.initial_particle_q):
            raise RuntimeError("Newton did not advance the deformables")

    @staticmethod
    def create_parser():
        """Create the deformables example's command-line parser."""
        parser = BasicExample.create_parser()
        parser.set_defaults(stage="scene_deformables.usda", solver="vbd", num_frames=180)
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
