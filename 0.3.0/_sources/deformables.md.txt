<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deformables

Ovnewton loads a subset of canonical AOUSD/Newton cables, cloth, and tetrahedral
volumes with Newton 1.8.0 or newer. Native PhysX deformables are unsupported.

## Run the example

The packaged example uses VBD to simulate cables on the left, cloth over a
sphere in the center, and a volume on the right:

```bash
python -m ovnewton.examples.example_ovnewton_deformables
```

Install the `examples` extra for rendering. CUDA and graph capture are enabled
by default. Add `--no-render` for simulation only. To save frames, use
`--headless --output-dir output/deformables`. For a short CPU check, use
`--no-render --device cpu --num-frames 8`. Rendering requires an RTX GPU.

## Load your own scene

Follow [the loading example](getting-started.md), using
`ovnewton.register_usd_schemas()` before population and
`ovnewton.population_desc()` to include explicitly authored deformable fields.
Rendering population alone does not include the required physics inputs.

Both `attach_ovstage(stage)` and caller-owned `add_ovstage(builder, stage)` work.
For VBD, register `SolverVBD`'s custom attributes and call
`builder.color(include_bending=True)` before finalizing a caller-owned builder.

Each geometry needs an enabled, non-kinematic `PhysicsDeformableBodyAPI` owner
on itself or an ancestor. Bind the physics material directly to the geometry
or its body. `PhysicsCollisionAPI` may be applied to the simulation geometry.
`PhysicsRigidBodyAPI` on the geometry or any ancestor is unsupported.

| Type | Geometry and simulation schema | Supported inputs |
| --- | --- | --- |
| Cable | `BasisCurves` + `PhysicsCurvesDeformableSimAPI` | Open linear curves, at least three points each; constant positive thickness; optional normals; uniform initial velocity per curve; fixed rigid transform. |
| Cloth | `Mesh` + `PhysicsSurfaceDeformableSimAPI` | Fixed rigid transform; constant thickness. |
| Volume | `TetMesh` + `PhysicsVolumeDeformableSimAPI` | Positive, right-handed tetrahedra without unused points, duplicates, or non-manifold faces; optional per-point velocities; fixed rigid transform. |

VBD supports the combined scene on CPU and CUDA. Cables require VBD;
volumes are also tested with XPBD. Cable display widths and physics thicknesses
are separate. A volume can render its own boundary through
`surfaceFaceVertexIndices` without a separate skin.

## Materials and mass

Density comes from the body, then material. Otherwise, cables and cloth use
`builder.default_shape_cfg.density` (1000 kg/m³), and volumes use
`builder.default_tet_density` (1 kg/m³), converted from SI to stage units.
Callers can override these builder defaults. Explicit body/per-element masses are
unsupported. Cable mass is density × cross-section area × length; capsule
endcaps add no mass.

Cable materials accept Young's modulus, Poisson's ratio, and
`physics:curvesStretchStiffness`, `physics:curvesShearStiffness`,
`physics:curvesBendStiffness`, and `physics:curvesTwistStiffness`. These are
section rigidities (EA, kGA, EI, GJ); missing values come from elastic properties
and cross-section geometry. Cable elasticity accepts finite `E >= 0` and
`-1 < nu < 0.5`; zero E leaves unauthored rigidities at zero. Defaults are
1 MPa in stage units and `nu = 0.3`.

Cloth materials accept `physics:surfaceStretchStiffness` and
`physics:surfaceBendStiffness` as structural stiffnesses, without multiplying by
thickness again. Missing values use `builder.default_tri_ke` and
`builder.default_edge_ke`; triangle damping uses `builder.default_tri_kd`.
Thickness still controls density-derived mass and particle radius.

Volume materials require Young's modulus and Poisson's ratio together
(`E > 0`, `0 <= nu < 0.5`), or use Newton's defaults when both are absent.

## Updates and limits

`update_to_ovstage()` publishes points in USD-local coordinates.
`update_from_ovstage()` preserves cable state while updating rigid bodies.
Read deformable state directly from Newton arrays; the generic `read()` API
has no deformable selections. Rebuild after topology or geometry-frame changes.

Attachments, authored rest arrays, separate collision/graphics geometry,
element filters, collision disabling, variable thickness, periodic cables, and general
inherited/collection material bindings are unsupported. Deformable transforms
must not scale, shear, or reflect.
