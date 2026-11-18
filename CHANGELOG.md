<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Changelog

Notable user-facing changes to ovnewton.

## [0.3.0] - 2026-11-18

### Upgrading from 0.2.1

Upgrade the runtime stack together: use Newton 1.8 or newer, OVStage 0.4,
and optional OVRTX 0.7 for rendering examples. The exact requirements are
listed below.

`StageBinding.read()` now separates prim groups from attribute columns.
Iterate `group.columns` to access values, tensor metadata and CUDA readiness.
Use each column's `data_row_index(i)` to map a group's prim row to its value
row. Newton-native units, zero-copy paths and result lifetime rules are
unchanged.

### Added

- Discover read fields through `ovnewton.read_fields()` and `ovnewton.attributes.list_attributes()` / `get_attribute()`. Their `ReadField` metadata and the read field reference describe source arrays, units, frames, joint layouts, device and storage behavior.
- Read Newton-native drive gains and limits through `joint_target_ke`, `joint_target_kd`, `joint_limit_lower`, `joint_limit_upper` and `joint_velocity_limit`, without conversion to USD units or authored body order.
- Read additional Newton-native body and joint properties through `body_mass`, `body_inertia`, `body_com`, `body_qdd`, `joint_effort_limit`, `joint_armature` and `joint_friction`.
- Read commanded joint effort through `joint_f`, preserving `Control.joint_f` units, DOF layout and device storage. This is commanded force/torque, not measured effort or a solver result.
- Add native `joint_target_q` / `joint_target_qd` reads and `StageBinding.write()` for selected scalar articulation-joint targets, using Newton-native units without stage exchange.
- Import open-curve cables with fixed rigid transforms, density-derived mass, VBD dynamics and simulated local-point publication.
- Support cloth in fixed translated and rotated geometry frames, publishing simulated points back to USD-local coordinates.
- Import tetrahedral volumes with fixed rigid transforms, density-derived mass and simulated local-point publication.
- Ship `example_ovnewton_deformables` and its scene, simulating cables, cloth and a tetrahedral volume together with VBD.
- Right-drag rigid bodies and robot links in the example viewers to apply spring forces, including with CUDA graphs. Cloth particles are not pickable.
- Import scalar MuJoCo position/velocity actuators with control and force limits and damping-ratio preservation, following Newton's USD importer.
- Import scalar MuJoCo joint actuator-force limits while preserving USD drive precedence.
- Select final releases or `dev (main)` in the documentation version menu, preserving the current page when available.

### Changed

- Require Newton `>=1.8.0`, OVStage `==0.4.0.1794520352`, and optional OVRTX `>=0.7.0.1794596436,<0.8`. These versions are available from public package sources.
- Support Newton USD schemas 0.6 while retaining compatibility with 0.5.
- Replay basic, robot, and deformables example physics with CUDA graphs by default; use `--cuda-graph false` to run the direct loop. CPU simulation stays on the direct loop.
- Reduce redundant array copies and stage reads during model import and attachment.

### Fixed

- Resolve collision-group membership through OVStage's USD collection resolver, including explicit-only selections, nested collections and root inclusion.
- Preserve authored joint direction when body-to-world joints are reordered, including supported scalar and D6 axes. Keep Newton-native coordinates, targets and limits unchanged, and keep standard reads and scalar stage updates consistent for caller-owned models.
- Match Newton 1.8's MuJoCo weld relative poses and loop-joint frames, keeping rotation free when torque scale is zero.
- Preserve omitted Newton joint overrides through `population_desc()` so MuJoCo or builder defaults remain effective. Explicit Newton values, including zero, retain their precedence.
- Preserve tetrahedral boundary faces during example population so OVRTX renders the deformable volume.
- Match Newton's USD importer for cloth stretch/bend stiffness and default triangle damping, without extra thickness scaling. Retune the example asset to preserve its previous effective material values.
- Preserve authored empty strings when resolving OVStage tokens, so an empty physics axis is rejected instead of using a default.
- Reject non-contiguous or non-`float32` joint state, control and DOF-property arrays instead of misreading them.

## [0.2.1] - 2026-10-22

### Upgrading from 0.2.0

Upgrade the runtime stack together: Newton 1.6 and OVStage 0.2 / OVRTX 0.5
are no longer supported. Use Newton 1.7 or newer, OVStage 0.3, and optional
OVRTX 0.6 with the exact requirements listed below. For application-owned
stages, populate with `population_desc()` or equivalent schema metadata.
Rendering examples now disable debug synchronization by default; pass
`--render-debug-sync true` when diagnosing rendering problems.

### Added

- Select body pose/velocity, single-DOF joint state, and drive targets with `update_from_ovstage(attributes=...)`. Target-only updates preserve live state; every selection preserves unselected arrays and unmapped joint coordinates.
- Add `--render-debug-sync true|false` to the rendering examples, with debug synchronization disabled by default.
- Import `MjcJointAPI` dynamics and registered MuJoCo solver parameters for revolute, prismatic, and D6 joints.
- Import MuJoCo Connect/Weld body equalities and scalar joint equalities, preserving their parameters through `population_desc()`.
- Add `using-ovnewton` and `physical-ai-physics-selection` agent skills and evaluation datasets to the public source distribution.
- Add `population_desc()` to preserve composed schema names for unsupported-physics validation, including schemas without a registered plugin.

### Changed

- Require Pillow 13.0.0 or newer for the examples extra to include upstream security fixes.
- Reuse fixed publication queries and ingress path lists across runtime updates instead of rebuilding them each time.
- Construct the OVStage stage before the OVRTX renderer in the shipped examples, following the OVStage 0.3 startup requirement.
- Require Newton 1.7.0 or newer, OVStage 0.3.0.1791986420, and optional OVRTX `>=0.6.0.1791947691,<0.7`. Dependencies remain available from public package sources.
- Require scalar-component OVStage reads and remove the Newton 1.6 compatibility paths. Python 3.11 through 3.14 remain supported.
- Populate application-owned stages with `population_desc()` or equivalent schema metadata; use GPU incremental hierarchy computation when rendering, including with CPU physics.

### Fixed

- Keep rigid-body collision filters from suppressing collisions with separately owned nested bodies.
- Accept joint endpoints on existing static prims as world anchors while retaining invalid-target checks.
- Advance path reads past empty or deleted groups without reprocessing or releasing a group twice.
- Initialize interactive cameras from the current sealed scene state so the basic viewer starts after its first simulation step with OVStage 0.3.
- Preserve logical rows and validate contiguous tensor layouts with OVStage 0.3.
- Preserve unsupported MuJoCo schema rejection when OVStage omits unregistered API names.
- Apply `PhysicsMassAPI` so the rigid-body example uses its authored 1 kg masses, and explicitly select compliant VBD contacts.
- Resolve the Newton 1.7 importer dependencies on ARM64/Python 3.13 with USD Exchange 3.0.1.

## [0.2.0] - 2026-10-08

### Added

- Include NVIDIA's security reporting policy in the source distribution.
- Add an example that renders several Ant robots using assets from the installed Newton package, including editable source installations.
- Ship PEP 561 typing metadata and complete annotations for the public ovnewton interface.
- Expose `ovnewton.__version__` from installed distribution metadata.
- Add `register_usd_schemas()` for registering the Newton and PhysX definitions consumed by ovnewton before the first USD schema read.
- Expose rigid-body mass, inertia, and local center-of-mass position through `StageBinding.read()` without copying Newton model arrays.
- Add split rigid-body pose, velocity, and optional acceleration fields to `StageBinding.read()`.
- Expose the effective physics-scene gravity through `StageBinding.read()`.
- Add per-body shape counts, friction, and restitution output reads.
- Add articulation-root pose and velocity output reads using the authored root prim as identity.
- Add shared per-axis joint position and velocity reads with ovphysx-compatible angular units.
- Add `add_ovstage()` so applications can configure a Newton model builder,
  populate it from ovstage, prepare it for a solver, and finalize it themselves.
- Add a VBD option to the basic and Ant examples.
- Add current joint targets and model-backed joint properties to output reads.
- Add explicit object-type selection and report the supported `ALL` output scope.

### Changed

- Document structured import diagnostics and leave logging configuration application-owned.
- Support Python 3.11 through 3.14. Python 3.10 is no longer supported.
- Require fast-simplification 0.1.13 or newer.
- Require Newton 1.6.0 or newer, including development and examples extras, and Warp 1.17.0 or newer. Update gravity, mimic-coefficient, and site-dimension handling for newer Newton versions.
- Use ovstage's DLPack adapter for device-resident publication and reusable read index buffers.
- Use Newton's `CollisionPipeline` in the shipped example.
- Require `ovstage>=0.2.0.377349,<0.3` and optional `ovrtx>=0.5.0.377615,<0.6` for schema registration, profile-driven stage metadata, scalar asset-path pairs, and direct device-resident body transform publication.

### Fixed

- Preserve USD collision-group filter rules, including inverted, merged, and overlapping groups, without overwriting Newton's default collision group.
- Fix missing degrees/s-to-radians/s conversion for initial angular joint velocities, including merged D6 joints. This matches the units expected by both Newton 1.6.0 and newer versions.
- Use Newton's per-joint mimic API when available and retain the Newton 1.6 fallback. Reject mimics involving joints merged into D6 because Newton's mimic APIs cannot target individual source axes after merging.
- Keep RTX viewer presentation on CUDA when physics runs on CPU.
- Honor distance-joint bounds, matching Newton 1.6's `add_usd()` importer.
- Allow unused visual prims with incomplete populated ancestry without weakening physics hierarchy validation.
- Keep the shipped cartpole articulation discoverable by Newton's `add_usd()` importer with OpenUSD 25.x.

## [0.1.1] - 2026-08-04

### Added

- Publish the initial public ovnewton release, including its wheel, source distribution, and documentation.
