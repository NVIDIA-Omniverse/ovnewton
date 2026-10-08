<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Changelog

Notable user-facing changes to ovnewton.

## [0.2.0]

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
