<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Support

## Requirements

The supported Python range is 3.11 through 3.14. The NumPy and SciPy floors are 1.26.0 and 1.11.2 on Python 3.11 and 3.12, 2.1.0 and 1.14.1 on Python 3.13, and NumPy 2.3.2 and SciPy 1.16.1 on Python 3.14. The fast-simplification floor is 0.1.13. Ovstage is pinned to the exact qualified build declared in `pyproject.toml`. Newton 1.8.0 or newer is required and installs from PyPI; the wheel metadata records the supported range. Newton USD schemas 0.5 and 0.6 are supported. The Warp floor remains 1.17. Release validation checks the declared dependency floors using the built ovnewton wheel on Linux x86-64 and aarch64.

Supported non-rendering workflows can run entirely on the CPU. OVRTX rendering requires a compatible NVIDIA RTX-capable GPU and supported driver, even when physics runs on the CPU.

## Supported

`ovnewton` supports:

- rigid bodies and articulations
- limited canonical cable, cloth, and tetrahedral volume inputs; see [deformable loading](deformables.md)
- sphere, cube, capsule, cylinder, cone, and mesh colliders
- mass, inertia, center of mass, physics materials, and collision filtering
- revolute, prismatic, fixed, spherical, distance, and D6 joints
- MuJoCo Connect/Weld body equalities with explicit {ref}`population selection <mujoco-body-equalities>`
- MuJoCo scalar joint equalities, including polynomial coefficients and disabled constraints
- `MjcJointAPI` parameters on revolute, prismatic, and D6 joints, including merged scalar axes
- Scalar MuJoCo position/velocity actuators with control and force limits (registered MuJoCo profile)
- initial body and joint state
- body state and single-axis joint state updates
- selected read-only Newton output on CPU and CUDA

Malformed data and unsupported physics that would change the model raise an error during attachment instead of creating an incorrect model.

Authored angular joint positions and velocities are converted from USD degrees
and degrees/s to Newton radians and radians/s, both at import and during runtime
updates, matching Newton's `add_usd()` importer. Linear joint state is unchanged.

## Known limits

- Native USD physics instances are rejected.
- MuJoCo body-equality endpoints must target rigid bodies directly or omit a target for world. Joint equalities require scalar revolute/prismatic joints, not source axes merged into D6. Site endpoints are not supported. Full equality semantics require `SolverMuJoCo`.
- `MjcJointAPI` is limited to revolute, prismatic, and D6 joints. Native D6 reference coordinates are rejected. MuJoCo-specific solver parameters require attribute registration before import; see {ref}`MuJoCo joint parameters <mujoco-joint-parameters>`.
- Physics `PointInstancer` content is rejected.
- Per-axis runtime state for D6 joints is not synchronized.
- Newton limitation: reversed multi-axis joints can preserve direction exactly only for translations, or one rotation with translations parallel to its axis. Other reversed combinations emit a warning and retain their authored axes. Author the rigid body as `physics:body1` against world to avoid this limitation.
- Newton limitation: its mimic APIs operate on whole joints and cannot target individual source axes after D6 merging. ovnewton rejects enabled mimics involving such merges with `UnsupportedPhysicsError` to preserve authored physics. This is intentionally stricter than Newton's USD importer, which can warn and continue when joint dimensions match.
- Inherited and collection-based material bindings have partial support.
- Cables, cloth, and volumes support only the documented canonical subset. Surface cloth supports enabled, non-kinematic bodies with fixed rigid world transforms, constant thickness, and density-derived mass; its material inputs are limited to density, stretch stiffness, and bend stiffness. Attachments, separate render skins, and broader deformable parity remain unsupported.
- Some recoverable inputs are ignored or normalized with a warning.
