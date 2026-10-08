<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Support

## Requirements

The supported Python range is 3.11 through 3.14. The NumPy and SciPy floors are 1.26.0 and 1.11.2 on Python 3.11 and 3.12, 2.1.0 and 1.14.1 on Python 3.13, and NumPy 2.3.2 and SciPy 1.16.1 on Python 3.14. The fast-simplification floor is 0.1.13. Ovstage supports `>=0.2.0.377349,<0.3`, and the optional examples require OVRTX `>=0.5.0.377615,<0.6`. The lockfile retains the tested versions, while installations may select newer versions within these ranges. The remaining dependency floors are Newton 1.6 and Warp 1.17. CI tests the minimum versions with the built ovnewton wheel on Linux x86-64.

Supported non-rendering workflows can run entirely on the CPU. OVRTX rendering requires a compatible NVIDIA RTX-capable GPU and supported driver, even when physics runs on the CPU.

## Supported

`ovnewton` supports:

- rigid bodies and articulations
- a limited subset of canonical surface-cloth inputs
- sphere, cube, capsule, cylinder, cone, and mesh colliders
- mass, inertia, center of mass, physics materials, and collision filtering
- revolute, prismatic, fixed, spherical, distance, and D6 joints
- initial body and joint state
- body state and single-axis joint state updates
- selected read-only Newton output on CPU and CUDA

Malformed data and unsupported physics that would change the model raise an error during attachment instead of creating an incorrect model.

Authored angular joint positions and velocities are converted from USD degrees
and degrees/s to Newton radians and radians/s, both at import and during runtime
updates. Initial angular velocities intentionally differ from Newton 1.6.0's
`add_usd()` importer, which left them unconverted. Linear joint state is unchanged.

## Known limits

- Native USD physics instances are rejected.
- Physics `PointInstancer` content is rejected.
- Per-axis runtime state for D6 joints is not synchronized.
- Newton limitation: its mimic APIs operate on whole joints and cannot target individual source axes after D6 merging. ovnewton rejects enabled mimics involving such merges with `UnsupportedPhysicsError` to preserve authored physics. This is intentionally stricter than Newton's USD importer, which can warn and continue when joint dimensions match.
- Inherited and collection-based material bindings have partial support.
- Surface cloth supports enabled, non-kinematic bodies with identity world transforms, constant thickness, and density-derived mass. Material inputs are limited to density, stretch stiffness, and bend stiffness.
- Newton actuators, cables, and volume deformables are not supported.
- Some recoverable inputs are ignored or normalized with a warning.
