<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Support

## Requirements

The minimum supported releases are ovstage 0.1.0.346039, NumPy 1.26.0, and SciPy 1.11.2. The release workflow tests these versions with the built ovnewton wheel and public dependencies from PyPI.

## Supported

`ovnewton` supports:

- rigid bodies and articulations
- sphere, cube, capsule, cylinder, cone, and mesh colliders
- mass, inertia, center of mass, physics materials, and collision filtering
- revolute, prismatic, fixed, spherical, distance, and D6 joints
- initial body and joint state
- body state and single-axis joint state updates
- selected read-only Newton output on CPU and CUDA

Malformed data and unsupported physics that would change the model raise an error during attachment instead of creating an incorrect model.

## Known limits

- Native USD physics instances are rejected.
- Physics `PointInstancer` content is rejected.
- Per-axis runtime state for D6 joints is not synchronized.
- Inherited and collection-based material bindings have partial support.
- Newton actuators, soft bodies, cables, cloth, and volumes are not available because ovstage does not expose the required data.
- Some recoverable inputs are ignored or normalized with a warning.
