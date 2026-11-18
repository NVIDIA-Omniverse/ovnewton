<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Example assets

- `scene_deformables.usda` is an NVIDIA-authored combined scene: cables on the left, cloth over a static sphere in the center, and a tetrahedral volume on the right. It is run by `example_ovnewton_deformables` and is included in the wheel. The geometry is self-contained and has no external asset references.
- `scene_rigid_bodies.usda` is an NVIDIA-authored rigid-body scene.
- `scene_cartpole.usda` is derived from Newton's [`cartpole.usda`](https://github.com/newton-physics/newton/blob/3c8d11761caff6efcf4a345b4f1b38b35e043b80/newton/examples/assets/cartpole.usda). The joint definitions and body mass properties are retained. The articulation schema, initial pole poses, colors, lighting, and render setup are adapted for ovnewton and ovrtx. Newton's copyright is retained in the asset header and in the third-party notices shipped with the distribution.

These scenes are licensed under Apache-2.0.
