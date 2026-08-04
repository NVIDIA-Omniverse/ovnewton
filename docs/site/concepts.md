<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Concepts

## Ordinals

Every stage write has an ordinal chosen by the application. An ordinal orders writes and acts as a consistency fence. It is not a frame identity, even when the application uses one ordinal per frame.

Written data becomes visible after the application advances the stage's write floor. The stage stores the latest value. It does not store a frame history.

- `attach_ovstage(stage, ordinal=1)` reads data at or below the conventional initial population ordinal.
- `update_to_ovstage(state, ordinal=n)` writes simulation state at ordinal `n`.
- `update_from_ovstage(state, control)` reads the latest sealed data.

An explicit read ordinal cannot be newer than the sealed write floor. Such a read raises an error instead of returning unsealed data.

## Labels

Newton uses body and joint indices. The stage uses prim paths. `body_label` and `joint_label` map the indices to those paths.

`attach_ovstage` fills the labels when it builds a model. A model supplied by the application must already have matching stage prim paths. All runtime synchronization uses these labels, and the binding validates them when it is created.

## Topology

`StageBinding` prepares its buffers and path mappings when it is created. Create a new binding after changing the model or stage structure.

See [Support](support.md) for the current feature set and known limits.
