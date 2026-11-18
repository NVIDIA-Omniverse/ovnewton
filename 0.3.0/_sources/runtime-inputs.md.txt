<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Update simulation inputs

`StageBinding.write()` replaces selected scalar articulation joint targets
directly in Newton's control buffers. No runtime OVStage write or read is needed.
It accepts `joint_target_q` and `joint_target_qd`, using Newton's native units
and signs: radians for angular position, radians/second for angular velocity,
and native length units for linear motion.

```python
import warp as wp

joint = binding.query(paths=["/Robot/Hinge"])
target = wp.array([0.5], dtype=wp.float32, device=binding.model.device)
binding.write({"joint_target_q": target}, control=control, query=joint)
```

The query must select only scalar articulation joints. Supply one float32 value
per selected joint, in the corresponding public read's logical prim order.
That order is not necessarily the input path order or the backing tensor's row
order. Use the read group's row map when consuming `joint_q` or `joint_qd`
on the GPU. The native target fields are also available through
`binding.read(control=control, query=joint, attributes=("joint_target_q",))`.

Target payloads stay on the model device. Shape, dtype, device, query ownership,
and finite values are checked before any target is changed. A small validation
flag is copied to the host. Inputs are copied on the device, so they can be reused
after return; unselected targets remain unchanged. Calls are synchronous and do
not support CUDA graph capture. Use the stream that produced the input arrays,
or establish a dependency on that stream before calling.

This initial API writes commands, not measured forces or simulation state.
It does not step the solver, enable drives, or change stiffness, damping,
activation, or topology. Configure the solver and drives first and perform
writes between steps, without concurrent consumers of the target buffers.
Body-state writes, multi-axis joints and model-property writes are separate work.

`update_from_ovstage()` remains the explicit stage adapter. Its ordinal,
USD-unit conversion and resynchronization behavior is unchanged. A USD edit
changes Newton only when the application invokes that adapter.

Queries belong to one binding and fixed topology. After rebuilding a model,
create a new binding and queries; an old query is rejected by the new binding.
