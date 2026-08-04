<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Read simulation output

Use `StageBinding.read()` when an application needs Newton data without writing it to ovstage.

Prepare a selection once, then reuse it. In the example, `stage` is the attached ovstage stage, `binding` is its stage binding, and `state` is the current Newton state from the simulation loop. Replace the path with a body path from your stage.

```python
import ovstage
import warp as wp

query = binding.query(paths=["/World/Robot/base"])
result = binding.read(
    state,
    query=query,
    attributes=("body_q", "body_qd"),
)

with ovstage.PathDictionary(stage) as path_dictionary:
    for group in result.groups:
        attribute = path_dictionary.token_to_string(group.attribute)
        prim_paths = path_dictionary.get_path_strings(group.prim_list)
        tensors = [
            wp.from_dlpack(group.dlpack(i))
            for i in range(group.tensor_count)
        ]
        data_rows = [
            group.data_row_index(i)
            for i in range(group.prim_count)
        ]
```

The available attributes are `body_q`, `body_qd`, `joint_q`, and `joint_qd`. A read may return more than one group because joints can have different coordinate widths. Each group identifies one attribute and one coordinate width, so more than one group may have the same attribute.

Groups are read-only. Some groups point directly at the supplied Newton state. Do not change or reuse that state until the consumer has finished.

`group.data_row_index(i)` maps logical prim `i` to its tensor row. Therefore, `prim_paths[i]` names the prim, and `data_rows[i]` gives its corresponding row in each group tensor. Sparse groups may also expose the row map through `data_index_dlpack()`.

CPU data is ready when `read()` returns. A consumer on another CUDA stream must wait for the reported event on its own `consumer_stream`:

```python
if group.cuda_sync.wait_event:
    event = wp.Event(
        device=binding.model.device,
        cuda_event=group.cuda_sync.wait_event,
    )
    consumer_stream.wait_event(event)
```
