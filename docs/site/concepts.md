<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Concepts

## Schema registration

Call `register_usd_schemas()` before the first USD schema read in a process,
including an ovstage population or export call. It registers Newton's USD schema
definitions with ovstage without loading generated schema code. It also
registers the PhysX compatibility definitions already consumed by ovnewton.
Registration is process-wide, irreversible, and safe to repeat.

## Model construction and binding

`add_ovstage(builder, stage, *, ordinal=1) -> OvstageImportResult` adds the
contents of one populated ovstage to an application-owned `ModelBuilder`. It
does not finalize the builder or keep a connection to the stage. The
application can register solver attributes before the call and perform steps
such as VBD coloring after the call.

The current implementation supports one ovstage per builder. The builder may
have defaults and custom attribute definitions configured, but it must not
already contain bodies, joints, shapes, or other model data. USD assets that
belong to the same simulation should be composed into the ovstage before this
call.

`attach_ovstage(stage, *, model=None, ordinal=1) -> StageBinding` creates the
persistent connection used by `update_to_ovstage()` and
`update_from_ovstage()`. When `model` is supplied, the function binds that
finalized model without importing or finalizing anything. When `model` is
omitted, the function creates a `ModelBuilder`, adds the ovstage contents,
finalizes the model, and creates the binding. Call `add_ovstage()` directly
when the application needs to configure the builder before finalization.

## Ordinals

Every stage write has an ordinal chosen by the application. An ordinal orders writes and acts as a consistency fence. It is not a frame identity, even when the application uses one ordinal per frame.

Written data becomes visible after the application advances the stage's write floor. The stage stores the latest value. It does not store a frame history.

- `add_ovstage(builder, stage, ordinal=1) -> OvstageImportResult` reads data at or below the conventional initial population ordinal.
- `attach_ovstage(stage, model=None, ordinal=1) -> StageBinding` uses that ordinal when it builds a model internally.
- `attach_ovstage(stage, model=model, ordinal=1) -> StageBinding` uses that ordinal to validate and bind an application-supplied model.
- `update_to_ovstage(state, ordinal=n)` writes simulation state at ordinal `n`.
- `update_from_ovstage(state, control)` reads the latest sealed data.

An explicit read ordinal cannot be newer than the sealed write floor. Such a read raises an error instead of returning unsealed data.

## Labels

Newton uses body and joint indices. The stage uses prim paths. `body_label` and `joint_label` map the indices to those paths.

`add_ovstage` fills the labels while populating a builder. A model supplied by
the application must have matching stage prim paths. All runtime
synchronization uses these labels, and the binding validates them when it is
created.

## Topology

`StageBinding` prepares its buffers and path mappings when it is created. Create a new binding after changing the model or stage structure.

## Diagnostics and observability

Recoverable import decisions are emitted through the standard `ovnewton` Python logger. The library does not install handlers, set levels, add filters, or register callbacks. Logging configuration is process-wide and application-owned:

```python
import logging

logger = logging.getLogger("ovnewton")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
```

Applications can use normal logging filters. Every diagnostic record contains:

- `diagnostic_code`: a stable machine-readable decision name;
- `prim_path`: the affected stage path, or `None`.

The standard `levelno` and `created` fields carry severity and timestamp. Disabling logging does not change exception behavior.

ovnewton does not configure the separate Newton, Warp, or ovstage logging facilities. Applications integrating those dependencies own their configuration as well.

ovnewton does not emit metrics or implement capture and replay. The application owns the solver and stepping loop, so it also owns performance counters, OTel integration, and capture of the composed Newton/ovstage workflow. The stage transport is a latest-value interchange surface, not a replay history.

See [Support](support.md) for the current feature set and known limits.
