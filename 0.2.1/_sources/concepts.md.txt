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

(mujoco-body-equalities)=
## MuJoCo equalities

For `MjcEqualityConnectAPI` on spherical joints, `MjcEqualityWeldAPI` on
fixed joints, and `MjcEqualityJointAPI` on revolute/prismatic joints, populate
with `ovnewton.population_desc()`. The ordinary physics
domain alone can omit MuJoCo properties and unregistered schema names, so the
importer cannot reliably detect an equality after those have been discarded.
The description includes physics data,
the equality parameters, and composed API-schema names, including unregistered
schema names. No additional schema package is required for authored equality
properties; unauthored parameters use Newton's MuJoCo defaults.

```python
ovnewton.register_usd_schemas()
population.open_usd_with_desc(
    stage, scene_path, ordinal=1, time=population.DEFAULT_TIME,
    descs=ovnewton.population_desc(),
)
stage.advance_write_floor(ordinal=1).wait()
```

Descriptions for rendering or other consumers can be passed in the same `descs`
list. Then use `add_ovstage()` or `attach_ovstage()` normally. Body-equality endpoints
must refer directly to rigid bodies; omitting an endpoint denotes the world.
Disabled equalities are omitted, matching Newton's default USD import policy.

The importer creates a ball/fixed loop joint and preserves the authoritative
MuJoCo equality row, including solver reference/impedance, weld relative pose,
and torque scale. Use `SolverMuJoCo` for these complete semantics. Other solvers
only see the projected loop joint, subject to their own loop-closure support;
successful import does not guarantee equivalent dynamics with another solver.
As with other orphan joints, finalize a caller-owned builder using
`skip_validation_joints=import_result.has_orphan_joints`.

Joint equalities couple a follower joint to the scalar joint referenced by
`newton:mimicJoint`. They preserve `newton:mimicCoef0`, `newton:mimicCoef1`,
`mjc:coef2` through `mjc:coef4`, and solver reference/impedance parameters.
The constant offset is converted from degrees to radians for a revolute follower;
the higher-order coefficients retain Newton's joint-coordinate convention.
Deprecated `mjc:target`, `mjc:coef0`, and `mjc:coef1` are fallbacks when the modern
properties are absent; the legacy angular offset is already in radians.
OVStage does not distinguish authored defaults from schema fallbacks. If a
modern coefficient resolves to its default (`0` for `mimicCoef0`, `1` for
`mimicCoef1`) while a different legacy coefficient is present, ovnewton rejects
the ambiguous input. Migrate the legacy values to the modern properties and
remove the old coefficient aliases, converting a legacy revolute offset from
radians to degrees.
`newton:mimicEnabled = false` preserves a disabled equality and its disabled
projection, while `physics:jointEnabled = false` omits the joint and equality.

As in Newton's USD importer, joint equalities create a linear mimic projection
and an authoritative MuJoCo equality row. `SolverMuJoCo` consumes the full
polynomial once, without a duplicate linear constraint. Other solvers only see
the linear projection, subject to their own mimic support. Both endpoints must
remain scalar revolute/prismatic joints; multi-DOF joints and source joints
merged into D6 are rejected because they cannot identify an individual coordinate.

Body-equality site endpoints remain unsupported.
Changing equality parameters after import requires rebuilding the model.

(mujoco-joint-parameters)=
## MuJoCo joint parameters

Use `population_desc()` when importing `MjcJointAPI` on revolute, prismatic,
or D6 joints. Armature, passive damping, and friction loss populate Newton's
ordinary joint fields. MuJoCo angular damping is already per radian and is not
scaled like USD's `newton:damping`.

For limit margins, limit/friction solver parameters, passive stiffness,
reference coordinates, and actuator gravity compensation, register the solver
attributes before importing:

```python
builder = newton.ModelBuilder()
newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
ovnewton.add_ovstage(builder, stage)
model = builder.finalize()
binding = ovnewton.attach_ovstage(stage, model=model)
```

Authored solver settings without the corresponding registration raise an error.
Scalar axes merged into D6 keep their own settings. `mjc:ref` and `mjc:springref`
on revolute joints follow the PhysicsScene's `mjc:compiler:angle` (`degree` by
default, or `radian`); linear references are unchanged. Joint limits are shifted
into Newton's reference-relative coordinates, matching its USD importer.
Native D6 joints have no equivalent reference-coordinate semantics and reject
these two properties.

Explicit non-default Newton joint parameters take precedence over MuJoCo values.
When both APIs are applied and a zero Newton value could be either authored or
a schema fallback that conflicts with MuJoCo, import fails instead of guessing.
With registered MuJoCo solver fields, Newton's `-inf` limit defaults have the
same ambiguity: an authored sentinel selects force-space gains, while a schema
fallback permits MJCF defaults. Use one parameter namespace, explicit finite
Newton limit gains, or `mjc:solreflimit` to resolve that case.
Changing these settings after import requires rebuilding the model.
Actuator-force ranges (`mjc:actuatorfrcrange`, `mjc:actuatorfrclimited`) and
`mjc:springdamper` are not imported yet and raise an error if authored.

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
- `update_from_ovstage(state, control)` resynchronizes all supported state and targets from the latest sealed data.
- `update_from_ovstage(state, control, attributes=("jointPositionTarget",))` updates only existing single-DOF position targets, preserving live state. `jointVelocityTarget`, `body_q`, `body_qd`, `jointPosition`, and `jointVelocity` use the same selector. Explicit selections preserve other fields and do not run kinematics; use full resynchronization to reconstruct body-derived joint coordinates.

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
