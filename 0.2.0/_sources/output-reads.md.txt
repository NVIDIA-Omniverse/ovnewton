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
    attributes=("body_q", "body_qd", "mass"),
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

## Select an object type

Pass `object_type` without a path query to select every bound object of one type:

```python
import ovnewton

query = binding.query(
    object_type=ovnewton.SimObjectType.RIGID_BODY,
    scope=ovnewton.ObjectScope.ALL,
)
```

The supported types are `RIGID_BODY`, `ARTICULATION_LINK`, `ARTICULATION_JOINT`, `ARTICULATION`, and `PHYSICS_SCENE`. The first four names and numeric values match the ovphysx output-read API. `PHYSICS_SCENE` is an ovnewton extension for reading effective gravity.

`RIGID_BODY` selects bodies outside an articulation. `ARTICULATION_LINK` selects bodies in an articulation. `ARTICULATION_JOINT` selects authored articulation joints with readable coordinates or velocities; it excludes generated free joints. `ARTICULATION` selects prims with `PhysicsArticulationRootAPI`. `PHYSICS_SCENE` selects the effective physics scene.

A type can also narrow `paths` or an ovstage query. An explicit path with the wrong type is an error. An ovstage query may match other prims; those prims are ignored.

Each group from a typed query reports that type through `group.object_type`, as ovphysx does. A query without `object_type` may combine several types, so its groups report `None`.

Only `ObjectScope.ALL` is supported. It is a reusable topology snapshot that remains valid until the binding's structure changes. `ObjectScope.ACTIVE` raises `NotImplementedError`. Ovnewton does not yet expose changed-since selection.

## Native output contract

Native fields return values as Newton stores them, with the same units, frames, component order, and meaning. The shared `jointPosition` and `jointVelocity` fields convert angular axes to degrees and degrees per second. See Newton's [State API](https://newton-physics.github.io/newton/1.6.0/api/_generated/newton.State.html), [conventions](https://newton-physics.github.io/newton/1.6.0/concepts/conventions.html), and [articulation model](https://newton-physics.github.io/newton/1.6.0/concepts/articulations.html) for details.

| Attribute | Object | Ovnewton group layout |
| --- | --- | --- |
| `body_q` | Rigid body or articulation link | one `float32x7` value per body |
| `body_qd` | Rigid body or articulation link | one `float32x6` value per body |
| `position` | Rigid body or articulation link | one world-space `float32x3` value per body |
| `orientation` | Rigid body or articulation link | one world-space XYZW `float32x4` value per body |
| `linearVelocity` | Rigid body or articulation link | one world-space `float32x3` value per body |
| `angularVelocity` | Rigid body or articulation link | one world-space `float32x3` value per body |
| `linearAcceleration` | Rigid body or articulation link | one world-space `float32x3` value per body, when `State.body_qdd` is allocated |
| `angularAcceleration` | Rigid body or articulation link | one world-space `float32x3` value per body, when `State.body_qdd` is allocated |
| `gravity` | Effective physics scene | one stage-space `float32x3` value |
| `mass` | Rigid body or articulation link | one `float32` value per body |
| `inertia` | Rigid body or articulation link | one row-major `float32x9` inertia matrix per body |
| `centerOfMassPosition` | Rigid body or articulation link | one local-frame `float32x3` value per body |
| `shapeCount` | Rigid body or articulation link | one `int32` value per body |
| `friction` | Rigid body or articulation link | one padded `float32xN` row per body |
| `restitution` | Rigid body or articulation link | one padded `float32xN` row per body |
| `rootPosition` | Articulation root | one world-space `float32x3` value per articulation |
| `rootOrientation` | Articulation root | one world-space XYZW `float32x4` value per articulation |
| `rootLinearVelocity` | Articulation root | one world-space `float32x3` value per articulation |
| `rootAngularVelocity` | Articulation root | one world-space `float32x3` value per articulation |
| `jointPosition` | Articulation joint | one or more per-axis groups; angular axes use degrees |
| `jointVelocity` | Articulation joint | one or more per-axis groups; angular axes use degrees per second |
| `jointPositionTarget` | Articulation joint | current position targets from `Control`; angular axes use degrees |
| `jointVelocityTarget` | Articulation joint | current velocity targets from `Control`; angular axes use degrees per second |
| `jointStiffness` | Articulation joint | position-drive stiffness per axis; angular axes use force per degree |
| `jointDamping` | Articulation joint | velocity-drive damping per axis; angular axes use force per degree per second |
| `jointLimit` | Articulation joint | interleaved low/high values per axis; angular axes use degrees |
| `jointMaxVelocity` | Articulation joint | one limit per axis; angular axes use degrees per second |
| `jointMaxForce` | Articulation joint | one effort limit per axis |
| `jointArmature` | Articulation joint | one armature value per axis |
| `jointFriction` | Articulation joint | one friction-effort value per axis |
| `joint_q` | Joint | one or more `float32xN` groups, partitioned by coordinate width `N` |
| `joint_qd` | Joint | one or more `float32xN` groups, partitioned by velocity width `N` |

`body_q` is `[px, py, pz, qx, qy, qz, qw]`: the body origin's world pose. `body_qd` is `[vx, vy, vz, wx, wy, wz]`: center-of-mass linear velocity followed by angular velocity, both in world space. Native `joint_q` and `joint_qd` values use Newton generalized coordinates. Their angles use radians and angular rates use radians per second. Free and distance joint velocities use the parent joint's frame. D6 joints store linear coordinates before angular coordinates.

The split pose and velocity fields have the same values as their `body_q` and `body_qd` components. The acceleration fields similarly split `State.body_qdd`; request that optional Newton state field before creating the state. Split fields are produced by a device kernel for the selected rows only and do not make a host round trip.

`gravity` aliases `Model.gravity` for the first populated `PhysicsScene`. It contains the effective direction and magnitude after ovnewton applies scene defaults, stage length units, and `NewtonSceneAPI.gravityEnabled`. A stage without a `PhysicsScene` has no scene output path, even though Newton still has a default gravity value.

For a caller-owned model, `gravity` borrows the row for the bound bodies' Newton world, including the global world. The mapping is resolved when the binding is created. A model with multiple gravity rows must have bound bodies in exactly one world; otherwise a gravity read raises `ValueError`. Other output reads remain available.

`mass`, `inertia`, and `centerOfMassPosition` expose the final values in `Model.body_mass`, `Model.body_inertia`, and `Model.body_com`. Inertia is relative to the center of mass and expressed in the body frame. The center-of-mass position is an offset in the body frame. Reading these values does not normalize units. Each group aliases its model array; no value is copied.

Ovnewton does not expose `centerOfMassOrientation`. Newton stores a complete body-frame inertia matrix rather than a separate principal-axis orientation, so reporting one would require a new shared decomposition contract.

`friction` and `restitution` expose `Model.shape_material_mu` and `Model.shape_material_restitution` in Newton's native shape order. Each row is padded with zeros to the largest `shapeCount` in the query; `dtype.lanes` is that width. Newton stores one friction coefficient, so ovnewton does not report it as distinct `staticFriction` and `dynamicFriction` values. These fields are gathered on the model device.

Material reads support at most 255 shapes per selected body, matching ovstage's DLTensor lane limit. A wider material read raises `ValueError`; `shapeCount` and other body fields remain readable. Selecting only bodies within the limit still works.

The four articulation-root fields report the pose and velocity of the root body. Select the prim that has `PhysicsArticulationRootAPI`, not the root body's path. Ovnewton preserves that authored prim path as the Newton articulation label. The split values are gathered into dense rows on the model device.

`jointPosition` and `jointVelocity` use the shared per-axis convention from ovphysx. Linear axes stay in the model's length units. Angular positions use degrees and angular velocities use degrees per second. Signed values follow the authored USD body order, even when Newton has reordered a world-attached joint. Revolute, prismatic, and scalar-coordinate D6 joints provide `jointPosition`. Spherical position is not returned because Newton stores it as a quaternion, but its three angular velocities are available. These values are converted on the model device. Use `joint_q` and `joint_qd` when an application needs Newton's generalized-coordinate layout, parent-child order, and radians.

Joint targets come from the `Control` passed to the same read; ovnewton does not substitute model defaults. `jointPositionTarget` follows the same supported position axes as `jointPosition`. `jointVelocityTarget` follows `jointVelocity`. Both use authored body order. Joint properties come from the connected model. Angular targets, limits, and maximum velocities use degrees. Angular stiffness and damping are converted from Newton's per-radian values to per-degree values. Effort limits, armature, and friction are unchanged. `jointLimit` stores `[low, high]` for each axis, so a group with `N` axes has `2N` lanes. An unlimited bound is reported as the positive or negative maximum `float32` value.

Newton has one friction-effort value per joint axis, so ovnewton does not expose separate joint static and dynamic friction fields. It also does not expose `jointActuationForce` because `Control.joint_f` is an input, not measured solver output. Newton's target mode does not carry the same force-versus-acceleration meaning as the shared `jointDriveType`, so that field is not exposed either.

The `state` argument may be omitted when a read requests only model-backed fields. State-backed fields still require it. The `control` argument is required only for target fields.

Lengths use the Newton model's units. When ovnewton builds the model, it uses the stage's length units. For example, a centimetre stage returns positions in centimetres and linear velocities in centimetres per second, not metres.

Rod-joint `joint_q` and `joint_qd` values are outside this contract: Newton does not use them as generalized positions and velocities. Read the rod segments' `body_q` and `body_qd` instead.

Generated free joints may have Newton labels instead of USD prim paths. To identify a standalone rigid body by its USD path, select the body and read `body_q` or `body_qd`.

A read may return several groups because joints can have different numbers of coordinates or axes. Each group contains one attribute and one width, so several groups may use the same attribute name. `group.tensor(0).dtype.lanes` gives the width. The raw `DLTensor` has one dimension: `shape[0]` gives the number of rows, and `dtype.lanes` gives the number of values in each row.

`query.attributes` lists the fields available for the selected objects. `query.object_type` and `query.scope` report the selection contract. An unknown attribute is an error. A known attribute returns no group when none of the selected objects provide it. Requesting no attributes also returns no groups. Selecting a path that is not a readable body, articulation root, physics scene, or joint is an error. Fixed joints are not readable because they have no coordinates.

Groups are read-only. Native fields point directly to the supplied Newton state or connected model; split fields own a result buffer. The current `ordinal == 0` and zero-valued metadata are placeholders; they do not identify a simulation step. The application owns Newton stepping, so it must call `read()` with the state and control from one sample and prevent them from changing while the read is being issued. A public tick/time/ordinal contract still needs agreement. Keep borrowed arrays unchanged until every consumer has finished reading them.

`group.data_row_index(i)` maps logical prim `i` to its tensor row. Therefore, `prim_paths[i]` names the prim, and `data_rows[i]` gives its corresponding row in each group tensor. Sparse groups may also expose the row map through `data_index_dlpack()`.

CPU data is ready when `read()` returns. On CUDA, call `read()` on the Warp stream that last wrote the requested state or model arrays, or first make the current stream wait for that work. The returned event covers only work ordered on that stream. A consumer on another CUDA stream must wait for the reported event on its own `consumer_stream`:

```python
if group.cuda_sync.wait_event:
    event = wp.Event(
        device=binding.model.device,
        cuda_event=group.cuda_sync.wait_event,
    )
    consumer_stream.wait_event(event)
```
