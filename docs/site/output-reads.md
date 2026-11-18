<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Read simulation output

{meth}`ovnewton.StageBinding.read` exposes selected Newton state, control and
model data without stepping simulation or publishing values to ovstage. Field
names follow two rules:

- **Newton fields.** A snake_case name, such as `body_q`, `joint_q` or
  `joint_target_ke`, is the name of a Newton array. The field returns Newton's
  values unchanged: Newton's units, frames and component order. The application
  owns any conversion.
- **Standard fields.** A camelCase name, such as `position` or `jointPosition`,
  is a standard field. Per-axis joint fields use the units of the matching USD
  joint attributes: degrees on angular axes, and the authored body order. All
  other values keep Newton's units.

The [read field reference](read-fields.md) lists every field with its objects,
source arrays, units, frame, layout, device and storage.
{func}`ovnewton.read_fields` returns the same declarations as
{class}`ovnewton.ReadField` objects.

## Select and discover

Prepare a reusable {class}`ovnewton.Query` from explicit paths, an ovstage query,
or an {class}`ovnewton.SimObjectType`. See the enum for object types and their
meaning. Types may also narrow a path/query selection. Invalid explicit paths
raise an error; unrelated prims in an ovstage query are ignored.

`query.attributes` contains field-name tokens for the selected topology. Resolve
them through `ovstage.PathDictionary`. This is candidate discovery, not a check
that a particular state allocates optional outputs. Only
{attr}`ovnewton.ObjectScope.ALL` is supported; active and changed-since queries
are unavailable. Recreate the binding and queries after structural changes.

This CPU example uses the packaged cartpole scene. The code below is executed
by the documentation tests, including checks of native values and row mappings.

```{testcode} read-output
from importlib.resources import files

import newton
import numpy as np
import ovnewton
import ovstage
import warp as wp
from ovstage import population

ovnewton.register_usd_schemas()
with wp.ScopedDevice("cpu"), ovstage.Stage() as stage:
    scene = str(files("ovnewton.examples").joinpath("assets/scene_cartpole.usda"))
    population.open_usd_with_desc(
        stage, scene, ordinal=1, time=population.DEFAULT_TIME,
        descs=(ovnewton.population_desc(),),
    )
    stage.advance_write_floor(1).wait()
    binding = ovnewton.attach_ovstage(stage)
    model = binding.model
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    query = binding.query(object_type=ovnewton.SimObjectType.ARTICULATION_JOINT)

    with ovstage.PathDictionary(stage) as pd:
        fields = {pd.token_to_string(token): token for token in query.attributes}
        result = binding.read(state, query=query, attributes=(fields["joint_q"],))
        joints = {path: j for j, path in enumerate(model.joint_label)}
        starts = model.joint_q_start.numpy()
        for group in result.groups:
            paths = pd.get_path_strings(group.prim_list)
            for column in group.columns:
                width = int(column.tensor(0).dtype.lanes)
                rows = column.array(0).reshape(-1, width)  # read-only CPU view
                for i, path in enumerate(paths):
                    row = rows[column.data_row_index(i)]
                    j = joints[path]
                    np.testing.assert_array_equal(row, state.joint_q.numpy()[starts[j]:starts[j + 1]])
                    assert not row.flags.writeable
```

Omit `query` to select all bound output objects. Pass `control=control` for drive
targets or commanded joint effort; control-only and model-only reads may omit
`state`. Unknown fields raise an error;
known fields with no compatible selected objects produce no groups.

## Available fields

The [read field reference](read-fields.md) is the complete list. It is
generated from `contracts/read_fields.json`, the same declarations that drive
the runtime and {func}`ovnewton.read_fields`. Tests check every declaration
against actual reads. The sections below explain the main field families. All
value components are `float32` except `shapeCount` (`int32`). Joint state,
control and DOF-property source arrays must be contiguous `float32` arrays.
Other layouts raise an error rather than being misread.

Body fields apply to rigid bodies and articulation links. `root*` fields apply
to authored articulation-root prims, `gravity` to the effective physics scene,
and joint fields to bound joint paths. Convenience joint fields require an
articulation joint; generated free-joint labels are not generally USD prim paths.
Fixed joints have no readable coordinates; rod generalized coordinates are
outside this contract. Read rod-segment body state instead.

**Bodies and scene.** `body_q` is world body-origin XYZ followed by an XYZW
quaternion. `body_qd` is world COM linear XYZ followed by angular XYZ velocity.
Split pose/velocity fields preserve those meanings; acceleration splits optional
`State.body_qdd` with the same world/COM reference. The caller must allocate it
and use a solver that populates it. Lengths use model units (stage units for an
ovnewton-built model), time is seconds, mass uses model mass units, angular
velocity/acceleration uses rad/s and rad/s². Inertia is a row-major body-frame
matrix about COM; `centerOfMassPosition` is a body-frame offset. `root*` uses the
root body's origin/COM and the authored root prim's identity. `gravity` borrows
the resolved `Model.gravity` world row; no scene or an ambiguous multi-world
mapping prevents this read.

**Shapes.** Counts come from `Model.shape_body`. Friction and restitution are
unitless, ordered by Newton shape index within each body and zero-padded to the
maximum selected shape count. Zero shapes produces no material column. More than
255 shapes on a selected body raises an error for material reads.

**Joints.** `joint_q` uses `Model.joint_q_start` coordinate spans; `joint_qd`
uses `Model.joint_qd_start` velocity-DOF spans. Prismatic/revolute joints have
one displacement/angle and one rate. Ball joints have an XYZW quaternion and
three angular velocity components, not quaternion derivatives. Free/distance
joints have XYZ + XYZW relative joint pose and six velocity DOFs, with child-COM
linear velocity then angular velocity in the joint parent frame. D6 stores
configured linear axes before angular coordinates/DOFs, following Newton's
rotation sequence rather than a generic XYZ Euler convention. Resolve paths
through `Model.joint_label`; inspect `joint_type`, `joint_dof_dim`, `joint_axis`
and parent/child joint frames to interpret axes. See Newton's
[articulation conventions](https://newton-physics.github.io/newton/1.7.0/concepts/articulations.html).

**Per-axis joint fields.** The angular unit of a standard joint field depends
on the quantity:

- `jointPosition`, `jointVelocity`, `jointPositionTarget`,
  `jointVelocityTarget`, `jointLimit` and `jointMaxVelocity` use degrees or
  degrees/s.
- `jointStiffness` and `jointDamping` use effort/degree and effort·s/degree.
- `jointMaxForce`, `jointArmature` and `jointFriction` are not converted.

Linear axes, such as a prismatic joint's axis, are never converted.
Positions, velocities, targets and limits also follow the authored USD body
order. Position and position-target reads support prismatic, revolute and
scalar-coordinate D6 joints, not spherical positions. Targets read the supplied
`Control`; position-target indexing uses `Model.joint_target_q_start`.
The mapping accounts for both endpoint order and axis direction, including
caller-owned models built with older Newton importers. `jointLimit` interleaves
low/high per axis, reverses bounds when a coordinate changes sign, and reports
unlimited bounds as ±maximum `float32`.

For Newton's own units and body order, read the Newton fields instead:
`joint_q`, `joint_qd`, `joint_target_q`, `joint_target_qd`, `joint_target_ke`,
`joint_target_kd`, `joint_limit_lower`, `joint_limit_upper`,
`joint_velocity_limit`, `joint_effort_limit`, `joint_armature` and
`joint_friction`. For bodies, the Newton fields are `body_q`, `body_qd`,
`body_qdd`, `body_mass`, `body_inertia` and `body_com`.

**Commanded joint effort.** `joint_f` reads one commanded force/torque value per
supported joint degree of freedom (DOF) from `Control.joint_f`. A hinge has one
torque entry, a slider one force entry, and a ball joint three torque components.
Reading it requires `control`, not `state`; the joint need not be moving.

The source is a contiguous `float32` array indexed by `Model.joint_qd_start`.
It shares DOF indexing with `State.joint_qd`, not its ownership or velocity units.
Selected joints are grouped by their DOF count `N`, with `N` components per data
row (`float32xN`). The read preserves Newton's force/torque units, axes, signs
and component order; no conversion to authored USD body order is performed.
Free/distance joint
wrenches use world-frame linear force followed by torque about the child COM,
not the joint-parent frame used by their velocities. Fixed joints have no DOFs.
This is an input buffer, not measured effort, actuator output or a solver result.
Reading it does not step a solver, apply forces or clear the control. The caller
owns writes and resets, including `Control.clear()`. Borrowing, selection gathers
and result lifetime follow the native joint rules below.

`jointMaxForce` reads the resolved symmetric effort limit from
`Model.joint_effort_limit`. For imported MuJoCo joints, enabled actuator-force
ranges contribute to this value, and a tighter USD drive `maxForce` takes
precedence. See {ref}`mujoco-joint-parameters` for the import rules and supported
joint types.

## Layout, storage and lifetime

Each {class}`ovnewton.ReadGroup` identifies shared, ordered prim rows and contains
one or more {class}`ovnewton.ReadColumn` objects. A column carries one attribute's
values, layout, row mapping and CUDA readiness. For example, `body_q` and
`body_qd` share one group for the same selected bodies despite their different
widths. Groups are ordered by first occurrence in the requested fields; columns
within each group follow request order, with duplicate fields removed.

Each column has one packed DLTensor: shape `(R,)`, 32-bit
components, `N` lanes, compact strides and zero byte offset. Row stride is `4*N`
bytes. Adapters expand lanes into scalar dimensions. Results and optional
`uint32` row maps reside on `binding.model.device`. Several groups can carry the
same field with different widths. `group.prim_list[i]` identifies logical prim
`i`; always use each column's `data_row_index(i)` to locate its storage row.
Two columns in the same group can have different row maps. `R` can exceed the
selected prim count. Explicit path order is preserved within each group.

Native body reads borrow full buffers with row indirection. Native joint and
unchanged DOF-property reads borrow a source span only when all selected widths
match and scalar offsets align to that width; gaps/reordering use a row map.
Otherwise they gather into dense device buffers. Split/root fields always
extract, materials gather/pad, and convenience joint fields gather/convert.
These operations allocate per-read output; they do not require a host round trip.
Borrowing avoids copying values, but still creates descriptors and CUDA events.

Groups, columns and managed DLPack exports retain storage. Retention keeps an allocation
alive; it does **not** freeze borrowed values. Prevent solver steps, state-buffer
reuse or model/control writes until every reader finishes. Retain the result or
group or column through asynchronous completion, including when using raw NumPy/tensor
views. Never mutate exported values or indices. Copy explicitly for history.
Results need no release; later reads do not overwrite per-read gathered buffers.
Keep the stage/binding alive while resolving path/token handles.

CPU values are ready on return. On CUDA, issue reads on the Warp stream that
last wrote the sources, or make the current stream wait for that work first.
A consumer on another stream must wait for `column.cuda_sync.wait_event` before
using `wp.from_dlpack(column.dlpack(0))` and its optional
`column.data_index_dlpack()` map. The event reports producer readiness, not
consumer completion. Column ordinals, semantics and generation metadata are zero
placeholders, not sample timestamps, representation IDs or invalidation tokens.

Particle/deformable state, contacts and measured joint forces are not exposed.
There are no separate static/dynamic friction, principal-axis COM orientation or
drive-type fields.

{func}`ovnewton.read_fields` gives static metadata for each field. This includes
units, frame, conversion, component names, layout and storage class. Its
values are text for people to read, not a typed schema. Do not parse them.

The catalog cannot describe a particular read. Inspect the result for those
details. The actual lane count, row mapping and device are in the group and
column layouts. A `borrowed_or_gathered` field is borrowed only when the
selection allows it. The result does not say which one happened, so treat it as
borrowed and follow the lifetime rules above. Do not infer formats
from shared names across engines.

### Migrating from flat output groups

Attribute and buffer accessors now live on `group.columns`, not on the group.
Move `group.attribute`, `group.tensor(0)`, `group.dlpack(0)`, row-map accessors,
`meta` and `cuda_sync` into an inner loop over columns. Keep prim lookup and
`prim_index(i)` on the group. Metadata types are now
{class}`ovnewton.ReadColumnMeta` and {class}`ovnewton.ReadColumnCudaSync`.

This follows OVStage's group/column separation, but not its storage-release
protocol: ovnewton does not pin stage storage and needs no explicit release.
Its packed Newton tensor layouts and units are unchanged.
