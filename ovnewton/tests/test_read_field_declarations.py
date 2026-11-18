# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check every read-field declaration against actual reads.

The expected values here are computed from Newton arrays with this file's own
rules. They do not use the declarations or the runtime descriptors, so a wrong
declaration or a wrong runtime conversion makes a test fail.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import ovstage
import pytest
import warp as wp

import ovnewton

from .test_output_read import (
    BODY_SHAPE_OUTPUTS,
    MIXED_JOINT_WIDTHS,
    MULTI_AXIS_D6_JOINT,
    REVERSED_WORLD_JOINT,
    _binding,
)
from .test_runtime import DRIVEN_REVOLUTE, _populate

DEG_PER_RAD = 180.0 / np.pi

ARTICULATION_WITH_SHAPES = """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "World" (
    apiSchemas = ["PhysicsArticulationRootAPI"]
)
{
    def Xform "Parent" (
        apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        def Sphere "Shape" (prepend apiSchemas = ["PhysicsCollisionAPI"])
        {
            double radius = 0.5
        }
    }
    def Xform "Child" (
        apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        def Cube "Shape0" (prepend apiSchemas = ["PhysicsCollisionAPI"])
        {
            double size = 1
        }
        def Sphere "Shape1" (prepend apiSchemas = ["PhysicsCollisionAPI"])
        {
            double radius = 0.25
        }
    }
    def PhysicsPrismaticJoint "Slider"
    {
        rel physics:body0 = </World/Parent>
        rel physics:body1 = </World/Child>
        uniform token physics:axis = "X"
    }
}
"""

# Authored physics:body1 of each articulation joint, taken from the USD text.
# The reversed-world fixture also reverses its axis, preserving coordinate signs.
SCENES = {
    "mixed_widths": (
        MIXED_JOINT_WIDTHS,
        {"/World/HingeA": "/World/Link1", "/World/HingeB": "/World/Link2", "/World/Ball": "/World/Link3"},
    ),
    "multi_axis_d6": (MULTI_AXIS_D6_JOINT, {"/World/Joint": "/World/Child"}),
    "reversed_world_joint": (REVERSED_WORLD_JOINT, {"/World/Joint": None}),
    "free_joint_pair": (DRIVEN_REVOLUTE, {}),
    "shapes": (BODY_SHAPE_OUTPUTS, {}),
    "articulation_with_shapes": (ARTICULATION_WITH_SHAPES, {"/World/Slider": "/World/Child"}),
}

OBJECT_TYPES = {
    "rigid_body": ovnewton.SimObjectType.RIGID_BODY,
    "articulation_link": ovnewton.SimObjectType.ARTICULATION_LINK,
    "articulation_root": ovnewton.SimObjectType.ARTICULATION,
    "physics_scene": ovnewton.SimObjectType.PHYSICS_SCENE,
    "articulation_joint": ovnewton.SimObjectType.ARTICULATION_JOINT,
}

CURRENT = tuple(ovnewton.read_fields())


@dataclass
class Scene:
    name: str
    stage: Any
    pd: Any
    binding: Any
    state: Any
    control: Any
    body1: dict[str, str | None]
    queries: dict[str, Any]

    @property
    def model(self):
        return self.binding.model


def _fill(array, rng, low=-2.0, high=2.0):
    values = rng.uniform(low, high, size=array.numpy().shape).astype(np.float32)
    array.assign(wp.array(values, dtype=array.dtype, device=array.device))


def _non_articulation_joints(binding):
    paths = []
    bodies = set(binding.model.body_label)
    for label in binding.model.joint_label:
        if not isinstance(label, str) or not label.startswith("/") or label in bodies:
            continue
        try:
            binding.query(paths=[label])
        except ValueError:
            continue
        try:
            binding.query(paths=[label], object_type=ovnewton.SimObjectType.ARTICULATION_JOINT)
        except ValueError:
            paths.append(label)
    return paths


def _open_scene(stack, name, usda, body1, seed):
    stage = stack.enter_context(ovstage.Stage(f"read-field-declarations-{name}"))
    pd = stack.enter_context(ovstage.PathDictionary(stage))
    _populate(stage, usda)
    binding = _binding(stage, pd)
    model = binding.model
    rng = np.random.default_rng(seed)
    state = model.state()
    control = model.control()
    state.body_qdd = wp.zeros_like(state.body_qd)
    for array in (state.body_q, state.body_qd, state.body_qdd, state.joint_q, state.joint_qd):
        _fill(array, rng)
    for name_ in ("joint_f", "joint_target_q", "joint_target_qd"):
        _fill(getattr(control, name_), rng)
    for name_ in (
        "body_mass",
        "body_inertia",
        "body_com",
        "gravity",
        "shape_material_mu",
        "shape_material_restitution",
        "joint_target_ke",
        "joint_target_kd",
        "joint_velocity_limit",
        "joint_effort_limit",
        "joint_armature",
        "joint_friction",
    ):
        _fill(getattr(model, name_), rng, 0.1, 3.0)
    _fill(model.joint_limit_lower, rng, -3.0, -0.1)
    _fill(model.joint_limit_upper, rng, 0.1, 3.0)

    queries = {}
    for kind, object_type in OBJECT_TYPES.items():
        query = binding.query(object_type=object_type)
        if query.prim_count:
            queries[kind] = query
    joints = _non_articulation_joints(binding)
    if joints:
        queries["joint"] = binding.query(paths=joints)
    return Scene(name, stage, pd, binding, state, control, body1, queries)


@pytest.fixture(scope="module")
def scenes():
    with contextlib.ExitStack() as stack, wp.ScopedDevice("cpu"):
        yield [
            _open_scene(stack, name, usda, body1, seed)
            for seed, (name, (usda, body1)) in enumerate(SCENES.items())
        ]


# ---------------------------------------------------------------------------
# Independent expectations: one function per current field.
# ---------------------------------------------------------------------------


def _body(scene, path):
    return list(scene.model.body_label).index(path)


def _joint(scene, path):
    return list(scene.model.joint_label).index(path)


def _root_body(scene, path):
    articulation = list(scene.model.articulation_label).index(path)
    root_joint = int(scene.model.articulation_start.numpy()[articulation])
    return int(scene.model.joint_child.numpy()[root_joint])


def _span(starts, joint):
    starts = starts.numpy()
    return slice(int(starts[joint]), int(starts[joint + 1]))


def _angular_axes(scene, joint, width):
    linear, angular = (int(value) for value in scene.model.joint_dof_dim.numpy()[joint])
    assert linear + angular == width, "joint convenience field covers a joint with non-axis coordinates"
    return np.array([False] * linear + [True] * angular)


def _sign(scene, joint, path):
    if scene.name == "reversed_world_joint":
        np.testing.assert_allclose(scene.model.joint_axis.numpy(), [(-1, 0, 0)])
        return 1.0
    child = int(scene.model.joint_child.numpy()[joint])
    child_path = scene.model.body_label[child] if child >= 0 else None
    return 1.0 if child_path == scene.body1[path] else -1.0


def _shapes(scene, body):
    return [shape for shape, owner in enumerate(scene.model.shape_body.numpy()) if int(owner) == body]


def _body_row(array, index):
    return lambda scene, path: getattr(scene.state, array).numpy()[_body(scene, path)][index]


def _model_body(array):
    return lambda scene, path: getattr(scene.model, array).numpy()[_body(scene, path)].reshape(-1)


def _root_row(array, index):
    return lambda scene, path: getattr(scene.state, array).numpy()[_root_body(scene, path)][index]


def _shape_values(array):
    def expected(scene, path, width):
        body = _body(scene, path)
        values = getattr(scene.model, array).numpy()[_shapes(scene, body)]
        return np.pad(values, (0, width - len(values)))

    return expected


def _native_joint(owner, array, starts):
    def expected(scene, path):
        joint = _joint(scene, path)
        source = getattr(getattr(scene, owner), array).numpy()
        return source[_span(getattr(scene.model, starts), joint)]

    return expected


def _converted_joint(owner, array, starts, *, scale, signed):
    def expected(scene, path):
        joint = _joint(scene, path)
        values = getattr(getattr(scene, owner), array).numpy()[_span(getattr(scene.model, starts), joint)].copy()
        angular = _angular_axes(scene, joint, len(values))
        values[angular] *= scale
        return values * (_sign(scene, joint, path) if signed else 1.0)

    return expected


def _joint_limit(scene, path):
    joint = _joint(scene, path)
    span = _span(scene.model.joint_qd_start, joint)
    lower = scene.model.joint_limit_lower.numpy()[span]
    upper = scene.model.joint_limit_upper.numpy()[span]
    if _sign(scene, joint, path) < 0:
        lower, upper = -upper, -lower
    scale = np.where(_angular_axes(scene, joint, len(lower)), DEG_PER_RAD, 1.0)
    return np.stack((lower * scale, upper * scale), axis=1).reshape(-1)


def _gravity(scene, _path):
    gravity = scene.model.gravity.numpy()
    assert len(gravity) == 1, "the test scenes use one Newton world"
    return gravity[0]


EXPECTED: dict[str, Callable[..., np.ndarray]] = {
    "body_q": _body_row("body_q", slice(0, 7)),
    "body_qd": _body_row("body_qd", slice(0, 6)),
    "body_qdd": _body_row("body_qdd", slice(0, 6)),
    "body_mass": _model_body("body_mass"),
    "body_inertia": _model_body("body_inertia"),
    "body_com": _model_body("body_com"),
    "position": _body_row("body_q", slice(0, 3)),
    "orientation": _body_row("body_q", slice(3, 7)),
    "linearVelocity": _body_row("body_qd", slice(0, 3)),
    "angularVelocity": _body_row("body_qd", slice(3, 6)),
    "linearAcceleration": _body_row("body_qdd", slice(0, 3)),
    "angularAcceleration": _body_row("body_qdd", slice(3, 6)),
    "gravity": _gravity,
    "mass": _model_body("body_mass"),
    "inertia": _model_body("body_inertia"),
    "centerOfMassPosition": _model_body("body_com"),
    "shapeCount": lambda scene, path: np.array([len(_shapes(scene, _body(scene, path)))]),
    "friction": _shape_values("shape_material_mu"),
    "restitution": _shape_values("shape_material_restitution"),
    "rootPosition": _root_row("body_q", slice(0, 3)),
    "rootOrientation": _root_row("body_q", slice(3, 7)),
    "rootLinearVelocity": _root_row("body_qd", slice(0, 3)),
    "rootAngularVelocity": _root_row("body_qd", slice(3, 6)),
    "joint_q": _native_joint("state", "joint_q", "joint_q_start"),
    "joint_qd": _native_joint("state", "joint_qd", "joint_qd_start"),
    "joint_f": _native_joint("control", "joint_f", "joint_qd_start"),
    "joint_target_q": _native_joint("control", "joint_target_q", "joint_target_q_start"),
    "joint_target_qd": _native_joint("control", "joint_target_qd", "joint_qd_start"),
    "joint_target_ke": _native_joint("model", "joint_target_ke", "joint_qd_start"),
    "joint_target_kd": _native_joint("model", "joint_target_kd", "joint_qd_start"),
    "joint_limit_lower": _native_joint("model", "joint_limit_lower", "joint_qd_start"),
    "joint_limit_upper": _native_joint("model", "joint_limit_upper", "joint_qd_start"),
    "joint_velocity_limit": _native_joint("model", "joint_velocity_limit", "joint_qd_start"),
    "joint_effort_limit": _native_joint("model", "joint_effort_limit", "joint_qd_start"),
    "joint_armature": _native_joint("model", "joint_armature", "joint_qd_start"),
    "joint_friction": _native_joint("model", "joint_friction", "joint_qd_start"),
    # Standard fields: angular axes in degrees, per degree, or unconverted, by quantity.
    "jointPosition": _converted_joint("state", "joint_q", "joint_q_start", scale=DEG_PER_RAD, signed=True),
    "jointVelocity": _converted_joint("state", "joint_qd", "joint_qd_start", scale=DEG_PER_RAD, signed=True),
    "jointPositionTarget": _converted_joint(
        "control", "joint_target_q", "joint_target_q_start", scale=DEG_PER_RAD, signed=True
    ),
    "jointVelocityTarget": _converted_joint(
        "control", "joint_target_qd", "joint_qd_start", scale=DEG_PER_RAD, signed=True
    ),
    "jointStiffness": _converted_joint(
        "model", "joint_target_ke", "joint_qd_start", scale=1.0 / DEG_PER_RAD, signed=False
    ),
    "jointDamping": _converted_joint(
        "model", "joint_target_kd", "joint_qd_start", scale=1.0 / DEG_PER_RAD, signed=False
    ),
    "jointLimit": _joint_limit,
    "jointMaxVelocity": _converted_joint(
        "model", "joint_velocity_limit", "joint_qd_start", scale=DEG_PER_RAD, signed=False
    ),
    "jointMaxForce": _native_joint("model", "joint_effort_limit", "joint_qd_start"),
    "jointArmature": _native_joint("model", "joint_armature", "joint_qd_start"),
    "jointFriction": _native_joint("model", "joint_friction", "joint_qd_start"),
}

# Fields whose row width depends on the other rows in the same group.
GROUP_WIDTH = {"friction", "restitution"}


# ---------------------------------------------------------------------------
# Helpers that read through the public API only.
# ---------------------------------------------------------------------------


def _source_array(scene, source):
    owner, array = source.split(".")
    return getattr({"State": scene.state, "Control": scene.control, "Model": scene.model}[owner], array)


def _inside(pointer, array):
    return array.ptr is not None and array.ptr <= pointer < array.ptr + array.capacity


def _rows(column, lanes):
    values = np.asarray(column.array(0)).reshape(-1, lanes)
    return [values[column.data_row_index(i)] for i in range(column.data_count)]


def _read(scene, query, names):
    return scene.binding.read(scene.state, control=scene.control, query=query, attributes=names)


def _check_column(scene, field, group, column):
    tensor = column.tensor(0)
    code = {"float32": ovstage.DLDataTypeCode.kDLFloat, "int32": ovstage.DLDataTypeCode.kDLInt}[field.dtype]
    assert (int(tensor.dtype.code), int(tensor.dtype.bits)) == (int(code), 32), f"{field.name}: dtype"
    lanes = int(tensor.dtype.lanes)
    if field.lanes is not None:
        assert lanes == field.lanes, f"{field.name}: lanes"

    # Device: the payload is on the connected model's device.
    assert wp.from_dlpack(column.dlpack(0)).device == scene.model.device, f"{field.name}: device"

    # Storage: borrowed payloads point into a declared source array; others do not.
    sources = [_source_array(scene, source) for source in field.sources]
    borrowed = any(_inside(tensor.data, source) for source in sources)
    if field.storage == "borrowed":
        assert borrowed, f"{field.name}: storage should borrow {field.sources}"
    elif field.storage in ("gathered", "query_owned"):
        assert not borrowed, f"{field.name}: storage should not alias {field.sources}"

    # Values, units, frames and layout: each row equals the independent expectation.
    paths = scene.pd.get_path_strings(group.prim_list)
    width = max(len(_shapes(scene, _body(scene, path))) for path in paths) if field.name in GROUP_WIDTH else None
    for path, row in zip(paths, _rows(column, lanes)):
        expected = EXPECTED[field.name](scene, path, width) if width is not None else EXPECTED[field.name](scene, path)
        expected = np.asarray(expected).reshape(-1)
        assert len(expected) == lanes, f"{field.name} {path}: lanes {lanes}, expected {len(expected)}"
        if field.conversion is None:
            np.testing.assert_array_equal(row, expected, err_msg=f"{field.name} {path}")
        else:
            np.testing.assert_allclose(row, expected, rtol=1.0e-6, atol=1.0e-6, err_msg=f"{field.name} {path}")
    return paths


def _covered_kinds(field):
    kinds = set(field.objects)
    if "joint" in kinds:
        kinds.add("articulation_joint")
    return kinds


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_every_current_field_has_an_independent_expectation():
    assert set(EXPECTED) == set(CURRENT)


def test_metadata_lists_exactly_the_accepted_names(scenes):
    scene = scenes[0]
    accepted = set(ovnewton.read_fields())
    for name in accepted:
        _read(scene, scene.queries["articulation_joint"], (name,))
    with pytest.raises(ValueError, match="unsupported output attribute"):
        _read(scene, scene.queries["articulation_joint"], ("jointPosition.degrees",))


@pytest.mark.parametrize("name", CURRENT)
def test_declaration_matches_reads(scenes, name):
    field = ovnewton.read_fields()[name]
    produced = set()
    for scene in scenes:
        for kind, query in scene.queries.items():
            result = _read(scene, query, (name,))
            columns = [(group, column) for group in result.groups for column in group.columns]
            if not columns:
                continue
            assert kind in _covered_kinds(field), f"{name} produced rows for undeclared {kind} in {scene.name}"
            produced.add(kind)
            for group, column in columns:
                assert scene.pd.token_to_string(column.attribute) == name
                _check_column(scene, field, group, column)
    assert produced == _covered_kinds(field), f"{name} declared objects never produced rows"


@pytest.mark.parametrize("name", CURRENT)
def test_declared_requirement_matches_reads(scenes, name):
    field = ovnewton.read_fields()[name]
    scene, query = next(
        (scene, query)
        for scene in scenes
        for kind, query in scene.queries.items()
        if kind in _covered_kinds(field) and _read(scene, query, (name,)).groups
    )
    inputs = {
        "state": {"state": scene.state},
        "control": {"control": scene.control},
        "model": {},
    }[field.requires]
    # The declared input alone is enough.
    assert scene.binding.read(**inputs, query=query, attributes=(name,)).groups
    if field.requires == "state":
        with pytest.raises(ValueError, match="state is required"):
            scene.binding.read(control=scene.control, query=query, attributes=(name,))
    if field.requires == "control":
        with pytest.raises(ValueError, match="control is required"):
            scene.binding.read(scene.state, query=query, attributes=(name,))


@pytest.mark.parametrize("name", [name for name in CURRENT if ovnewton.read_fields()[name].storage == "query_owned"])
def test_query_owned_storage_is_stable_between_reads(scenes, name):
    for scene in scenes:
        for query in scene.queries.values():
            first = _read(scene, query, (name,))
            second = _read(scene, query, (name,))
            pointers = [column.tensor(0).data for group in first.groups for column in group.columns]
            assert pointers == [column.tensor(0).data for group in second.groups for column in group.columns]


@pytest.mark.parametrize("name", [name for name in CURRENT if ovnewton.read_fields()[name].storage == "gathered"])
def test_gathered_storage_is_a_snapshot(scenes, name):
    field = ovnewton.read_fields()[name]
    for scene in scenes:
        for query in scene.queries.values():
            result = _read(scene, query, (name,))
            columns = [column for group in result.groups for column in group.columns]
            if not columns:
                continue
            before = [np.array(column.array(0), copy=True) for column in columns]
            sources = [_source_array(scene, source) for source in field.sources]
            saved = [source.numpy().copy() for source in sources]
            try:
                for source in sources:
                    source.assign(wp.array(source.numpy() + 1, dtype=source.dtype, device=source.device))
                for column, values in zip(columns, before):
                    np.testing.assert_array_equal(np.asarray(column.array(0)), values)
            finally:
                for source, values in zip(sources, saved):
                    source.assign(wp.array(values, dtype=source.dtype, device=source.device))
            return
    pytest.fail(f"{name} produced no rows")


def test_query_attributes_list_every_available_field(scenes):
    scene = scenes[0]
    query = scene.queries["articulation_joint"]
    attributes = {scene.pd.token_to_string(token) for token in query.attributes}
    available = {name for name in CURRENT if _read(scene, query, (name,)).groups}
    assert attributes == available


LINKED = [(name, linked) for name, field in ovnewton.read_fields().items() for linked in field.newton_fields]


def _row_paths(scene, query, name):
    return {
        path
        for group in _read(scene, query, (name,)).groups
        for path in scene.pd.get_path_strings(group.prim_list)
    }


@pytest.mark.parametrize("name,linked", LINKED)
def test_newton_fields_cover_the_standard_field(scenes, name, linked):
    """Each linked Newton field reads the same array, and has a row wherever the standard field
    has one, unless the standard field reports on a different prim (articulation roots)."""
    field = ovnewton.read_fields()[name]
    newton = ovnewton.read_fields()[linked]
    assert newton.sources[0] in field.sources
    if not _covered_kinds(field) <= _covered_kinds(newton):
        return
    checked = False
    for scene in scenes:
        for kind, query in scene.queries.items():
            if kind not in _covered_kinds(field):
                continue
            standard_paths = _row_paths(scene, query, name)
            assert standard_paths <= _row_paths(scene, query, linked), f"{name} rows without {linked} rows"
            checked = checked or bool(standard_paths)
    assert checked, f"{name} produced no rows"


@pytest.mark.parametrize("path", ["/World/Slider", "/World/HingeA"])
def test_linear_axes_are_never_converted(scenes, path):
    """A prismatic axis returns Newton's value in every joint field; a revolute axis is converted."""
    scene = next(scene for scene in scenes if path in scene.body1)
    query = scene.binding.query(paths=[path], object_type=ovnewton.SimObjectType.ARTICULATION_JOINT)
    # (standard field, Newton field, angular scale, follows authored body order)
    pairs = [
        ("jointPosition", "joint_q", DEG_PER_RAD, True),
        ("jointStiffness", "joint_target_ke", 1.0 / DEG_PER_RAD, False),
        ("jointMaxVelocity", "joint_velocity_limit", DEG_PER_RAD, False),
    ]
    joint = _joint(scene, path)
    angular = bool(_angular_axes(scene, joint, 1)[0])
    assert angular == (path == "/World/HingeA")
    for converted, native, scale, signed in pairs:
        result = _read(scene, query, (converted, native))
        values = {
            scene.pd.token_to_string(column.attribute): float(np.asarray(column.array(0)).reshape(-1)[0])
            for group in result.groups
            for column in group.columns
        }
        factor = (scale if angular else 1.0) * (_sign(scene, joint, path) if signed else 1.0)
        np.testing.assert_allclose(values[converted], values[native] * factor, rtol=1.0e-6, err_msg=converted)
