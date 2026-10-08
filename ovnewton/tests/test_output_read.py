# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import gc
import inspect
import weakref
from types import SimpleNamespace

import newton
import numpy as np
import ovstage
import pytest
import warp as wp

import ovnewton
from ovnewton._src import _build

from .test_runtime import (
    D6_JOINT,
    DISTANCE_JOINT,
    DRIVEN_PRISMATIC,
    DRIVEN_REVOLUTE,
    FALLING_BODY,
    SPHERICAL_JOINT,
    _populate,
    _revolute_indices,
    _write_scalar,
)

MIXED_JOINT_WIDTHS = """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "World" (
    apiSchemas = ["PhysicsArticulationRootAPI"]
)
{
    def Xform "Root" (
        apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
    }
    def Xform "Link1" (
        apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
    }
    def Xform "Link2" (
        apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
    }
    def Xform "Link3" (
        apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
    }
    def PhysicsRevoluteJoint "HingeA"
    {
        rel physics:body0 = </World/Root>
        rel physics:body1 = </World/Link1>
    }
    def PhysicsRevoluteJoint "HingeB"
    {
        rel physics:body0 = </World/Link1>
        rel physics:body1 = </World/Link2>
    }
    def PhysicsSphericalJoint "Ball"
    {
        rel physics:body0 = </World/Link2>
        rel physics:body1 = </World/Link3>
    }
}
"""

BODY_OUTPUT_SEMANTICS = """#usda 1.0
(
    metersPerUnit = 0.01
    upAxis = "Z"
)
def Xform "Body" (
    apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{
    float physics:mass = 2
    point3f physics:centerOfMass = (0.25, 0.5, 0.75)
    vector3f physics:diagonalInertia = (1, 2, 3)
    quatf physics:principalAxes = (1, 0, 0, 0)
    vector3f physics:velocity = (1, 2, 3)
    vector3f physics:angularVelocity = (10, 20, 30)
    double3 xformOp:translate = (4, 5, 6)
    quatd xformOp:orient = (0.7071067811865476, 0, 0, 0.7071067811865476)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
}
"""

BODY_SHAPE_OUTPUTS = """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "A" (
    apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{
    def Sphere "Shape0" (prepend apiSchemas = ["PhysicsCollisionAPI"])
    {
        double radius = 1
    }
    def Cube "Shape1" (prepend apiSchemas = ["PhysicsCollisionAPI"])
    {
        double size = 1
    }
}
def Xform "B" (
    apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{
    def Capsule "Shape2" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        double radius = 0.5
        double height = 1
    }
}
"""

FIXED_JOINT = SPHERICAL_JOINT.replace("PhysicsSphericalJoint", "PhysicsFixedJoint")

MULTI_AXIS_D6_JOINT = D6_JOINT.replace(
    'apiSchemas = ["PhysicsLimitAPI:rotX"]',
    'apiSchemas = ["PhysicsLimitAPI:transY", "PhysicsLimitAPI:rotX"]',
).replace(
    "        float limit:rotX:physics:low = -45",
    """        float limit:transY:physics:low = -2
        float limit:transY:physics:high = 2
        float limit:rotX:physics:low = -45""",
)

REVERSED_WORLD_JOINT = """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "World" (
    apiSchemas = ["PhysicsArticulationRootAPI"]
)
{
    def Xform "Body" (
        apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
    }
    def PhysicsRevoluteJoint "Joint"
    {
        rel physics:body0 = </World/Body>
    }
}
"""


def _binding(stage, pd, device="cpu"):
    with wp.ScopedDevice(device):
        model = _build.build_model(stage, pd, ordinal=1)
    return ovnewton.attach_ovstage(stage, model=model)


def _state(binding):
    state = binding.model.state()
    body_q = state.body_q.numpy()
    for index in range(binding.model.body_count):
        body_q[index, :3] = (10.0 + index, 20.0 + index, 30.0 + index)
        body_q[index, 3:7] = (0.0, 0.0, 0.0, 1.0)
    state.body_q.assign(wp.array(body_q, dtype=wp.transformf, device=binding.model.device))

    body_qd = np.zeros((binding.model.body_count, 6), dtype=np.float32)
    for index in range(binding.model.body_count):
        body_qd[index] = (1.0 + index, 2.0 + index, 3.0 + index, 0.1, 0.2, 0.3)
    state.body_qd.assign(wp.array(body_qd, dtype=wp.spatial_vectorf, device=binding.model.device))

    q_index, qd_index, _ = _revolute_indices(binding.model)
    joint_q = state.joint_q.numpy()
    joint_qd = state.joint_qd.numpy()
    joint_q[q_index] = np.deg2rad(35.0)
    joint_qd[qd_index] = np.deg2rad(45.0)
    state.joint_q.assign(wp.array(joint_q, dtype=wp.float32, device=binding.model.device))
    state.joint_qd.assign(wp.array(joint_qd, dtype=wp.float32, device=binding.model.device))
    return state


def _body_path_at(binding, index):
    """Return a body path by model row without assuming builder ordering."""
    return list(binding.model.body_label)[index]


def _groups_by_name(result, pd):
    return {pd.token_to_string(group.attribute): group for group in result.groups}


def _logical_values(group):
    values = wp.from_dlpack(group.dlpack(0)).numpy()
    if group.has_data_index_map:
        values = values[wp.from_dlpack(group.data_index_dlpack()).numpy()]
    else:
        values = values[:group.data_count]
    return values


def _dense_group_layout(group):
    return {
        "is_array": group.is_array,
        "prim_offset": group.prim_offset,
        "prim_count": group.prim_count,
        "prim_rows": tuple(group.prim_index(i) for i in range(group.prim_count)),
        "tensor_count": group.tensor_count,
        "data_count": group.data_count,
        "data_rows": tuple(group.data_row_index(i) for i in range(group.data_count)),
        "has_prim_index_map": group.has_prim_index_map,
        "has_data_index_map": group.has_data_index_map,
    }


def test_output_read_api_is_keyword_scoped():
    query = inspect.signature(ovnewton.StageBinding.query).parameters
    read = inspect.signature(ovnewton.StageBinding.read).parameters

    assert tuple(query) == ("self", "stage_query", "paths", "object_type", "scope")
    assert query["stage_query"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert query["paths"].kind is inspect.Parameter.KEYWORD_ONLY
    assert query["object_type"].kind is inspect.Parameter.KEYWORD_ONLY
    assert query["object_type"].default is None
    assert query["scope"].kind is inspect.Parameter.KEYWORD_ONLY
    assert query["scope"].default is ovnewton.ObjectScope.ALL
    assert tuple(read) == ("self", "state", "control", "query", "attributes")
    assert read["state"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert read["state"].default is None
    assert read["control"].kind is inspect.Parameter.KEYWORD_ONLY
    assert read["control"].default is None
    assert read["query"].kind is inspect.Parameter.KEYWORD_ONLY
    assert read["attributes"].kind is inspect.Parameter.KEYWORD_ONLY
    for value_type in (ovnewton.Query, ovnewton.ReadResult, ovnewton.ReadGroup):
        assert not hasattr(value_type, "release")
        assert not hasattr(value_type, "close")
        assert not hasattr(value_type, "__enter__")


def test_query_selects_explicit_object_types_and_reports_scope():
    assert int(ovnewton.SimObjectType.RIGID_BODY) == 0
    assert int(ovnewton.SimObjectType.ARTICULATION_LINK) == 1
    assert int(ovnewton.SimObjectType.ARTICULATION_JOINT) == 2
    assert int(ovnewton.SimObjectType.ARTICULATION) == 9
    assert int(ovnewton.ObjectScope.ALL) == 0
    assert int(ovnewton.ObjectScope.ACTIVE) == 1

    with ovstage.Stage("output-read-object-types") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, MIXED_JOINT_WIDTHS)
        binding = _binding(stage, pd)

        links = binding.query(object_type=ovnewton.SimObjectType.ARTICULATION_LINK)
        joints = binding.query(object_type=ovnewton.SimObjectType.ARTICULATION_JOINT)
        articulation = binding.query(object_type=ovnewton.SimObjectType.ARTICULATION)
        rigid = binding.query(object_type=ovnewton.SimObjectType.RIGID_BODY)

        assert links.object_type is ovnewton.SimObjectType.ARTICULATION_LINK
        assert links.scope is ovnewton.ObjectScope.ALL
        assert links.prim_count == 4
        assert joints.prim_count == 3
        assert articulation.prim_count == 1
        assert rigid.prim_count == 0
        link_groups = binding.read(
            binding.model.state(),
            query=links,
            attributes=("position",),
        ).groups
        assert link_groups
        assert all(
            group.object_type is ovnewton.SimObjectType.ARTICULATION_LINK
            for group in link_groups
        )
        assert {pd.token_to_string(token) for token in articulation.attributes} == {
            "rootPosition",
            "rootOrientation",
            "rootLinearVelocity",
            "rootAngularVelocity",
        }
        assert "jointPosition" in {
            pd.token_to_string(token) for token in joints.attributes
        }

        with stage.query() as stage_query:
            stage_joints = binding.query(
                stage_query,
                object_type=ovnewton.SimObjectType.ARTICULATION_JOINT,
            )
        assert stage_joints.prim_count == 3
        with pytest.raises(ValueError, match="output-capable"):
            binding.query(
                paths=["/World/Root"],
                object_type=ovnewton.SimObjectType.RIGID_BODY,
            )

        with pytest.raises(NotImplementedError, match="ACTIVE"):
            binding.query(
                object_type=ovnewton.SimObjectType.RIGID_BODY,
                scope=ovnewton.ObjectScope.ACTIVE,
            )
        with pytest.raises(TypeError, match="object_type"):
            binding.query(object_type=0)
        with pytest.raises(TypeError, match="scope"):
            binding.query(
                object_type=ovnewton.SimObjectType.RIGID_BODY,
                scope=0,
            )

    with ovstage.Stage("output-read-scene-type") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, FALLING_BODY)
        binding = _binding(stage, pd)
        scene = binding.query(object_type=ovnewton.SimObjectType.PHYSICS_SCENE)
        rigid = binding.query(object_type=ovnewton.SimObjectType.RIGID_BODY)
        links = binding.query(object_type=ovnewton.SimObjectType.ARTICULATION_LINK)
        articulations = binding.query(object_type=ovnewton.SimObjectType.ARTICULATION)
        assert scene.prim_count == 1
        assert rigid.prim_count == 1
        assert links.prim_count == 0
        assert articulations.prim_count == 0
        assert {pd.token_to_string(token) for token in scene.attributes} == {"gravity"}


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA not available"))],
)
@pytest.mark.parametrize("articulation_label", [None, "caller_robot"])
@pytest.mark.parametrize("free_joints_only", [False, True])
def test_query_classifies_caller_owned_articulation_links(device, articulation_label, free_joints_only):
    with ovstage.Stage("output-read-caller-articulation") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, MIXED_JOINT_WIDTHS + '''
def Xform "Loose" (apiSchemas = ["PhysicsRigidBodyAPI"]) {}
def Xform "Unassigned" (apiSchemas = ["PhysicsRigidBodyAPI"]) {}
''')
        builder = newton.ModelBuilder()
        link_paths = ["/World/Root", "/World/Link1", "/World/Link2", "/World/Link3"]
        links = [builder.add_link(label=path) for path in link_paths]
        joints = [builder.add_joint_free(links[0])]
        if free_joints_only:
            joints.extend(builder.add_joint_free(child, parent=parent) for parent, child in zip(links, links[1:]))
        else:
            joints.extend((
                builder.add_joint_revolute(links[0], links[1], label="/World/HingeA"),
                builder.add_joint_revolute(links[1], links[2], label="/World/HingeB"),
                builder.add_joint_ball(links[2], links[3], label="/World/Ball"),
            ))
        builder.add_articulation(joints, label=articulation_label)
        builder.add_body(label="/Loose")
        # This joint is not part of the preceding standalone body's articulation.
        builder.add_joint_free(builder.add_link(label="/Unassigned"))
        binding = ovnewton.attach_ovstage(stage, model=builder.finalize(device=device))
        state = binding.model.state()

        for object_type, expected_paths in (
            (ovnewton.SimObjectType.ARTICULATION_LINK, link_paths),
            (ovnewton.SimObjectType.RIGID_BODY, ["/Loose", "/Unassigned"]),
        ):
            query = binding.query(object_type=object_type)
            assert query.prim_count == len(expected_paths)
            group = binding.read(state, query=query, attributes=("position",)).groups[0]
            assert pd.get_path_strings(group.prim_list) == expected_paths
            assert binding.query(paths=expected_paths, object_type=object_type).prim_count == len(expected_paths)


def test_query_discovers_native_arrays_and_ignores_other_prims():
    with ovstage.Stage("output-read-discovery") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, DRIVEN_REVOLUTE)
        binding = _binding(stage, pd)

        with stage.query() as stage_query:
            query = binding.query(stage_query)
        names = {pd.token_to_string(token) for token in query.attributes}
        assert isinstance(query, ovnewton.Query)
        assert query.prim_count == binding.model.body_count + 2
        assert names == {
            "angularAcceleration",
            "angularVelocity",
            "body_q",
            "body_qd",
            "centerOfMassPosition",
            "friction",
            "inertia",
            "joint_q",
            "joint_qd",
            "linearAcceleration",
            "linearVelocity",
            "gravity",
            "mass",
            "orientation",
            "position",
            "restitution",
            "shapeCount",
        }


def test_query_reuses_binding_catalog(monkeypatch):
    with ovstage.Stage("output-read-catalog") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, DRIVEN_REVOLUTE)
        binding = _binding(stage, pd)

        def unexpected_host_read(_array):
            raise AssertionError("query rebuilt static model metadata")

        monkeypatch.setattr(type(binding.model.joint_q_start), "numpy", unexpected_host_read)
        query = binding.query(paths=["/Arm", "/Joint"])
        names = {pd.token_to_string(token) for token in query.attributes}
        assert query.prim_count == 2
        assert names == {
            "angularAcceleration",
            "angularVelocity",
            "body_q",
            "body_qd",
            "centerOfMassPosition",
            "friction",
            "inertia",
            "joint_q",
            "joint_qd",
            "linearAcceleration",
            "linearVelocity",
            "mass",
            "orientation",
            "position",
            "restitution",
            "shapeCount",
        }


def test_output_read_indexes_borrowed_newton_body_arrays():
    with ovstage.Stage("output-read-values") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, DRIVEN_REVOLUTE)
        binding = _binding(stage, pd)
        state = _state(binding)
        selected_path = _body_path_at(binding, 1)
        selected_row = 1

        query = binding.query(paths=[selected_path])
        result = binding.read(state, query=query, attributes=("body_q", "body_qd"))
        groups = _groups_by_name(result, pd)
        assert isinstance(result, ovnewton.ReadResult)
        assert all(isinstance(group, ovnewton.ReadGroup) for group in result.groups)
        assert set(groups) == {"body_q", "body_qd"}
        assert all(pd.get_path_strings(group.prim_list) == [selected_path] for group in groups.values())
        assert all(group.prim_count == 1 and group.tensor_count == 1 for group in groups.values())
        assert all(not group.is_array for group in groups.values())
        assert all(group.has_data_index_map for group in groups.values())
        assert groups["body_q"].tensor(0).data == state.body_q.ptr
        assert groups["body_qd"].tensor(0).data == state.body_qd.ptr
        assert groups["body_q"].data_row_index(0) == selected_row
        assert groups["body_q"].data_index_tensor().dtype.code == ovstage.DLDataTypeCode.kDLUInt
        data_indices = groups["body_q"].data_index_array()
        assert not data_indices.flags.writeable
        np.testing.assert_array_equal(data_indices, (selected_row,))
        np.testing.assert_array_equal(
            wp.from_dlpack(groups["body_q"].data_index_dlpack()).numpy(),
            (selected_row,),
        )
        with pytest.raises(ValueError, match="read-only"):
            groups["body_q"].data_index_dlpack(readonly=False)
        with pytest.raises(IndexError, match="out of range"):
            groups["body_q"].data_row_index(1)
        np.testing.assert_allclose(_logical_values(groups["body_q"])[0], state.body_q.numpy()[selected_row])
        np.testing.assert_allclose(_logical_values(groups["body_qd"])[0], state.body_qd.numpy()[selected_row])


def test_output_read_preserves_body_origin_pose_and_world_com_twist_in_model_units():
    with ovstage.Stage("output-read-body-semantics") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, BODY_OUTPUT_SEMANTICS)
        binding = _binding(stage, pd)
        state = binding.model.state()

        query = binding.query(paths=["/Body"])
        result = binding.read(state, query=query, attributes=("body_q", "body_qd"))
        groups = _groups_by_name(result, pd)
        pose = _logical_values(groups["body_q"])[0]
        twist = _logical_values(groups["body_qd"])[0]

        # Pose translation names the body origin, not the offset center of mass.
        # Values remain in the model's units: this stage uses centimeters.
        np.testing.assert_allclose(pose, (4.0, 5.0, 6.0, 0.0, 0.0, 2.0**-0.5, 2.0**-0.5), atol=1.0e-6)
        np.testing.assert_allclose(binding.model.body_com.numpy()[0], (0.25, 0.5, 0.75))

        # Newton/ovnewton currently rotate authored velocities by the body
        # orientation and convert angular rates from degrees/s to radians/s.
        # This characterizes their importer, not ovphysx import parity.
        np.testing.assert_allclose(
            twist,
            (-2.0, 1.0, 3.0, -np.deg2rad(20.0), np.deg2rad(10.0), np.deg2rad(30.0)),
            atol=1.0e-6,
        )


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA not available"))],
)
def test_output_read_splits_body_pose_and_twist_on_device(device):
    with ovstage.Stage("output-read-body-fields") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, BODY_OUTPUT_SEMANTICS)
        binding = _binding(stage, pd, device)
        state = binding.model.state()

        result = binding.read(
            state,
            query=binding.query(paths=["/Body"]),
            attributes=("position", "orientation", "linearVelocity", "angularVelocity"),
        )
        groups = _groups_by_name(result, pd)

        assert groups["position"].tensor(0).data != state.body_q.ptr
        assert groups["orientation"].tensor(0).data != state.body_q.ptr
        assert groups["linearVelocity"].tensor(0).data != state.body_qd.ptr
        assert groups["angularVelocity"].tensor(0).data != state.body_qd.ptr
        assert groups["position"].tensor(0).dtype.lanes == 3
        assert groups["orientation"].tensor(0).dtype.lanes == 4
        np.testing.assert_allclose(_logical_values(groups["position"]), ((4.0, 5.0, 6.0),), atol=1.0e-6)
        np.testing.assert_allclose(
            _logical_values(groups["orientation"]),
            ((0.0, 0.0, 2.0**-0.5, 2.0**-0.5),),
            atol=1.0e-6,
        )
        np.testing.assert_allclose(_logical_values(groups["linearVelocity"]), ((-2.0, 1.0, 3.0),), atol=1.0e-6)
        np.testing.assert_allclose(
            _logical_values(groups["angularVelocity"]),
            ((-np.deg2rad(20.0), np.deg2rad(10.0), np.deg2rad(30.0)),),
            atol=1.0e-6,
        )
        assert all(bool(group.cuda_sync.wait_event) == (device != "cpu") for group in groups.values())


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA not available"))],
)
def test_filtered_split_body_reads_gather_only_selected_rows(device):
    with ovstage.Stage("output-read-filtered-body-fields") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, DRIVEN_REVOLUTE)
        binding = _binding(stage, pd, device)
        state = _state(binding)
        selected_path = _body_path_at(binding, 1)

        result = binding.read(
            state,
            query=binding.query(paths=[selected_path]),
            attributes=("position", "linearVelocity"),
        )
        groups = _groups_by_name(result, pd)

        assert all(group.tensor(0).shape[0] == 1 for group in groups.values())
        assert all(not group.has_data_index_map for group in groups.values())
        np.testing.assert_allclose(
            _logical_values(groups["position"]),
            state.body_q.numpy()[1:2, :3],
        )
        np.testing.assert_allclose(
            _logical_values(groups["linearVelocity"]),
            state.body_qd.numpy()[1:2, :3],
        )


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA not available"))],
)
def test_output_read_splits_optional_body_acceleration_on_device(device):
    with ovstage.Stage("output-read-body-acceleration") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, BODY_OUTPUT_SEMANTICS)
        binding = _binding(stage, pd, device)

        with pytest.raises(ValueError, match=r"State\.body_qdd"):
            binding.read(
                binding.model.state(),
                query=binding.query(paths=["/Body"]),
                attributes=("linearAcceleration",),
            )

        binding.model.request_state_attributes("body_qdd")
        state = binding.model.state()
        state.body_qdd.assign(
            wp.array(
                [(1.0, 2.0, 3.0, 0.1, 0.2, 0.3)],
                dtype=wp.spatial_vectorf,
                device=binding.model.device,
            )
        )
        groups = _groups_by_name(
            binding.read(
                state,
                query=binding.query(paths=["/Body"]),
                attributes=("linearAcceleration", "angularAcceleration"),
            ),
            pd,
        )
        np.testing.assert_allclose(_logical_values(groups["linearAcceleration"]), ((1.0, 2.0, 3.0),))
        np.testing.assert_allclose(_logical_values(groups["angularAcceleration"]), ((0.1, 0.2, 0.3),))


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA not available"))],
)
def test_output_read_borrows_body_mass(device):
    with ovstage.Stage("output-read-body-mass") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, BODY_OUTPUT_SEMANTICS)
        binding = _binding(stage, pd, device)

        query = binding.query(paths=["/Body"])
        result = binding.read(query=query, attributes=("mass",))
        group = result.groups[0]

        assert pd.token_to_string(group.attribute) == "mass"
        assert pd.get_path_strings(group.prim_list) == ["/Body"]
        assert group.tensor(0).data == binding.model.body_mass.ptr
        assert group.tensor(0).dtype.lanes == 1
        assert bool(group.cuda_sync.wait_event) == (device != "cpu")
        np.testing.assert_allclose(_logical_values(group), (2.0,))


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA not available"))],
)
@pytest.mark.parametrize("world", [None, 0, 1, -1], ids=["implicit", "world-0", "world-1", "global"])
def test_output_read_borrows_effective_scene_gravity(device, world):
    with ovstage.Stage("output-read-scene-gravity") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, FALLING_BODY)
        if world is not None:
            source = newton.ModelBuilder()
            ovnewton.add_ovstage(source, stage)
            builder = newton.ModelBuilder()
            builder.gravity = (1.0, 2.0, 3.0)
            if world != 0:
                builder.begin_world()
                builder.add_particle(pos=(0, 0, 0), vel=(0, 0, 0), mass=1.0)
                builder.end_world()
            if world >= 0:
                builder.add_world(source)
            else:
                builder.add_builder(source)
            binding = ovnewton.attach_ovstage(stage, model=builder.finalize(device=device))
        else:
            binding = _binding(stage, pd, device)

        query = binding.query(paths=["/Scene"])
        assert query.prim_count == 1
        assert {pd.token_to_string(token) for token in query.attributes} == {"gravity"}
        result = binding.read(query=query, attributes=("gravity",))
        group = result.groups[0]

        assert pd.get_path_strings(group.prim_list) == ["/Scene"]
        assert group.data_count == 1
        row = 0 if world is None else (1 if world == -1 else world)
        assert group.tensor(0).data == binding.model.gravity.ptr + row * binding.model.gravity.strides[0]
        assert group.tensor(0).shape[0] == 1
        assert group.tensor(0).dtype.lanes == 3
        np.testing.assert_allclose(_logical_values(group), ((0.0, 0.0, -9.81),), atol=1.0e-6)


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA not available"))],
)
def test_output_read_rejects_ambiguous_scene_gravity(device):
    with ovstage.Stage("output-read-multiple-worlds") as stage:
        _populate(stage, BODY_SHAPE_OUTPUTS + '\ndef PhysicsScene "Scene" {}\n')
        builder = newton.ModelBuilder()
        for path, gravity in (("/A", (1, 2, 3)), ("/B", (0, 0, -9.81))):
            builder.begin_world(gravity=gravity)
            body = builder.add_body(label=path)
            builder.add_shape_sphere(body, radius=1.0)
            builder.end_world()
        binding = ovnewton.attach_ovstage(stage, model=builder.finalize(device=device))

        with pytest.raises(ValueError, match="cannot map.*gravity.*single Newton world"):
            binding.read(query=binding.query(paths=["/Scene"]), attributes=("gravity",))
        masses = binding.read(query=binding.query(paths=["/A", "/B"]), attributes=("mass",))
        assert masses.groups[0].data_count == 2


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA not available"))],
)
def test_output_read_groups_shape_materials_by_body(device):
    with ovstage.Stage("output-read-shape-materials") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, BODY_SHAPE_OUTPUTS)
        binding = _binding(stage, pd, device)
        labels = list(binding.model.shape_label)
        friction_by_path = {"/A/Shape0": 0.1, "/A/Shape1": 0.2, "/B/Shape2": 0.3}
        restitution_by_path = {"/A/Shape0": 0.4, "/A/Shape1": 0.5, "/B/Shape2": 0.6}
        binding.model.shape_material_mu.assign(
            wp.array([friction_by_path[path] for path in labels], dtype=wp.float32, device=binding.model.device)
        )
        binding.model.shape_material_restitution.assign(
            wp.array([restitution_by_path[path] for path in labels], dtype=wp.float32, device=binding.model.device)
        )

        query = binding.query(paths=["/B", "/A"])
        names = {pd.token_to_string(token) for token in query.attributes}
        assert {"friction", "restitution", "shapeCount"} <= names
        counts = binding.read(query=query, attributes=("shapeCount",)).groups[0]
        assert bool(counts.cuda_sync.wait_event) == (device != "cpu")
        np.testing.assert_array_equal(_logical_values(counts), (1, 2))
        groups = _groups_by_name(
            binding.read(query=query, attributes=("shapeCount", "friction", "restitution")),
            pd,
        )

        assert groups["shapeCount"].tensor(0).dtype.code == ovstage.DLDataTypeCode.kDLInt
        assert groups["shapeCount"].tensor(0).dtype.lanes == 1
        assert groups["friction"].tensor(0).dtype.lanes == 2
        assert groups["restitution"].tensor(0).dtype.lanes == 2
        np.testing.assert_array_equal(_logical_values(groups["shapeCount"]), (1, 2))
        shapes_by_body = {
            path: [
                shape
                for shape, body in enumerate(binding.model.shape_body.numpy())
                if int(body) == list(binding.model.body_label).index(path)
            ]
            for path in ("/B", "/A")
        }
        expected_friction = [
            [friction_by_path[labels[shape]] for shape in shapes_by_body[path]]
            + [0.0] * (2 - len(shapes_by_body[path]))
            for path in ("/B", "/A")
        ]
        expected_restitution = [
            [restitution_by_path[labels[shape]] for shape in shapes_by_body[path]]
            + [0.0] * (2 - len(shapes_by_body[path]))
            for path in ("/B", "/A")
        ]
        np.testing.assert_allclose(_logical_values(groups["friction"]), expected_friction)
        np.testing.assert_allclose(_logical_values(groups["restitution"]), expected_restitution)


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA not available"))],
)
@pytest.mark.parametrize("shape_count", [255, 256])
def test_output_read_shape_material_lane_limit(device, shape_count):
    with ovstage.Stage("output-read-shape-lane-limit") as stage:
        _populate(stage, BODY_SHAPE_OUTPUTS)
        builder = newton.ModelBuilder()
        ovnewton.add_ovstage(builder, stage)
        body = builder.body_label.index("/A")
        for _ in range(shape_count - builder.shape_body.count(body)):
            builder.add_shape_sphere(body, radius=0.01)
        binding = ovnewton.attach_ovstage(stage, model=builder.finalize(device=device))
        binding.model.shape_material_mu.fill_(0.25)
        binding.model.shape_material_restitution.fill_(0.5)

        query = binding.query(paths=["/A"])
        counts = binding.read(query=query, attributes=("shapeCount",)).groups[0]
        np.testing.assert_array_equal(_logical_values(counts), (shape_count,))
        assert binding.read(query=query, attributes=("mass",)).groups[0].data_count == 1
        for name, value in (("friction", 0.25), ("restitution", 0.5)):
            if shape_count > 255:
                with pytest.raises(ValueError, match=f"{name}.*255 shapes.*256"):
                    binding.read(query=query, attributes=(name,))
            else:
                group = binding.read(query=query, attributes=(name,)).groups[0]
                assert group.tensor(0).dtype.lanes == 255
                np.testing.assert_array_equal(_logical_values(group), np.full((1, 255), value))
                if device == "cpu":
                    np.testing.assert_array_equal(group.array(0), np.full(255, value))

            # A wide body elsewhere in the model must not prevent narrow queries.
            narrow = binding.read(query=binding.query(paths=["/B"]), attributes=(name,)).groups[0]
            np.testing.assert_array_equal(_logical_values(narrow), (value,))


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA not available"))],
)
def test_output_read_borrows_body_mass_frame_properties(device):
    with ovstage.Stage("output-read-body-mass-frame") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, BODY_OUTPUT_SEMANTICS)
        binding = _binding(stage, pd, device)

        result = binding.read(
            query=binding.query(paths=["/Body"]),
            attributes=("inertia", "centerOfMassPosition"),
        )
        groups = _groups_by_name(result, pd)

        assert groups["inertia"].tensor(0).data == binding.model.body_inertia.ptr
        assert groups["inertia"].tensor(0).dtype.lanes == 9
        assert groups["centerOfMassPosition"].tensor(0).data == binding.model.body_com.ptr
        assert groups["centerOfMassPosition"].tensor(0).dtype.lanes == 3
        np.testing.assert_allclose(
            _logical_values(groups["inertia"]).reshape(3, 3),
            np.diag((1.0, 2.0, 3.0)),
        )
        np.testing.assert_allclose(
            _logical_values(groups["centerOfMassPosition"]),
            ((0.25, 0.5, 0.75),),
        )


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA not available"))],
)
def test_output_read_uses_authored_articulation_root_identity(device):
    with ovstage.Stage("output-read-articulation-root") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, MIXED_JOINT_WIDTHS)
        binding = _binding(stage, pd, device)
        assert list(binding.model.articulation_label) == ["/World"]
        state = binding.model.state()
        root_body = list(binding.model.body_label).index("/World/Root")
        body_q = state.body_q.numpy()
        body_q[root_body] = (1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0)
        state.body_q.assign(wp.array(body_q, dtype=wp.transformf, device=binding.model.device))
        body_qd = state.body_qd.numpy()
        body_qd[root_body] = (4.0, 5.0, 6.0, 0.1, 0.2, 0.3)
        state.body_qd.assign(wp.array(body_qd, dtype=wp.spatial_vectorf, device=binding.model.device))

        query = binding.query(paths=["/World"])
        assert query.prim_count == 1
        assert {pd.token_to_string(token) for token in query.attributes} == {
            "rootPosition",
            "rootOrientation",
            "rootLinearVelocity",
            "rootAngularVelocity",
        }
        groups = _groups_by_name(
            binding.read(
                state,
                query=query,
                attributes=(
                    "rootPosition",
                    "rootOrientation",
                    "rootLinearVelocity",
                    "rootAngularVelocity",
                ),
            ),
            pd,
        )

        assert all(pd.get_path_strings(group.prim_list) == ["/World"] for group in groups.values())
        assert all(not group.has_data_index_map for group in groups.values())
        assert all(group.data_row_index(0) == 0 for group in groups.values())
        np.testing.assert_allclose(_logical_values(groups["rootPosition"]), ((1.0, 2.0, 3.0),))
        np.testing.assert_allclose(_logical_values(groups["rootOrientation"]), ((0.0, 0.0, 0.0, 1.0),))
        np.testing.assert_allclose(_logical_values(groups["rootLinearVelocity"]), ((4.0, 5.0, 6.0),))
        np.testing.assert_allclose(_logical_values(groups["rootAngularVelocity"]), ((0.1, 0.2, 0.3),))


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA not available"))],
)
@pytest.mark.parametrize("root_location", ["body", "ancestor", None])
def test_output_read_single_body_uses_only_authored_roots(device, root_location):
    ancestor_schema = 'apiSchemas = ["PhysicsArticulationRootAPI"]' if root_location == "ancestor" else ""
    body_schema = ', "PhysicsArticulationRootAPI"' if root_location == "body" else ""
    scene = f'''#usda 1.0
def Xform "World" ({ancestor_schema}) {{
    def Xform "Body" (apiSchemas = ["PhysicsRigidBodyAPI"{body_schema}]) {{
        double3 xformOp:translate = (1, 2, 3)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}
}}
'''
    with ovstage.Stage("output-read-single-body-root") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, scene)
        with wp.ScopedDevice(device):
            binding = ovnewton.attach_ovstage(stage)
        state = binding.model.state()
        attributes = ("rootPosition", "rootOrientation", "rootLinearVelocity", "rootAngularVelocity")

        # Newton's generated articulation label is not a USD prim path.
        with pytest.raises(ValueError, match="not an output-capable bound object"):
            binding.query(paths=["/World/Body_articulation"])

        if root_location is None:
            assert binding.read(state, attributes=attributes).groups == ()
        else:
            root_path = "/World/Body" if root_location == "body" else "/World"
            query = binding.query(paths=[root_path])
            assert query.prim_count == 1
            groups = _groups_by_name(binding.read(state, query=query, attributes=attributes), pd)
            assert set(groups) == set(attributes)
            assert all(pd.get_path_strings(group.prim_list) == [root_path] for group in groups.values())
            np.testing.assert_allclose(_logical_values(groups["rootPosition"]), ((1.0, 2.0, 3.0),))
            np.testing.assert_allclose(_logical_values(groups["rootOrientation"]), ((0.0, 0.0, 0.0, 1.0),))
            np.testing.assert_array_equal(_logical_values(groups["rootLinearVelocity"]), ((0.0, 0.0, 0.0),))
            np.testing.assert_array_equal(_logical_values(groups["rootAngularVelocity"]), ((0.0, 0.0, 0.0),))

        body = binding.read(state, query=binding.query(paths=["/World/Body"]), attributes=("body_q",)).groups[0]
        np.testing.assert_allclose(_logical_values(body), ((1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0),))


def test_output_read_ambiguous_free_body_root_preserves_import():
    scene = '''#usda 1.0
def Xform "World" (apiSchemas = ["PhysicsArticulationRootAPI"]) {
    def Xform "A" (apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def Xform "B" (apiSchemas = ["PhysicsRigidBodyAPI"]) {}
}
'''
    with ovstage.Stage("output-read-ambiguous-free-body-root") as stage:
        _populate(stage, scene)
        with wp.ScopedDevice("cpu"):
            binding = ovnewton.attach_ovstage(stage)
        state = binding.model.state()
        # The existing importer accepts these independent bodies. Do not turn
        # an ambiguous output identity into a simulation import failure.
        bodies = binding.read(state, attributes=("body_q",)).groups[0]
        assert bodies.prim_count == 2
        assert binding.read(state, attributes=("rootPosition",)).groups == ()


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA not available"))],
)
def test_output_read_converts_shared_joint_state_to_axis_units(device):
    with ovstage.Stage("output-read-shared-joint-state") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, MIXED_JOINT_WIDTHS)
        binding = _binding(stage, pd, device)
        state = binding.model.state()
        labels = list(binding.model.joint_label)
        q_starts = binding.model.joint_q_start.numpy()
        qd_starts = binding.model.joint_qd_start.numpy()

        joint_q = state.joint_q.numpy()
        joint_q[int(q_starts[labels.index("/World/HingeA")])] = np.deg2rad(35.0)
        state.joint_q.assign(wp.array(joint_q, dtype=wp.float32, device=binding.model.device))

        joint_qd = state.joint_qd.numpy()
        joint_qd[int(qd_starts[labels.index("/World/HingeA")])] = np.deg2rad(45.0)
        ball_start = int(qd_starts[labels.index("/World/Ball")])
        joint_qd[ball_start : ball_start + 3] = np.deg2rad((10.0, 20.0, 30.0))
        state.joint_qd.assign(wp.array(joint_qd, dtype=wp.float32, device=binding.model.device))

        query = binding.query(paths=["/World/HingeA", "/World/Ball"])
        names = {pd.token_to_string(token) for token in query.attributes}
        assert {"jointPosition", "jointVelocity"} <= names
        result = binding.read(
            state,
            query=query,
            attributes=("jointPosition", "jointVelocity"),
        )
        position = next(
            group for group in result.groups if pd.token_to_string(group.attribute) == "jointPosition"
        )
        velocities = [
            group for group in result.groups if pd.token_to_string(group.attribute) == "jointVelocity"
        ]

        assert pd.get_path_strings(position.prim_list) == ["/World/HingeA"]
        np.testing.assert_allclose(_logical_values(position), (35.0,), atol=1.0e-5)
        assert [pd.get_path_strings(group.prim_list) for group in velocities] == [
            ["/World/HingeA"],
            ["/World/Ball"],
        ]
        np.testing.assert_allclose(_logical_values(velocities[0]), (45.0,), atol=1.0e-5)
        np.testing.assert_allclose(_logical_values(velocities[1]), ((10.0, 20.0, 30.0),), atol=1.0e-5)


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA not available"))],
)
def test_output_read_exposes_current_joint_targets_and_properties(device):
    with ovstage.Stage("output-read-joint-properties") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, MIXED_JOINT_WIDTHS)
        binding = _binding(stage, pd, device)
        model = binding.model
        state = model.state()
        control = model.control()
        joint = list(model.joint_label).index("/World/HingeA")
        qd_index = int(model.joint_qd_start.numpy()[joint])
        target_index = int(model.joint_target_q_start.numpy()[joint])

        def assign(array, index, value):
            values = array.numpy()
            values[index] = value
            array.assign(wp.array(values, dtype=array.dtype, device=model.device))

        assign(control.joint_target_q, target_index, np.deg2rad(15.0))
        assign(control.joint_target_qd, qd_index, np.deg2rad(25.0))
        assign(model.joint_target_ke, qd_index, 100.0)
        assign(model.joint_target_kd, qd_index, 20.0)
        assign(model.joint_limit_lower, qd_index, np.deg2rad(-30.0))
        assign(model.joint_limit_upper, qd_index, np.deg2rad(60.0))
        assign(model.joint_velocity_limit, qd_index, np.deg2rad(90.0))
        assign(model.joint_effort_limit, qd_index, 50.0)
        assign(model.joint_armature, qd_index, 0.2)
        assign(model.joint_friction, qd_index, 0.3)

        query = binding.query(paths=["/World/HingeA"])
        attributes = (
            "jointPositionTarget",
            "jointVelocityTarget",
            "jointStiffness",
            "jointDamping",
            "jointLimit",
            "jointMaxVelocity",
            "jointMaxForce",
            "jointArmature",
            "jointFriction",
        )
        assert set(attributes) <= {pd.token_to_string(token) for token in query.attributes}
        with pytest.raises(ValueError, match="control is required"):
            binding.read(state, query=query, attributes=("jointPositionTarget",))

        groups = _groups_by_name(
            binding.read(state, control=control, query=query, attributes=attributes),
            pd,
        )
        assert _logical_values(groups["jointPositionTarget"])[0] == pytest.approx(15.0)
        assert _logical_values(groups["jointVelocityTarget"])[0] == pytest.approx(25.0)
        assert _logical_values(groups["jointStiffness"])[0] == pytest.approx(100.0 / (180.0 / np.pi))
        assert _logical_values(groups["jointDamping"])[0] == pytest.approx(20.0 / (180.0 / np.pi))
        np.testing.assert_allclose(_logical_values(groups["jointLimit"]), ((-30.0, 60.0),), atol=1.0e-5)
        assert _logical_values(groups["jointMaxVelocity"])[0] == pytest.approx(90.0)
        assert _logical_values(groups["jointMaxForce"])[0] == pytest.approx(50.0)
        assert _logical_values(groups["jointArmature"])[0] == pytest.approx(0.2)
        assert _logical_values(groups["jointFriction"])[0] == pytest.approx(0.3)
        assert groups["jointMaxForce"].tensor(0).data == model.joint_effort_limit.ptr + qd_index * 4
        assert groups["jointArmature"].tensor(0).data == model.joint_armature.ptr + qd_index * 4
        assert groups["jointFriction"].tensor(0).data == model.joint_friction.ptr + qd_index * 4

        model_only = _groups_by_name(
            binding.read(
                query=query,
                attributes=("jointMaxForce", "jointArmature", "jointFriction"),
            ),
            pd,
        )
        assert set(model_only) == {"jointMaxForce", "jointArmature", "jointFriction"}


def test_output_read_preserves_multi_axis_joint_limit_layout():
    with ovstage.Stage("output-read-joint-limits") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, MULTI_AXIS_D6_JOINT)
        binding = _binding(stage, pd)
        model = binding.model
        joint = list(model.joint_label).index("/World/Joint")
        start = int(model.joint_qd_start.numpy()[joint])

        lower = model.joint_limit_lower.numpy()
        upper = model.joint_limit_upper.numpy()
        lower[start : start + 2] = (-2.0, np.deg2rad(-45.0))
        upper[start : start + 2] = (3.0, np.deg2rad(60.0))
        model.joint_limit_lower.assign(wp.array(lower, dtype=wp.float32, device=model.device))
        model.joint_limit_upper.assign(wp.array(upper, dtype=wp.float32, device=model.device))

        result = binding.read(
            query=binding.query(paths=["/World/Joint"]),
            attributes=("jointLimit",),
        )
        group = result.groups[0]
        assert group.tensor(0).dtype.lanes == 4
        np.testing.assert_allclose(
            _logical_values(group),
            ((-2.0, 3.0, -45.0, 60.0),),
            atol=1.0e-5,
        )


def test_output_read_uses_authored_joint_body_order_and_unlimited_bounds():
    with ovstage.Stage("output-read-joint-body-order") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, REVERSED_WORLD_JOINT)
        binding = _binding(stage, pd)
        model = binding.model
        state = model.state()
        control = model.control()
        joint = list(model.joint_label).index("/World/Joint")
        q = int(model.joint_q_start.numpy()[joint])
        qd = int(model.joint_qd_start.numpy()[joint])
        target = int(model.joint_target_q_start.numpy()[joint])

        def assign(array, index, value):
            values = array.numpy()
            values[index] = value
            array.assign(wp.array(values, dtype=array.dtype, device=model.device))

        assign(state.joint_q, q, np.deg2rad(10.0))
        assign(state.joint_qd, qd, np.deg2rad(20.0))
        assign(control.joint_target_q, target, np.deg2rad(30.0))
        assign(control.joint_target_qd, qd, np.deg2rad(40.0))

        groups = _groups_by_name(
            binding.read(
                state,
                control=control,
                query=binding.query(paths=["/World/Joint"]),
                attributes=(
                    "jointPosition",
                    "jointVelocity",
                    "jointPositionTarget",
                    "jointVelocityTarget",
                    "jointLimit",
                ),
            ),
            pd,
        )
        assert _logical_values(groups["jointPosition"])[0] == pytest.approx(-10.0)
        assert _logical_values(groups["jointVelocity"])[0] == pytest.approx(-20.0)
        assert _logical_values(groups["jointPositionTarget"])[0] == pytest.approx(-30.0)
        assert _logical_values(groups["jointVelocityTarget"])[0] == pytest.approx(-40.0)
        np.testing.assert_array_equal(
            _logical_values(groups["jointLimit"]),
            ((-np.finfo(np.float32).max, np.finfo(np.float32).max),),
        )


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA not available"))],
)
@pytest.mark.parametrize("explicit_query", [False, True])
def test_output_read_borrows_complete_native_state(device, explicit_query, monkeypatch):
    with ovstage.Stage("output-read-borrow") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, DRIVEN_REVOLUTE)
        binding = _binding(stage, pd, device)
        state = _state(binding)
        all_paths = [*binding.model.body_label, "/Joint"]
        query = binding.query(paths=all_paths) if explicit_query else None
        joint = list(binding.model.joint_label).index("/Joint")
        q_start = int(binding.model.joint_q_start.numpy()[joint])
        qd_start = int(binding.model.joint_qd_start.numpy()[joint])
        launches = []
        launch = wp.launch

        def tracked_launch(*args, **kwargs):
            launches.append(None)
            return launch(*args, **kwargs)

        monkeypatch.setattr(wp, "launch", tracked_launch)
        if device != "cpu":
            def unexpected_sync(*args, **kwargs):
                raise AssertionError("zero-copy output read synchronized the host")

            monkeypatch.setattr(wp, "synchronize_device", unexpected_sync)
            monkeypatch.setattr(wp, "synchronize_stream", unexpected_sync)
            monkeypatch.setattr(wp, "synchronize_event", unexpected_sync)
        stream = wp.Stream(binding.model.device) if device != "cpu" else None
        scope = wp.ScopedStream(stream) if stream is not None else contextlib.nullcontext()
        with scope:
            result = binding.read(
                state,
                query=query,
                attributes=("body_q", "body_qd", "joint_q", "joint_qd"),
            )
        assert launches == []
        groups = _groups_by_name(result, pd)
        assert groups["body_q"].tensor(0).data == state.body_q.ptr
        assert groups["body_qd"].tensor(0).data == state.body_qd.ptr
        assert groups["body_q"].data_index_tensor() is None
        assert groups["body_q"].data_index_array() is None
        assert groups["body_q"].data_index_dlpack() is None

        assert groups["joint_q"].tensor(0).data == state.joint_q.ptr + q_start * 4
        assert groups["joint_qd"].tensor(0).data == state.joint_qd.ptr + qd_start * 4

        with pytest.raises(ValueError, match="read-only"):
            groups["body_q"].dlpack(0, readonly=False)
        if device == "cpu":
            body_q = groups["body_q"].array(0)
            assert not body_q.flags.writeable
            updated = state.body_q.numpy()
            updated[0, 0] += 1.0
            state.body_q.assign(wp.array(updated, dtype=wp.transformf, device=binding.model.device))
            assert body_q.reshape(-1, 7)[0, 0] == updated[0, 0]
            assert all(group.cuda_sync.wait_event == 0 for group in result.groups)
        else:
            assert all(group.cuda_sync.wait_event for group in result.groups)


def test_output_group_matches_ovstage_read_group_accessors():
    with ovstage.Stage("output-read-protocol") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, DRIVEN_REVOLUTE)
        binding = _binding(stage, pd)

        selected_path = _body_path_at(binding, 1)
        query = binding.query(paths=[selected_path])
        result = binding.read(binding.model.state(), query=query, attributes=("body_q",))
        output_group = result.groups[0]
        # Newton output groups borrow caller-owned state and have no pinned
        # ovstage storage to release.
        required = {
            name
            for name in dir(ovstage.ReadGroup)
            if not name.startswith("_") and name not in {"release", "released"}
        }
        assert required <= {name for name in dir(output_group) if not name.startswith("_")}
        assert isinstance(output_group.meta, ovnewton.ReadGroupMeta)
        assert isinstance(output_group.cuda_sync, ovnewton.ReadGroupCudaSync)
        assert output_group.ordinal == 0
        assert not output_group.is_delete
        assert output_group.semantic == int(ovstage.AttributeSemantic.NONE)
        assert output_group.meta.attribute_write_floor_ordinal == 0
        assert output_group.meta.layout_generation == 0
        assert output_group.cuda_sync.stream == 0
        assert output_group.cuda_sync.wait_event == 0

        with stage.query_from_path_list(output_group.prim_list) as stage_query:
            stage_query.wait()
            with stage.read_attributes(
                stage_query,
                [pd.intern_token("omni:fabric:worldMatrix")],
                ovstage.OrdinalRange.latest(1),
            ) as read:
                read.wait()
                stage_group = read.fetch_next()
                try:
                    assert stage_group is not None
                    assert not stage_group.is_delete
                    output_layout = _dense_group_layout(output_group)
                    stage_layout = _dense_group_layout(stage_group)
                    for name in output_layout.keys() - {"data_rows", "has_data_index_map"}:
                        assert output_layout[name] == stage_layout[name]
                    assert output_group.has_data_index_map
                    assert output_group.data_row_index(0) == 1
                finally:
                    if stage_group is not None:
                        stage.release_group(stage_group)


def test_output_read_accepts_discovery_tokens_without_reverse_lookup(monkeypatch):
    with ovstage.Stage("output-read-tokens") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, DRIVEN_REVOLUTE)
        binding = _binding(stage, pd)

        query = binding.query(paths=["/Arm"])
        body_q = next(token for token in query.attributes if pd.token_to_string(token) == "body_q")

        def unexpected_reverse_lookup(_token):
            raise AssertionError("unexpected reverse token lookup")

        monkeypatch.setattr(binding._pd, "token_to_string", unexpected_reverse_lookup)
        result = binding.read(binding.model.state(), query=query, attributes=(body_q, "body_q"))
        assert len(result.groups) == 1
        assert pd.token_to_string(result.groups[0].attribute) == "body_q"

        result = binding.read(binding.model.state(), query=query, attributes=())
        assert result.groups == ()


def test_output_views_retain_query_storage_until_last_reference_drops(monkeypatch):
    with ovstage.Stage("output-read-owner") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, DRIVEN_REVOLUTE)
        binding = _binding(stage, pd)
        destroyed = []
        destroy_path_list = binding._pd.destroy_path_list

        def tracked_destroy(path_list):
            destroyed.append(path_list)
            destroy_path_list(path_list)

        monkeypatch.setattr(binding._pd, "destroy_path_list", tracked_destroy)
        query = binding.query(paths=[_body_path_at(binding, 1)])
        result = binding.read(binding.model.state(), query=query, attributes=("body_q",))
        group = result.groups[0]
        prim_list = group.prim_list
        dlpack = group.dlpack(0)
        data_index_dlpack = group.data_index_dlpack()
        query_ref = weakref.ref(query)
        result_ref = weakref.ref(result)
        group_ref = weakref.ref(group)

        del result, query, group
        gc.collect()
        assert result_ref() is None
        assert group_ref() is not None
        assert query_ref() is not None
        np.testing.assert_equal(wp.from_dlpack(dlpack).shape, (binding.model.body_count, 7))
        np.testing.assert_equal(wp.from_dlpack(data_index_dlpack).shape, (1,))

        del dlpack
        gc.collect()
        assert group_ref() is not None
        assert query_ref() is not None

        del data_index_dlpack
        gc.collect()
        assert group_ref() is None
        assert query_ref() is None
        assert prim_list in destroyed


def test_output_read_validates_only_requested_sources():
    with ovstage.Stage("output-read-sources") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, DRIVEN_REVOLUTE)
        binding = _binding(stage, pd)
        state = _state(binding)

        query = binding.query(paths=["/Arm"])
        result = binding.read(
            SimpleNamespace(body_q=state.body_q),
            query=query,
            attributes=("body_q",),
        )
        assert len(result.groups) == 1

        query = binding.query(paths=["/Joint"])
        result = binding.read(
            SimpleNamespace(joint_q=state.joint_q),
            query=query,
            attributes=("joint_q",),
        )
        assert len(result.groups) == 1

        with pytest.raises(ValueError, match="state is required"):
            binding.read(query=query, attributes=("joint_q",))


def test_output_read_indexes_in_explicit_path_order():
    with ovstage.Stage("output-read-order") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, DRIVEN_REVOLUTE)
        binding = _binding(stage, pd)
        state = _state(binding)

        query = binding.query(paths=["/Arm", "/Base"])
        result = binding.read(state, query=query, attributes=("body_q",))
        group = result.groups[0]
        assert pd.get_path_strings(group.prim_list) == ["/Arm", "/Base"]
        expected = [state.body_q.numpy()[list(binding.model.body_label).index(path)] for path in ("/Arm", "/Base")]
        np.testing.assert_allclose(_logical_values(group), expected)


@pytest.mark.parametrize(
    ("usda", "path", "q_values", "qd_values", "expected_axes"),
    (
        (
            SPHERICAL_JOINT,
            "/World/Joint",
            (0.1, 0.2, 0.3, 0.9),
            (0.4, 0.5, 0.6),
            None,
        ),
        (
            DISTANCE_JOINT,
            "/World/Joint",
            (1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9),
            (4.0, 5.0, 6.0, 0.4, 0.5, 0.6),
            None,
        ),
        (MULTI_AXIS_D6_JOINT, "/World/Joint", (7.0, 8.0), (9.0, 10.0), ((0, 1, 0), (1, 0, 0))),
        (
            BODY_OUTPUT_SEMANTICS,
            None,
            (11.0, 12.0, 13.0, 0.4, 0.3, 0.2, 0.8),
            (14.0, 15.0, 16.0, 0.7, 0.8, 0.9),
            None,
        ),
    ),
    ids=("ball", "distance", "d6", "generated-free"),
)
def test_output_read_preserves_newton_multivalue_joint_layouts(usda, path, q_values, qd_values, expected_axes):
    with ovstage.Stage("output-read-multivalue-joint") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, usda)
        binding = _binding(stage, pd)
        labels = list(binding.model.joint_label)
        if path is None:
            assert len(labels) == 1
            path = labels[0]
        joint = labels.index(path)
        q_start = binding.model.joint_q_start.numpy()
        qd_start = binding.model.joint_qd_start.numpy()
        q_slice = slice(int(q_start[joint]), int(q_start[joint + 1]))
        qd_slice = slice(int(qd_start[joint]), int(qd_start[joint + 1]))
        assert q_slice.stop - q_slice.start == len(q_values)
        assert qd_slice.stop - qd_slice.start == len(qd_values)

        joint_q = binding.model.joint_q.numpy()
        joint_qd = binding.model.joint_qd.numpy()
        joint_q[q_slice] = q_values
        joint_qd[qd_slice] = qd_values
        state = binding.model.state()
        state.joint_q.assign(wp.array(joint_q, dtype=wp.float32, device=binding.model.device))
        state.joint_qd.assign(wp.array(joint_qd, dtype=wp.float32, device=binding.model.device))

        groups = _groups_by_name(
            binding.read(
                state,
                query=binding.query(paths=[path]),
                attributes=("joint_q", "joint_qd"),
            ),
            pd,
        )
        np.testing.assert_array_equal(
            _logical_values(groups["joint_q"]).reshape(-1),
            np.asarray(q_values, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            _logical_values(groups["joint_qd"]).reshape(-1),
            np.asarray(qd_values, dtype=np.float32),
        )

        if expected_axes is not None:
            np.testing.assert_array_equal(binding.model.joint_axis.numpy()[qd_slice], expected_axes)


def test_output_read_excludes_fixed_joints_without_coordinates():
    with ovstage.Stage("output-read-fixed-joint") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, FIXED_JOINT)
        binding = _binding(stage, pd)

        with pytest.raises(ValueError, match="not an output-capable bound object"):
            binding.query(paths=["/World/Joint"])


@pytest.mark.parametrize(
    ("usda", "channel", "angular"),
    ((DRIVEN_REVOLUTE, "angular", True), (DRIVEN_PRISMATIC, "linear", False)),
    ids=("revolute", "prismatic"),
)
def test_output_read_preserves_newton_single_dof_units(usda, channel, angular):
    with ovstage.Stage(f"output-read-{channel}-units") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, usda)
        binding = _binding(stage, pd)
        _write_scalar(stage, pd, "/Joint", f"state:{channel}:physics:position", 12.0, 2)
        _write_scalar(stage, pd, "/Joint", f"state:{channel}:physics:velocity", 34.0, 2)
        stage.advance_write_floor(ordinal=2).wait()

        state = binding.model.state()
        binding.update_from_ovstage(state, control=binding.model.control(), ordinal=2)
        query = binding.query(paths=["/Joint"])
        groups = _groups_by_name(
            binding.read(state, query=query, attributes=("joint_q", "joint_qd")),
            pd,
        )

        scale = np.pi / 180.0 if angular else 1.0
        np.testing.assert_allclose(_logical_values(groups["joint_q"]), (12.0 * scale,))
        np.testing.assert_allclose(_logical_values(groups["joint_qd"]), (34.0 * scale,))


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA not available"))],
)
def test_output_read_batches_joint_groups_by_coordinate_width(device, monkeypatch):
    with ovstage.Stage("output-read-joint-widths") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, MIXED_JOINT_WIDTHS)
        binding = _binding(stage, pd, device)
        state = binding.model.state()
        q_start = binding.model.joint_q_start.numpy()
        qd_start = binding.model.joint_qd_start.numpy()
        labels = list(binding.model.joint_label)
        joint_q = state.joint_q.numpy()
        joint_qd = state.joint_qd.numpy()
        values = {
            "/World/HingeA": ((1.0,), (10.0,)),
            "/World/HingeB": ((2.0,), (20.0,)),
            "/World/Ball": ((0.1, 0.2, 0.3, 0.9), (3.0, 4.0, 5.0)),
        }
        for path, (q_values, qd_values) in values.items():
            joint = labels.index(path)
            joint_q[int(q_start[joint]) : int(q_start[joint + 1])] = q_values
            joint_qd[int(qd_start[joint]) : int(qd_start[joint + 1])] = qd_values
        state.joint_q.assign(wp.array(joint_q, dtype=wp.float32, device=binding.model.device))
        state.joint_qd.assign(wp.array(joint_qd, dtype=wp.float32, device=binding.model.device))

        query = binding.query(paths=["/World/HingeB", "/World/Ball", "/World/HingeA"])
        launches = []
        launch = wp.launch

        def tracked_launch(*args, **kwargs):
            launches.append(None)
            return launch(*args, **kwargs)

        monkeypatch.setattr(wp, "launch", tracked_launch)
        result = binding.read(state, query=query, attributes=("joint_q", "joint_qd"))
        assert len(launches) == 2
        groups = {}
        for group in result.groups:
            name = pd.token_to_string(group.attribute)
            width = int(group.tensor(0).dtype.lanes)
            groups[name, width] = group

        assert set(groups) == {("joint_q", 1), ("joint_q", 4), ("joint_qd", 1), ("joint_qd", 3)}
        assert all(not group.is_array and group.tensor_count == 1 for group in result.groups)

        q_scalar = groups["joint_q", 1]
        assert q_scalar.prim_count == 2
        assert pd.get_path_strings(q_scalar.prim_list) == ["/World/HingeB", "/World/HingeA"]
        assert not q_scalar.has_data_index_map
        np.testing.assert_array_equal(_logical_values(q_scalar).reshape(2, 1), ((2.0,), (1.0,)))

        q_ball = groups["joint_q", 4]
        assert pd.get_path_strings(q_ball.prim_list) == ["/World/Ball"]
        np.testing.assert_allclose(
            _logical_values(q_ball).reshape(1, 4),
            ((0.1, 0.2, 0.3, 0.9),),
        )

        qd_scalar = groups["joint_qd", 1]
        assert pd.get_path_strings(qd_scalar.prim_list) == ["/World/HingeB", "/World/HingeA"]
        assert not qd_scalar.has_data_index_map
        np.testing.assert_array_equal(_logical_values(qd_scalar).reshape(2, 1), ((20.0,), (10.0,)))

        qd_ball = groups["joint_qd", 3]
        assert pd.get_path_strings(qd_ball.prim_list) == ["/World/Ball"]
        np.testing.assert_array_equal(
            _logical_values(qd_ball).reshape(1, 3),
            ((3.0, 4.0, 5.0),),
        )

        launches.clear()
        complete = binding.read(state, attributes=("joint_q", "joint_qd"))
        assert len(launches) == 2
        for group in complete.groups:
            name = pd.token_to_string(group.attribute)
            value_index = 0 if name == "joint_q" else 1
            expected = [values[path][value_index] for path in pd.get_path_strings(group.prim_list)]
            np.testing.assert_allclose(
                _logical_values(group).reshape(group.prim_count, -1),
                expected,
            )
            assert not group.has_data_index_map

        hinges = binding.query(paths=["/World/HingeB", "/World/HingeA"])
        launches.clear()
        selected = binding.read(state, query=hinges, attributes=("joint_q", "joint_qd"))
        assert launches == []
        groups = _groups_by_name(selected, pd)
        assert groups["joint_q"].tensor(0).data == state.joint_q.ptr + int(q_start[labels.index("/World/HingeA")]) * 4
        assert groups["joint_qd"].tensor(0).data == (
            state.joint_qd.ptr + int(qd_start[labels.index("/World/HingeA")]) * 4
        )
        assert all(group.has_data_index_map for group in groups.values())
        np.testing.assert_array_equal(_logical_values(groups["joint_q"]).reshape(2, 1), ((2.0,), (1.0,)))
        np.testing.assert_array_equal(_logical_values(groups["joint_qd"]).reshape(2, 1), ((20.0,), (10.0,)))


def test_selected_output_results_share_sources_with_distinct_indices():
    with ovstage.Stage("output-read-storage") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, DRIVEN_REVOLUTE)
        binding = _binding(stage, pd)
        state = _state(binding)

        first = binding.query(paths=[_body_path_at(binding, 0)])
        second = binding.query(paths=[_body_path_at(binding, 1)])
        first_result = binding.read(state, query=first, attributes=("body_q",))
        second_result = binding.read(state, query=second, attributes=("body_q",))
        assert first_result.groups[0].tensor(0).data == state.body_q.ptr
        assert second_result.groups[0].tensor(0).data == state.body_q.ptr
        assert first_result.groups[0].data_index_tensor() is None
        assert second_result.groups[0].data_index_tensor() is not None
        assert first_result.groups[0].data_row_index(0) != second_result.groups[0].data_row_index(0)
        binding.update_to_ovstage(state, ordinal=2)

        result = binding.read(state, attributes=("body_q", "body_qd"))
        assert len(result.groups) == 2
        groups = _groups_by_name(result, pd)
        assert groups["body_q"].tensor(0).data == state.body_q.ptr
        assert groups["body_qd"].tensor(0).data == state.body_qd.ptr


def test_query_validation_is_strict_only_for_explicit_paths():
    with ovstage.Stage("output-read-validation") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, DRIVEN_REVOLUTE)
        binding = _binding(stage, pd)

        with pytest.raises(TypeError, match="stage_query, paths, or object_type"):
            binding.query()
        with pytest.raises(ValueError, match="output-capable"):
            binding.query(paths=["/NotReadable"])
        with pytest.raises(ValueError, match="absolute prim paths"):
            binding.query(paths=[[]])
        with pytest.raises(ValueError, match="duplicate"):
            binding.query(paths=["/Arm", "/Arm"])

        query = binding.query(paths=["/Arm"])
        for attribute in (
            "centerOfMassOrientation",
            "staticFriction",
            "dynamicFriction",
            "temperature",
            "jointActuationForce",
            "jointDriveType",
            "jointStaticFriction",
            "jointDynamicFriction",
        ):
            with pytest.raises(ValueError, match="unsupported output attribute"):
                binding.read(binding.model.state(), query=query, attributes=(attribute,))
        result = binding.read(binding.model.state(), query=query, attributes=("joint_q",))
        assert result.groups == ()

        query = binding.query(paths=[])
        assert query.prim_count == 0
        result = binding.read(binding.model.state(), query=query, attributes=("body_q",))
        assert result.groups == ()


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA not available")
def test_output_read_cuda_group_is_device_resident(monkeypatch):
    with ovstage.Stage("output-read-cuda") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, DRIVEN_REVOLUTE)
        binding = _binding(stage, pd, "cuda:0")
        state = _state(binding)
        stream = wp.Stream(binding.model.device)
        launches = []
        launch = wp.launch

        def unexpected_sync(*args, **kwargs):
            raise AssertionError("output read synchronized the host")

        def tracked_launch(*args, **kwargs):
            launches.append(kwargs.get("stream"))
            return launch(*args, **kwargs)

        monkeypatch.setattr(wp, "synchronize_device", unexpected_sync)
        monkeypatch.setattr(wp, "synchronize_stream", unexpected_sync)
        monkeypatch.setattr(wp, "synchronize_event", unexpected_sync)
        monkeypatch.setattr(wp, "launch", tracked_launch)
        query = binding.query(paths=[_body_path_at(binding, 1)])
        with wp.ScopedStream(stream):
            result = binding.read(state, query=query, attributes=("body_q", "body_qd"))
            assert len(result.groups) == 2
            for group in result.groups:
                assert group.tensor(0).device.device_type.value == ovstage.DLDeviceType.kDLCUDA
                assert group.data_index_tensor().device.device_type.value == ovstage.DLDeviceType.kDLCUDA
                assert group.cuda_sync.stream == 0
                assert group.cuda_sync.wait_event
                with pytest.raises(NotImplementedError):
                    group.array(0)
        assert launches == []


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA not available")
def test_output_reads_do_not_wait_for_other_results(monkeypatch):
    with ovstage.Stage("output-read-cuda-independent") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, DRIVEN_REVOLUTE)
        binding = _binding(stage, pd, "cuda:0")
        state = _state(binding)
        first_stream = wp.Stream(binding.model.device)
        second_stream = wp.Stream(binding.model.device)
        waits = []
        wait_event = wp.Stream.wait_event

        def tracked_wait_event(stream, event, external=False):
            waits.append((stream, int(event.cuda_event)))
            return wait_event(stream, event, external=external)

        monkeypatch.setattr(wp.Stream, "wait_event", tracked_wait_event)
        query = binding.query(paths=["/Arm"])
        with wp.ScopedStream(first_stream):
            first = binding.read(state, query=query, attributes=("body_q",))
        with wp.ScopedStream(second_stream):
            second = binding.read(state, query=query, attributes=("body_q",))
        assert first.groups[0].cuda_sync.wait_event
        assert second.groups[0].cuda_sync.wait_event
        assert first.groups[0].cuda_sync.wait_event != second.groups[0].cuda_sync.wait_event
        assert waits == []


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA not available")
def test_output_read_cuda_graph_captures_zero_copy_reads():
    with ovstage.Stage("output-read-cuda-graph") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, DRIVEN_REVOLUTE)
        binding = _binding(stage, pd, "cuda:0")
        state = _state(binding)
        stream = wp.Stream(binding.model.device)

        query = binding.query(paths=["/Arm"])
        with wp.ScopedStream(stream):
            binding.read(state, query=query, attributes=("body_q", "body_qd"))
            wp.synchronize_stream(stream)

            with wp.ScopedCapture(stream=stream) as capture:
                result = binding.read(state, query=query, attributes=("body_q", "body_qd"))

            wp.capture_launch(capture.graph, stream=stream)
            wp.synchronize_stream(stream)
            groups = _groups_by_name(result, pd)
            arm = list(binding.model.body_label).index("/Arm")
            np.testing.assert_allclose(
                _logical_values(groups["body_q"])[0],
                state.body_q.numpy()[arm],
            )
            np.testing.assert_allclose(
                _logical_values(groups["body_qd"])[0],
                state.body_qd.numpy()[arm],
            )
