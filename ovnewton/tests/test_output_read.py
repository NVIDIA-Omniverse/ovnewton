# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import gc
import inspect
import weakref
from types import SimpleNamespace

import numpy as np
import ovstage
import pytest
import warp as wp

import ovnewton
from ovnewton._src import _build

from .test_runtime import DRIVEN_REVOLUTE, SPHERICAL_JOINT, _populate, _revolute_indices

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


def _groups_by_name(result, pd):
    return {pd.token_to_string(group.attribute): group for group in result.groups}


def _logical_values(group):
    values = wp.from_dlpack(group.dlpack(0)).numpy()
    if group.has_data_index_map:
        values = values[wp.from_dlpack(group.data_index_dlpack()).numpy()]
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

    assert tuple(query) == ("self", "stage_query", "paths")
    assert query["stage_query"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert query["paths"].kind is inspect.Parameter.KEYWORD_ONLY
    assert tuple(read) == ("self", "state", "query", "attributes")
    assert read["state"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert read["query"].kind is inspect.Parameter.KEYWORD_ONLY
    assert read["attributes"].kind is inspect.Parameter.KEYWORD_ONLY
    for value_type in (ovnewton.Query, ovnewton.ReadResult, ovnewton.ReadGroup):
        assert not hasattr(value_type, "release")
        assert not hasattr(value_type, "close")
        assert not hasattr(value_type, "__enter__")


def test_query_discovers_native_arrays_and_ignores_other_prims():
    with ovstage.Stage("output-read-discovery") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, DRIVEN_REVOLUTE)
        binding = _binding(stage, pd)

        with stage.query() as stage_query:
            query = binding.query(stage_query)
        names = {pd.token_to_string(token) for token in query.attributes}
        assert isinstance(query, ovnewton.Query)
        assert query.prim_count == binding.model.body_count + 1
        assert names == {"body_q", "body_qd", "joint_q", "joint_qd"}


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
        assert names == {"body_q", "body_qd", "joint_q", "joint_qd"}


def test_output_read_indexes_borrowed_newton_body_arrays():
    with ovstage.Stage("output-read-values") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, DRIVEN_REVOLUTE)
        binding = _binding(stage, pd)
        state = _state(binding)
        arm = list(binding.model.body_label).index("/Arm")

        query = binding.query(paths=["/Arm"])
        result = binding.read(state, query=query, attributes=("body_q", "body_qd"))
        groups = _groups_by_name(result, pd)
        assert isinstance(result, ovnewton.ReadResult)
        assert all(isinstance(group, ovnewton.ReadGroup) for group in result.groups)
        assert set(groups) == {"body_q", "body_qd"}
        assert all(pd.get_path_strings(group.prim_list) == ["/Arm"] for group in groups.values())
        assert all(group.prim_count == 1 and group.tensor_count == 1 for group in groups.values())
        assert all(not group.is_array for group in groups.values())
        assert all(group.has_data_index_map for group in groups.values())
        assert groups["body_q"].tensor(0).data == state.body_q.ptr
        assert groups["body_qd"].tensor(0).data == state.body_qd.ptr
        assert groups["body_q"].data_row_index(0) == arm
        assert groups["body_q"].data_index_tensor().dtype.code == ovstage.DLDataTypeCode.kDLUInt
        data_indices = groups["body_q"].data_index_array()
        assert not data_indices.flags.writeable
        np.testing.assert_array_equal(data_indices, (arm,))
        np.testing.assert_array_equal(
            wp.from_dlpack(groups["body_q"].data_index_dlpack()).numpy(),
            (arm,),
        )
        with pytest.raises(ValueError, match="read-only"):
            groups["body_q"].data_index_dlpack(readonly=False)
        with pytest.raises(IndexError, match="out of range"):
            groups["body_q"].data_row_index(1)
        np.testing.assert_allclose(_logical_values(groups["body_q"])[0], state.body_q.numpy()[arm])
        np.testing.assert_allclose(_logical_values(groups["body_qd"])[0], state.body_qd.numpy()[arm])


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

        query = binding.query(paths=["/Arm"])
        result = binding.read(binding.model.state(), query=query, attributes=("body_q",))
        output_group = result.groups[0]
        required = {name for name in dir(ovstage.ReadGroup) if not name.startswith("_")}
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
                    assert output_group.data_row_index(0) == list(binding.model.body_label).index("/Arm")
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
        query = binding.query(paths=["/Arm"])
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


def test_output_read_preserves_newton_joint_layout_and_units():
    with ovstage.Stage("output-read-joint") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, SPHERICAL_JOINT)
        binding = _binding(stage, pd)
        state = binding.model.state()
        joint = list(binding.model.joint_label).index("/World/Joint")
        q_start = binding.model.joint_q_start.numpy()
        qd_start = binding.model.joint_qd_start.numpy()
        q_slice = slice(int(q_start[joint]), int(q_start[joint + 1]))
        qd_slice = slice(int(qd_start[joint]), int(qd_start[joint + 1]))
        joint_q = state.joint_q.numpy()
        joint_qd = state.joint_qd.numpy()
        joint_q[q_slice] = np.asarray((0.1, 0.2, 0.3, 0.9), dtype=np.float32)
        joint_qd[qd_slice] = np.asarray((0.4, 0.5, 0.6), dtype=np.float32)
        state.joint_q.assign(wp.array(joint_q, dtype=wp.float32, device=binding.model.device))
        state.joint_qd.assign(wp.array(joint_qd, dtype=wp.float32, device=binding.model.device))

        query = binding.query(paths=["/World/Joint"])
        result = binding.read(state, query=query, attributes=("joint_q", "joint_qd"))
        groups = _groups_by_name(result, pd)
        assert set(groups) == {"joint_q", "joint_qd"}
        assert all(not group.is_array for group in groups.values())
        assert all(group.prim_count == 1 and group.tensor_count == 1 for group in groups.values())
        np.testing.assert_array_equal(groups["joint_q"].array(0).reshape(-1), joint_q[q_slice])
        np.testing.assert_array_equal(groups["joint_qd"].array(0).reshape(-1), joint_qd[qd_slice])


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

        first = binding.query(paths=["/Base"])
        second = binding.query(paths=["/Arm"])
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

        with pytest.raises(TypeError, match="exactly one"):
            binding.query()
        with pytest.raises(ValueError, match="output-capable"):
            binding.query(paths=["/Scene"])
        with pytest.raises(ValueError, match="absolute prim paths"):
            binding.query(paths=[[]])
        with pytest.raises(ValueError, match="duplicate"):
            binding.query(paths=["/Arm", "/Arm"])

        query = binding.query(paths=["/Arm"])
        with pytest.raises(ValueError, match="unsupported output attribute"):
            binding.read(binding.model.state(), query=query, attributes=("position",))
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
        query = binding.query(paths=["/Arm"])
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
