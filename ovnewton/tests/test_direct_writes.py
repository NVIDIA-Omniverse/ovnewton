# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext

import numpy as np
import ovstage
import pytest
import warp as wp

import ovnewton

from .test_output_read import MIXED_JOINT_WIDTHS, _logical_values, _unique_columns_by_name
from .test_runtime import DRIVEN_PRISMATIC, DRIVEN_REVOLUTE, _populate


@wp.kernel
def _offset_target(
    positions: wp.array(dtype=wp.float32),
    target: wp.array(dtype=wp.float32),
    row: int,
):
    target[0] = positions[row] + 0.25


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("scene", [DRIVEN_REVOLUTE, DRIVEN_PRISMATIC], ids=["revolute", "prismatic"])
@pytest.mark.parametrize("reversed_order", [False, True])
def test_native_target_write_round_trip(device, scene, reversed_order):
    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    scene = scene.replace("rel physics:body0 = </Base>", "")
    scene = scene.replace(
        'apiSchemas = ["PhysicsDriveAPI:', 'apiSchemas = ["PhysicsArticulationRootAPI", "PhysicsDriveAPI:'
    )
    if reversed_order:
        scene = scene.replace("rel physics:body1 = </Arm>", "rel physics:body0 = </Arm>")
    with wp.ScopedDevice(device), ovstage.Stage("direct-targets") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, scene)
        binding = ovnewton.attach_ovstage(stage)
        state, control = binding.model.state(), binding.model.control()
        before_pose = state.body_q.numpy().copy()
        query = binding.query(paths=["/Joint"])
        values = {
            "joint_target_q": wp.array([0.75], dtype=wp.float32, device=device),
            "joint_target_qd": wp.array([-0.5], dtype=wp.float32, device=device),
        }
        binding.write(values, control=control, query=query)
        read = _unique_columns_by_name(binding.read(control=control, query=query, attributes=tuple(values)), pd)
        np.testing.assert_allclose(_logical_values(read["joint_target_q"]), 0.75)
        np.testing.assert_allclose(_logical_values(read["joint_target_qd"]), -0.5)
        np.testing.assert_array_equal(state.body_q.numpy(), before_pose)
        for source in values.values():
            source.fill_(999)
        np.testing.assert_allclose(_logical_values(read["joint_target_q"]), 0.75)


@pytest.mark.parametrize("device,nondefault_stream", [("cpu", False), ("cuda:0", False), ("cuda:0", True)])
def test_native_controller_reads_state_and_writes_only_selected_target(device, nondefault_stream):
    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    stream = wp.ScopedStream(wp.Stream(device)) if nondefault_stream else nullcontext()
    with (
        wp.ScopedDevice(device), stream,
        ovstage.Stage("direct-controller") as stage, ovstage.PathDictionary(stage) as pd,
    ):
        _populate(stage, MIXED_JOINT_WIDTHS)
        binding = ovnewton.attach_ovstage(stage)
        state, control = binding.model.state(), binding.model.control()
        control.joint_target_q.fill_(4)
        control.joint_target_qd.fill_(5)
        query = binding.query(paths=["/World/HingeB"])
        result = binding.read(state, query=query, attributes=("joint_q",))
        column = result.groups[0].columns[0]
        # Same-stream device consumer: only the public row map is inspected on the host.
        position = wp.from_dlpack(column.dlpack(0))
        target = wp.empty(1, dtype=wp.float32, device=device)
        wp.launch(_offset_target, dim=1, inputs=[position, target, column.data_row_index(0)], device=device)
        binding.write({"joint_target_q": target}, control=control, query=query)
        result = _unique_columns_by_name(binding.read(control=control, query=query, attributes=("joint_target_q",)), pd)
        np.testing.assert_allclose(_logical_values(result["joint_target_q"]), 0.25)
        other = binding.query(paths=["/World/HingeA"])
        other_result = _unique_columns_by_name(
            binding.read(control=control, query=other, attributes=("joint_target_q",)), pd
        )
        np.testing.assert_allclose(_logical_values(other_result["joint_target_q"]), 4)
        np.testing.assert_array_equal(control.joint_target_qd.numpy(), 5)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_native_target_write_rejects_invalid_batch_before_mutation(device):
    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    with wp.ScopedDevice(device), ovstage.Stage("direct-validation") as stage:
        _populate(stage, MIXED_JOINT_WIDTHS)
        binding = ovnewton.attach_ovstage(stage)
        control = binding.model.control()
        query = binding.query(paths=["/World/HingeB"])
        original = control.joint_target_q.numpy().copy()
        valid = wp.ones(1, dtype=wp.float32, device=device)
        with pytest.raises(ValueError, match="finite"):
            binding.write({
                "joint_target_q": valid,
                "joint_target_qd": wp.array([float("nan")], dtype=wp.float32, device=device),
            }, control=control, query=query)
        np.testing.assert_array_equal(control.joint_target_q.numpy(), original)
        with pytest.raises(ValueError, match="shape"):
            binding.write({"joint_target_q": wp.ones(2, device=device)}, control=control, query=query)
        with pytest.raises(ValueError, match="float32"):
            binding.write({"joint_target_q": wp.ones(1, dtype=wp.float64, device=device)}, control=control, query=query)
        if device != "cpu":
            with pytest.raises(ValueError, match="input must be on"):
                binding.write({"joint_target_q": wp.ones(1, device="cpu")}, control=control, query=query)
        with pytest.raises(ValueError, match="unsupported writable"):
            binding.write({"jointPositionTarget": valid}, control=control, query=query)
        with pytest.raises(ValueError, match="scalar articulation joints"):
            binding.write({"joint_target_q": valid}, control=control,
                          query=binding.query(paths=["/World/Ball"]))
        with pytest.raises(ValueError, match="scalar articulation joints"):
            binding.write({"joint_target_q": valid}, control=control,
                          query=binding.query(paths=["/World/HingeB", "/World/Root"]))
        other = ovnewton.attach_ovstage(stage)
        with pytest.raises(ValueError, match="different StageBinding"):
            binding.write({"joint_target_q": valid}, control=control, query=other.query(paths=["/World/HingeB"]))
        binding.write({}, control=control, query=query)
        np.testing.assert_array_equal(control.joint_target_q.numpy(), original)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_native_target_batch_follows_read_order_and_copies_aliased_inputs(device):
    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    with wp.ScopedDevice(device), ovstage.Stage("direct-batch") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, MIXED_JOINT_WIDTHS)
        binding = ovnewton.attach_ovstage(stage)
        control = binding.model.control()
        query = binding.query(paths=["/World/HingeB", "/World/HingeA"])
        group = binding.read(control=control, query=query, attributes=("joint_target_q",)).groups[0]
        paths = pd.get_path_strings(group.prim_list)
        targets = {"/World/HingeA": 0.75, "/World/HingeB": -0.5}
        values = wp.array([targets[path] for path in paths], dtype=wp.float32, device=device)
        binding.write({"joint_target_q": values}, control=control, query=query)
        for path, expected in targets.items():
            selected = binding.query(paths=[path])
            read = binding.read(control=control, query=selected, attributes=("joint_target_q",))
            np.testing.assert_allclose(_logical_values(read.groups[0].columns[0]), expected)

        # Swap two aliased target buffers in one atomic batch.
        single = binding.query(paths=["/World/HingeA"])
        binding.write(
            {"joint_target_qd": wp.array([2.0], dtype=wp.float32, device=device)}, control=control, query=single
        )
        joint = list(binding.model.joint_label).index("/World/HingeA")
        q_index = int(binding.model.joint_target_q_start.numpy()[joint])
        qd_index = int(binding.model.joint_qd_start.numpy()[joint])
        # Read results are gathered copies; slice the control buffers to alias the destinations.
        q_view = control.joint_target_q[q_index:q_index + 1]
        qd_view = control.joint_target_qd[qd_index:qd_index + 1]
        binding.write({
            "joint_target_q": qd_view,
            "joint_target_qd": q_view,
        }, control=control, query=single)
        result = _unique_columns_by_name(binding.read(
            control=control, query=single, attributes=("joint_target_q", "joint_target_qd")
        ), pd)
        np.testing.assert_allclose(_logical_values(result["joint_target_q"]), 2.0)
        np.testing.assert_allclose(_logical_values(result["joint_target_qd"]), 0.75)
