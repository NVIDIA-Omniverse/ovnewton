# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime transport scenarios adapted from ovphysx's ovstage tests.

The reference tests establish sealed-ordinal reads, unchanged/repeated reads,
and an independent rigid-body output readback after simulation.  These tests use
Newton's state/control arrays and the canonical USD physics columns instead of
the PhysX descriptor/change-dispatch layer.
"""

import gc
import inspect
import threading
import weakref

import newton
import numpy as np
import ovstage
import pytest
import warp as wp
from ovstage import PopulationDomain, population

import ovnewton
from ovnewton._src import _build, _runtime, _stage
from ovnewton._src._errors import OvstageContractError

from .runtime_helpers import lanes_tensor

DRIVEN_REVOLUTE = """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)

def Cube "Base" (
    apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "PhysicsMassAPI"]
)
{
    bool physics:collisionEnabled = 1
    bool physics:kinematicEnabled = 1
    float physics:mass = 1
    vector3f physics:velocity = (0, 0, 0)
    vector3f physics:angularVelocity = (0, 0, 0)
    double size = 1
}

def Cube "Arm" (
    apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "PhysicsMassAPI"]
)
{
    bool physics:collisionEnabled = 1
    float physics:mass = 1
    vector3f physics:velocity = (0, 0, 0)
    vector3f physics:angularVelocity = (0, 0, 0)
    double size = 1
    double3 xformOp:translate = (1, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]
}

def PhysicsRevoluteJoint "Joint" (
    apiSchemas = ["PhysicsDriveAPI:angular", "PhysicsJointStateAPI:angular"]
)
{
    rel physics:body0 = </Base>
    rel physics:body1 = </Arm>
    uniform token physics:axis = "Z"
    point3f physics:localPos0 = (0.5, 0, 0)
    point3f physics:localPos1 = (-0.5, 0, 0)
    quatf physics:localRot0 = (1, 0, 0, 0)
    quatf physics:localRot1 = (1, 0, 0, 0)
    float drive:angular:physics:targetPosition = 5
    float drive:angular:physics:targetVelocity = 6
    float drive:angular:physics:stiffness = 10
    float drive:angular:physics:damping = 2
    float drive:angular:physics:maxForce = 100
    float state:angular:physics:position = 1
    float state:angular:physics:velocity = 2
}

def PhysicsScene "Scene"
{
    vector3f physics:gravityDirection = (0, 0, -1)
    float physics:gravityMagnitude = 9.81
}
"""

DRIVEN_PRISMATIC = (
    DRIVEN_REVOLUTE.replace("PhysicsRevoluteJoint", "PhysicsPrismaticJoint")
    .replace(":angular", ":linear")
    .replace('physics:axis = "Z"', 'physics:axis = "X"')
)


FALLING_BODY = """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)

def Cube "Body" (
    apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "PhysicsMassAPI"]
)
{
    bool physics:collisionEnabled = 1
    float physics:mass = 1
    vector3f physics:velocity = (0, 0, 0)
    vector3f physics:angularVelocity = (0, 0, 0)
    double size = 1
    double3 xformOp:translate = (0, 0, 5)
    uniform token[] xformOpOrder = ["xformOp:translate"]
}

def PhysicsScene "Scene"
{
    vector3f physics:gravityDirection = (0, 0, -1)
    float physics:gravityMagnitude = 9.81
}
"""


NESTED_BODY = """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Parent"
{
    double3 xformOp:translate = (5, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Cube "Body" (
        apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "PhysicsMassAPI"]
    )
    {
        bool physics:collisionEnabled = 1
        float physics:mass = 1
        vector3f physics:velocity = (0, 0, 0)
        vector3f physics:angularVelocity = (0, 0, 0)
        double size = 1
        double3 xformOp:translate = (2, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}
"""


SPHERICAL_JOINT = """#usda 1.0
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
    }
    def Xform "Child" (
        apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
    }
    def PhysicsSphericalJoint "Joint"
    {
        rel physics:body0 = </World/Parent>
        rel physics:body1 = </World/Child>
    }
}
"""


DISTANCE_JOINT = SPHERICAL_JOINT.replace("PhysicsSphericalJoint", "PhysicsDistanceJoint")


D6_JOINT = """#usda 1.0
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
    }
    def Xform "Child" (
        apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
    }
    def PhysicsJoint "Joint" (
        apiSchemas = ["PhysicsLimitAPI:rotX"]
    )
    {
        rel physics:body0 = </World/Parent>
        rel physics:body1 = </World/Child>
        float limit:rotX:physics:low = -45
        float limit:rotX:physics:high = 45
    }
}
"""


def _populate(stage, usda):
    population.open_usd_from_string(stage, usda, ordinal=1, domains=PopulationDomain.ALL)
    stage.advance_write_floor(ordinal=1).wait()


def _write_scalar(stage, pd, path, attribute, value, ordinal):
    with _stage._path_list_query(stage, pd, [path]) as query:
        stage.write_attribute(
            query,
            attribute,
            ordinal,
            lanes_tensor(np.asarray([value], dtype=np.float32), 1),
            is_array=False,
        ).wait()


def _shifted_state(model, shift, velocity):
    state = model.state()
    body_q = state.body_q.numpy()
    body_q[:, :3] += np.asarray(shift, dtype=np.float32)
    state.body_q.assign(wp.array(body_q, dtype=wp.transformf, device=model.device))
    body_qd = np.tile(np.asarray(velocity, dtype=np.float32), (model.body_count, 1))
    state.body_qd.assign(wp.array(body_qd, dtype=wp.spatial_vectorf, device=model.device))
    return state


def _single_dof_indices(model, expected_type):
    joint_types = model.joint_type.numpy()
    joint = next(i for i, joint_type in enumerate(joint_types) if int(joint_type) == int(expected_type))
    q_index = int(model.joint_q_start.numpy()[joint])
    qd_index = int(model.joint_qd_start.numpy()[joint])
    target_starts = getattr(model, "joint_target_q_start", None)
    target_q_index = int(target_starts.numpy()[joint]) if target_starts is not None else qd_index
    return q_index, qd_index, target_q_index


def _revolute_indices(model):
    return _single_dof_indices(model, newton.JointType.REVOLUTE)


def _control_targets(control):
    return control.joint_target_q, control.joint_target_qd


def test_runtime_configuration_arguments_are_keyword_only():
    binding = inspect.signature(ovnewton.StageBinding).parameters
    to_stage = inspect.signature(ovnewton.StageBinding.update_to_ovstage).parameters
    from_stage = inspect.signature(ovnewton.StageBinding.update_from_ovstage).parameters

    assert tuple(binding) == ("stage", "model", "ordinal")
    assert binding["stage"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert binding["model"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert binding["ordinal"].kind is inspect.Parameter.KEYWORD_ONLY
    assert to_stage["state"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert from_stage["state"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert tuple(to_stage) == ("self", "state", "ordinal")
    assert to_stage["ordinal"].kind is inspect.Parameter.KEYWORD_ONLY
    assert tuple(from_stage) == ("self", "state", "control", "ordinal")
    assert from_stage["control"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert from_stage["ordinal"].kind is inspect.Parameter.KEYWORD_ONLY
    assert from_stage["control"].default is inspect.Parameter.empty


def test_runtime_ingress_keeps_composed_body_pose_below_transformed_parent():
    with ovstage.Stage("runtime-nested-body") as stage:
        _populate(stage, NESTED_BODY)
        binding = ovnewton.attach_ovstage(stage)
        state = binding.model.state()

        assert state.body_q.numpy()[0, 0] == pytest.approx(7.0)
        binding.update_from_ovstage(state, control=binding.model.control())
        assert state.body_q.numpy()[0, 0] == pytest.approx(7.0)


def test_ingress_uses_latest_sealed_payload_and_rejects_historical_read():
    with ovstage.Stage("runtime-snapshots") as stage:
        _populate(stage, FALLING_BODY)
        binding = ovnewton.attach_ovstage(stage)
        model = binding.model
        control = model.control()

        at_two = _shifted_state(model, (1, 0, 0), (1, 2, 3, 0.1, 0.2, 0.3))
        binding.update_to_ovstage(at_two, ordinal=2)
        stage.advance_write_floor(ordinal=2).wait()

        destination = model.state()
        binding.update_from_ovstage(destination, control, ordinal=2)
        np.testing.assert_allclose(destination.body_q.numpy(), at_two.body_q.numpy(), atol=1.0e-5)
        np.testing.assert_allclose(destination.body_qd.numpy(), at_two.body_qd.numpy(), atol=1.0e-5)

        at_three = _shifted_state(model, (2, 0, 0), (4, 5, 6, 0.4, 0.5, 0.6))
        binding.update_to_ovstage(at_three, ordinal=3)
        stage.advance_write_floor(ordinal=3).wait()

        binding.update_from_ovstage(destination, control=control)
        np.testing.assert_allclose(destination.body_q.numpy(), at_three.body_q.numpy(), atol=1.0e-5)
        np.testing.assert_allclose(destination.body_qd.numpy(), at_three.body_qd.numpy(), atol=1.0e-5)

        # Re-reading an unchanged level-triggered value is valid and idempotent.
        binding.update_from_ovstage(destination, control=control, ordinal=3)
        np.testing.assert_allclose(destination.body_q.numpy(), at_three.body_q.numpy(), atol=1.0e-5)

        with pytest.raises(
            OvstageContractError,
            match="payload ordinal 3 newer than requested ordinal 2",
        ):
            binding.update_from_ovstage(destination, control=control, ordinal=2)

        with pytest.raises(ValueError, match="newer than the sealed write floor"):
            binding.update_from_ovstage(destination, control=control, ordinal=4)


def test_attach_rejects_payload_newer_than_requested_ordinal():
    with ovstage.Stage("attach-ordinal-ceiling") as stage:
        _populate(stage, FALLING_BODY)
        binding = ovnewton.attach_ovstage(stage)
        state = _shifted_state(binding.model, (1, 0, 0), (0, 0, 0, 0, 0, 0))
        binding.update_to_ovstage(state, ordinal=2)
        stage.advance_write_floor(ordinal=2).wait()

        with pytest.raises(OvstageContractError, match="newer than requested ordinal 1"):
            ovnewton.attach_ovstage(stage, ordinal=1)


def test_attach_rejects_unsealed_read_ceiling():
    with ovstage.Stage("attach-unsealed-ceiling") as stage:
        _populate(stage, FALLING_BODY)

        with pytest.raises(ValueError, match="newer than the sealed write floor"):
            ovnewton.attach_ovstage(stage, ordinal=2)


def test_missing_required_body_column_fails_without_partial_commit():
    with ovstage.Stage("runtime-missing") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, FALLING_BODY)
        binding = ovnewton.attach_ovstage(stage)
        state = binding.model.state()
        control = binding.model.control()
        original_q = state.body_q.numpy().copy()
        original_qd = state.body_qd.numpy().copy()

        with _stage._path_list_query(stage, pd, binding.model.body_label) as query:
            stage.delete_attributes(query, ["physics:velocity"], ordinal=2).wait()
        stage.advance_write_floor(ordinal=2).wait()

        with pytest.raises(OvstageContractError, match="physics:velocity"):
            binding.update_from_ovstage(state, control=control)
        np.testing.assert_array_equal(state.body_q.numpy(), original_q)
        np.testing.assert_array_equal(state.body_qd.numpy(), original_qd)


@pytest.mark.parametrize(
    "attribute",
    ["state:angular:physics:position", "state:angular:physics:velocity"],
)
def test_missing_required_joint_column_fails_without_partial_commit(attribute):
    with ovstage.Stage(f"runtime-missing-{attribute.rsplit(':', 1)[-1]}") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, DRIVEN_REVOLUTE)
        binding = ovnewton.attach_ovstage(stage)
        model = binding.model
        state = model.state()
        control = model.control()
        original_q = state.body_q.numpy().copy()
        original_joint_q = state.joint_q.numpy().copy()
        original_joint_qd = state.joint_qd.numpy().copy()

        external = _shifted_state(model, (3, 0, 0), (1, 2, 3, 0.1, 0.2, 0.3))
        binding.update_to_ovstage(external, ordinal=2)
        with _stage._path_list_query(stage, pd, ["/Joint"]) as query:
            stage.delete_attributes(query, [attribute], ordinal=2).wait()
        stage.advance_write_floor(ordinal=2).wait()

        with pytest.raises(OvstageContractError, match=attribute):
            binding.update_from_ovstage(state, control=control)
        np.testing.assert_array_equal(state.body_q.numpy(), original_q)
        np.testing.assert_array_equal(state.joint_q.numpy(), original_joint_q)
        np.testing.assert_array_equal(state.joint_qd.numpy(), original_joint_qd)


def test_output_requires_body_velocity_before_writing():
    with ovstage.Stage("runtime-output-body-validation") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, FALLING_BODY)
        binding = ovnewton.attach_ovstage(stage)
        state = binding.model.state()
        body_q = state.body_q.numpy()
        body_q[:, 0] = 10.0
        state.body_q.assign(wp.array(body_q, dtype=wp.transformf, device=binding.model.device))
        state.body_qd = None

        with pytest.raises(ValueError, match="State.body_qd"):
            binding.update_to_ovstage(state, ordinal=2)
        stage.advance_write_floor(ordinal=2).wait()
        with _stage._path_list_query(stage, pd, binding.model.body_label) as query:
            matrices = _stage.read_fixed(stage, pd, query, "omni:xform", ordinal=2)
        translation, _, _ = _build._decode_pose(np.stack([matrices[i] for i in range(binding.model.body_count)]))
        assert translation[0, 0] == 0.0


def test_output_requires_joint_velocity_before_writing_body_state():
    with ovstage.Stage("runtime-output-joint-validation") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, DRIVEN_REVOLUTE)
        binding = ovnewton.attach_ovstage(stage)
        state = binding.model.state()
        original_body_q = state.body_q.numpy().copy()
        body_q = original_body_q.copy()
        body_q[:, 2] += 10.0
        state.body_q.assign(wp.array(body_q, dtype=wp.transformf, device=binding.model.device))
        state.joint_qd = None

        with pytest.raises(ValueError, match="State.joint_qd"):
            binding.update_to_ovstage(state, ordinal=2)
        stage.advance_write_floor(ordinal=2).wait()
        with _stage._path_list_query(stage, pd, binding.model.body_label) as query:
            matrices = _stage.read_fixed(stage, pd, query, "omni:xform", ordinal=2)
        translation, _, _ = _build._decode_pose(np.stack([matrices[i] for i in range(binding.model.body_count)]))
        np.testing.assert_allclose(translation, original_body_q[:, :3])


def test_single_dof_state_and_drive_targets_share_one_snapshot():
    with ovstage.Stage("runtime-joint-control") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, DRIVEN_REVOLUTE)
        binding = ovnewton.attach_ovstage(stage)
        model = binding.model
        q_index, qd_index, target_q_index = _revolute_indices(model)

        values_two = {
            "state:angular:physics:position": 20.0,
            "state:angular:physics:velocity": 30.0,
            "drive:angular:physics:targetPosition": 40.0,
            "drive:angular:physics:targetVelocity": 50.0,
        }
        for attribute, value in values_two.items():
            _write_scalar(stage, pd, "/Joint", attribute, value, 2)
        stage.advance_write_floor(ordinal=2).wait()

        state = model.state()
        control = model.control()
        position_targets, velocity_targets = _control_targets(control)
        binding.update_from_ovstage(state, control=control, ordinal=2)
        assert state.joint_q.numpy()[q_index] == pytest.approx(np.deg2rad(20.0))
        assert state.joint_qd.numpy()[qd_index] == pytest.approx(np.deg2rad(30.0))
        assert position_targets.numpy()[target_q_index] == pytest.approx(np.deg2rad(40.0))
        assert velocity_targets.numpy()[qd_index] == pytest.approx(np.deg2rad(50.0))

        values_three = {attribute: value + 10.0 for attribute, value in values_two.items()}
        for attribute, value in values_three.items():
            _write_scalar(stage, pd, "/Joint", attribute, value, 3)
        stage.advance_write_floor(ordinal=3).wait()

        binding.update_from_ovstage(state, control=control)
        assert state.joint_q.numpy()[q_index] == pytest.approx(np.deg2rad(30.0))
        assert state.joint_qd.numpy()[qd_index] == pytest.approx(np.deg2rad(40.0))
        assert position_targets.numpy()[target_q_index] == pytest.approx(np.deg2rad(50.0))
        assert velocity_targets.numpy()[qd_index] == pytest.approx(np.deg2rad(60.0))

        before_state = state.joint_q.numpy().copy()
        before_target = position_targets.numpy().copy()
        with _stage._path_list_query(stage, pd, ["/Joint"]) as query:
            stage.delete_attributes(query, ["drive:angular:physics:targetPosition"], ordinal=4).wait()
        stage.advance_write_floor(ordinal=4).wait()
        with pytest.raises(OvstageContractError, match="targetPosition"):
            binding.update_from_ovstage(state, control=control)
        np.testing.assert_array_equal(state.joint_q.numpy(), before_state)
        np.testing.assert_array_equal(position_targets.numpy(), before_target)


def test_prismatic_state_and_drive_targets_share_one_snapshot():
    with ovstage.Stage("runtime-prismatic-control") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, DRIVEN_PRISMATIC)
        binding = ovnewton.attach_ovstage(stage)
        model = binding.model
        q_index, qd_index, target_q_index = _single_dof_indices(model, newton.JointType.PRISMATIC)

        values = {
            "state:linear:physics:position": 2.0,
            "state:linear:physics:velocity": 3.0,
            "drive:linear:physics:targetPosition": 4.0,
            "drive:linear:physics:targetVelocity": 5.0,
        }
        for attribute, value in values.items():
            _write_scalar(stage, pd, "/Joint", attribute, value, 2)
        stage.advance_write_floor(ordinal=2).wait()

        state = model.state()
        control = model.control()
        position_targets, velocity_targets = _control_targets(control)
        binding.update_from_ovstage(state, control=control)
        assert state.joint_q.numpy()[q_index] == pytest.approx(2.0)
        assert state.joint_qd.numpy()[qd_index] == pytest.approx(3.0)
        assert position_targets.numpy()[target_q_index] == pytest.approx(4.0)
        assert velocity_targets.numpy()[qd_index] == pytest.approx(5.0)


def test_coordinate_layout_drive_target():
    previous = newton.use_coord_layout_targets
    newton.use_coord_layout_targets = True
    try:
        with ovstage.Stage("runtime-coordinate-layout") as stage, ovstage.PathDictionary(stage) as pd:
            _populate(stage, DRIVEN_REVOLUTE)
            binding = ovnewton.attach_ovstage(stage)
            model = binding.model
            _, _, target_q_index = _revolute_indices(model)
            _write_scalar(stage, pd, "/Joint", "drive:angular:physics:targetPosition", 40.0, 2)
            stage.advance_write_floor(ordinal=2).wait()

            control = model.control()
            binding.update_from_ovstage(model.state(), control=control, ordinal=2)
            assert control.joint_target_q.numpy()[target_q_index] == pytest.approx(np.deg2rad(40.0))
    finally:
        newton.use_coord_layout_targets = previous


@pytest.mark.parametrize(
    "stiffness,damping,mode,required,optional",
    [
        (10, 0, newton.JointTargetMode.POSITION, "targetPosition", "targetVelocity"),
        (0, 2, newton.JointTargetMode.VELOCITY, "targetVelocity", "targetPosition"),
    ],
)
def test_drive_mode_requires_only_its_active_target(stiffness, damping, mode, required, optional):
    usda = DRIVEN_REVOLUTE.replace("physics:stiffness = 10", f"physics:stiffness = {stiffness}")
    usda = usda.replace("physics:damping = 2", f"physics:damping = {damping}")
    with ovstage.Stage(f"runtime-drive-mode-{required}") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, usda)
        binding = ovnewton.attach_ovstage(stage)
        model = binding.model
        _, qd_index, target_q_index = _revolute_indices(model)
        assert int(model.joint_target_mode.numpy()[qd_index]) == int(mode)
        control = model.control()
        position_targets, velocity_targets = _control_targets(control)
        target_array = position_targets if required == "targetPosition" else velocity_targets
        target_index = target_q_index if required == "targetPosition" else qd_index
        optional_array = velocity_targets if optional == "targetVelocity" else position_targets
        optional_before = optional_array.numpy().copy()

        required_attr = f"drive:angular:physics:{required}"
        optional_attr = f"drive:angular:physics:{optional}"
        _write_scalar(stage, pd, "/Joint", required_attr, 70.0, 2)
        with _stage._path_list_query(stage, pd, ["/Joint"]) as query:
            stage.delete_attributes(query, [optional_attr], ordinal=2).wait()
        stage.advance_write_floor(ordinal=2).wait()

        binding.update_from_ovstage(model.state(), control=control)
        assert target_array.numpy()[target_index] == pytest.approx(np.deg2rad(70.0))
        np.testing.assert_array_equal(optional_array.numpy(), optional_before)

        before_required = target_array.numpy().copy()
        with _stage._path_list_query(stage, pd, ["/Joint"]) as query:
            stage.delete_attributes(query, [required_attr], ordinal=3).wait()
        stage.advance_write_floor(ordinal=3).wait()
        with pytest.raises(OvstageContractError, match=required):
            binding.update_from_ovstage(model.state(), control=control)
        np.testing.assert_array_equal(target_array.numpy(), before_required)


def test_runtime_batches_reads_and_pipelines_writes():
    with ovstage.Stage("runtime-operation-shape") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, DRIVEN_REVOLUTE)
        binding = ovnewton.attach_ovstage(stage)

        read_sizes = []
        original_read = stage.read_attributes

        def tracked_read(query, attributes, ordinal_range):
            read_sizes.append(len(attributes))
            return original_read(query, attributes, ordinal_range)

        stage.read_attributes = tracked_read
        binding.update_from_ovstage(binding.model.state(), control=binding.model.control())
        assert read_sizes == [7]

        events = []
        batches = []
        original_write = stage.write_attributes

        class TrackedOperation:
            def __init__(self, operation):
                self.operation = operation

            def wait(self):
                events.append("wait")
                return self.operation.wait()

        def tracked_write(query, writes, ordinal, **kwargs):
            events.append("enqueue")
            batches.append(tuple(writes))
            return TrackedOperation(original_write(query, writes, ordinal, **kwargs))

        stage.write_attributes = tracked_write
        binding.update_to_ovstage(binding.model.state(), ordinal=2)
        assert events == ["enqueue"] * 2 + ["wait"] * 2
        assert [len(batch) for batch in batches] == [4, 2]
        reset = batches[0][0]
        assert reset.attribute == pd.intern_token("omni:resetXformStack")
        assert int(reset.tensors.dtype.code) == int(ovstage.DLDataTypeCode.kDLBool)
        assert int(reset.tensors.dtype.bits) == 8
        assert int(reset.tensors.dtype.lanes) == 1
        assert _runtime._tensor_device(reset.tensors) == binding.model.device
        assert int(reset.tensors.data) == int(binding._runtime.reset_xform_stack_out.ptr)
        expected_event = binding._runtime.producer_event is not None
        assert all(bool(write.cuda_event) == expected_event for batch in batches for write in batch)


def test_runtime_owns_each_publication_path_list_once(monkeypatch):
    with ovstage.Stage("runtime-publication-path-lists") as stage:
        _populate(stage, DRIVEN_REVOLUTE)
        binding = ovnewton.attach_ovstage(stage)
        path_dictionary = binding._pd
        destroyed = []
        destroy_path_list = path_dictionary.destroy_path_list

        def tracked_destroy(path_list):
            destroyed.append(path_list)
            destroy_path_list(path_list)

        monkeypatch.setattr(path_dictionary, "destroy_path_list", tracked_destroy)
        binding_ref = weakref.ref(binding)
        del binding
        gc.collect()

        assert binding_ref() is None
        assert destroyed
        assert len(destroyed) == len(set(destroyed))


def test_runtime_releases_publication_path_lists_when_setup_fails(monkeypatch):
    destroyed = []
    destroy_path_lists = _runtime._destroy_path_lists

    def tracked_destroy(path_dictionary, path_lists):
        destroyed.extend(path_lists)
        destroy_path_lists(path_dictionary, path_lists)

    def fail_finalizer(*_args):
        raise RuntimeError("publication finalizer failed")

    monkeypatch.setattr(_runtime, "_destroy_path_lists", tracked_destroy)
    monkeypatch.setattr(_runtime, "_finalize_path_lists", fail_finalizer)
    with ovstage.Stage("runtime-publication-setup-failure") as stage:
        _populate(stage, DRIVEN_REVOLUTE)
        with pytest.raises(RuntimeError, match="publication finalizer failed"):
            ovnewton.attach_ovstage(stage)

    assert len(destroyed) == 2
    assert len(destroyed) == len(set(destroyed))


def test_runtime_cpu_path_needs_no_device_synchronization(monkeypatch):
    with ovstage.Stage("runtime-cpu-residency") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, DRIVEN_REVOLUTE)
        with wp.ScopedDevice("cpu"):
            model = _build.build_model(stage, pd, ordinal=1)
        binding = ovnewton.attach_ovstage(stage, model=model)
        assert binding._runtime.producer_event is None

        def unexpected_sync(*_args, **_kwargs):
            raise AssertionError("CPU runtime transport must not synchronize a device or stream")

        monkeypatch.setattr(wp, "synchronize_device", unexpected_sync)
        monkeypatch.setattr(wp, "synchronize_stream", unexpected_sync)
        binding.update_to_ovstage(model.state(), ordinal=2)
        stage.advance_write_floor(ordinal=2).wait()
        binding.update_from_ovstage(model.state(), control=model.control(), ordinal=2)


def test_runtime_cuda_path_never_uses_a_device_wide_barrier(monkeypatch):
    if not wp.is_cuda_available():
        pytest.skip("CUDA not available")

    with ovstage.Stage("runtime-no-device-barrier") as stage:
        _populate(stage, DRIVEN_REVOLUTE)
        binding = ovnewton.attach_ovstage(stage)
        assert binding.model.device.is_cuda
        transport_stream = wp.Stream(binding.model.device)
        synchronized_streams = []
        synchronize_stream = wp.synchronize_stream

        def unexpected_sync(*_args, **_kwargs):
            raise AssertionError("runtime transport must not synchronize the CUDA device")

        def tracked_stream_sync(stream):
            synchronized_streams.append(stream)
            synchronize_stream(stream)

        monkeypatch.setattr(wp, "synchronize_device", unexpected_sync)
        monkeypatch.setattr(wp, "synchronize_stream", tracked_stream_sync)
        with wp.ScopedStream(transport_stream):
            binding.update_to_ovstage(binding.model.state(), ordinal=2)
        stage.advance_write_floor(ordinal=2).wait()
        with wp.ScopedStream(transport_stream):
            binding.update_from_ovstage(
                binding.model.state(), control=binding.model.control(), ordinal=2
            )

        assert synchronized_streams == [transport_stream]


def test_runtime_ingress_reuses_binding_owned_storage(monkeypatch):
    with ovstage.Stage("runtime-storage-reuse") as stage:
        _populate(stage, DRIVEN_REVOLUTE)
        binding = ovnewton.attach_ovstage(stage)
        state = binding.model.state()
        control = binding.model.control()
        runtime = binding._runtime
        pointers = (
            runtime.body_q_in.ptr,
            runtime.body_qd_in.ptr,
            runtime.joint_q_in.ptr,
            runtime.joint_qd_in.ptr,
            runtime.target_q_in.ptr,
            runtime.target_qd_in.ptr,
            *(layout.staging.ptr for column in runtime.read_plan.columns for layout in column.slices),
        )

        def unexpected_allocation(*_args, **_kwargs):
            raise AssertionError("steady-state ingress must reuse binding-owned storage")

        monkeypatch.setattr(wp, "clone", unexpected_allocation)
        monkeypatch.setattr(wp, "empty", unexpected_allocation)
        binding.update_from_ovstage(state, control=control)
        binding.update_from_ovstage(state, control=control)

        assert pointers == (
            runtime.body_q_in.ptr,
            runtime.body_qd_in.ptr,
            runtime.joint_q_in.ptr,
            runtime.joint_qd_in.ptr,
            runtime.target_q_in.ptr,
            runtime.target_qd_in.ptr,
            *(layout.staging.ptr for column in runtime.read_plan.columns for layout in column.slices),
        )


def test_deferred_runtime_layout_rolls_back_and_reuses_storage(monkeypatch):
    usda = DRIVEN_REVOLUTE.replace(
        ', "PhysicsJointStateAPI:angular"',
        "",
    ).replace(
        "    float state:angular:physics:position = 1\n",
        "",
    ).replace(
        "    float state:angular:physics:velocity = 2\n",
        "",
    )
    with ovstage.Stage("runtime-deferred-layout") as stage:
        _populate(stage, usda)
        binding = ovnewton.attach_ovstage(stage)
        state = binding.model.state()
        control = binding.model.control()
        deferred = [column for column in binding._runtime.read_plan.columns if column.deferred]
        assert len(deferred) >= 2

        binding.update_to_ovstage(state, ordinal=2)
        stage.advance_write_floor(ordinal=2).wait()

        capture_group_slice = _runtime._capture_group_slice
        captures = 0

        def fail_second_capture(column, group, runtime):
            nonlocal captures
            if column.deferred:
                captures += 1
                if captures == 2:
                    raise RuntimeError("injected deferred-layout failure")
            return capture_group_slice(column, group, runtime)

        monkeypatch.setattr(_runtime, "_capture_group_slice", fail_second_capture)
        with pytest.raises(RuntimeError, match="injected deferred-layout failure"):
            binding.update_from_ovstage(state, control=control, ordinal=2)
        assert all(column.deferred and not column.slices for column in deferred)

        monkeypatch.setattr(_runtime, "_capture_group_slice", capture_group_slice)
        binding.update_from_ovstage(state, control=control, ordinal=2)
        assert all(not column.deferred for column in deferred)

        def unexpected_allocation(*_args, **_kwargs):
            raise AssertionError("materialized runtime layout must reuse binding-owned storage")

        monkeypatch.setattr(wp, "clone", unexpected_allocation)
        monkeypatch.setattr(wp, "empty", unexpected_allocation)
        binding.update_from_ovstage(state, control=control, ordinal=2)


def test_runtime_ingress_rejects_stage_topology_changes():
    with ovstage.Stage("runtime-topology-change") as stage:
        _populate(stage, FALLING_BODY)
        binding = ovnewton.attach_ovstage(stage)

        stage.clone("/Body", ["/Other"], ordinal=2)
        stage.advance_write_floor(ordinal=2).wait()

        with pytest.raises(OvstageContractError, match="layout generation changed.*recreate the StageBinding"):
            binding.update_from_ovstage(
                binding.model.state(), control=binding.model.control(), ordinal=2
            )


def test_equal_sized_bindings_do_not_share_output_storage():
    barrier = threading.Barrier(2)
    errors = []
    with (
        ovstage.Stage("runtime-isolation-a") as stage_a,
        ovstage.PathDictionary(stage_a) as pd_a,
        ovstage.Stage("runtime-isolation-b") as stage_b,
        ovstage.PathDictionary(stage_b) as pd_b,
    ):
        _populate(stage_a, FALLING_BODY)
        _populate(stage_b, FALLING_BODY)
        binding_a = ovnewton.attach_ovstage(stage_a)
        binding_b = ovnewton.attach_ovstage(stage_b)
        state_a = _shifted_state(binding_a.model, (3, 0, 0), (0, 0, 0, 0, 0, 0))
        state_b = _shifted_state(binding_b.model, (7, 0, 0), (0, 0, 0, 0, 0, 0))

        def synchronize_first_write(original_write):
            first = True

            def synchronized_write(*args, **kwargs):
                nonlocal first
                if first:
                    first = False
                    barrier.wait(timeout=10)
                return original_write(*args, **kwargs)

            return synchronized_write

        for stage in (stage_a, stage_b):
            stage.write_attributes = synchronize_first_write(stage.write_attributes)

        def publish(binding, state):
            try:
                binding.update_to_ovstage(state, ordinal=2)
            except Exception as error:
                errors.append(error)

        thread_a = threading.Thread(target=publish, args=(binding_a, state_a))
        thread_b = threading.Thread(target=publish, args=(binding_b, state_b))
        thread_a.start()
        thread_b.start()
        thread_a.join(timeout=15)
        thread_b.join(timeout=15)
        assert not thread_a.is_alive() and not thread_b.is_alive()
        assert not errors

        for stage in (stage_a, stage_b):
            stage.advance_write_floor(ordinal=2).wait()
        for stage, pd, binding, expected in (
            (stage_a, pd_a, binding_a, 3.0),
            (stage_b, pd_b, binding_b, 7.0),
        ):
            with _stage._path_list_query(stage, pd, binding.model.body_label) as query:
                matrices = _stage.read_fixed(stage, pd, query, "omni:xform", ordinal=2)
            rows = np.stack([matrices[i] for i in range(binding.model.body_count)])
            translation, _, _ = _build._decode_pose(rows)
            assert translation[0, 0] == pytest.approx(expected)


def test_newton_step_is_observable_through_independent_ovstage_read():
    with ovstage.Stage("runtime-falling-body") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, FALLING_BODY)
        binding = ovnewton.attach_ovstage(stage)
        model = binding.model
        state_in = model.state()
        state_out = model.state()
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        initial_z = float(state_in.body_q.numpy()[0, 2])

        solver = newton.solvers.SolverSemiImplicit(model)
        solver.step(state_in, state_out, model.control(), None, 0.1)
        binding.update_to_ovstage(state_out, ordinal=2)
        stage.advance_write_floor(ordinal=2).wait()

        with _stage._path_list_query(stage, pd, binding.model.body_label) as query:
            matrices = _stage.read_fixed(stage, pd, query, "omni:xform", ordinal=2)
        matrix = np.asarray(matrices[0], dtype=np.float64).reshape(1, 16)
        translation, _, _ = _build._decode_pose(matrix)
        assert translation[0, 2] < initial_z


def test_spherical_runtime_state_roundtrips_through_body_state():
    with ovstage.Stage("runtime-spherical") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, SPHERICAL_JOINT)
        binding = ovnewton.attach_ovstage(stage)
        model = binding.model
        external = model.state()
        body_q = external.body_q.numpy()
        child = list(binding.model.body_label).index("/World/Child")
        angle = np.deg2rad(60.0)
        expected_joint_q = np.asarray([0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)], dtype=np.float32)
        body_q[child, :3] = (3.0, 0.0, 0.0)
        body_q[child, 3:7] = expected_joint_q
        external.body_q.assign(wp.array(body_q, dtype=wp.transformf, device=model.device))
        body_qd = np.zeros((model.body_count, 6), dtype=np.float32)
        expected_joint_qd = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
        body_qd[child, 3:6] = expected_joint_qd
        external.body_qd.assign(wp.array(body_qd, dtype=wp.spatial_vectorf, device=model.device))

        # SphericalJoint has no joint-state attribute; body state is canonical.
        binding.update_to_ovstage(external, ordinal=2)
        stage.advance_write_floor(ordinal=2).wait()
        with _stage._path_list_query(stage, pd, binding.model.body_label) as query:
            matrices = _stage.read_fixed(stage, pd, query, "omni:xform", ordinal=2)
        rows = np.stack([matrices[i] for i in range(model.body_count)])
        translations, _, _ = _build._decode_pose(rows)
        assert translations[child, 0] == pytest.approx(3.0)

        # Inbound body state is converted back to Newton's quaternion BALL
        # coordinate rather than requiring a nonexistent scalar state column.
        destination = model.state()
        binding.update_from_ovstage(destination, control=model.control(), ordinal=2)
        ball = next(i for i, kind in enumerate(model.joint_type.numpy()) if int(kind) == int(newton.JointType.BALL))
        q_start = int(model.joint_q_start.numpy()[ball])
        qd_start = int(model.joint_qd_start.numpy()[ball])
        actual_joint_q = destination.joint_q.numpy()[q_start : q_start + 4]
        actual_joint_qd = destination.joint_qd.numpy()[qd_start : qd_start + 3]
        assert abs(float(np.dot(actual_joint_q, expected_joint_q))) == pytest.approx(1.0, abs=1.0e-5)
        np.testing.assert_allclose(actual_joint_qd, expected_joint_qd, atol=1.0e-5)


def test_attach_initializes_spherical_coordinates_from_body_state():
    oriented = SPHERICAL_JOINT.replace(
        'def Xform "Child" (\n        apiSchemas = ["PhysicsRigidBodyAPI"]\n    )\n    {\n    }',
        'def Xform "Child" (\n        apiSchemas = ["PhysicsRigidBodyAPI"]\n    )\n    {\n'
        '        quatf xformOp:orient = (0.70710678, 0, 0, 0.70710678)\n'
        '        uniform token[] xformOpOrder = ["xformOp:orient"]\n    }',
    )
    with ovstage.Stage("runtime-spherical-initial") as stage:
        _populate(stage, oriented)
        model = ovnewton.attach_ovstage(stage).model

        ball = next(i for i, kind in enumerate(model.joint_type.numpy()) if int(kind) == int(newton.JointType.BALL))
        q_start = int(model.joint_q_start.numpy()[ball])
        actual = model.joint_q.numpy()[q_start : q_start + 4]
        expected = np.asarray([0.0, 0.0, 2.0**-0.5, 2.0**-0.5], dtype=np.float32)
        assert abs(float(np.dot(actual, expected))) == pytest.approx(1.0, abs=1.0e-5)


def test_spherical_body_update_survives_featherstone_fk():
    with ovstage.Stage("runtime-spherical-featherstone") as stage:
        _populate(stage, SPHERICAL_JOINT)
        binding = ovnewton.attach_ovstage(stage)
        model = binding.model
        child = list(binding.model.body_label).index("/World/Child")
        expected = np.asarray([0.0, 0.0, 0.5, 3.0**0.5 / 2.0], dtype=np.float32)
        external = model.state()
        body_q = external.body_q.numpy()
        body_q[child, 3:7] = expected
        external.body_q.assign(wp.array(body_q, dtype=wp.transformf, device=model.device))

        binding.update_to_ovstage(external, ordinal=2)
        stage.advance_write_floor(ordinal=2).wait()
        state_in = model.state()
        binding.update_from_ovstage(state_in, control=model.control(), ordinal=2)

        state_out = model.state()
        solver = newton.solvers.SolverFeatherstone(model, angular_damping=0.0)
        solver.step(state_in, state_out, model.control(), None, 1.0e-6)

        # Featherstone refreshes its input body poses from generalized state.
        actual = state_in.body_q.numpy()[child, 3:7]
        assert abs(float(np.dot(actual, expected))) == pytest.approx(1.0, abs=1.0e-5)


def test_distance_runtime_state_roundtrips_through_body_state():
    with ovstage.Stage("runtime-distance") as stage:
        _populate(stage, DISTANCE_JOINT)
        binding = ovnewton.attach_ovstage(stage)
        model = binding.model
        external = model.state()
        child = list(binding.model.body_label).index("/World/Child")

        body_q = external.body_q.numpy()
        body_q[child, :3] = (3.0, 2.0, 1.0)
        external.body_q.assign(wp.array(body_q, dtype=wp.transformf, device=model.device))
        body_qd = external.body_qd.numpy()
        body_qd[child] = (0.1, 0.2, 0.3, -0.4, -0.5, -0.6)
        external.body_qd.assign(wp.array(body_qd, dtype=wp.spatial_vectorf, device=model.device))

        expected = model.state()
        newton.eval_ik(model, external, expected.joint_q, expected.joint_qd)

        binding.update_to_ovstage(external, ordinal=2)
        stage.advance_write_floor(ordinal=2).wait()
        destination = model.state()
        binding.update_from_ovstage(destination, control=model.control(), ordinal=2)

        np.testing.assert_allclose(destination.body_q.numpy(), external.body_q.numpy(), atol=1.0e-5)
        np.testing.assert_allclose(destination.body_qd.numpy(), external.body_qd.numpy(), atol=1.0e-5)
        np.testing.assert_allclose(destination.joint_q.numpy(), expected.joint_q.numpy(), atol=1.0e-5)
        np.testing.assert_allclose(destination.joint_qd.numpy(), expected.joint_qd.numpy(), atol=1.0e-5)


def test_d6_sync_preserves_unsupported_joint_coordinates():
    with ovstage.Stage("runtime-multidof-d6") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(
            stage,
            D6_JOINT
            + '''
def Xform "Free" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    double3 xformOp:translate = (0, 2, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]
}
''',
        )
        binding = ovnewton.attach_ovstage(stage)
        model = binding.model
        state = model.state()
        body_q = state.body_q.numpy()
        body_q[:, 0] = 10.0
        state.body_q.assign(wp.array(body_q, dtype=wp.transformf, device=model.device))

        binding.update_to_ovstage(state, ordinal=2)
        stage.advance_write_floor(ordinal=2).wait()
        with _stage._path_list_query(stage, pd, binding.model.body_label) as query:
            matrices = _stage.read_fixed(stage, pd, query, "omni:xform", ordinal=2)
        rows = np.stack([matrices[i] for i in range(model.body_count)])
        translations, _, _ = _build._decode_pose(rows)
        assert np.all(translations[:, 0] == 10.0)

        d6 = next(i for i, kind in enumerate(model.joint_type.numpy()) if int(kind) == int(newton.JointType.D6))
        q_start = model.joint_q_start.numpy()
        qd_start = model.joint_qd_start.numpy()
        destination = model.state()
        joint_q = destination.joint_q.numpy()
        joint_qd = destination.joint_qd.numpy()
        joint_q[int(q_start[d6]) : int(q_start[d6 + 1])] = 7.0
        joint_qd[int(qd_start[d6]) : int(qd_start[d6 + 1])] = 8.0
        destination.joint_q.assign(wp.array(joint_q, dtype=wp.float32, device=model.device))
        destination.joint_qd.assign(wp.array(joint_qd, dtype=wp.float32, device=model.device))

        binding.update_from_ovstage(destination, control=model.control(), ordinal=2)

        np.testing.assert_allclose(destination.body_q.numpy(), state.body_q.numpy(), atol=1.0e-5)
        np.testing.assert_array_equal(
            destination.joint_q.numpy()[int(q_start[d6]) : int(q_start[d6 + 1])],
            joint_q[int(q_start[d6]) : int(q_start[d6 + 1])],
        )
        np.testing.assert_array_equal(
            destination.joint_qd.numpy()[int(qd_start[d6]) : int(qd_start[d6 + 1])],
            joint_qd[int(qd_start[d6]) : int(qd_start[d6 + 1])],
        )
