# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import newton
import numpy as np
import ovstage
import pytest
import warp as wp
from ovstage import PopulationDomain, population

import ovnewton

_RIGID_BODY = """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Body" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI"]
) {
    def Sphere "Collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    ) {
        double radius = 0.5
    }
}
"""

_MULTI_BODY_STATE = """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def PhysicsScene "Scene" {
    vector3f physics:gravityDirection = (0, -3, -4)
    float physics:gravityMagnitude = 10
}
def Cube "First" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "PhysicsMassAPI"]
) {
    float physics:mass = 2
    point3f physics:centerOfMass = (0.1, 0.2, 0.3)
    double3 xformOp:translate = (1, 2, 3)
    float3 xformOp:rotateXYZ = (0, 0, 90)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
}
def Cube "Second" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "PhysicsMassAPI"]
) {
    float physics:mass = 3
    point3f physics:centerOfMass = (-0.4, 0.5, -0.6)
    double3 xformOp:translate = (-4, 5, -6)
    float3 xformOp:rotateXYZ = (90, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
}
"""

_ROOTLESS_JOINTS = """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World" {
    def Xform "Base" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def Xform "Link" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def PhysicsFixedJoint "Root" {
        rel physics:body1 = </World/Base>
    }
    def PhysicsRevoluteJoint "Child" {
        rel physics:body0 = </World/Base>
        rel physics:body1 = </World/Link>
        token physics:axis = "Z"
    }
}
"""

_Y_UP_BODY = """#usda 1.0
(
    metersPerUnit = 0.01
    upAxis = "Y"
)

def PhysicsScene "Scene" {
    vector3f physics:gravityDirection = (0, -1, 0)
    float physics:gravityMagnitude = -inf
}

def Sphere "Body" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]
) {
    double radius = 0.5
}
"""

_D6_JOINT = """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World" (
    prepend apiSchemas = ["PhysicsArticulationRootAPI"]
) {
    def Xform "Parent" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def Xform "Child" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def PhysicsJoint "Joint" (
        prepend apiSchemas = ["PhysicsLimitAPI:transX"]
    ) {
        rel physics:body0 = </World/Parent>
        rel physics:body1 = </World/Child>
        float limit:transX:physics:low = -1
        float limit:transX:physics:high = 1
    }
}
"""

_D6_JOINT_WITH_DAMPING = """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World" (
    prepend apiSchemas = ["PhysicsArticulationRootAPI"]
) {
    def Xform "Parent" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def Xform "Child" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def PhysicsJoint "Joint" (
        prepend apiSchemas = ["PhysicsLimitAPI:transX", "PhysicsDriveAPI:transX", "NewtonJointAPI"]
    ) {
        rel physics:body0 = </World/Parent>
        rel physics:body1 = </World/Child>
        float limit:transX:physics:low = -1
        float limit:transX:physics:high = 1
        float drive:transX:physics:maxForce = 567
        float drive:transX:physics:stiffness = 1
        float newton:damping = 456
    }
}
"""

_MERGED_D6_JOINTS = """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World" (
    prepend apiSchemas = ["PhysicsArticulationRootAPI"]
) {
    def Xform "Parent" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def Xform "Child" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def PhysicsPrismaticJoint "Linear" {
        rel physics:body0 = </World/Parent>
        rel physics:body1 = </World/Child>
        uniform token physics:axis = "X"
    }
    def PhysicsRevoluteJoint "Angular" {
        rel physics:body0 = </World/Parent>
        rel physics:body1 = </World/Child>
        uniform token physics:axis = "Z"
    }
}
"""


def _d6_dof_values(builder, values):
    joint = next(index for index, kind in enumerate(builder.joint_type) if kind == newton.JointType.D6)
    start = builder.joint_qd_start[joint]
    count = sum(builder.joint_dof_dim[joint])
    return values[start : start + count]


def test_vbd_can_prepare_caller_owned_builder_before_finalizing():
    builder = newton.ModelBuilder()
    newton.solvers.SolverVBD.register_custom_attributes(builder)

    with ovstage.Stage("add-ovstage-vbd") as stage:
        population.open_usd_from_string(stage, _RIGID_BODY, ordinal=1, domains=PopulationDomain.PHYSICS)
        stage.advance_write_floor(ordinal=1).wait()

        result = ovnewton.add_ovstage(builder, stage)
        builder.color()
        model = builder.finalize()
        solver = newton.solvers.SolverVBD(model)
        binding = ovnewton.attach_ovstage(stage, model=model)

    assert binding.model is model
    assert not result.has_orphan_joints
    assert model.body_label == ["/Body"]
    assert model.body_color_groups
    assert model.vbd.joint_is_hard.shape == (model.joint_count,)
    assert isinstance(solver, newton.solvers.SolverVBD)


def test_add_ovstage_rejects_a_builder_that_already_contains_model_entities():
    builder = newton.ModelBuilder()
    builder.add_body(label="/Existing")

    with ovstage.Stage("add-ovstage-nonempty") as stage:
        population.open_usd_from_string(stage, _RIGID_BODY, ordinal=1, domains=PopulationDomain.PHYSICS)
        stage.advance_write_floor(ordinal=1).wait()

        with pytest.raises(ValueError, match="builder must not contain model entities"):
            ovnewton.add_ovstage(builder, stage)


def test_add_ovstage_rejects_preexisting_collision_filter_pairs():
    builder = newton.ModelBuilder()
    builder.add_shape_collision_filter_pair(0, 1)

    with ovstage.Stage("add-ovstage-collision-filter-pair") as stage:
        population.open_usd_from_string(stage, _RIGID_BODY, ordinal=1, domains=PopulationDomain.PHYSICS)
        stage.advance_write_floor(ordinal=1).wait()

        with pytest.raises(ValueError, match="shape_collision_filter_pairs=1"):
            ovnewton.add_ovstage(builder, stage)


def test_add_ovstage_rejects_populated_custom_frequencies():
    builder = newton.ModelBuilder()
    builder.add_custom_values(**{"mujoco:equality_constraint_type": 1})

    with ovstage.Stage("add-ovstage-custom-frequency-values") as stage:
        population.open_usd_from_string(stage, _RIGID_BODY, ordinal=1, domains=PopulationDomain.PHYSICS)
        stage.advance_write_floor(ordinal=1).wait()

        with pytest.raises(ValueError, match="custom_frequency:mujoco:equality_constraint=1"):
            ovnewton.add_ovstage(builder, stage)


def test_add_ovstage_rejects_populated_custom_attributes():
    builder = newton.ModelBuilder()
    builder.add_custom_attribute(
        newton.ModelBuilder.CustomAttribute(
            name="stale",
            dtype=int,
            frequency=newton.Model.AttributeFrequency.BODY,
            values=[7],
        )
    )

    with ovstage.Stage("add-ovstage-custom-attribute-values") as stage:
        population.open_usd_from_string(stage, _RIGID_BODY, ordinal=1, domains=PopulationDomain.PHYSICS)
        stage.advance_write_floor(ordinal=1).wait()

        with pytest.raises(ValueError, match="custom_attribute:stale=1"):
            ovnewton.add_ovstage(builder, stage)


def test_add_ovstage_rejects_preexisting_actuator_entries():
    builder = newton.ModelBuilder()
    builder.add_actuator(newton.actuators.ControllerPD, index=0, kp=1.0)

    with ovstage.Stage("add-ovstage-actuator-entry") as stage:
        population.open_usd_from_string(stage, _RIGID_BODY, ordinal=1, domains=PopulationDomain.PHYSICS)
        stage.advance_write_floor(ordinal=1).wait()

        with pytest.raises(ValueError, match="actuator_entries=1"):
            ovnewton.add_ovstage(builder, stage)


def test_add_ovstage_reports_rootless_joints_that_need_validation_skipped():
    builder = newton.ModelBuilder()

    with ovstage.Stage("add-ovstage-rootless") as stage:
        population.open_usd_from_string(stage, _ROOTLESS_JOINTS, ordinal=1, domains=PopulationDomain.PHYSICS)
        stage.advance_write_floor(ordinal=1).wait()

        result = ovnewton.add_ovstage(builder, stage)
        model = builder.finalize(skip_validation_joints=result.has_orphan_joints)
        binding = ovnewton.attach_ovstage(stage, model=model)

    assert result.has_orphan_joints
    assert binding.model.articulation_count == 0


def test_add_ovstage_preserves_builder_defaults_and_applies_stage_settings():
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 4321.0

    with ovstage.Stage("add-ovstage-settings") as stage:
        population.open_usd_from_string(stage, _Y_UP_BODY, ordinal=1, domains=PopulationDomain.PHYSICS)
        stage.advance_write_floor(ordinal=1).wait()

        ovnewton.add_ovstage(builder, stage)
        model = builder.finalize()

    assert model.up_axis == newton.Axis.Y
    np.testing.assert_allclose(model.gravity.numpy()[0], (0.0, -981.0, 0.0))
    np.testing.assert_allclose(model.shape_material_ke.numpy(), 4321.0)


@pytest.mark.parametrize("import_api", ["add_ovstage", "attach_ovstage"])
def test_public_import_preserves_distinct_scalar_vector_and_matrix_rows(import_api):
    with wp.ScopedDevice("cpu"), ovstage.Stage("multi-body-import") as stage:
        population.open_usd_from_string(stage, _MULTI_BODY_STATE, ordinal=1, domains=PopulationDomain.PHYSICS)
        stage.advance_write_floor(ordinal=1).wait()

        if import_api == "add_ovstage":
            builder = newton.ModelBuilder()
            ovnewton.add_ovstage(builder, stage)
            model = builder.finalize()
        else:
            model = ovnewton.attach_ovstage(stage).model

        assert set(model.body_label) == {"/First", "/Second"}
        rows = [model.body_label.index(path) for path in ("/First", "/Second")]
        np.testing.assert_allclose(model.gravity.numpy()[0], [0, -6, -8], atol=1e-6)
        np.testing.assert_allclose(model.body_mass.numpy()[rows], [2, 3])
        np.testing.assert_allclose(model.body_com.numpy()[rows], [[0.1, 0.2, 0.3], [-0.4, 0.5, -0.6]])
        poses = model.body_q.numpy()[rows]
        np.testing.assert_allclose(poses[:, :3], [[1, 2, 3], [-4, 5, -6]], atol=1e-6)
        # Quaternion signs are equivalent; these are 90-degree rotations about Z and X.
        rotations = [[0, 0, 0.70710678, 0.70710678], [0.70710678, 0, 0, 0.70710678]]
        np.testing.assert_allclose(np.abs(np.sum(poses[:, 3:] * rotations, axis=1)), [1, 1], atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_public_binding_roundtrips_distinct_body_state(device):
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")

    with wp.ScopedDevice(device), ovstage.Stage("multi-body-roundtrip") as stage:
        population.open_usd_from_string(stage, _MULTI_BODY_STATE, ordinal=1, domains=PopulationDomain.PHYSICS)
        stage.advance_write_floor(ordinal=1).wait()
        builder = newton.ModelBuilder()
        ovnewton.add_ovstage(builder, stage)
        model = builder.finalize()
        binding = ovnewton.attach_ovstage(stage, model=model)
        rows = [model.body_label.index(path) for path in ("/First", "/Second")]
        expected_poses = np.asarray(
            [[-7, 8, 9, 0.70710678, 0, 0, 0.70710678], [10, -11, 12, 0, 0, 0.70710678, 0.70710678]],
            dtype=np.float32,
        )
        expected_velocities = np.asarray([[1, 2, 3, 0.1, -0.2, 0.3], [-4, 5, -6, -0.4, 0.5, -0.6]], dtype=np.float32)
        source = model.state()
        poses = source.body_q.numpy()
        velocities = source.body_qd.numpy()
        poses[rows] = expected_poses
        velocities[rows] = expected_velocities
        source.body_q.assign(wp.array(poses, dtype=wp.transformf, device=device))
        source.body_qd.assign(wp.array(velocities, dtype=wp.spatial_vectorf, device=device))
        binding.update_to_ovstage(source, ordinal=2)
        stage.advance_write_floor(ordinal=2).wait()

        destination = model.state()
        destination.body_q.fill_(float("nan"))
        destination.body_qd.fill_(float("nan"))
        binding.update_from_ovstage(destination, model.control(), ordinal=2)

        actual_poses = destination.body_q.numpy()[rows]
        np.testing.assert_allclose(actual_poses[:, :3], expected_poses[:, :3], atol=1e-5)
        np.testing.assert_allclose(
            np.abs(np.sum(actual_poses[:, 3:] * expected_poses[:, 3:], axis=1)), [1, 1], atol=1e-5
        )
        np.testing.assert_allclose(destination.body_qd.numpy()[rows], expected_velocities, atol=1e-5)


def test_add_ovstage_preserves_builder_joint_defaults_for_d6():
    builder = newton.ModelBuilder()
    builder.default_joint_cfg.damping = 123.0
    builder.default_joint_cfg.effort_limit = 234.0
    builder.default_joint_cfg.actuator_mode = newton.JointTargetMode.EFFORT

    with ovstage.Stage("add-ovstage-d6-defaults") as stage:
        population.open_usd_from_string(stage, _D6_JOINT, ordinal=1, domains=PopulationDomain.PHYSICS)
        stage.advance_write_floor(ordinal=1).wait()

        ovnewton.add_ovstage(builder, stage)

    assert _d6_dof_values(builder, builder.joint_damping) == pytest.approx([123.0])
    assert _d6_dof_values(builder, builder.joint_effort_limit) == pytest.approx([234.0])
    assert _d6_dof_values(builder, builder.joint_target_mode) == [newton.JointTargetMode.EFFORT]


def test_add_ovstage_preserves_builder_joint_defaults_for_merged_d6():
    builder = newton.ModelBuilder()
    builder.default_joint_cfg.damping = 123.0

    with ovstage.Stage("add-ovstage-merged-d6-defaults") as stage:
        population.open_usd_from_string(stage, _MERGED_D6_JOINTS, ordinal=1, domains=PopulationDomain.PHYSICS)
        stage.advance_write_floor(ordinal=1).wait()

        ovnewton.add_ovstage(builder, stage)

    assert _d6_dof_values(builder, builder.joint_damping) == pytest.approx([123.0, 123.0])


def test_add_ovstage_stage_joint_values_override_builder_defaults_for_d6():
    builder = newton.ModelBuilder()
    builder.default_joint_cfg.damping = 123.0
    builder.default_joint_cfg.effort_limit = 234.0
    builder.default_joint_cfg.actuator_mode = newton.JointTargetMode.EFFORT

    with ovstage.Stage("add-ovstage-d6-stage-overrides") as stage:
        population.open_usd_from_string(stage, _D6_JOINT_WITH_DAMPING, ordinal=1, domains=PopulationDomain.PHYSICS)
        stage.advance_write_floor(ordinal=1).wait()

        ovnewton.add_ovstage(builder, stage)

    assert _d6_dof_values(builder, builder.joint_damping) == pytest.approx([456.0])
    assert _d6_dof_values(builder, builder.joint_effort_limit) == pytest.approx([567.0])
    assert _d6_dof_values(builder, builder.joint_target_mode) == [newton.JointTargetMode.POSITION]


def test_mujoco_can_register_attributes_before_add_ovstage():
    pytest.importorskip("mujoco")
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

    with ovstage.Stage("add-ovstage-mujoco") as stage:
        population.open_usd_from_string(stage, _RIGID_BODY, ordinal=1, domains=PopulationDomain.PHYSICS)
        stage.advance_write_floor(ordinal=1).wait()

        ovnewton.add_ovstage(builder, stage)
        model = builder.finalize()
        solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True)
        binding = ovnewton.attach_ovstage(stage, model=model)

    assert binding.model is model
    assert model.mujoco.gravcomp.shape == (model.body_count,)
    assert isinstance(solver, newton.solvers.SolverMuJoCo)
