# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo joint parameters through the public importer and binding APIs."""

import newton
import numpy as np
import ovstage
import pytest
import warp as wp
from newton.usd import SchemaResolverMjc, SchemaResolverNewton
from ovstage import population

import ovnewton

JOINT = """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
    float physics:mass = 1
    float3 physics:diagonalInertia = (1, 1, 1)
}
def PhysicsRevoluteJoint "Joint" (prepend apiSchemas = ["MjcJointAPI"]) {
    rel physics:body1 = </Body>
    float physics:lowerLimit = -45
    float physics:upperLimit = 45
    double mjc:armature = 0.2
    double mjc:damping = 3
    double mjc:frictionloss = 0.4
}
"""


def _populate(stage, usda):
    population.open_usd_from_string_with_desc(
        stage, usda, ordinal=1, time=population.DEFAULT_TIME, descs=ovnewton.population_desc()
    )
    stage.advance_write_floor(1).wait()


@pytest.mark.parametrize("joint_type", ["PhysicsRevoluteJoint", "PhysicsPrismaticJoint"])
def test_mjc_joint_dynamics_match_add_usd(tmp_path, joint_type):
    usda = JOINT.replace("PhysicsRevoluteJoint", joint_type)
    asset = tmp_path / "joint.usda"
    asset.write_text(usda)
    with wp.ScopedDevice("cpu"), ovstage.Stage("mjc-joint-dynamics") as stage:
        reference = newton.ModelBuilder()
        reference.add_usd(
            str(asset), collapse_fixed_joints=False, schema_resolvers=[SchemaResolverNewton(), SchemaResolverMjc()]
        )
        _populate(stage, usda)
        model = ovnewton.attach_ovstage(stage).model
        for field, expected in (("joint_armature", [0.2]), ("joint_damping", [3]), ("joint_friction", [0.4])):
            np.testing.assert_allclose(getattr(model, field).numpy(), expected)
            np.testing.assert_allclose(getattr(model, field).numpy(), getattr(reference, field))


@pytest.mark.parametrize("joint_type", ["PhysicsRevoluteJoint", "PhysicsPrismaticJoint"])
@pytest.mark.parametrize("solref", ["[0.04, 0.8]", "[-200, -10]", None])
def test_mjc_joint_solver_parameters_match_add_usd(tmp_path, joint_type, solref, warp_device):
    pytest.importorskip("mujoco")
    pytest.importorskip("mujoco_warp")
    properties = """double mjc:margin = 0.03
    double[] mjc:solimplimit = [0.8, 0.9, 0.002, 0.4, 3]
    double[] mjc:solreffriction = [0.03, 0.7]
    double[] mjc:solimpfriction = [0.7, 0.85, 0.004, 0.6, 4]
    double mjc:stiffness = 4
    double mjc:springref = 0
    double mjc:ref = 0
    bool mjc:actuatorgravcomp = true"""
    if solref is not None:
        properties += f"\n    double[] mjc:solreflimit = {solref}"
    usda = JOINT.replace("PhysicsRevoluteJoint", joint_type).replace(
        "double mjc:armature", properties + "\n    double mjc:armature"
    )
    asset = tmp_path / "joint.usda"
    asset.write_text(usda)
    with wp.ScopedDevice(warp_device), ovstage.Stage("mjc-joint-solver") as stage:
        reference = newton.ModelBuilder()
        builder = newton.ModelBuilder()
        for target in (reference, builder):
            newton.solvers.SolverMuJoCo.register_custom_attributes(target)
        reference.add_usd(str(asset), schema_resolvers=[SchemaResolverNewton(), SchemaResolverMjc()])
        _populate(stage, usda)
        ovnewton.add_ovstage(builder, stage)
        model, reference_model = builder.finalize(), reference.finalize()
        assert ovnewton.attach_ovstage(stage, model=model).model is model
        for field in ("joint_stiffness", "joint_rest_q"):
            np.testing.assert_allclose(
                getattr(model, field).numpy(), getattr(reference_model, field).numpy(), err_msg=field
            )
        for field in (
            "limit_margin",
            "solreflimit",
            "solimplimit",
            "solreffriction",
            "solimpfriction",
            "dof_ref",
            "jnt_actgravcomp",
            "solreflimit_mode",
            "solreflimit_gain_baseline",
        ):
            np.testing.assert_allclose(
                getattr(model.mujoco, field).numpy(), getattr(reference_model.mujoco, field).numpy(), err_msg=field
            )
        solver = newton.solvers.SolverMuJoCo(model)
        reference_solver = newton.solvers.SolverMuJoCo(reference_model)
        for field in ("jnt_solref", "jnt_solimp", "jnt_margin", "jnt_stiffness", "dof_damping", "dof_armature"):
            np.testing.assert_allclose(
                getattr(solver.mjw_model, field).numpy(), getattr(reference_solver.mjw_model, field).numpy()
            )


@pytest.mark.parametrize("property_type,values", [("float2", "(0.04, 0.8)"), ("double[]", "[0.04, 0.8]")])
def test_mjc_solver_vectors_and_arrays_preserve_multiple_joint_rows(property_type, values):
    usda = JOINT.replace("double mjc:armature", f"{property_type} mjc:solreflimit = {values}\n    double mjc:armature")
    other_joint = (
        usda[usda.index("def Xform") :]
        .replace('"Body"', '"OtherBody"')
        .replace("</Body>", "</OtherBody>")
        .replace('"Joint"', '"OtherJoint"')
        .replace("0.04, 0.8", "0.05, 0.9")
    )
    with wp.ScopedDevice("cpu"), ovstage.Stage("mjc-solver-value-layout") as stage:
        _populate(stage, usda + other_joint)
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        ovnewton.add_ovstage(builder, stage)
        model = builder.finalize()
        for path, expected in (("/Joint", [0.04, 0.8]), ("/OtherJoint", [0.05, 0.9])):
            dof = model.joint_qd_start.numpy()[model.joint_label.index(path)]
            np.testing.assert_allclose(model.mujoco.solreflimit.numpy()[dof], expected)


def test_merged_mjc_axes_preserve_each_dof(tmp_path):
    pytest.importorskip("mujoco")
    usda = JOINT.replace("double mjc:armature", "double[] mjc:solreflimit = [0.04, 0.8]\n    double mjc:armature")
    # Deliberately author the angular joint first; Newton's D6 order is linear first.
    usda += """def PhysicsPrismaticJoint "Slider" (prepend apiSchemas = ["MjcJointAPI"]) {
    rel physics:body1 = </Body>
    double mjc:armature = 0.7
    double mjc:damping = 5
    double mjc:frictionloss = 0.6
    double[] mjc:solreflimit = [-300, -20]
}
"""
    asset = tmp_path / "merged.usda"
    asset.write_text(usda)
    with wp.ScopedDevice("cpu"), ovstage.Stage("merged-mjc") as stage:
        reference, builder = newton.ModelBuilder(), newton.ModelBuilder()
        for target in (reference, builder):
            newton.solvers.SolverMuJoCo.register_custom_attributes(target)
        reference.add_usd(str(asset), schema_resolvers=[SchemaResolverNewton(), SchemaResolverMjc()])
        _populate(stage, usda)
        ovnewton.add_ovstage(builder, stage)
        model, oracle = builder.finalize(), reference.finalize()
        assert model.joint_count == 1
        np.testing.assert_allclose(model.joint_damping.numpy(), [5, 3])
        np.testing.assert_allclose(model.joint_armature.numpy(), [0.7, 0.2])
        np.testing.assert_allclose(model.joint_friction.numpy(), [0.6, 0.4])
        np.testing.assert_allclose(model.mujoco.solreflimit.numpy(), [[-300, -20], [0.04, 0.8]])
        np.testing.assert_allclose(model.mujoco.solreflimit.numpy(), oracle.mujoco.solreflimit.numpy())


def test_mjc_solver_parameters_require_registration():
    usda = JOINT.replace("double mjc:armature", "double mjc:stiffness = 5\n    double mjc:armature")
    with wp.ScopedDevice("cpu"), ovstage.Stage("mjc-registration-required") as stage:
        _populate(stage, usda)
        with pytest.raises(ovnewton.UnsupportedPhysicsError, match=r"register_custom_attributes.*before add_ovstage"):
            ovnewton.attach_ovstage(stage)


@pytest.mark.parametrize("property_name", ["armature", "damping", "frictionloss"])
def test_invalid_mjc_joint_dynamics_are_rejected(property_name):
    usda = JOINT.replace(f"double mjc:{property_name} = ", f"double mjc:{property_name} = -")
    with wp.ScopedDevice("cpu"), ovstage.Stage("invalid-mjc-dynamics") as stage:
        _populate(stage, usda)
        with pytest.raises(ovnewton.InvalidPhysicsError, match=f"invalid mjc:{property_name} at /Joint"):
            ovnewton.attach_ovstage(stage)


def test_mjc_spring_state_roundtrip(warp_device):
    pytest.importorskip("mujoco")
    pytest.importorskip("mujoco_warp")
    usda = JOINT.replace(
        "double mjc:armature", "double mjc:stiffness = 20\n    double mjc:springref = 0.3\n    double mjc:armature"
    )
    usda += '\ndef PhysicsScene "Physics" {\n    uniform token mjc:compiler:angle = "radian"\n}\n'
    with wp.ScopedDevice(warp_device), ovstage.Stage("mjc-spring-roundtrip") as stage:
        _populate(stage, usda)
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        ovnewton.add_ovstage(builder, stage)
        model = builder.finalize()
        binding = ovnewton.attach_ovstage(stage, model=model)
        solver = newton.solvers.SolverMuJoCo(model)
        state, next_state = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        for _ in range(120):
            solver.step(state, next_state, None, None, 1 / 240)
            state, next_state = next_state, state
        assert state.joint_q.numpy()[0] > 0.1
        binding.update_to_ovstage(state, ordinal=2)
        stage.advance_write_floor(2).wait()
        roundtrip = model.state()
        binding.update_from_ovstage(roundtrip, model.control(), ordinal=2)
        np.testing.assert_allclose(roundtrip.joint_q.numpy(), state.joint_q.numpy(), atol=1e-6)
        np.testing.assert_allclose(roundtrip.joint_qd.numpy(), state.joint_qd.numpy(), atol=1e-6)


@pytest.mark.parametrize("override_defaults", [False, True])
@pytest.mark.parametrize("newton_api", [False, True])
def test_mjc_d6_defaults_and_solver_parameters(tmp_path, override_defaults, newton_api):
    pytest.importorskip("mujoco")
    usda = (
        JOINT.replace("PhysicsRevoluteJoint", "PhysicsJoint")
        .replace('"MjcJointAPI"', '"MjcJointAPI", "PhysicsLimitAPI:transX", "PhysicsLimitAPI:rotZ"')
        .replace(
            "float physics:lowerLimit = -45\n    float physics:upperLimit = 45",
            """float limit:transX:physics:low = -1
    float limit:transX:physics:high = 1
    float limit:rotZ:physics:low = -45
    float limit:rotZ:physics:high = 45
    double[] mjc:solimplimit = [0.8, 0.9, 0.002, 0.4, 3]""",
        )
    )
    if newton_api:
        usda = usda.replace('"MjcJointAPI"', '"MjcJointAPI", "NewtonJointAPI"')
    asset = tmp_path / "d6.usda"
    asset.write_text(usda)
    with wp.ScopedDevice("cpu"), ovstage.Stage("mjc-d6") as stage:
        reference, builder = newton.ModelBuilder(), newton.ModelBuilder()
        for target in (reference, builder):
            newton.solvers.SolverMuJoCo.register_custom_attributes(target)
            if override_defaults:
                target.default_joint_cfg.limit_ke = 200
                target.default_joint_cfg.limit_kd = 20
        reference.add_usd(str(asset), schema_resolvers=[SchemaResolverNewton(), SchemaResolverMjc()])
        _populate(stage, usda)
        ovnewton.add_ovstage(builder, stage)
        model, oracle = builder.finalize(), reference.finalize()
        np.testing.assert_allclose(model.joint_damping.numpy(), [3, 3])
        np.testing.assert_allclose(model.joint_damping.numpy(), oracle.joint_damping.numpy())
        np.testing.assert_allclose(model.mujoco.solimplimit.numpy(), oracle.mujoco.solimplimit.numpy())
        np.testing.assert_array_equal(model.mujoco.solreflimit_mode.numpy(), oracle.mujoco.solreflimit_mode.numpy())
        np.testing.assert_array_equal(model.mujoco.solreflimit_mode.numpy(), [0, 0] if override_defaults else [2, 2])


@pytest.mark.parametrize("joint_type", ["PhysicsRevoluteJoint", "PhysicsPrismaticJoint"])
@pytest.mark.parametrize("newton_value", [None, 0, 2])
@pytest.mark.parametrize("mjc_values", [False, True])
def test_newton_joint_dynamics_precedence(tmp_path, joint_type, newton_value, mjc_values):
    usda = JOINT.replace("PhysicsRevoluteJoint", joint_type).replace(
        '"MjcJointAPI"', '"MjcJointAPI", "NewtonJointAPI"'
    )
    if not mjc_values:
        usda = "\n".join(line for line in usda.splitlines() if "double mjc:" not in line)
    if newton_value is not None:
        authored = "\n".join(
            f"float newton:{name} = {newton_value * scale}"
            for scale, name in enumerate(("armature", "damping", "friction"), start=1)
        )
        usda = usda.replace("rel physics:body1", authored + "\n    rel physics:body1")
    asset = tmp_path / "precedence.usda"
    asset.write_text(usda)
    with wp.ScopedDevice("cpu"), ovstage.Stage("joint-parameter-precedence") as stage:
        reference, builder = newton.ModelBuilder(), newton.ModelBuilder()
        for target in (reference, builder):
            target.default_joint_cfg.armature = 0.7
            target.default_joint_cfg.damping = 5
            target.default_joint_cfg.friction = 0.6
        reference.add_usd(str(asset), schema_resolvers=[SchemaResolverNewton(), SchemaResolverMjc()])
        _populate(stage, usda)
        ovnewton.add_ovstage(builder, stage)
        model = builder.finalize()
        for field in ("joint_damping", "joint_armature", "joint_friction"):
            np.testing.assert_allclose(getattr(model, field).numpy(), getattr(reference, field))


@pytest.mark.parametrize(
    "limit_settings,custom_defaults,mode",
    [
        ("", False, 2),
        ("float newton:limitStiffness = -inf", False, 0),
        ("float newton:limitDamping = -inf", False, 0),
        ("float newton:limitStiffness = 20", False, 0),
        ("double[] mjc:solreflimit = [0.04, 0.8]", False, 1),
        ("", True, 0),
    ],
)
def test_explicit_limit_settings_resolve_mixed_schema_defaults(
    tmp_path, limit_settings, custom_defaults, mode, warp_device
):
    pytest.importorskip("mujoco")
    pytest.importorskip("mujoco_warp")
    usda = JOINT.replace('"MjcJointAPI"', '"MjcJointAPI", "NewtonJointAPI"').replace(
        "double mjc:armature",
        "float newton:armature = 1\n    float newton:damping = 2\n"
        f"    float newton:friction = 3\n    {limit_settings}\n    double mjc:armature",
    )
    asset = tmp_path / "mixed-limit-settings.usda"
    asset.write_text(usda)
    with wp.ScopedDevice(warp_device), ovstage.Stage("explicit-mjc-limit-settings") as stage:
        reference, builder = newton.ModelBuilder(), newton.ModelBuilder()
        for target in (reference, builder):
            newton.solvers.SolverMuJoCo.register_custom_attributes(target)
            if custom_defaults:
                target.default_joint_cfg.limit_ke = 200
        reference.add_usd(str(asset), schema_resolvers=[SchemaResolverNewton(), SchemaResolverMjc()])
        _populate(stage, usda)
        ovnewton.add_ovstage(builder, stage)
        model, oracle = builder.finalize(), reference.finalize()
        np.testing.assert_array_equal(model.mujoco.solreflimit_mode.numpy(), [mode])
        np.testing.assert_array_equal(model.mujoco.solreflimit_mode.numpy(), oracle.mujoco.solreflimit_mode.numpy())
        solver = newton.solvers.SolverMuJoCo(model)
        reference_solver = newton.solvers.SolverMuJoCo(oracle)
        np.testing.assert_allclose(solver.mjw_model.jnt_solref.numpy(), reference_solver.mjw_model.jnt_solref.numpy())


def test_removed_mjc_schema_does_not_enable_its_parameters(tmp_path):
    base = tmp_path / "base.usda"
    base.write_text(JOINT)
    scene = tmp_path / "scene.usda"
    scene.write_text("""#usda 1.0
(subLayers = [@base.usda@])
over "Joint" (delete apiSchemas = ["MjcJointAPI"]) {}
""")
    with wp.ScopedDevice("cpu"), ovstage.Stage("removed-mjc-schema") as stage:
        population.open_usd_with_desc(
            stage, str(scene), ordinal=1, time=population.DEFAULT_TIME, descs=ovnewton.population_desc()
        )
        stage.advance_write_floor(1).wait()
        model = ovnewton.attach_ovstage(stage).model
        np.testing.assert_allclose(model.joint_damping.numpy(), [newton.ModelBuilder().default_joint_cfg.damping])


@pytest.mark.parametrize(
    "angle_unit,expected", [(None, 0.2617993877991494), ("degree", 0.2617993877991494), ("radian", 15)]
)
@pytest.mark.parametrize("joint_type", ["PhysicsRevoluteJoint", "PhysicsPrismaticJoint"])
def test_mjc_reference_coordinate_units_and_limits(tmp_path, angle_unit, expected, joint_type):
    pytest.importorskip("mujoco")
    usda = JOINT.replace("PhysicsRevoluteJoint", joint_type).replace(
        "double mjc:armature", "double mjc:ref = 15\n    double mjc:springref = 45\n    double mjc:armature"
    )
    if angle_unit is not None:
        usda += f'\ndef PhysicsScene "Physics" {{\n    uniform token mjc:compiler:angle = "{angle_unit}"\n}}\n'
    asset = tmp_path / "reference-coordinate.usda"
    asset.write_text(usda)
    with wp.ScopedDevice("cpu"), ovstage.Stage("mjc-reference-coordinate") as stage:
        _populate(stage, usda)
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        ovnewton.add_ovstage(builder, stage)
        model = builder.finalize()
        if joint_type == "PhysicsPrismaticJoint":
            expected = 15
        np.testing.assert_allclose(model.mujoco.dof_ref.numpy(), [expected])
        # Newton stores the spring rest coordinate relative to mjc:ref.
        np.testing.assert_allclose(model.joint_rest_q.numpy(), [2 * expected])
        limit = 0.7853981633974483 if joint_type == "PhysicsRevoluteJoint" else 45
        np.testing.assert_allclose(model.joint_limit_lower.numpy(), [-limit - expected])
        np.testing.assert_allclose(model.joint_limit_upper.numpy(), [limit - expected])
        reference = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(reference)
        reference.add_usd(str(asset), schema_resolvers=[SchemaResolverNewton(), SchemaResolverMjc()])
        oracle = reference.finalize()
        np.testing.assert_allclose(model.mujoco.dof_ref.numpy(), oracle.mujoco.dof_ref.numpy())
        np.testing.assert_allclose(model.joint_rest_q.numpy(), oracle.joint_rest_q.numpy())
        np.testing.assert_allclose(model.joint_limit_lower.numpy(), oracle.joint_limit_lower.numpy())


@pytest.mark.parametrize("angle_unit,expected", [("degree", np.pi / 12), ("radian", 15)])
def test_matching_scene_angle_conventions_are_supported(angle_unit, expected):
    usda = JOINT.replace("double mjc:armature", "double mjc:ref = 15\n    double mjc:armature")
    for name in ("First", "Second"):
        usda += f'\ndef PhysicsScene "{name}" {{\n    uniform token mjc:compiler:angle = "{angle_unit}"\n}}\n'
    with wp.ScopedDevice("cpu"), ovstage.Stage("matching-mjc-scene-angles") as stage:
        _populate(stage, usda)
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        ovnewton.add_ovstage(builder, stage)
        np.testing.assert_allclose(builder.finalize().mujoco.dof_ref.numpy(), [expected])


@pytest.mark.parametrize("other_unit", [None, "degree", "invalid"])
def test_conflicting_scene_angle_conventions_are_rejected(other_unit):
    usda = JOINT.replace("double mjc:armature", "double mjc:ref = 15\n    double mjc:armature")
    usda += '\ndef PhysicsScene "First" {\n    uniform token mjc:compiler:angle = "radian"\n}\n'
    property_text = f'uniform token mjc:compiler:angle = "{other_unit}"' if other_unit else ""
    usda += f'\ndef PhysicsScene "Second" {{\n    {property_text}\n}}\n'
    with wp.ScopedDevice("cpu"), ovstage.Stage("conflicting-mjc-scene-angles") as stage:
        _populate(stage, usda)
        with pytest.raises(ovnewton.UnsupportedPhysicsError, match="conflicting or invalid mjc:compiler:angle"):
            ovnewton.attach_ovstage(stage)


@pytest.mark.parametrize("other_unit", ["degree", "invalid"])
@pytest.mark.parametrize("joint_case", ["ordinary-usd", "no-reference", "prismatic", "disabled"])
def test_unused_mjc_angle_conventions_do_not_block_import(other_unit, joint_case):
    usda = JOINT
    if joint_case == "ordinary-usd":
        usda = usda.replace(' (prepend apiSchemas = ["MjcJointAPI"])', "")
    elif joint_case in ("prismatic", "disabled"):
        usda = usda.replace("double mjc:armature", "double mjc:ref = 15\n    double mjc:armature")
        if joint_case == "prismatic":
            usda = usda.replace("PhysicsRevoluteJoint", "PhysicsPrismaticJoint")
        else:
            usda = usda.replace("rel physics:body1", "bool physics:jointEnabled = false\n    rel physics:body1")
    usda += '\ndef PhysicsScene "First" {\n    uniform token mjc:compiler:angle = "radian"\n}\n'
    usda += f'\ndef PhysicsScene "Second" {{\n    uniform token mjc:compiler:angle = "{other_unit}"\n}}\n'
    with wp.ScopedDevice("cpu"), ovstage.Stage("unused-mjc-scene-angles") as stage:
        _populate(stage, usda)
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        ovnewton.add_ovstage(builder, stage)
        model = builder.finalize()
        assert model.body_count == 1
        assert ("/Joint" in model.joint_label) == (joint_case != "disabled")
        if joint_case == "prismatic":
            np.testing.assert_allclose(model.mujoco.dof_ref.numpy(), [15])


def test_mjc_springdamper_is_not_silently_dropped():
    usda = JOINT.replace("double mjc:armature", "double[] mjc:springdamper = [0.03, 1]\n    double mjc:armature")
    with wp.ScopedDevice("cpu"), ovstage.Stage("unsupported-mjc-setting") as stage:
        _populate(stage, usda)
        with pytest.raises(ovnewton.UnsupportedPhysicsError, match="unsupported mjc:"):
            ovnewton.attach_ovstage(stage)


@pytest.mark.parametrize("property_name", ["ref", "springref"])
def test_native_d6_reference_coordinates_are_rejected(property_name):
    usda = JOINT.replace("PhysicsRevoluteJoint", "PhysicsJoint").replace(
        "double mjc:armature", f"double mjc:{property_name} = 0.3\n    double mjc:armature"
    )
    with wp.ScopedDevice("cpu"), ovstage.Stage("unsupported-d6-reference") as stage:
        _populate(stage, usda)
        with pytest.raises(ovnewton.UnsupportedPhysicsError, match="no native D6 reference-coordinate semantics"):
            ovnewton.attach_ovstage(stage)


@pytest.mark.parametrize("joint_type", ["PhysicsRevoluteJoint", "PhysicsPrismaticJoint"])
@pytest.mark.parametrize("limits,limited,drive_limit,expected", [
    ((-2, 5), "auto", None, 5),
    ((-2, 5), "false", None, 1.0e6),
    ((0, 0), "true", None, 0),
    ((0, 0), "auto", None, 1.0e6),
    ((-7, None), "auto", None, 7),
    ((None, 4), "auto", None, 4),
    ((None, None), "true", None, 1.0e6),
    ((-2, 5), "auto", 3, 3),
    ((-2, 5), "true", 8, 5),
])
def test_mjc_joint_force_limits_match_reference(tmp_path, joint_type, limits, limited, drive_limit, expected):
    properties = f'token mjc:actuatorfrclimited = "{limited}"'
    for name, value in zip(("min", "max"), limits):
        if value is not None:
            properties += f"\n    double mjc:actuatorfrcrange:{name} = {value}"
    usda = JOINT.replace("PhysicsRevoluteJoint", joint_type).replace(
        "double mjc:armature", properties + "\n    double mjc:armature"
    )
    if drive_limit is not None:
        axis = "angular" if joint_type == "PhysicsRevoluteJoint" else "linear"
        usda = usda.replace('"MjcJointAPI"', f'"MjcJointAPI", "PhysicsDriveAPI:{axis}"').replace(
            "double mjc:armature", f"float drive:{axis}:physics:maxForce = {drive_limit}\n    double mjc:armature"
        )
    asset = tmp_path / "force-limits.usda"
    asset.write_text(usda)
    with wp.ScopedDevice("cpu"), ovstage.Stage("mjc-force-limits") as stage:
        reference = newton.ModelBuilder()
        reference.add_usd(str(asset), schema_resolvers=[SchemaResolverNewton(), SchemaResolverMjc()])
        _populate(stage, usda)
        model = ovnewton.attach_ovstage(stage).model
        np.testing.assert_allclose(model.joint_effort_limit.numpy(), [expected])
        np.testing.assert_allclose(model.joint_effort_limit.numpy(), reference.joint_effort_limit)


@pytest.mark.parametrize("endpoint", ["min", "max"])
def test_native_d6_force_ranges_are_rejected(tmp_path, endpoint):
    usda = (
        JOINT.replace("PhysicsRevoluteJoint", "PhysicsJoint")
        .replace('"MjcJointAPI"', '"MjcJointAPI", "PhysicsLimitAPI:transX", "PhysicsLimitAPI:rotZ"')
        .replace(
            "float physics:lowerLimit = -45\n    float physics:upperLimit = 45",
            """float limit:transX:physics:low = -1
    float limit:transX:physics:high = 1
    float limit:rotZ:physics:low = -45
    float limit:rotZ:physics:high = 45""",
        )
        .replace("double mjc:armature", f"double mjc:actuatorfrcrange:{endpoint} = 5\n    double mjc:armature")
    )
    asset = tmp_path / "d6-force-range.usda"
    asset.write_text(usda)
    with wp.ScopedDevice("cpu"), ovstage.Stage("unsupported-d6-force-range") as stage:
        reference = newton.ModelBuilder()
        reference.add_usd(str(asset), schema_resolvers=[SchemaResolverNewton(), SchemaResolverMjc()])
        np.testing.assert_array_equal(reference.joint_effort_limit, [np.inf, np.inf])
        _populate(stage, usda)
        with pytest.raises(ovnewton.UnsupportedPhysicsError, match="native D6 actuator-force ranges"):
            ovnewton.attach_ovstage(stage)


def test_mjc_force_limits_keep_merged_axis_order():
    usda = JOINT.replace("double mjc:armature", "double mjc:actuatorfrcrange:max = 5\n    double mjc:armature")
    usda += '''def PhysicsPrismaticJoint "Slider" (prepend apiSchemas = ["MjcJointAPI"]) {
        rel physics:body1 = </Body>
        double mjc:actuatorfrcrange:min = -7
    }
'''
    with wp.ScopedDevice("cpu"), ovstage.Stage("merged-force-limits") as stage:
        _populate(stage, usda)
        model = ovnewton.attach_ovstage(stage).model
        np.testing.assert_allclose(model.joint_effort_limit.numpy(), [7, 5])


def test_mjc_force_limit_clamps_solver_drive(warp_device):
    pytest.importorskip("mujoco")
    pytest.importorskip("mujoco_warp")
    usda = JOINT.replace('"MjcJointAPI"', '"MjcJointAPI", "PhysicsDriveAPI:angular"').replace(
        "double mjc:armature",
        """double mjc:actuatorfrcrange:min = -2
        double mjc:actuatorfrcrange:max = 5
        float drive:angular:physics:stiffness = 100
        float drive:angular:physics:targetPosition = 40
        double mjc:armature""",
    )
    with wp.ScopedDevice(warp_device), ovstage.Stage("force-limited-drive") as stage:
        _populate(stage, usda)
        binding = ovnewton.attach_ovstage(stage)
        model = binding.model
        solver = newton.solvers.SolverMuJoCo(model)
        state, next_state, control = model.state(), model.state(), model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        solver.step(state, next_state, control, None, 1 / 240)
        np.testing.assert_allclose(solver.mjw_data.qfrc_actuator.numpy(), [[5]])
