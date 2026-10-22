# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo equality import through ovnewton's public builder and binding APIs."""

import newton
import numpy as np
import ovstage
import pytest
import warp as wp
from ovstage import PopulationDomain, population

import ovnewton

CONNECT = """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "World" {
    def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
        float physics:mass = 1
        float3 physics:diagonalInertia = (1, 1, 1)
    }
    def PhysicsSphericalJoint "Equality" (prepend apiSchemas = ["MjcEqualityConnectAPI"]) {
        rel physics:body1 = </World/Body>
        point3f physics:localPos0 = (1, 2, 3)
        double[] mjc:solref = [0.04, 0.8]
        double[] mjc:solimp = [0.8, 0.9, 0.002, 0.4, 3]
    }
}
"""

WELD = (
    CONNECT.replace("PhysicsSphericalJoint", "PhysicsFixedJoint")
    .replace("MjcEqualityConnectAPI", "MjcEqualityWeldAPI")
    .replace(
        "point3f physics:localPos0 = (1, 2, 3)",
        """point3f physics:localPos0 = (1, 2, 3)
        point3f physics:localPos1 = (0.5, -1, 2)
        quatf physics:localRot0 = (0.70710678, 0, 0, 0.70710678)
        float mjc:torqueScale = 2.5""",
    )
)


JOINT = """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "World" {
    def Xform "Body1" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
        float physics:mass = 1
        float3 physics:diagonalInertia = (1, 1, 1)
    }
    def Xform "Body2" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
        float physics:mass = 1
        float3 physics:diagonalInertia = (1, 1, 1)
    }
    def PhysicsPrismaticJoint "Leader" {
        rel physics:body1 = </World/Body1>
        uniform token physics:axis = "Z"
    }
    def PhysicsPrismaticJoint "Follower" (prepend apiSchemas = ["NewtonMimicAPI", "MjcEqualityJointAPI"]) {
        rel physics:body1 = </World/Body2>
        uniform token physics:axis = "Z"
        rel newton:mimicJoint = </World/Leader>
        float newton:mimicCoef0 = 0
        float newton:mimicCoef1 = 2
        double[] mjc:solref = [0.04, 0.8]
        double[] mjc:solimp = [0.8, 0.9, 0.002, 0.4, 3]
    }
}
"""


def _populate(stage, usda):
    population.open_usd_from_string_with_desc(
        stage, usda, ordinal=1, time=population.DEFAULT_TIME, descs=ovnewton.population_desc()
    )
    stage.advance_write_floor(ordinal=1).wait()


def _legacy_joint_equality(usda):
    return (
        usda.replace('"NewtonMimicAPI", ', "")
        .replace("newton:mimicJoint", "mjc:target")
        .replace("newton:mimicCoef", "mjc:coef")
    )


@pytest.mark.parametrize("joint_type", ["PhysicsPrismaticJoint", "PhysicsRevoluteJoint"])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("polynomial", [False, True])
def test_joint_equality_matches_newton_add_usd(tmp_path, joint_type, legacy, polynomial):
    usda = JOINT.replace("PhysicsPrismaticJoint", joint_type).replace("mimicCoef0 = 0", "mimicCoef0 = 30")
    if legacy:
        usda = _legacy_joint_equality(usda)
    if polynomial:
        usda = usda.replace(
            "double[] mjc:solref",
            """float mjc:coef2 = 0.3
        float mjc:coef3 = -0.2
        float mjc:coef4 = 0.1
        double[] mjc:solref""",
        )
    asset = tmp_path / "joint-equality.usda"
    asset.write_text(usda)
    with wp.ScopedDevice("cpu"), ovstage.Stage("joint-equality-parity") as stage:
        reference_builder = newton.ModelBuilder()
        reference_builder.add_usd(str(asset), collapse_fixed_joints=False)
        reference = reference_builder.finalize()
        _populate(stage, usda)
        builder = newton.ModelBuilder()
        result = ovnewton.add_ovstage(builder, stage)
        model = builder.finalize(skip_validation_joints=result.has_orphan_joints)
        assert ovnewton.attach_ovstage(stage, model=model).model is model
        assert model.mujoco.equality_constraint_count == reference.mujoco.equality_constraint_count == 1
        for field in (
            "equality_constraint_type",
            "equality_constraint_polycoef",
            "equality_constraint_enabled",
            "equality_constraint_objtype",
            "equality_constraint_target_kind",
            "eq_solref",
            "eq_solimp",
        ):
            np.testing.assert_allclose(
                getattr(model.mujoco, field).numpy(),
                getattr(reference.mujoco, field).numpy(),
                atol=1e-6,
                err_msg=field,
            )
        for field in ("equality_constraint_joint1", "equality_constraint_joint2"):
            actual = int(getattr(model.mujoco, field).numpy()[0])
            expected = int(getattr(reference.mujoco, field).numpy()[0])
            assert model.joint_label[actual] == reference.joint_label[expected]
        assert model.constraint_mimic_count == reference.constraint_mimic_count == 1
        offset = np.deg2rad(30) if joint_type == "PhysicsRevoluteJoint" and not legacy else 30
        np.testing.assert_allclose(
            model.mujoco.equality_constraint_polycoef.numpy(),
            [[offset, 2, 0.3, -0.2, 0.1] if polynomial else [offset, 2, 0, 0, 0]],
        )


@pytest.mark.parametrize("legacy", [False, True])
def test_multiple_joint_equalities_preserve_each_coupling(legacy):
    first = JOINT.replace("double[] mjc:solref", "float mjc:coef2 = 0\n        double[] mjc:solref").replace(
        "rel newton:mimicJoint", "bool newton:mimicEnabled = true\n        rel newton:mimicJoint"
    )
    second = (
        first[first.index('def Xform "World"') :]
        .replace("World", "Other")
        .replace("mimicCoef0 = 0", "mimicCoef0 = 3")
        .replace("mimicCoef1 = 2", "mimicCoef1 = -2")
        .replace("[0.04, 0.8]", "[0.06, 0.9]")
        .replace("[0.8, 0.9, 0.002, 0.4, 3]", "[0.7, 0.85, 0.003, 0.3, 4]")
        .replace("mjc:coef2 = 0", "mjc:coef2 = 0.5")
        .replace("mimicEnabled = true", "mimicEnabled = false")
    )
    usda = first + second
    if legacy:
        usda = _legacy_joint_equality(usda)
    expected = {
        "/World/Follower": ("/World/Leader", [0, 2, 0, 0, 0], [0.04, 0.8], [0.8, 0.9, 0.002, 0.4, 3], True),
        "/Other/Follower": ("/Other/Leader", [3, -2, 0.5, 0, 0], [0.06, 0.9], [0.7, 0.85, 0.003, 0.3, 4], False),
    }
    with wp.ScopedDevice("cpu"), ovstage.Stage("multiple-joint-equalities") as stage:
        _populate(stage, usda)
        builder = newton.ModelBuilder()
        result = ovnewton.add_ovstage(builder, stage)
        model = builder.finalize(skip_validation_joints=result.has_orphan_joints)
        assert ovnewton.attach_ovstage(stage, model=model).model is model
        eq = model.mujoco
        assert eq.equality_constraint_count == model.constraint_mimic_count == 2
        assert set(eq.equality_constraint_label) == set(expected)
        for i, path in enumerate(eq.equality_constraint_label):
            leader, coefficients, solref, solimp, enabled = expected[path]
            assert model.joint_label[int(eq.equality_constraint_joint1.numpy()[i])] == path
            assert model.joint_label[int(eq.equality_constraint_joint2.numpy()[i])] == leader
            np.testing.assert_allclose(eq.equality_constraint_polycoef.numpy()[i], coefficients)
            np.testing.assert_allclose(eq.eq_solref.numpy()[i], solref)
            np.testing.assert_allclose(eq.eq_solimp.numpy()[i], solimp)
            assert bool(eq.equality_constraint_enabled.numpy()[i]) == enabled


@pytest.mark.parametrize("flag", ["newton:mimicEnabled", "physics:jointEnabled"])
def test_disabled_joint_equality_matches_newton(tmp_path, flag):
    usda = JOINT.replace("rel newton:mimicJoint", f"bool {flag} = false\n        rel newton:mimicJoint")
    asset = tmp_path / "disabled-joint-equality.usda"
    asset.write_text(usda)
    with wp.ScopedDevice("cpu"), ovstage.Stage("disabled-joint-equality") as stage:
        reference_builder = newton.ModelBuilder()
        reference_builder.add_usd(str(asset), collapse_fixed_joints=False)
        reference = reference_builder.finalize()
        _populate(stage, usda)
        model = ovnewton.attach_ovstage(stage).model
        assert model.mujoco.equality_constraint_count == reference.mujoco.equality_constraint_count
        assert model.constraint_mimic_count == reference.constraint_mimic_count
        if flag == "newton:mimicEnabled":
            np.testing.assert_array_equal(model.mujoco.equality_constraint_enabled.numpy(), [False])
            np.testing.assert_array_equal(model.constraint_mimic_enabled.numpy(), [False])
        else:
            assert "/World/Follower" not in model.joint_label


@pytest.mark.parametrize("polynomial", [False, True])
@pytest.mark.parametrize("angular", [False, True])
def test_mujoco_enforces_joint_equality_and_state_roundtrip(polynomial, angular):
    pytest.importorskip("mujoco")
    usda = (
        JOINT.replace("float newton:mimicCoef0 = 0", "float newton:mimicCoef0 = 0\n        float mjc:coef2 = 0.5")
        if polynomial
        else JOINT
    )
    if angular:
        usda = usda.replace("PhysicsPrismaticJoint", "PhysicsRevoluteJoint")
    with wp.ScopedDevice("cpu"), ovstage.Stage("joint-equality-solver") as stage:
        _populate(stage, usda)
        binding = ovnewton.attach_ovstage(stage)
        model = binding.model
        solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True)
        assert solver.mj_model.neq == 1
        state, next_state = model.state(), model.state()
        velocity = model.joint_qd.numpy()
        velocity_starts = model.joint_qd_start.numpy()
        velocity[velocity_starts[model.joint_label.index("/World/Leader")]] = 1
        velocity[velocity_starts[model.joint_label.index("/World/Follower")]] = 2
        state.joint_qd.assign(velocity)
        newton.eval_fk(model, model.joint_q, state.joint_qd, state)
        for _ in range(120):
            solver.step(state, next_state, None, None, 1.0 / 240.0)
            state, next_state = next_state, state
        starts = model.joint_q_start.numpy()
        leader = starts[model.joint_label.index("/World/Leader")]
        follower = starts[model.joint_label.index("/World/Follower")]
        q = state.joint_q.numpy()
        assert abs(q[leader]) > 0.05
        expected = 2 * q[leader] + (0.5 * q[leader] ** 2 if polynomial else 0)
        assert q[follower] == pytest.approx(expected, abs=0.02)
        binding.update_to_ovstage(state, ordinal=2)
        stage.advance_write_floor(2).wait()
        roundtrip = model.state()
        binding.update_from_ovstage(roundtrip, model.control(), ordinal=2)
        np.testing.assert_allclose(roundtrip.joint_q.numpy(), q, atol=1e-6)
        np.testing.assert_allclose(roundtrip.joint_qd.numpy(), state.joint_qd.numpy(), atol=1e-6)


def test_joint_equality_modern_properties_override_legacy_aliases():
    usda = JOINT.replace(
        "float newton:mimicCoef0 = 0",
        """float newton:mimicCoef0 = 3
        rel mjc:target = </World/Missing>
        float mjc:coef0 = inf
        float mjc:coef1 = nan""",
    )
    with wp.ScopedDevice("cpu"), ovstage.Stage("joint-equality-precedence") as stage:
        _populate(stage, usda)
        model = ovnewton.attach_ovstage(stage).model
        np.testing.assert_allclose(model.mujoco.equality_constraint_polycoef.numpy(), [[3, 2, 0, 0, 0]])


def test_joint_equality_polynomial_projection_has_structured_diagnostic(caplog):
    usda = JOINT.replace("double[] mjc:solref", "float mjc:coef2 = 0.5\n        double[] mjc:solref")
    with wp.ScopedDevice("cpu"), ovstage.Stage("polynomial-joint-equality") as stage:
        _populate(stage, usda)
        model = ovnewton.attach_ovstage(stage).model
        records = [record for record in caplog.records if record.name == "ovnewton"]
        assert [(record.diagnostic_code, record.prim_path) for record in records] == [
            ("joint-equality-linear-projection", "/World/Follower")
        ]
        np.testing.assert_allclose(model.mujoco.equality_constraint_polycoef.numpy(), [[0, 2, 0.5, 0, 0]])


@pytest.mark.parametrize("coefficient,default,legacy", [(0, 0, 30), (1, 1, 2)])
@pytest.mark.parametrize("authored_default", [False, True])
def test_joint_equality_rejects_ambiguous_coefficient_fallback(coefficient, default, legacy, authored_default):
    usda = "\n".join(line for line in JOINT.splitlines() if f"newton:mimicCoef{coefficient}" not in line)
    properties = f"float mjc:coef{coefficient} = {legacy}"
    if authored_default:
        properties += f"\n        float newton:mimicCoef{coefficient} = {default}"
    usda = usda.replace("double[] mjc:solref", f"{properties}\n        double[] mjc:solref")
    with wp.ScopedDevice("cpu"), ovstage.Stage("ambiguous-joint-equality") as stage:
        _populate(stage, usda)
        with pytest.raises(ovnewton.UnsupportedPhysicsError, match=rf"ambiguous.*mimicCoef{coefficient}"):
            ovnewton.attach_ovstage(stage)


@pytest.mark.parametrize("legacy", [False, True])
def test_joint_equality_defaults_and_relative_target(legacy):
    usda = "\n".join(line for line in JOINT.splitlines() if "mimicCoef" not in line and "mjc:" not in line)
    usda = usda.replace("</World/Leader>", "<../Leader>")
    if legacy:
        usda = _legacy_joint_equality(usda)
    with wp.ScopedDevice("cpu"), ovstage.Stage("joint-equality-defaults") as stage:
        _populate(stage, usda)
        model = ovnewton.attach_ovstage(stage).model
        np.testing.assert_allclose(model.mujoco.equality_constraint_polycoef.numpy(), [[0, 1, 0, 0, 0]])
        np.testing.assert_allclose(model.mujoco.eq_solref.numpy(), [[0.02, 1]])
        np.testing.assert_allclose(model.mujoco.eq_solimp.numpy(), [[0.9, 0.95, 0.001, 0.5, 2]])


@pytest.mark.parametrize(
    "old,new,error,message",
    [
        (
            "rel newton:mimicJoint = </World/Leader>",
            "",
            ovnewton.InvalidPhysicsError,
            "has no newton:mimicJoint target",
        ),
        ("</World/Leader>", "</World/Missing>", ovnewton.InvalidPhysicsError, "unknown joint"),
        ("</World/Leader>", "</World/Body1>", ovnewton.InvalidPhysicsError, "unknown joint"),
        ("</World/Leader>", "</World/Follower>", ovnewton.InvalidPhysicsError, "cannot target itself"),
        (
            "</World/Leader>",
            "[</World/Leader>, </World/Follower>]",
            ovnewton.InvalidPhysicsError,
            "multiple newton:mimicJoint targets",
        ),
        (
            'PhysicsPrismaticJoint "Follower"',
            'PhysicsSphericalJoint "Follower"',
            ovnewton.UnsupportedPhysicsError,
            "requires a revolute or prismatic joint",
        ),
        (
            'PhysicsPrismaticJoint "Leader"',
            'PhysicsSphericalJoint "Leader"',
            ovnewton.UnsupportedPhysicsError,
            "requires scalar joint target",
        ),
        ("mimicCoef0 = 0", "mimicCoef0 = inf", ovnewton.InvalidPhysicsError, "invalid newton:mimicCoef0"),
        ("mimicCoef1 = 2", "mimicCoef1 = nan", ovnewton.InvalidPhysicsError, "invalid newton:mimicCoef1"),
        ("float newton:mimicCoef0 = 0", "float mjc:coef2 = inf", ovnewton.InvalidPhysicsError, "invalid mjc:coef2"),
        ("[0.04, 0.8]", "[0.04]", ovnewton.InvalidPhysicsError, "invalid mjc:solref"),
        ("[0.8, 0.9, 0.002, 0.4, 3]", "[0.8]", ovnewton.InvalidPhysicsError, "invalid mjc:solimp"),
        (
            '"MjcEqualityJointAPI"',
            '"MjcEqualityJointAPI", "MjcEqualityWeldAPI"',
            ovnewton.InvalidPhysicsError,
            "conflicting body and joint equality schemas",
        ),
    ],
)
def test_invalid_joint_equality_is_rejected(old, new, error, message):
    with wp.ScopedDevice("cpu"), ovstage.Stage("invalid-joint-equality") as stage:
        _populate(stage, JOINT.replace(old, new))
        with pytest.raises(error, match=message):
            ovnewton.attach_ovstage(stage)


@pytest.mark.parametrize("target,body", [("Follower", "Body2"), ("Leader", "Body1")])
def test_joint_equality_rejects_merged_joint_axes(target, body):
    usda = (
        JOINT.rsplit("}", 1)[0]
        + f'''
    def PhysicsRevoluteJoint "{target}Angular" {{
        rel physics:body1 = </World/{body}>
        uniform token physics:axis = "X"
    }}
}}
'''
    )
    with wp.ScopedDevice("cpu"), ovstage.Stage("merged-joint-equality") as stage:
        _populate(stage, usda)
        with pytest.raises(ovnewton.UnsupportedPhysicsError, match=rf"{target} merged into a D6 joint"):
            ovnewton.attach_ovstage(stage)


def test_connect_preserves_equality_and_loop_joint():
    with wp.ScopedDevice("cpu"), ovstage.Stage("connect-equality") as stage:
        _populate(stage, CONNECT)
        builder = newton.ModelBuilder()
        result = ovnewton.add_ovstage(builder, stage)
        model = builder.finalize(skip_validation_joints=result.has_orphan_joints)
        assert result.has_orphan_joints
        assert model.body_count == 1
        assert model.mujoco.equality_constraint_count == 1
        eq = model.mujoco
        np.testing.assert_allclose(eq.equality_constraint_anchor.numpy(), [[1, 2, 3]])
        np.testing.assert_allclose(eq.eq_solref.numpy(), [[0.04, 0.8]])
        np.testing.assert_allclose(eq.eq_solimp.numpy(), [[0.8, 0.9, 0.002, 0.4, 3]])
        target = int(eq.equality_constraint_target.numpy()[0])
        assert model.joint_label[target] == "/World/Equality"
        assert model.joint_type.numpy()[target] == newton.JointType.BALL
        assert model.joint_count == 2  # The free body plus its equality loop, not a tree joint.
        binding = ovnewton.attach_ovstage(stage, model=model)
        assert binding.model is model


@pytest.mark.parametrize("usda", [CONNECT, WELD], ids=["connect", "weld"])
def test_equality_population_composes_with_other_descriptions(usda):
    with wp.ScopedDevice("cpu"), ovstage.Stage("combined-population") as stage:
        population.open_usd_from_string_with_desc(
            stage,
            usda,
            ordinal=1,
            time=population.DEFAULT_TIME,
            descs=[population.Desc(domains=PopulationDomain.ALL), ovnewton.population_desc()],
        )
        stage.advance_write_floor(ordinal=1).wait()
        model = ovnewton.attach_ovstage(stage).model
        assert model.mujoco.equality_constraint_count == 1
        np.testing.assert_allclose(model.mujoco.eq_solref.numpy(), [[0.04, 0.8]])


@pytest.mark.parametrize(
    "usda,schema,path",
    [
        (CONNECT, "MjcEqualityConnectAPI", "/World/Equality"),
        (WELD, "MjcEqualityWeldAPI", "/World/Equality"),
        (JOINT, "MjcEqualityJointAPI", "/World/Follower"),
    ],
)
def test_equality_requires_preserved_metadata(usda, schema, path):
    with ovstage.Stage("equality-without-metadata") as stage, ovstage.PathDictionary(stage) as pd:
        population.open_usd_from_string(stage, usda, ordinal=1, domains=PopulationDomain.PHYSICS)
        stage.advance_write_floor(ordinal=1).wait()
        paths = pd.create_path_list_from_strings([path])
        try:
            with stage.query_from_path_list(paths) as query:
                stage.write_attribute(
                    query,
                    "usd-schemas",
                    ordinal=2,
                    tensors=np.asarray([pd.intern_token(schema)], dtype=np.uint64),
                    is_array=True,
                ).wait()
            stage.advance_write_floor(ordinal=2).wait()
            with pytest.raises(
                ovnewton.OvstageContractError, match=r"requires population with ovnewton\.population_desc\(\)"
            ):
                ovnewton.attach_ovstage(stage, ordinal=2)
        finally:
            pd.destroy_path_list(paths)


def test_joint_equality_requires_readable_prim_type():
    with ovstage.Stage("joint-equality-without-prim-type") as stage, ovstage.PathDictionary(stage) as pd:
        _populate(stage, JOINT)
        paths = pd.create_path_list_from_strings(["/Follower"])
        try:
            with stage.query_from_path_list(paths) as query:
                stage.write_attribute(
                    query,
                    "usd-metadata:apiSchemas",
                    ordinal=2,
                    tensors=np.asarray([pd.intern_token("MjcEqualityJointAPI")], dtype=np.uint64),
                    is_array=True,
                ).wait()
            stage.advance_write_floor(ordinal=2).wait()
            with pytest.raises(ovnewton.OvstageContractError, match="no readable 'usd-prim-type'.*/Follower"):
                ovnewton.attach_ovstage(stage, ordinal=2)
        finally:
            pd.destroy_path_list(paths)


def test_parallel_equalities_remain_separate_from_tree_joints():
    weld_joint = WELD[WELD.index("    def PhysicsFixedJoint") :].rsplit("}", 1)[0]
    usda = CONNECT.rsplit("}", 1)[0] + weld_joint.replace('"Equality"', '"Weld"') + "}\n"
    with wp.ScopedDevice("cpu"), ovstage.Stage("parallel-equalities") as stage:
        _populate(stage, usda)
        model = ovnewton.attach_ovstage(stage).model
        eq = model.mujoco
        assert eq.equality_constraint_count == 2
        assert model.joint_count == 3
        targets = eq.equality_constraint_target.numpy()
        assert len(set(targets)) == 2
        assert {model.joint_label[int(target)] for target in targets} == {"/World/Equality", "/World/Weld"}
        assert set(eq.equality_constraint_type.numpy()) == {0, 1}


@pytest.mark.parametrize("usda", [CONNECT, WELD], ids=["connect", "weld"])
@pytest.mark.parametrize("endpoint", ["world-body", "body-world", "body-body"])
@pytest.mark.parametrize("scaled", [False, True])
def test_equality_matches_newton_add_usd(tmp_path, usda, endpoint, scaled):
    if endpoint == "body-world":
        usda = usda.replace("rel physics:body1", "rel physics:body0")
    elif endpoint == "body-body":
        usda = usda.replace(
            '    def Xform "Body"',
            '''    def Xform "Other" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
        float physics:mass = 1
        float3 physics:diagonalInertia = (1, 1, 1)
        double3 xformOp:translate = (-2, 1, 3)
        quatf xformOp:orient = (0.70710678, 0, 0.70710678, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
    }
    def Xform "Body"''',
        ).replace("        rel physics:body1", "        rel physics:body0 = </World/Other>\n        rel physics:body1")
    if scaled:
        usda = usda.replace(
            '    def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {',
            """    def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
        double3 xformOp:translate = (2, -1, 0.5)
        quatf xformOp:orient = (0.70710678, 0.70710678, 0, 0)
        double3 xformOp:scale = (2, 3, 4)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]""",
        )
    asset = tmp_path / "equality.usda"
    asset.write_text(usda)
    with wp.ScopedDevice("cpu"), ovstage.Stage("equality-parity") as stage:
        reference_builder = newton.ModelBuilder()
        reference_builder.add_usd(str(asset), collapse_fixed_joints=False)
        reference = reference_builder.finalize(skip_validation_joints=True)
        _populate(stage, usda)
        binding = ovnewton.attach_ovstage(stage)
        model = binding.model
        assert model.mujoco.equality_constraint_count == reference.mujoco.equality_constraint_count == 1
        for field in (
            "equality_constraint_type",
            "equality_constraint_anchor",
            "equality_constraint_relpose",
            "equality_constraint_torquescale",
            "equality_constraint_enabled",
            "equality_constraint_objtype",
            "equality_constraint_target_kind",
            "eq_solref",
            "eq_solimp",
        ):
            np.testing.assert_allclose(
                getattr(model.mujoco, field).numpy(),
                getattr(reference.mujoco, field).numpy(),
                rtol=1e-6,
                atol=1e-6,
                err_msg=field,
            )
        actual_joint = model.joint_label.index("/World/Equality")
        expected_joint = reference.joint_label.index("/World/Equality")
        for field in ("joint_type", "joint_X_p", "joint_X_c"):
            np.testing.assert_allclose(
                getattr(model, field).numpy()[actual_joint],
                getattr(reference, field).numpy()[expected_joint],
                rtol=1e-6,
                atol=1e-6,
                err_msg=field,
            )
        for field in ("equality_constraint_body1", "equality_constraint_body2"):

            def body_label(m):
                index = int(getattr(m.mujoco, field).numpy()[0])
                return m.body_label[index] if index >= 0 else "world"

            assert body_label(model) == body_label(reference)


@pytest.mark.parametrize("usda", [CONNECT, WELD], ids=["connect", "weld"])
def test_disabled_equality_is_omitted(usda):
    usda = usda.replace("rel physics:body1 = </World/Body>", "bool physics:jointEnabled = false")
    with wp.ScopedDevice("cpu"), ovstage.Stage("disabled-equality") as stage:
        _populate(stage, usda)
        model = ovnewton.attach_ovstage(stage).model
        assert model.mujoco.equality_constraint_count == 0
        assert "/World/Equality" not in model.joint_label


@pytest.mark.parametrize("usda", [CONNECT, WELD], ids=["connect", "weld"])
def test_equality_defaults(usda):
    usda = "\n".join(line for line in usda.splitlines() if "mjc:" not in line)
    with wp.ScopedDevice("cpu"), ovstage.Stage("equality-defaults") as stage:
        _populate(stage, usda)
        eq = ovnewton.attach_ovstage(stage).model.mujoco
        np.testing.assert_allclose(eq.eq_solref.numpy(), [[0.02, 1]])
        np.testing.assert_allclose(eq.eq_solimp.numpy(), [[0.9, 0.95, 0.001, 0.5, 2]])
        np.testing.assert_allclose(eq.equality_constraint_torquescale.numpy(), [1 if "Weld" in usda else 0])


@pytest.mark.parametrize("usda", [CONNECT, WELD], ids=["connect", "weld"])
def test_mujoco_enforces_equality_without_duplicate_constraints(usda):
    pytest.importorskip("mujoco")
    # Put both frames at the body origin and let gravity act for half a second.
    usda = "\n".join(line for line in usda.splitlines() if "physics:local" not in line and "mjc:" not in line)
    with wp.ScopedDevice("cpu"), ovstage.Stage("equality-solver") as stage:
        _populate(stage, usda)
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        result = ovnewton.add_ovstage(builder, stage)
        model = builder.finalize(skip_validation_joints=result.has_orphan_joints)
        binding = ovnewton.attach_ovstage(stage, model=model)
        solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True)
        assert solver.mj_model.neq == 1
        state, next_state = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        for _ in range(120):
            solver.step(state, next_state, None, None, 1.0 / 240.0)
            state, next_state = next_state, state
        np.testing.assert_allclose(state.body_q.numpy()[0, :3], [0, 0, 0], atol=0.01)
        binding.update_to_ovstage(state, ordinal=2)
        stage.advance_write_floor(ordinal=2).wait()
        roundtrip = model.state()
        binding.update_from_ovstage(roundtrip, model.control(), ordinal=2)
        np.testing.assert_allclose(roundtrip.body_q.numpy(), state.body_q.numpy(), atol=1e-6)
        query = binding.query(paths=["/World/Body"])
        result = binding.read(state, query=query, attributes=["body_q"])
        assert result.groups
        np.testing.assert_allclose(result.groups[0].array(0).reshape(-1, 7), state.body_q.numpy())


@pytest.mark.parametrize(
    "old,new,error,message",
    [
        ("[0.04, 0.8]", "[0.04]", ovnewton.InvalidPhysicsError, "invalid mjc:solref"),
        ("[0.8, 0.9, 0.002, 0.4, 3]", "[0.8, 0.9]", ovnewton.InvalidPhysicsError, "invalid mjc:solimp"),
        ("</World/Body>", "[</World/Body>, </World/Other>]", ovnewton.InvalidPhysicsError, "multiple physics:body1"),
        ("</World/Body>", "</World/Missing>", ovnewton.UnsupportedPhysicsError, "rigid body or world"),
        ("rel physics:body1 = </World/Body>", "", ovnewton.InvalidPhysicsError, "distinct bodies"),
        (
            "MjcEqualityConnectAPI",
            "MjcEqualityJointAPI",
            ovnewton.UnsupportedPhysicsError,
            "requires a revolute or prismatic joint",
        ),
        ("MjcEqualityConnectAPI", "MjcJointAPI", ovnewton.UnsupportedPhysicsError, "MjcJointAPI"),
        ("PhysicsSphericalJoint", "Xform", ovnewton.UnsupportedPhysicsError, "requires PhysicsSphericalJoint"),
    ],
)
def test_invalid_or_unsupported_equalities_are_rejected(old, new, error, message):
    with wp.ScopedDevice("cpu"), ovstage.Stage("invalid-equality") as stage:
        _populate(stage, CONNECT.replace(old, new))
        with pytest.raises(error, match=message):
            ovnewton.attach_ovstage(stage)


@pytest.mark.parametrize("endpoint", [0, 1])
@pytest.mark.parametrize(
    "usda,old,new,attribute",
    [
        (CONNECT, "(1, 2, 3)", "(nan, 2, 3)", "physics:localPos0"),
        (CONNECT, "(1, 2, 3)", "(inf, 2, 3)", "physics:localPos0"),
        (WELD, "(0.70710678, 0, 0, 0.70710678)", "(0, 0, 0, 0)", "physics:localRot0"),
        (WELD, "(0.70710678, 0, 0, 0.70710678)", "(nan, 0, 0, 0)", "physics:localRot0"),
    ],
    ids=["nan-position", "infinite-position", "zero-quaternion", "nan-quaternion"],
)
def test_invalid_equality_frames_are_rejected(usda, old, new, attribute, endpoint):
    renamed = attribute.replace("0", str(endpoint))
    usda = usda.replace(attribute, renamed)
    with wp.ScopedDevice("cpu"), ovstage.Stage("invalid-equality-frame") as stage:
        _populate(stage, usda.replace(old, new))
        with pytest.raises(ovnewton.InvalidPhysicsError, match=rf"{renamed}.*World/Equality"):
            ovnewton.attach_ovstage(stage)
