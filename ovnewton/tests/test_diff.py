# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-level differential of the reader against ModelBuilder.add_usd.

Builds scenes from a populated ovstage (bodies + colliders + mass + joints)
and compares against add_usd via a full label-aligned array comparison:
* per-body mass / full inertia tensor / CoM / pose + orientation, keyed by label;
* per-shape geometry / transform / flags / material / collision groups and
  label-normalized filter pairs;
* per-joint type + anchor frames (X_p / X_c), aligned by parent/child body edge;
  per-DOF state, limits, and actuation fields;
* articulation count and per-joint membership.

"""

import logging
import os
from types import SimpleNamespace

import newton
import numpy as np
import ovstage
import pytest
import warp as wp
from ovstage import PopulationDomain, population

import ovnewton
from ovnewton._src import _build, _stage
from ovnewton._src._errors import UnsupportedPhysicsError
from ovnewton.examples import get_asset, get_asset_directory

from . import diff_harness as dh

CARTPOLE = get_asset("scene_cartpole.usda")
NEWTON_ASSETS = os.path.abspath(
    os.path.join(os.path.dirname(newton.__file__), "tests", "assets")
)
REPO_ASSETS = get_asset_directory()
_REQUIRES_CUDA = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA is unavailable")


@pytest.fixture
def import_log(caplog):
    caplog.set_level(logging.INFO, logger="ovnewton")
    return caplog


def _diagnostics(import_log):
    return [
        record
        for record in import_log.records
        if record.name == "ovnewton" and hasattr(record, "diagnostic_code")
    ]


def _diagnostic_pairs(import_log):
    return [(record.diagnostic_code, record.prim_path) for record in _diagnostics(import_log)]


# Articulated and free-body scenes ovnewton fully reconstructs (model + topology).
# cartpole: single chain; ant / ant_mixed: one floating-base articulation.
# Free and nested bodies are covered by identity materialized from query-result
# prim lists. Capsule/cylinder `axis` is read directly in the stage-wide collider
# query.
_SCHEMA_FILTER_GAP_FIELDS = {
    "ant_mixed": {"joint_q", "joint_qd"},
    "humanoid": {"joint_q", "joint_qd"},
}

# Newton's default add_usd() resolver does not consume PhysicsJointStateAPI positions. The
# shipped cartpole authors its first angle so MuJoCo-Warp starts from the same
# leaning pose as the body transforms instead of snapping upright.
_INTENTIONAL_ADD_USD_DIFFERENCES = {"cartpole": {"joint_q"}}

MODEL_SCENES = [
    ("cartpole", CARTPOLE),
    ("ant", os.path.join(NEWTON_ASSETS, "ant.usda")),
    ("ant_mixed", os.path.join(NEWTON_ASSETS, "ant_mixed.usda")),
    ("cube_cylinder", os.path.join(NEWTON_ASSETS, "cube_cylinder.usda")),
    ("humanoid", os.path.join(NEWTON_ASSETS, "humanoid.usda")),
    ("rigid_bodies", os.path.join(REPO_ASSETS, "scene_rigid_bodies.usda")),
]

UNSUPPORTED_SCENES = [
    ("humanoid_mjc", os.path.join(NEWTON_ASSETS, "humanoid_mjc.usda"), "MjcJointAPI"),
]

NEGATIVE_SCALE_CUBE = """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Body" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
) {
    float physics:mass = 1
    float3 physics:diagonalInertia = (1, 1, 1)
    point3f physics:centerOfMass = (0, 0, 0)
    quatf physics:principalAxes = (1, 0, 0, 0)
    def Cube "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        double size = 1
        double3 xformOp:scale = (-1, 0.5, 1.5)
        uniform token[] xformOpOrder = ["xformOp:scale"]
    }
}
"""

MIRRORED_MESH = """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Body" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
) {
    float physics:mass = 1
    float3 physics:diagonalInertia = (1, 1, 1)
    point3f physics:centerOfMass = (0, 0, 0)
    quatf physics:principalAxes = (1, 0, 0, 0)
    def Mesh "Collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    ) {
        uniform token physics:approximation = "none"
        uniform token subdivisionScheme = "none"
        double3 xformOp:scale = __SCALE__
        uniform token[] xformOpOrder = ["xformOp:scale"]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        int[] faceVertexCounts = [3, 3, 3, 3]
        int[] faceVertexIndices = [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]
    }
}
"""


def _ovstage_model(asset, stage_name="ovstage-diff"):
    with ovstage.Stage(stage_name) as stage, ovstage.PathDictionary(stage) as pd:
        population.open_usd(stage, asset, ordinal=1, domains=PopulationDomain.ALL)
        stage.advance_write_floor(ordinal=1).wait()
        return _build.build_model(stage, pd, ordinal=1)


def _ovstage_model_from_string(usda, stage_name="ovstage-strict"):
    with ovstage.Stage(stage_name) as stage, ovstage.PathDictionary(stage) as pd:
        population.open_usd_from_string(stage, usda, ordinal=1, domains=PopulationDomain.ALL)
        stage.advance_write_floor(ordinal=1).wait()
        return _build.build_model(stage, pd, ordinal=1)


@pytest.mark.parametrize(
    "name,asset",
    MODEL_SCENES,
    ids=["cartpole", "ant", "ant_mixed", "cube_cylinder", "humanoid", "rigid_bodies"],
)
def test_model_matches_add_usd(name, asset):
    if not os.path.exists(asset):
        pytest.skip(f"asset not found: {asset}")
    ref = dh.reference_model(asset)
    model = _ovstage_model(asset)
    ours = dh._model_arrays(model)

    mismatches = dh.compare_models(ref, ours, check_joints=True, check_articulations=True)
    gap_fields = _SCHEMA_FILTER_GAP_FIELDS.get(name, set())
    excluded_fields = gap_fields | _INTENTIONAL_ADD_USD_DIFFERENCES.get(name, set())
    unexpected = [m for m in mismatches if m.partition(":")[0].rsplit(".", 1)[-1] not in excluded_fields]
    assert not unexpected, f"{name} unexpected mismatches vs add_usd:\n  " + "\n  ".join(unexpected)
    observed_gaps = {
        m.partition(":")[0].rsplit(".", 1)[-1]
        for m in mismatches
        if m.partition(":")[0].rsplit(".", 1)[-1] in gap_fields
    }
    if observed_gaps:
        pytest.xfail(
            "the pinned ovstage cannot mirror the unregistered/unapplied joint schema fields: "
            + ", ".join(sorted(observed_gaps))
        )


def _correct_newton_16_angular_velocity(ref, label, dof=0):
    # Newton 1.6.0 and 1.6.1 kept initial angular velocities in degrees/s.
    # Keep their other fields as the oracle, but require USD-correct units here.
    if newton.__version__ in ("1.6.0", "1.6.1"):
        joint = list(ref["joint_label"]).index(label)
        index = ref["joint_qd_start"][joint] + dof
        assert ref["joint_qd"][index] == pytest.approx(0.25)
        ref["joint_qd"][index] = np.deg2rad(0.25)


def test_authored_single_dof_initial_state_matches_add_usd(tmp_path):
    asset = tmp_path / "joint-state.usda"
    asset.write_text(
        """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "World" (prepend apiSchemas = ["PhysicsArticulationRootAPI"]) {
    def Xform "RevoluteBody" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def PhysicsRevoluteJoint "Revolute" (
        prepend apiSchemas = ["PhysicsJointStateAPI:angular"]
    ) {
        rel physics:body1 = </World/RevoluteBody>
        float state:angular:physics:position = 30
        float state:angular:physics:velocity = 0.25
    }
    def Xform "PrismaticBody" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def PhysicsPrismaticJoint "Prismatic" (
        prepend apiSchemas = ["PhysicsJointStateAPI:linear"]
    ) {
        rel physics:body0 = </World/RevoluteBody>
        rel physics:body1 = </World/PrismaticBody>
        float state:linear:physics:position = 0.4
        float state:linear:physics:velocity = -0.2
    }
}
""",
        encoding="utf-8",
    )
    ref = dh.reference_model(str(asset), include_physx_resolver=True)
    model = _ovstage_model(str(asset), "joint-initial-state")
    starts = model.joint_qd_start.numpy()
    velocities = model.joint_qd.numpy()
    assert velocities[starts[model.joint_label.index("/World/Revolute")]] == pytest.approx(np.deg2rad(0.25))
    assert velocities[starts[model.joint_label.index("/World/Prismatic")]] == pytest.approx(-0.2)
    _correct_newton_16_angular_velocity(ref, "/World/Revolute")
    mismatches = dh.compare_models(ref, dh._model_arrays(model), check_joints=True)
    assert not mismatches, "initial-state mismatches vs add_usd:\n  " + "\n  ".join(mismatches)


def test_authored_merged_d6_initial_state_matches_add_usd(tmp_path):
    asset = tmp_path / "merged-joint-state.usda"
    asset.write_text(
        """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "World" (prepend apiSchemas = ["PhysicsArticulationRootAPI"]) {
    def Xform "Parent" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def Xform "Child" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def PhysicsPrismaticJoint "Linear" (
        prepend apiSchemas = ["PhysicsJointStateAPI:linear"]
    ) {
        rel physics:body0 = </World/Parent>
        rel physics:body1 = </World/Child>
        float state:linear:physics:position = 0.4
        float state:linear:physics:velocity = -0.2
    }
    def PhysicsRevoluteJoint "Angular" (
        prepend apiSchemas = ["PhysicsJointStateAPI:angular"]
    ) {
        rel physics:body0 = </World/Parent>
        rel physics:body1 = </World/Child>
        float state:angular:physics:position = 30
        float state:angular:physics:velocity = 0.25
    }
}
""",
        encoding="utf-8",
    )
    ref = dh.reference_model(str(asset), include_physx_resolver=True)
    model = _ovstage_model(str(asset), "merged-joint-initial-state")
    joint = model.joint_label.index("/World/Angular")
    start = model.joint_qd_start.numpy()[joint]
    assert model.joint_qd.numpy()[start:start + 2] == pytest.approx([-0.2, np.deg2rad(0.25)])
    _correct_newton_16_angular_velocity(ref, "/World/Angular", dof=1)
    mismatches = dh.compare_models(ref, dh._model_arrays(model), check_joints=True)
    assert not mismatches, "merged initial-state mismatches vs add_usd:\n  " + "\n  ".join(mismatches)


def test_physx_joint_properties_are_outside_default_profile(tmp_path):
    asset = tmp_path / "physx-joint-properties.usda"
    asset.write_text(
        """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "World" (prepend apiSchemas = ["PhysicsArticulationRootAPI"]) {
    def Xform "Root" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def PhysicsFixedJoint "RootJoint" {
        rel physics:body1 = </World/Root>
    }
    def PhysicsRevoluteJoint "Joint" (
        prepend apiSchemas = ["PhysxJointAPI", "PhysxLimitAPI:angular"]
    ) {
        rel physics:body0 = </World/Root>
        rel physics:body1 = </World/Body>
        float physics:lowerLimit = -45
        float physics:upperLimit = 45
        float physxJoint:armature = 0.75
        float physxJoint:maxJointVelocity = 12
        float physxLimit:angular:stiffness = 5
        float physxLimit:angular:damping = 6
    }
}
""",
        encoding="utf-8",
    )
    model = _ovstage_model(str(asset), "physx-joint-properties")
    ours = dh._model_arrays(model)

    default_ref = dh.reference_model(str(asset))
    mismatches = dh.compare_models(default_ref, ours, check_joints=True)
    assert not mismatches, "PhysX-only properties affected the default profile:\n  " + "\n  ".join(mismatches)

    physx_ref = dh.reference_model(str(asset), include_physx_resolver=True)
    physx_fields = {
        mismatch.partition(":")[0].rsplit(".", 1)[-1]
        for mismatch in dh.compare_models(physx_ref, ours, check_joints=True)
    }
    assert {
        "joint_armature",
        "joint_velocity_limit",
        "joint_limit_ke",
        "joint_limit_kd",
    } <= physx_fields


@pytest.mark.parametrize(
    ("broadcast_ke", "broadcast_kd", "expected_ke", "expected_kd"),
    (
        (None, None, None, None),
        (200.0, 5.0, 200.0 * 180.0 / np.pi, 5.0 * 180.0 / np.pi),
        (200.0, float("inf"), 200.0 * 180.0 / np.pi, 0.0),
        (float("inf"), float("-inf"), 1.0e8 * 180.0 / np.pi, 0.0),
        (float("-inf"), float("-inf"), None, None),
    ),
)
def test_newton_joint_broadcast_limit_translation(broadcast_ke, broadcast_kd, expected_ke, expected_kd):
    dof = SimpleNamespace(
        broadcast_limit_stiffness=broadcast_ke,
        broadcast_limit_damping=broadcast_kd,
        drive_stiffness=None,
        drive_damping=None,
        drive_target=None,
        drive_velocity=None,
        velocity_limit=None,
        damping=2.0,
        armature=None,
        friction=None,
        drive_force_limit=None,
        drive_enabled=False,
    )

    kwargs = _build._dof_kw(dof, rotational=True, lo=None, hi=None)

    if expected_ke is None:
        assert "limit_ke" not in kwargs
    else:
        assert kwargs["limit_ke"] == pytest.approx(expected_ke)
    if expected_kd is None:
        assert "limit_kd" not in kwargs
    else:
        assert kwargs["limit_kd"] == pytest.approx(expected_kd)
    assert kwargs["damping"] == pytest.approx(2.0 * 180.0 / np.pi)

    linear_kwargs = _build._dof_kw(dof, rotational=False, lo=None, hi=None)
    if expected_ke is None:
        assert "limit_ke" not in linear_kwargs
    else:
        assert linear_kwargs["limit_ke"] == pytest.approx(expected_ke / (180.0 / np.pi))
    if expected_kd is None:
        assert "limit_kd" not in linear_kwargs
    else:
        assert linear_kwargs["limit_kd"] == pytest.approx(expected_kd / (180.0 / np.pi))
    assert linear_kwargs["damping"] == pytest.approx(2.0)


def test_incomplete_newton_joint_api_fails_explicitly(monkeypatch):
    usda = '''#usda 1.0
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
def PhysicsRevoluteJoint "Joint" (prepend apiSchemas = ["NewtonJointAPI"]) {
    rel physics:body1 = </Body>
}
'''

    read_columns = _stage.read_columns

    def omit_damping(*args, **kwargs):
        columns = read_columns(*args, **kwargs)
        if "newton:damping" in columns:
            columns["newton:damping"] = {}
        return columns

    monkeypatch.setattr(_stage, "read_columns", omit_damping)
    with pytest.raises(UnsupportedPhysicsError, match="NewtonJointAPI is not completely populated at /Joint"):
        _ovstage_model_from_string(usda, "incomplete-newton-joint-api")


@pytest.mark.parametrize("joint_type", ("PhysicsFixedJoint", "PhysicsSphericalJoint", "PhysicsDistanceJoint"))
def test_newton_joint_api_is_ignored_for_joints_without_configurable_dofs(joint_type):
    usda = f'''#usda 1.0
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {{}}
def {joint_type} "Joint" (prepend apiSchemas = ["NewtonJointAPI"]) {{
    rel physics:body1 = </Body>
}}
'''

    _ovstage_model_from_string(usda, f"ignored-newton-joint-api-{joint_type}")


def test_drive_max_force_matches_add_usd(tmp_path):
    asset = tmp_path / "drive-max-force.usda"
    asset.write_text(
        """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "World" (prepend apiSchemas = ["PhysicsArticulationRootAPI"]) {
    def Xform "A" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def Xform "B" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def Xform "C" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def Xform "D" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def PhysicsRevoluteJoint "Angular" (prepend apiSchemas = ["PhysicsDriveAPI:angular"]) {
        rel physics:body0 = </World/A>
        rel physics:body1 = </World/B>
        float drive:angular:physics:maxForce = 11
    }
    def PhysicsPrismaticJoint "Linear" (prepend apiSchemas = ["PhysicsDriveAPI:linear"]) {
        rel physics:body0 = </World/B>
        rel physics:body1 = </World/C>
        float drive:linear:physics:maxForce = 22
    }
    def PhysicsJoint "D6" (
        prepend apiSchemas = ["PhysicsLimitAPI:rotX", "PhysicsDriveAPI:rotX"]
    ) {
        rel physics:body0 = </World/C>
        rel physics:body1 = </World/D>
        float limit:rotX:physics:low = -45
        float limit:rotX:physics:high = 45
        float drive:rotX:physics:maxForce = 33
    }
}
""",
        encoding="utf-8",
    )
    ref = dh.reference_model(str(asset))
    model = _ovstage_model(str(asset), "drive-max-force")
    mismatches = dh.compare_models(ref, dh._model_arrays(model), check_joints=True)
    assert not mismatches, "drive maxForce mismatches vs add_usd:\n  " + "\n  ".join(mismatches)

    starts = model.joint_qd_start.numpy()
    effort = model.joint_effort_limit.numpy()
    for label, expected in (("/World/Angular", 11.0), ("/World/Linear", 22.0), ("/World/D6", 33.0)):
        joint = model.joint_label.index(label)
        assert effort[int(starts[joint])] == pytest.approx(expected)


MIMIC_SCENE = """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "World" (prepend apiSchemas = ["PhysicsArticulationRootAPI"]) {
    def Xform "Root" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def Xform "LeaderBody" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def Xform "FollowerBody" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}

    def PhysicsFixedJoint "RootToWorld" {
        rel physics:body1 = </World/Root>
    }
    def PhysicsRevoluteJoint "Leader" {
        rel physics:body0 = </World/Root>
        rel physics:body1 = </World/LeaderBody>
        uniform token physics:axis = "Z"
    }
    def PhysicsRevoluteJoint "Follower" (prepend apiSchemas = ["NewtonMimicAPI"]) {
        rel physics:body0 = </World/LeaderBody>
        rel physics:body1 = </World/FollowerBody>
        uniform token physics:axis = "Z"
        rel newton:mimicJoint = </World/Leader>
        float newton:mimicCoef0 = 0.5
        float newton:mimicCoef1 = -2
    }
}
"""


def _mimic_signatures(model):
    return dh.mimic_signatures(dh._model_arrays(model))


@pytest.mark.parametrize("solver_name", ["mujoco", "featherstone"])
@pytest.mark.parametrize("rotational", [True, False])
def test_imported_mimic_is_enforced_by_solver(solver_name, rotational):
    if solver_name == "featherstone" and not hasattr(newton.ModelBuilder, "set_joint_mimic"):
        pytest.skip("Newton 1.6 Featherstone does not support mimic constraints")
    usda = MIMIC_SCENE.replace(
        '(prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}',
        '(prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {'
        ' float physics:mass = 1\n float3 physics:diagonalInertia = (1, 1, 1)\n }',
    )
    if not rotational:
        usda = usda.replace("PhysicsRevoluteJoint", "PhysicsPrismaticJoint")
    with ovstage.Stage("solver-mimic") as stage:
        population.open_usd_from_string(stage, usda, ordinal=1, domains=PopulationDomain.ALL)
        stage.advance_write_floor(ordinal=1).wait()
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        if solver_name == "mujoco":
            pytest.importorskip("mujoco")
            newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        ovnewton.add_ovstage(builder, stage)
        model = builder.finalize()
    leader = model.joint_label.index("/World/Leader")
    follower = model.joint_label.index("/World/Follower")
    starts = model.joint_q_start.numpy()
    q = model.joint_q.numpy()
    q[starts[leader]], q[starts[follower]] = 0.35, 0.8
    model.joint_q.assign(q)
    state, next_state = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    solver = (
        newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True)
        if solver_name == "mujoco" else newton.solvers.SolverFeatherstone(model)
    )
    for _ in range(120):
        solver.step(state, next_state, None, None, 1.0 / 240.0)
        state, next_state = next_state, state
    result = state.joint_q.numpy()
    offset = np.deg2rad(0.5) if rotational else 0.5
    assert result[starts[follower]] == pytest.approx(offset - 2 * result[starts[leader]], abs=2e-3)


@pytest.mark.skipif(
    not hasattr(newton.ModelBuilder, "set_joint_mimic"), reason="Newton 1.6 lacks joint mimic validation"
)
def test_newton_joint_mimic_rejects_self_reference():
    usda = MIMIC_SCENE.replace("rel newton:mimicJoint = </World/Leader>", "rel newton:mimicJoint = </World/Follower>")
    with pytest.raises(ovnewton.InvalidPhysicsError, match="/World/Follower:.*cannot mimic itself"):
        _ovstage_model_from_string(usda, "self-mimic")


def test_newton_mimic_matches_add_usd(tmp_path):
    asset = tmp_path / "mimic.usda"
    asset.write_text(MIMIC_SCENE, encoding="utf-8")
    ref = dh.reference_model(str(asset))
    model = _ovstage_model(str(asset), "newton-mimic")

    assert _mimic_signatures(model) == dh.mimic_signatures(ref)


def _merged_mimic_scene(merged_joints):
    extra_joints = ""
    for joint in merged_joints:
        parent = "Root" if joint == "Leader" else "LeaderBody"
        extra_joints += f'''
    def PhysicsPrismaticJoint "{joint}Linear" {{
        rel physics:body0 = </World/{parent}>
        rel physics:body1 = </World/{joint}Body>
    }}
'''
    return MIMIC_SCENE.rsplit("}", 1)[0] + extra_joints + "}\n"


@pytest.mark.parametrize("merged_joints", [("Follower",), ("Leader",), ("Follower", "Leader")])
def test_mimic_rejects_merged_joint_axes(merged_joints):
    usda = _merged_mimic_scene(merged_joints)
    role = merged_joints[0].lower()
    with ovstage.Stage("merged-mimic") as stage:
        population.open_usd_from_string(stage, usda, ordinal=1, domains=PopulationDomain.ALL)
        stage.advance_write_floor(ordinal=1).wait()
        builder = newton.ModelBuilder()
        with pytest.raises(
            UnsupportedPhysicsError,
            match=(
                rf"NewtonMimicAPI at /World/Follower: {role} joint /World/{merged_joints[0]} .*merged"
                r".*Newton's mimic APIs cannot target individual source axes"
            ),
        ):
            ovnewton.add_ovstage(builder, stage)


def test_disabled_mimic_allows_merged_joint_axes():
    usda = _merged_mimic_scene(("Follower", "Leader")).replace(
        "        rel newton:mimicJoint = </World/Leader>\n",
        "        bool newton:mimicEnabled = false\n",
    )
    with ovstage.Stage("disabled-merged-mimic") as stage:
        population.open_usd_from_string(stage, usda, ordinal=1, domains=PopulationDomain.ALL)
        stage.advance_write_floor(ordinal=1).wait()
        builder = newton.ModelBuilder()
        ovnewton.add_ovstage(builder, stage)
        model = builder.finalize()
    for joint in ("Follower", "Leader"):
        assert model.joint_type.numpy()[model.joint_label.index(f"/World/{joint}")] == newton.JointType.D6
    assert _mimic_signatures(model) == []


def test_newton_prismatic_mimic_matches_add_usd(tmp_path):
    usda = MIMIC_SCENE.replace('def PhysicsRevoluteJoint "Follower"', 'def PhysicsPrismaticJoint "Follower"')
    asset = tmp_path / "prismatic-mimic.usda"
    asset.write_text(usda, encoding="utf-8")
    ref = dh.reference_model(str(asset))
    model = _ovstage_model(str(asset), "newton-prismatic-mimic")

    assert _mimic_signatures(model) == dh.mimic_signatures(ref)


def test_newton_mimic_schema_defaults_match_add_usd(tmp_path):
    usda = MIMIC_SCENE.replace("        float newton:mimicCoef0 = 0.5\n", "").replace(
        "        float newton:mimicCoef1 = -2\n", ""
    )
    asset = tmp_path / "mimic-defaults.usda"
    asset.write_text(usda, encoding="utf-8")
    ref = dh.reference_model(str(asset))
    model = _ovstage_model(str(asset), "newton-mimic-defaults")

    assert _mimic_signatures(model) == dh.mimic_signatures(ref)


def test_multiple_newton_mimics_match_add_usd_by_identity(tmp_path):
    usda = MIMIC_SCENE.replace(
        '    def Xform "FollowerBody" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}\n',
        '    def Xform "FollowerBody" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}\n'
        '    def Xform "FollowerBody2" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}\n',
    ).replace(
        "    }\n}\n",
        "    }\n"
        '    def PhysicsRevoluteJoint "Follower2" (prepend apiSchemas = ["NewtonMimicAPI"]) {\n'
        "        rel physics:body0 = </World/Root>\n"
        "        rel physics:body1 = </World/FollowerBody2>\n"
        "        uniform token physics:axis = \"Z\"\n"
        "        rel newton:mimicJoint = </World/Leader>\n"
        "        float newton:mimicCoef0 = -0.25\n"
        "        float newton:mimicCoef1 = 3\n"
        "    }\n"
        "}\n",
        1,
    )
    asset = tmp_path / "multiple-mimics.usda"
    asset.write_text(usda, encoding="utf-8")
    ref = dh.reference_model(str(asset))
    model = _ovstage_model(str(asset), "multiple-newton-mimics")

    assert set(_mimic_signatures(model)) == set(dh.mimic_signatures(ref))


def test_newton_mimic_rejects_missing_target():
    usda = MIMIC_SCENE.replace("        rel newton:mimicJoint = </World/Leader>\n", "")
    with pytest.raises(ovnewton.InvalidPhysicsError, match="has no newton:mimicJoint target"):
        _ovstage_model_from_string(usda, "newton-mimic-missing-target")


def test_disabled_newton_mimic_needs_no_target():
    usda = MIMIC_SCENE.replace(
        "        rel newton:mimicJoint = </World/Leader>\n",
        "        bool newton:mimicEnabled = false\n",
    )
    model = _ovstage_model_from_string(usda, "newton-mimic-disabled")
    assert _mimic_signatures(model) == []


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("[</World/Leader>, </World/RootToWorld>]", "multiple newton:mimicJoint targets"),
        ("</World/Unknown>", "references unknown joint /World/Unknown"),
    ],
)
def test_newton_mimic_rejects_invalid_target(target, message):
    usda = MIMIC_SCENE.replace("</World/Leader>\n", f"{target}\n")
    with pytest.raises(ovnewton.InvalidPhysicsError, match=message):
        _ovstage_model_from_string(usda, "newton-mimic-invalid-target")


def test_newton_mimic_rejects_non_finite_coefficient():
    usda = MIMIC_SCENE.replace("float newton:mimicCoef0 = 0.5", "float newton:mimicCoef0 = inf")
    with pytest.raises(ovnewton.InvalidPhysicsError, match="has non-finite coefficients"):
        _ovstage_model_from_string(usda, "newton-mimic-non-finite")


def test_newton_mimic_rejects_non_joint_prim():
    usda = MIMIC_SCENE.replace(
        '    def Xform "Root" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}\n',
        '    def Xform "Root" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "NewtonMimicAPI"]) {\n'
        "        rel newton:mimicJoint = </World/Leader>\n"
        "    }\n",
    )
    with pytest.raises(ovnewton.InvalidPhysicsError, match="is not applied to an imported joint"):
        _ovstage_model_from_string(usda, "newton-mimic-non-joint")


def test_nested_authoring_matches_add_usd(tmp_path):
    asset = tmp_path / "nested.usda"
    asset.write_text(
        """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "World" (prepend apiSchemas = ["PhysicsArticulationRootAPI"]) {
    def Xform "Robot" {
        def Xform "Base" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
            def Xform "Geometry" {
                def Cube "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
                    double size = 0.5
                }
            }
        }
        def Xform "Link" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
            double3 xformOp:translate = (0, 0, 1)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
        def Scope "Joints" {
            def PhysicsRevoluteJoint "Hinge" {
                rel physics:body0 = </World/Robot/Base>
                rel physics:body1 = </World/Robot/Link>
                token physics:axis = "Y"
            }
        }
    }
}
""",
        encoding="utf-8",
    )
    ref = dh.reference_model(str(asset))
    model = _ovstage_model(str(asset), "nested-authoring")
    mismatches = dh.compare_models(ref, dh._model_arrays(model), check_joints=True, check_articulations=True)
    assert not mismatches, "nested-authoring mismatches vs add_usd:\n  " + "\n  ".join(mismatches)
    shape = model.shape_label.index("/World/Robot/Base/Geometry/Collider")
    assert model.body_label[int(model.shape_body.numpy()[shape])] == "/World/Robot/Base"


def test_rootless_joints_remain_orphans_against_newton_14(tmp_path):
    asset = tmp_path / "rootless-joints.usda"
    asset.write_text(
        """#usda 1.0
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
""",
        encoding="utf-8",
    )
    ref = dh.reference_model(str(asset))
    model = _ovstage_model(str(asset), "rootless-joints")
    ours = dh._model_arrays(model)
    mismatches = dh.compare_models(ref, ours, check_joints=True, check_articulations=True)
    assert not mismatches, "rootless-joint mismatches vs add_usd:\n  " + "\n  ".join(mismatches)
    assert model.articulation_count == 0
    assert set(model.joint_articulation.numpy()) == {-1}


@pytest.mark.parametrize("body_relation", ["body0", "body1"])
def test_world_fixed_joint_preserves_composed_body_pose(tmp_path, body_relation):
    asset = tmp_path / "world-fixed-joint.usda"
    asset.write_text(
        """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "World" {
    double3 xformOp:translate = (1, 2, 3)
    quatd xformOp:orient = (0.9238795325, 0, 0, 0.3826834324)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
    def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
        double3 xformOp:translate = (0, 0, 0.5)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def PhysicsFixedJoint "Root" {
        rel physics:__BODY_RELATION__ = </World/Body>
    }
}
""".replace("__BODY_RELATION__", body_relation),
        encoding="utf-8",
    )
    ref = dh.reference_model(str(asset))
    model = _ovstage_model(str(asset), "world-fixed-joint")
    mismatches = dh.compare_models(ref, dh._model_arrays(model), check_joints=True)
    assert not mismatches, "world-fixed-joint mismatches vs add_usd:\n  " + "\n  ".join(mismatches)


def test_authored_and_orphan_components_match_newton_14(tmp_path):
    asset = tmp_path / "mixed-articulations.usda"
    asset.write_text(
        """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "World" {
    def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"]) {
        def Xform "Base" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
        def Xform "Link" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
        def PhysicsRevoluteJoint "Joint" {
            rel physics:body0 = </World/Robot/Base>
            rel physics:body1 = </World/Robot/Link>
            token physics:axis = "Z"
        }
    }
    def Xform "A" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def Xform "B" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def PhysicsFixedJoint "Orphan" {
        rel physics:body0 = </World/A>
        rel physics:body1 = </World/B>
    }
    def Cube "Loose" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {}
}
""",
        encoding="utf-8",
    )
    ref = dh.reference_model(str(asset))
    model = _ovstage_model(str(asset), "mixed-articulations")
    ours = dh._model_arrays(model)
    mismatches = dh.compare_models(ref, ours, check_joints=True, check_articulations=True)
    assert not mismatches, "mixed-articulation mismatches vs add_usd:\n  " + "\n  ".join(mismatches)
    orphan = model.joint_label.index("/World/Orphan")
    assert model.joint_articulation.numpy()[orphan] == -1


@pytest.mark.parametrize("name,asset,reason", UNSUPPORTED_SCENES, ids=[s[0] for s in UNSUPPORTED_SCENES])
def test_scene_with_unsupported_physics_fails(name, asset, reason):
    """Real regression assets must fail at the first semantic feature ovnewton
    cannot preserve, rather than producing a partial or approximate model."""
    if not os.path.exists(asset):
        pytest.skip(f"asset not found: {asset}")
    with pytest.raises(ovnewton.UnsupportedPhysicsError, match=reason):
        _ovstage_model(asset, f"unsupported-{name}")


def test_cartpole_behavioral_matches_add_usd():
    """Step the ovstage-built cartpole under gravity and compare body poses to an
    add_usd-built model stepped identically — the end-to-end check that the ovstage
    model (frames + axis + mass) *simulates* the same, not just looks the same."""
    n = 10
    ref = dh.reference_poses(CARTPOLE, n)
    model = _ovstage_model(CARTPOLE, "ovstage-beh")
    body_paths = model.body_label
    q = dh._step_model(model, n)
    ours = {body_paths[i]: (q[i, :3], q[i, 3:7]) for i in range(len(body_paths))}
    pos_err, common = dh.max_pos_error(ref, ours)
    ori_err, _ = dh.max_ori_error(ref, ours)
    assert len(common) == len(body_paths), f"shared bodies {len(common)} != {len(body_paths)}"
    assert pos_err < 0.05, f"max position error {pos_err:.4f} m over {n} frames"
    assert ori_err < 0.05, f"max orientation error {ori_err:.4f} rad over {n} frames"


def test_cartpole_behavioral_perturbed_matches_add_usd():
    import warp as wp

    n = 10
    ref = dh.reference_poses(CARTPOLE, n, perturb_joint_q=0.3)
    model = _ovstage_model(CARTPOLE, "ovstage-beh-p")
    body_paths = model.body_label
    s = model.state()
    jq = s.joint_q.numpy()
    jq[:] += 0.3
    s.joint_q.assign(wp.array(jq, dtype=wp.float32, device=model.device))
    q = dh._step_model(model, n, state=s)
    ours = {body_paths[i]: (q[i, :3], q[i, 3:7]) for i in range(len(body_paths))}
    assert dh.max_pos_error(ref, ours)[0] < 0.05 and dh.max_ori_error(ref, ours)[0] < 0.05


def test_parse_model_is_labeled():
    """build_model labels bodies with their prim paths, so model.body_label is the
    body->path mapping (parity with add_usd-built models)."""
    model = _ovstage_model(CARTPOLE, "labels")
    assert len(model.body_label) == model.body_count
    assert all(lbl.startswith("/") for lbl in model.body_label)


def test_rigid_bodies_includes_static_ground_plane():
    """The demo's orphan Plane collider must survive as static world geometry.

    The model differential intentionally compares bodies and joints only, so it
    cannot detect a missing ground shape even though the rendered demo depends
    on it.
    """
    asset = os.path.join(REPO_ASSETS, "scene_rigid_bodies.usda")
    model = _ovstage_model(asset, "static-ground")

    static = [i for i, body in enumerate(model.shape_body.numpy()) if body == -1]
    assert len(static) == 1
    assert model.shape_label[static[0]] == "/World/GroundPlane"
    assert model.shape_count == model.body_count + 1


def test_attach_empty_physics_stage(tmp_path):
    """A valid stage with no rigid bodies produces an empty binding, not a
    ``numpy.stack`` failure. This is also the path the example handles."""
    asset = tmp_path / "empty.usda"
    asset.write_text('#usda 1.0\n\ndef Xform "World" {}\n', encoding="utf-8")
    with ovstage.Stage("empty") as stage:
        population.open_usd(stage, str(asset), ordinal=1, domains=PopulationDomain.ALL)
        stage.advance_write_floor(ordinal=1).wait()
        binding = ovnewton.attach_ovstage(stage)

    assert binding.model.body_count == 0
    assert binding.model.body_label == []


def test_attach_uses_requested_population_ordinal():
    usda = """#usda 1.0
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
"""
    with ovstage.Stage("nondefault-attach-ordinal") as stage:
        population.open_usd_from_string(stage, usda, ordinal=4, domains=PopulationDomain.ALL)
        stage.advance_write_floor(ordinal=4).wait()
        binding = ovnewton.attach_ovstage(stage, ordinal=4)

    assert binding.model.body_count == 1
    assert binding.model.body_label == ["/Body"]


def test_raw_triangle_mesh_is_supported():
    model = _ovstage_model_from_string(
        """#usda 1.0
def Xform "World" {
    def Mesh "Collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    ) {
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        token physics:approximation = "none"
    }
}
""",
        "raw-mesh",
    )

    assert model.shape_count == 1
    assert model.shape_label[0] == "/World/Collider"
    assert int(model.shape_type.numpy()[0]) == int(newton.GeoType.MESH)


@pytest.mark.parametrize(
    "max_hull_attribute",
    ("int newton:maxHullVertices = 5", ""),
    ids=("authored", "schema-default"),
)
def test_newton_mesh_max_hull_vertices_matches_add_usd(tmp_path, max_hull_attribute):
    asset = tmp_path / "mesh-max-hull-vertices.usda"
    asset.write_text(
        f'''#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Mesh "Collider" (
    prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "NewtonMeshCollisionAPI"]
) {{
    uniform token physics:approximation = "convexHull"
    {max_hull_attribute}
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    int[] faceVertexCounts = [3, 3, 3, 3]
    int[] faceVertexIndices = [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]
}}
''',
        encoding="utf-8",
    )
    ref = dh.reference_model(str(asset))
    model = _ovstage_model(str(asset), "mesh-max-hull-vertices")
    mismatches = dh.compare_models(ref, dh._model_arrays(model), check_joints=False, check_shapes=True)
    assert not mismatches, "mesh max hull vertices mismatch vs add_usd:\n  " + "\n  ".join(mismatches)


def test_unlimited_mesh_hull_uses_finite_default():
    def build(max_hull_attribute, stage_name):
        return _ovstage_model_from_string(
            f'''#usda 1.0
def Mesh "Collider" (
    prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "NewtonMeshCollisionAPI"]
) {{
    uniform token physics:approximation = "convexHull"
    {max_hull_attribute}
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    int[] faceVertexCounts = [3, 3, 3, 3]
    int[] faceVertexIndices = [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]
}}
''',
            stage_name,
        )

    default_model = build("", "mesh-default-hull-limit")
    unlimited_model = build("int newton:maxHullVertices = -1", "mesh-unlimited-hull-limit")
    default_limit = default_model.shape_source[0].maxhullvert
    assert default_limit > 0
    assert unlimited_model.shape_source[0].maxhullvert == default_limit


def test_fan_triangulate_polygon():
    counts = np.array([4, 3], dtype=np.int32)
    indices = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int32)
    assert _build._fan_triangulate(counts, indices).tolist() == [0, 1, 2, 0, 2, 3, 4, 5, 6]


@pytest.mark.parametrize("prim_type", ["Sphere", "Capsule", "Cylinder", "Cone"])
def test_analytic_geometry_preserves_zero_radius(prim_type):
    _, dimensions = _build._resolve_geometry(
        prim_type,
        half=None,
        radius=0.0,
        size=None,
        height=2.0,
        scale=(2.0, 3.0, 4.0),
    )
    assert dimensions["radius"] == 0.0


def test_invalid_mesh_topology_fails():
    collider = {
        "path": "/World/BadMesh",
        "points": np.zeros((3, 3), dtype=np.float32),
        "face_counts": np.array([4], dtype=np.int32),
        "face_indices": np.array([0, 1, 2], dtype=np.int32),
        "orientation": "rightHanded",
    }
    with pytest.raises(ValueError, match="invalid mesh topology at /World/BadMesh"):
        _build._mesh_geometry(collider)


@pytest.mark.parametrize("bounds", [(0.25, 1.5), (0.0, 1.5), (-1.0, 1.5), (0.25, -1.0), (None, None)])
def test_distance_joint_matches_add_usd(tmp_path, bounds):
    asset = tmp_path / "distance-joint.usda"
    asset.write_text(
        """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "World" (prepend apiSchemas = ["PhysicsArticulationRootAPI"]) {
    def Xform "Parent" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def Xform "Child" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def PhysicsDistanceJoint "Distance" {
        rel physics:body0 = </World/Parent>
        rel physics:body1 = </World/Child>
        __BOUNDS__
    }
}
""".replace(
            "__BOUNDS__",
            "\n".join(
                f"float physics:{name} = {value}"
                for name, value in zip(("minDistance", "maxDistance"), bounds)
                if value is not None
            ),
        ),
        encoding="utf-8",
    )
    ref = dh.reference_model(str(asset))
    model = _ovstage_model(str(asset), "distance-joint")
    mismatches = dh.compare_models(ref, dh._model_arrays(model), check_joints=True, check_articulations=True)
    assert not mismatches, "distance-joint mismatches vs add_usd:\n  " + "\n  ".join(mismatches)


def test_spherical_joint_matches_add_usd(tmp_path):
    joint_type = "PhysicsSphericalJoint"
    asset = tmp_path / f"{joint_type}.usda"
    asset.write_text(
        f"""#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "World" (prepend apiSchemas = ["PhysicsArticulationRootAPI"]) {{
    def Xform "Parent" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {{}}
    def Xform "Child" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {{}}
    def {joint_type} "Joint" {{
        rel physics:body0 = </World/Parent>
        rel physics:body1 = </World/Child>
    }}
}}
""",
        encoding="utf-8",
    )
    ref = dh.reference_model(str(asset))
    model = _ovstage_model(str(asset), f"supported-{joint_type}")
    mismatches = dh.compare_models(ref, dh._model_arrays(model), check_joints=True, check_articulations=True)
    assert not mismatches, f"{joint_type} mismatches vs add_usd:\n  " + "\n  ".join(mismatches)


def test_nonuniform_scaled_sphere_matches_add_usd(tmp_path):
    asset = tmp_path / "scaled-sphere.usda"
    asset.write_text(
        """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Sphere "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        double radius = 0.5
        float3 xformOp:scale = (2, 3, 4)
        uniform token[] xformOpOrder = ["xformOp:scale"]
    }
}
""",
        encoding="utf-8",
    )
    ref = dh.reference_model(str(asset))
    model = _ovstage_model(str(asset), "scaled-sphere")
    mismatches = dh.compare_models(ref, dh._model_arrays(model), check_joints=False, check_shapes=True)
    assert not mismatches, "scaled sphere mismatches vs add_usd:\n  " + "\n  ".join(mismatches)


def test_scaled_authored_center_of_mass_matches_newton_14(tmp_path):
    asset = tmp_path / "scaled-center-of-mass.usda"
    asset.write_text(
        """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Scaled" {
    double3 xformOp:scale = (-2, 3, 4)
    uniform token[] xformOpOrder = ["xformOp:scale"]
    def Xform "Body" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    ) {
        point3f physics:centerOfMass = (0.3, 0.2, 0.1)
        float physics:mass = 1
        def Cube "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {}
    }
}
""",
        encoding="utf-8",
    )
    model = _ovstage_model(str(asset), "scaled-center-of-mass")
    body = model.body_label.index("/Scaled/Body")
    np.testing.assert_allclose(model.body_com.numpy()[body], (-0.6, -0.6, -0.4))
    if os.environ.get("OVNEWTON_NEWTON_SOURCE"):
        ref = dh.reference_model(str(asset))
        mismatches = dh.compare_models(ref, dh._model_arrays(model), check_joints=False)
        assert not mismatches, "scaled center of mass mismatches vs add_usd:\n  " + "\n  ".join(mismatches)


@pytest.mark.parametrize("angle", (45, 90))
def test_scaled_rotated_hierarchy_fails_explicitly(angle):
    usda = '''#usda 1.0
def Xform "Scaled" {
    double3 xformOp:scale = (2, 3, 4)
    uniform token[] xformOpOrder = ["xformOp:scale"]
    def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
        float xformOp:rotateZ = __ANGLE__
        uniform token[] xformOpOrder = ["xformOp:rotateZ"]
    }
}
'''.replace("__ANGLE__", str(angle))
    with pytest.raises(
        ovnewton.UnsupportedPhysicsError,
        match="non-uniform scale across a rotated hierarchy.* /Scaled/Body",
    ):
        _ovstage_model_from_string(usda, f"unsupported-scaled-rotated-transform-{angle}")


@pytest.mark.parametrize("shape", ["Capsule", "Cylinder", "Cone"])
def test_nonuniform_scaled_axial_shape_matches_add_usd(tmp_path, shape):
    asset = tmp_path / f"scaled-{shape.lower()}.usda"
    asset.write_text(
        f"""#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {{
    def {shape} "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {{
        uniform token axis = "X"
        double radius = 0.5
        double height = 2
        float3 xformOp:scale = (2, 3, 4)
        uniform token[] xformOpOrder = ["xformOp:scale"]
    }}
}}
""",
        encoding="utf-8",
    )
    ref = dh.reference_model(str(asset))
    model = _ovstage_model(str(asset), f"scaled-{shape.lower()}")
    mismatches = dh.compare_models(ref, dh._model_arrays(model), check_joints=False, check_shapes=True)
    assert not mismatches, f"scaled {shape} mismatches vs add_usd:\n  " + "\n  ".join(mismatches)


def test_negative_scale_cube_matches_add_usd(tmp_path):
    asset = tmp_path / "negative-scale-cube.usda"
    asset.write_text(NEGATIVE_SCALE_CUBE, encoding="utf-8")
    ref = dh.reference_model(str(asset))
    model = _ovstage_model(str(asset), "negative-scale-cube")
    mismatches = dh.compare_models(ref, dh._model_arrays(model), check_shapes=True)
    assert not mismatches, "negative cube scale mismatches vs add_usd:\n  " + "\n  ".join(mismatches)


@pytest.mark.parametrize("scale", [(1, -1, 1), (1, 1, -1), (-1, -1, -1)])
def test_mirrored_mesh_axes_match_add_usd(tmp_path, scale):
    asset = tmp_path / "mirrored-mesh.usda"
    asset.write_text(MIRRORED_MESH.replace("__SCALE__", str(scale)), encoding="utf-8")

    ref = dh.reference_model(str(asset))
    model = _ovstage_model(str(asset), f"mirrored-mesh-{scale}")
    mismatches = dh.compare_models(ref, dh._model_arrays(model), check_shapes=True)
    assert not mismatches, f"scale {scale} mismatches vs add_usd:\n  " + "\n  ".join(mismatches)


def test_dynamic_and_static_cones_match_add_usd(tmp_path):
    asset = tmp_path / "cones.usda"
    asset.write_text(
        """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Material "Material" (prepend apiSchemas = ["PhysicsMaterialAPI", "NewtonMaterialAPI"]) {
    float physics:density = 500
    float physics:dynamicFriction = 0.3
    float physics:staticFriction = 0.4
    float physics:restitution = 0.2
    float newton:torsionalFriction = 0.01
    float newton:rollingFriction = 0.02
}
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Cone "Dynamic" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"]
    ) {
        rel material:binding:physics = </Material>
        uniform token axis = "X"
        double radius = 0.5
        double height = 2
        float3 xformOp:scale = (2, 3, 4)
        uniform token[] xformOpOrder = ["xformOp:scale"]
    }
}
def Cone "Static" (
    prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"]
) {
    rel material:binding:physics = </Material>
    uniform token axis = "Y"
    double radius = 0.75
    double height = 3
    double3 xformOp:translate = (4, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]
}
""",
        encoding="utf-8",
    )
    ref = dh.reference_model(str(asset))
    model = _ovstage_model(str(asset), "cones")
    mismatches = dh.compare_models(ref, dh._model_arrays(model), check_joints=False, check_shapes=True)
    assert not mismatches, "cone mismatches vs add_usd:\n  " + "\n  ".join(mismatches)


def test_collision_api_on_unsupported_prim_type_fails():
    usda = """#usda 1.0
def Scope "UnsupportedCollider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {}
"""
    with pytest.raises(ovnewton.UnsupportedPhysicsError, match="unsupported prim type at /UnsupportedCollider"):
        _ovstage_model_from_string(usda, "unsupported-collider")


def test_physics_point_instancer_fails_explicitly():
    usda = '''#usda 1.0
def Xform "World" {
    def Cube "Prototype" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {}
    def PointInstancer "Instances" {
        rel prototypes = </World/Prototype>
        int[] protoIndices = [0, 0]
        point3f[] positions = [(-1, 0, 0), (1, 0, 0)]
    }
}
'''
    with pytest.raises(
        ovnewton.UnsupportedPhysicsError,
        match="physics PointInstancer is unsupported at /World/Instances",
    ):
        _ovstage_model_from_string(usda, "unsupported-point-instancer")


def test_native_collision_instance_fails_when_composed_properties_are_unavailable(tmp_path):
    usda = '''#usda 1.0
def Xform "CollisionPrototype" {
    def Sphere "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        double radius = 0.4
        bool physics:collisionEnabled = false
    }
}
def Xform "World" {
    def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
        def "Collisions" (
            instanceable = true
            prepend references = </CollisionPrototype>
        ) {
            double3 xformOp:translate = (2, 3, 4)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
    }
}
'''
    asset = tmp_path / "native-collision-instance.usda"
    asset.write_text(usda, encoding="utf-8")
    ref = dh.reference_model(str(asset))
    assert list(ref["shape_label"]) == [
        "/CollisionPrototype/Collider",
        "/World/Body/Collisions/Collider",
    ]
    for flags in ref["shape_flags"]:
        assert not int(flags) & newton.ShapeFlags.COLLIDE_SHAPES
    with pytest.raises(
        ovnewton.UnsupportedPhysicsError,
        match="cannot preserve composed collider properties at /World/Body/Collisions/Collider",
    ):
        _ovstage_model(str(asset), "unsupported-native-collision-properties")


def test_native_mesh_approximation_instance_fails_explicitly(tmp_path):
    usda = '''#usda 1.0
def Xform "CollisionPrototype" {
    def Mesh "Collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    ) {
        uniform token physics:approximation = "convexHull"
        uniform token subdivisionScheme = "none"
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        int[] faceVertexCounts = [3, 3, 3, 3]
        int[] faceVertexIndices = [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]
    }
}
def Xform "World" {
    def Xform "BodyA" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
        def "Collisions" (
            instanceable = true
            prepend references = </CollisionPrototype>
        ) {}
    }
    def Xform "BodyB" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
        def "Collisions" (
            instanceable = true
            prepend references = </CollisionPrototype>
        ) {
            double3 xformOp:translate = (2, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
    }
}
'''
    asset = tmp_path / "native-mesh-instances.usda"
    asset.write_text(usda, encoding="utf-8")
    ref = dh.reference_model(str(asset))
    assert list(ref["shape_label"]) == [
        "/CollisionPrototype/Collider",
        "/World/BodyA/Collisions/Collider",
        "/World/BodyB/Collisions/Collider",
    ]
    assert all(int(shape_type) == int(newton.GeoType.CONVEX_MESH) for shape_type in ref["shape_type"])
    with pytest.raises(
        ovnewton.UnsupportedPhysicsError,
        match="cannot preserve composed collider properties at /World/BodyA/Collisions/Collider",
    ):
        _ovstage_model(str(asset), "unsupported-native-mesh-properties")


def test_native_rigid_body_instance_fails_explicitly(tmp_path):
    usda = '''#usda 1.0
def Xform "RobotPrototype" {
    def Xform "Link" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
}
def Xform "World" {
    def Xform "Robot" (
        instanceable = true
        prepend references = </RobotPrototype>
    ) {}
}
'''
    asset = tmp_path / "native-rigid-body-instance.usda"
    asset.write_text(usda, encoding="utf-8")
    ref = dh.reference_model(str(asset))
    assert list(ref["body_label"]) == ["/RobotPrototype/Link", "/World/Robot/Link"]
    with pytest.raises(
        ovnewton.UnsupportedPhysicsError,
        match="native USD rigid-body instancing is unsupported at .+; flatten physics instances before attachment",
    ):
        _ovstage_model(str(asset), "unsupported-native-rigid-body-instancing")


def test_native_joint_instance_fails_explicitly(tmp_path):
    usda = '''#usda 1.0
def Xform "JointPrototype" {
    def PhysicsFixedJoint "Joint" {
        rel physics:body1 = </World/Body>
    }
}
def Xform "World" {
    def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def Xform "Robot" (
        instanceable = true
        prepend references = </JointPrototype>
    ) {}
}
'''
    asset = tmp_path / "native-joint-instance.usda"
    asset.write_text(usda, encoding="utf-8")
    ref = dh.reference_model(str(asset))
    assert list(ref["joint_label"]) == ["/JointPrototype/Joint", "/World/Robot/Joint"]
    with pytest.raises(
        ovnewton.UnsupportedPhysicsError,
        match="native USD joint instancing is unsupported at .+; flatten physics instances before attachment",
    ):
        _ovstage_model(str(asset), "unsupported-native-joint-instancing")


def test_nonphysics_point_instancer_is_ignored():
    usda = '''#usda 1.0
def Xform "World" {
    def Cube "Prototype" {}
    def PointInstancer "Instances" {
        rel prototypes = </World/Prototype>
        int[] protoIndices = [0]
        point3f[] positions = [(0, 0, 0)]
    }
}
'''
    model = _ovstage_model_from_string(usda, "nonphysics-point-instancer")
    assert model.body_count == 0
    assert model.shape_count == 0


@pytest.mark.parametrize("schema", ("MjcEqualityConnectAPI", "MjcEqualityWeldAPI", "MjcEqualityJointAPI"))
def test_mujoco_equality_schemas_fail_explicitly(schema):
    usda = f'''#usda 1.0
def Xform "World" {{
    def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {{}}
    def PhysicsFixedJoint "Equality" (prepend apiSchemas = ["{schema}"]) {{
        rel physics:body1 = </World/Body>
    }}
}}
'''
    with pytest.raises(ovnewton.UnsupportedPhysicsError, match=f"unsupported {schema} properties"):
        _ovstage_model_from_string(usda, f"unsupported-{schema}")


def test_non_axis_gravity_is_normalized_and_applied(import_log):
    usda = """#usda 1.0
def PhysicsScene "Scene" {
    vector3f physics:gravityDirection = (3, 4, 0)
    float physics:gravityMagnitude = 10
}
"""
    def check(binding):
        np.testing.assert_allclose(binding.model.gravity.numpy()[0], (6.0, 8.0, 0.0))
        assert not _diagnostics(import_log)

    _binding_from_string(usda, "directed-gravity", check)


def test_newton_scene_gravity_disabled_matches_add_usd(tmp_path):
    asset = tmp_path / "gravity-disabled.usda"
    asset.write_text(
        '''#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def PhysicsScene "Scene" (prepend apiSchemas = ["NewtonSceneAPI"]) {
    bool newton:gravityEnabled = false
}
''',
        encoding="utf-8",
    )
    ref = dh.reference_model(str(asset))
    model = _ovstage_model(str(asset), "gravity-disabled")
    mismatches = dh.compare_models(ref, dh._model_arrays(model), check_joints=False)
    assert not mismatches, "disabled gravity mismatches vs add_usd:\n  " + "\n  ".join(mismatches)


def test_population_resolves_default_gravity_from_stage_units():
    usda = """#usda 1.0
(
    metersPerUnit = 0.01
    upAxis = "Y"
)
def PhysicsScene "Scene" {}
"""

    def check(binding):
        np.testing.assert_allclose(binding.model.gravity.numpy()[0], (0.0, -981.0, 0.0))
        assert binding.model.up_axis == newton.Axis.Y

    _binding_from_string(usda, "default-y-gravity", check)


def test_y_up_physics_profile_matches_add_usd(tmp_path):
    asset = tmp_path / "y-up.usda"
    asset.write_text(
        """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Y"
)
def PhysicsScene "Scene" {}
def Cube "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {}
""",
        encoding="utf-8",
    )
    ref = dh.reference_model(str(asset))
    model = _ovstage_model(str(asset), "y-up-profile")
    mismatches = dh.compare_models(ref, dh._model_arrays(model), check_joints=False, check_shapes=True)
    assert not mismatches, "Y-up mismatches vs add_usd:\n  " + "\n  ".join(mismatches)
    assert model.up_axis == newton.Axis.Y


def test_gravity_sentinels_resolve_at_consumer_boundary():
    usda = """#usda 1.0
(
    metersPerUnit = 0.5
    upAxis = "Y"
)
def PhysicsScene "Scene" {
    vector3f physics:gravityDirection = (0, 0, 0)
    float physics:gravityMagnitude = -inf
}
"""

    def check(binding):
        np.testing.assert_allclose(binding.model.gravity.numpy()[0], (0.0, -19.62, 0.0))

    _binding_from_string(usda, "sentinel-gravity", check)


def test_kinematic_body_matches_add_usd(tmp_path):
    asset = tmp_path / "kinematic.usda"
    asset.write_text(
        """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    bool physics:kinematicEnabled = 1
}
""",
        encoding="utf-8",
    )
    ref = dh.reference_model(str(asset))
    model = _ovstage_model(str(asset), "kinematic")
    mismatches = dh.compare_models(ref, dh._model_arrays(model), check_joints=True, check_articulations=True)
    assert not mismatches, "kinematic-body mismatches vs add_usd:\n  " + "\n  ".join(mismatches)


def test_nonidentity_principal_axes_matches_add_usd(tmp_path):
    asset = tmp_path / "principal-axes.usda"
    asset.write_text(
        """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
    float physics:mass = 2
    float3 physics:diagonalInertia = (2, 3, 4)
    quatf physics:principalAxes = (0.7071068, 0, 0, 0.7071068)
}
""",
        encoding="utf-8",
    )
    ref = dh.reference_model(str(asset))
    model = _ovstage_model(str(asset), "principal-axes")
    mismatches = dh.compare_models(ref, dh._model_arrays(model), check_joints=True)
    assert not mismatches, "principal-axis mismatches vs add_usd:\n  " + "\n  ".join(mismatches)


def test_newton_mass_api_matches_add_usd(tmp_path):
    asset = tmp_path / "newton-mass-api.usda"
    asset.write_text(
        """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "ShellThickness" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
) {
    float physics:mass = 10
    def Sphere "Collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "NewtonMassAPI"]
    ) {
        double radius = 0.5
        uniform token newton:massModel = "shell"
        float newton:shellThickness = 0.05
    }
}
def Xform "ShellMargin" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
) {
    float physics:mass = 10
    def Sphere "Collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "NewtonCollisionAPI", "NewtonMassAPI"]
    ) {
        double radius = 0.5
        uniform token newton:massModel = "shell"
        float newton:contactMargin = 0.03
    }
}
def Xform "Solid" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
) {
    float physics:mass = 10
    def Sphere "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        double radius = 0.5
    }
}
def Xform "ShellDensity" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
    def Sphere "Collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "NewtonMassAPI"]
    ) {
        double radius = 0.5
        uniform token newton:massModel = "shell"
        float newton:shellThickness = 0.05
    }
}
def Xform "Tensor" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "NewtonMassAPI"]
) {
    float physics:mass = 5
    float3 physics:diagonalInertia = (9, 9, 9)
    quatf physics:principalAxes = (0.7071068, 0, 0, 0.7071068)
    double[] newton:inertia = [1, 2, 3, 0.1, 0.2, 0.3]
    def Sphere "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {}
}
def Xform "SingularTensor" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "NewtonMassAPI"]
) {
    float physics:mass = 2
    double[] newton:inertia = [1, 0, 0, 0, 0, 0]
    def Sphere "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {}
}
""",
        encoding="utf-8",
    )
    ref = dh.reference_model(str(asset))
    with pytest.warns(UserWarning, match="Inertia validation corrected"):
        model = _ovstage_model(str(asset), "newton-mass-api")
    mismatches = dh.compare_models(ref, dh._model_arrays(model), check_shapes=True)
    assert not mismatches, "NewtonMassAPI mismatches vs add_usd:\n  " + "\n  ".join(mismatches)


def test_unapplied_newton_mass_attributes_are_ignored(import_log):
    usda = """#usda 1.0
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
    float physics:mass = 3
    float3 physics:diagonalInertia = (2, 3, 4)
    custom double[] newton:inertia = [9, 9, 9, 0, 0, 0]
    def Sphere "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        custom uniform token newton:massModel = "shell"
        custom float newton:shellThickness = 0.05
    }
}
"""

    def check(binding):
        np.testing.assert_allclose(binding.model.body_inertia.numpy()[0], np.diag((2, 3, 4)))
        assert binding.model.shape_is_solid.numpy()[0] == 1
        assert not _diagnostics(import_log)

    _binding_from_string(usda, "unapplied-newton-mass", check)


def test_negative_shell_thickness_uses_contact_margin(import_log):
    usda = """#usda 1.0
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
    float physics:mass = 10
    def Sphere "Collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "NewtonCollisionAPI", "NewtonMassAPI"]
    ) {
        double radius = 0.5
        uniform token newton:massModel = "shell"
        float newton:shellThickness = -0.5
        float newton:contactMargin = 0.03
    }
}
def Xform "Reference" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
    float physics:mass = 10
    def Sphere "Collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "NewtonCollisionAPI", "NewtonMassAPI"]
    ) {
        double radius = 0.5
        uniform token newton:massModel = "shell"
        float newton:contactMargin = 0.03
    }
}
"""

    def check(binding):
        model = binding.model
        body = model.body_label.index("/Body")
        reference = model.body_label.index("/Reference")
        shape = model.shape_label.index("/Body/Collider")
        assert model.shape_is_solid.numpy()[shape] == 0
        np.testing.assert_allclose(model.shape_margin.numpy()[shape], 0.03)
        np.testing.assert_allclose(model.body_inertia.numpy()[body], model.body_inertia.numpy()[reference])
        assert _diagnostic_pairs(import_log) == [
            ("negative-shell-thickness-ignored", "/Body/Collider")
        ]

    _binding_from_string(usda, "negative-shell-thickness", check)


@pytest.mark.parametrize(
    "inertia",
    (
        "[1, 2, inf, 0, 0, 0]",
        "[-1, 2, 3, 0, 0, 0]",
        "[1, 2, 3]",
        "[1, 1, 1, 5, 5, 5]",
    ),
)
def test_invalid_newton_inertia_falls_back_to_standard_inertia(inertia, import_log):
    usda = f"""#usda 1.0
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "NewtonMassAPI"]) {{
    float physics:mass = 2
    float3 physics:diagonalInertia = (2, 3, 4)
    double[] newton:inertia = {inertia}
}}
"""

    def check(binding):
        np.testing.assert_allclose(binding.model.body_inertia.numpy()[0], np.diag((2, 3, 4)))
        assert _diagnostic_pairs(import_log) == [("newton-inertia-ignored", "/Body")]

    _binding_from_string(usda, "invalid-newton-inertia", check)


def test_singular_newton_inertia_has_zero_inverse():
    builder = newton.ModelBuilder()
    body = builder.add_body()
    inertia = np.diag((1.0, 0.0, 0.0))
    _build._finalize_body_mass(
        builder,
        body,
        body_scale=(1.0, 1.0, 1.0),
        mass=2.0,
        inertia_tensor=inertia,
        inertia_diag=None,
        principal_axes=None,
        com=None,
    )
    np.testing.assert_allclose(np.asarray(builder.body_inertia[body]).reshape(3, 3), inertia)
    np.testing.assert_allclose(np.asarray(builder.body_inv_inertia[body]).reshape(3, 3), np.zeros((3, 3)))


@_REQUIRES_CUDA
def test_newton_sdf_api_matches_add_usd(tmp_path, monkeypatch):
    asset = tmp_path / "newton-sdf-api.usda"
    asset.write_text(
        """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "MeshBody" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Mesh "Collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "NewtonSDFCollisionAPI"]
    ) {
        point3f[] points = [
            (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5),
            (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5),
            (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5),
            (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)
        ]
        int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
        int[] faceVertexIndices = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 5, 4, 2, 3, 7, 6, 0, 3, 7, 4, 1, 2, 6, 5]
        int newton:sdfMaxResolution = 16
        float newton:sdfNarrowBandInner = -0.02
        float newton:sdfNarrowBandOuter = 0.03
        token newton:sdfTextureFormat = "uint8"
        float newton:sdfPadding = 0.08
        bool newton:hydroelasticEnabled = true
        float newton:hydroelasticStiffness = 1e7
        float newton:contactGap = 0.07
    }
}
def Xform "SphereBody" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Sphere "Collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "NewtonSDFCollisionAPI"]
    ) {
        double radius = 0.5
        float newton:sdfTargetVoxelSize = 0.1
        bool newton:hydroelasticEnabled = true
        float newton:hydroelasticStiffness = 2e7
    }
}
def Xform "DefaultBody" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Sphere "Collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "NewtonSDFCollisionAPI"]
    ) {
        double radius = 0.25
        float newton:hydroelasticStiffness = 3e7
    }
}
def Xform "GapBody" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Sphere "Collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "NewtonCollisionAPI"]
    ) {
        double radius = 0.25
        float newton:contactGap = 0.02
    }
}
""",
        encoding="utf-8",
    )
    ref = dh.reference_model(str(asset))
    captured = {}
    original_finalize = newton.ModelBuilder.finalize

    def capture_finalize(builder, *args, **kwargs):
        captured.update(
            labels=list(builder.shape_label),
            flags=list(builder.shape_flags),
            kh=list(builder.shape_material_kh),
            max_resolution=list(builder.shape_sdf_max_resolution),
            target_voxel_size=list(builder.shape_sdf_target_voxel_size),
            narrow_band_range=list(builder.shape_sdf_narrow_band_range),
            texture_format=list(builder.shape_sdf_texture_format),
            padding=list(builder.shape_sdf_padding),
        )
        return original_finalize(builder, *args, **kwargs)

    monkeypatch.setattr(newton.ModelBuilder, "finalize", capture_finalize)
    with pytest.warns(UserWarning, match="Inertia validation corrected"):
        model = _ovstage_model(str(asset), "newton-sdf-api")
    mismatches = dh.compare_models(ref, dh._model_arrays(model), check_joints=False, check_shapes=True)
    assert not mismatches, "NewtonSDFCollisionAPI mismatches vs add_usd:\n  " + "\n  ".join(mismatches)

    mesh = captured["labels"].index("/MeshBody/Collider")
    sphere = captured["labels"].index("/SphereBody/Collider")
    default = captured["labels"].index("/DefaultBody/Collider")
    assert captured["max_resolution"][mesh] == 16
    assert captured["target_voxel_size"][mesh] is None
    np.testing.assert_allclose(captured["narrow_band_range"][mesh], (-0.02, 0.03))
    assert captured["texture_format"][mesh] == "uint8"
    assert captured["padding"][mesh] == pytest.approx(0.08)
    assert captured["flags"][mesh] & newton.ShapeFlags.HYDROELASTIC
    assert captured["kh"][mesh] == pytest.approx(1e7)

    assert captured["max_resolution"][sphere] is None
    assert captured["target_voxel_size"][sphere] == pytest.approx(0.1)
    assert captured["flags"][sphere] & newton.ShapeFlags.HYDROELASTIC
    assert captured["kh"][sphere] == pytest.approx(2e7)

    assert captured["max_resolution"][default] == 64
    assert captured["target_voxel_size"][default] is None
    assert not captured["flags"][default] & newton.ShapeFlags.HYDROELASTIC
    assert captured["kh"][default] == pytest.approx(3e7)


def test_bound_physics_material_matches_add_usd(tmp_path):
    asset = tmp_path / "physics-material.usda"
    asset.write_text(
        """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Material "Material" (prepend apiSchemas = ["PhysicsMaterialAPI", "NewtonMaterialAPI"]) {
    float physics:density = 500
    float physics:dynamicFriction = 0.3
    float physics:staticFriction = 0.4
    float physics:restitution = 0.2
    float newton:torsionalFriction = 0.01
    float newton:rollingFriction = 0.02
}
def Cube "Body" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "MaterialBindingAPI"]
) {
    rel material:binding:physics = </Material>
}
""",
        encoding="utf-8",
    )
    ref = dh.reference_model(str(asset))
    model = _ovstage_model(str(asset), "physics-material")
    mismatches = dh.compare_models(ref, dh._model_arrays(model), check_joints=True, check_shapes=True)
    assert not mismatches, "physics-material mismatches vs add_usd:\n  " + "\n  ".join(mismatches)


def test_ground_plane_bound_material_matches_add_usd(tmp_path):
    asset = tmp_path / "ground-material.usda"
    asset.write_text(
        """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Material "Material" (prepend apiSchemas = ["PhysicsMaterialAPI"]) {
    float physics:dynamicFriction = 0.3
    float physics:staticFriction = 0.4
    float physics:restitution = 0.2
}
def Plane "Ground" (
    prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"]
) {
    rel material:binding:physics = </Material>
}
""",
        encoding="utf-8",
    )
    ref = dh.reference_model(str(asset))
    model = _ovstage_model(str(asset), "ground-material")
    mismatches = dh.compare_models(ref, dh._model_arrays(model), check_joints=False, check_shapes=True)
    assert not mismatches, "ground-material mismatches vs add_usd:\n  " + "\n  ".join(mismatches)


def test_newton_contact_material_matches_add_usd(tmp_path):
    asset = tmp_path / "newton-contact-material.usda"
    asset.write_text(
        """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Material "All" (prepend apiSchemas = ["PhysicsMaterialAPI", "NewtonMaterialAPI"]) {
    float newton:contactStiffness = 5000
    float newton:contactDamping = 200
    float newton:contactFrictionGain = 800
    float newton:contactAdhesion = 0.01
}
def Material "Partial" (prepend apiSchemas = ["PhysicsMaterialAPI", "NewtonMaterialAPI"]) {
    float newton:contactStiffness = 3000
    float newton:contactDamping = 150
}
def Material "Defaults" (prepend apiSchemas = ["PhysicsMaterialAPI", "NewtonMaterialAPI"]) {}
def Material "DisabledAdhesion" (prepend apiSchemas = ["PhysicsMaterialAPI", "NewtonMaterialAPI"]) {
    float newton:contactAdhesion = -1
}
def Material "PhysicsOnly" (prepend apiSchemas = ["PhysicsMaterialAPI"]) {}

def Cube "AllBody" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "MaterialBindingAPI"]
) {
    rel material:binding:physics = </All>
}
def Cube "PartialBody" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "MaterialBindingAPI"]
) {
    rel material:binding:physics = </Partial>
}
def Cube "DefaultBody" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "MaterialBindingAPI"]
) {
    rel material:binding:physics = </Defaults>
}
def Cube "DisabledAdhesionBody" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "MaterialBindingAPI"]
) {
    rel material:binding:physics = </DisabledAdhesion>
}
def Cube "PhysicsOnlyBody" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "MaterialBindingAPI"]
) {
    rel material:binding:physics = </PhysicsOnly>
}
""",
        encoding="utf-8",
    )
    ref = dh.reference_model(str(asset))
    model = _ovstage_model(str(asset), "newton-contact-material")
    mismatches = dh.compare_models(ref, dh._model_arrays(model), check_joints=False, check_shapes=True)
    assert not mismatches, "NewtonMaterialAPI mismatches vs add_usd:\n  " + "\n  ".join(mismatches)


def _binding_from_string(usda, stage_name, check):
    with ovstage.Stage(stage_name) as stage:
        population.open_usd_from_string(stage, usda, ordinal=1, domains=PopulationDomain.ALL)
        stage.advance_write_floor(ordinal=1).wait()
        check(ovnewton.attach_ovstage(stage))


def _binding_with_builder_from_string(monkeypatch, usda, stage_name, check):
    original_finalize = newton.ModelBuilder.finalize
    captured = {}

    def capture_finalize(builder, *args, **kwargs):
        captured["builder"] = builder
        return original_finalize(builder, *args, **kwargs)

    monkeypatch.setattr(newton.ModelBuilder, "finalize", capture_finalize)

    def check_builder(binding):
        check(binding, captured["builder"])

    _binding_from_string(usda, stage_name, check_builder)


def test_newton_contact_material_policy(monkeypatch, import_log):
    usda = """#usda 1.0
def Material "Sentinel" (prepend apiSchemas = ["NewtonMaterialAPI"]) {
    float newton:contactStiffness = -inf
    float newton:contactDamping = -inf
    float newton:contactFrictionGain = -inf
    float newton:contactAdhesion = -inf
}
def Material "Invalid" (prepend apiSchemas = ["NewtonMaterialAPI"]) {
    float newton:torsionalFriction = -1
    float newton:rollingFriction = nan
    float newton:contactStiffness = nan
    float newton:contactDamping = nan
    float newton:contactFrictionGain = inf
    float newton:contactAdhesion = inf
}
def Material "Unapplied" (prepend apiSchemas = ["PhysicsMaterialAPI"]) {
    custom float newton:torsionalFriction = 0.2
    custom float newton:rollingFriction = 0.3
    custom float newton:contactStiffness = 4000
    custom float newton:contactDamping = 500
    custom float newton:contactFrictionGain = 600
    custom float newton:contactAdhesion = 0.4
}
def Cube "SentinelBody" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "MaterialBindingAPI"]
) {
    rel material:binding:physics = </Sentinel>
}
def Cube "InvalidBody" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "MaterialBindingAPI"]
) {
    rel material:binding:physics = </Invalid>
}
def Cube "UnappliedBody" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "MaterialBindingAPI"]
) {
    rel material:binding:physics = </Unapplied>
}
"""

    def check(binding, builder):
        defaults = builder.default_shape_cfg
        labels = {label: i for i, label in enumerate(builder.shape_label)}
        fields = (
            "mu_torsional",
            "mu_rolling",
            "ke",
            "kd",
            "kf",
            "ka",
        )
        arrays = {field: getattr(builder, f"shape_material_{field}") for field in fields}
        for label in ("/SentinelBody", "/InvalidBody", "/UnappliedBody"):
            shape = labels[label]
            for field in fields:
                assert arrays[field][shape] == pytest.approx(getattr(defaults, field))

        diagnostics = _diagnostics(import_log)
        assert len(diagnostics) == 6
        assert {(item.diagnostic_code, item.prim_path) for item in diagnostics} == {
            ("newton-material-value-ignored", "/Invalid")
        }
        assert {item.getMessage().split(": ", 1)[1].split(" must", 1)[0] for item in diagnostics} == {
            "newton:torsionalFriction",
            "newton:rollingFriction",
            "newton:contactStiffness",
            "newton:contactDamping",
            "newton:contactFrictionGain",
            "newton:contactAdhesion",
        }

    _binding_with_builder_from_string(monkeypatch, usda, "newton-contact-material-policy", check)


def test_newton_sites_match_add_usd(tmp_path):
    asset = tmp_path / "newton-sites.usda"
    asset.write_text(
        """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
    float physics:mass = 2
    float3 physics:diagonalInertia = (1, 2, 3)
    double3 xformOp:translate = (1, 2, 3)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Sphere "SphereSite" (prepend apiSchemas = ["NewtonSiteAPI"]) {
        double radius = 0.1
        double3 xformOp:translate = (0.1, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Cube "BoxSite" (prepend apiSchemas = ["NewtonSiteAPI"]) {
        double size = 0.2
        double3 xformOp:translate = (0, 0.2, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Capsule "CapsuleSite" (prepend apiSchemas = ["NewtonSiteAPI"]) {
        uniform token axis = "X"
        double radius = 0.05
        double height = 0.3
        float3 xformOp:scale = (2, 3, 4)
        uniform token[] xformOpOrder = ["xformOp:scale"]
    }
    def Cylinder "CylinderSite" (prepend apiSchemas = ["NewtonSiteAPI"]) {
        uniform token axis = "Y"
        double radius = 0.06
        double height = 0.4
        float3 xformOp:scale = (4, 2, 3)
        uniform token[] xformOpOrder = ["xformOp:scale"]
    }
    def Cone "ConeSite" (prepend apiSchemas = ["NewtonSiteAPI"]) {
        uniform token axis = "Z"
        double radius = 0.07
        double height = 0.5
        float3 xformOp:scale = (3, 4, 2)
        uniform token[] xformOpOrder = ["xformOp:scale"]
    }
    def Xform "Frame" {
        def Sphere "NestedSite" (prepend apiSchemas = ["NewtonSiteAPI"]) {
            double radius = 0.08
        }
    }
}
""",
        encoding="utf-8",
    )
    # add_usd 1.4 prunes non-site containers when visuals are disabled.
    ref = dh.reference_model(str(asset), load_visual_shapes=True)
    model = _ovstage_model(str(asset), "newton-sites")
    mismatches = dh.compare_models(ref, dh._model_arrays(model), check_joints=False, check_shapes=True)
    assert not mismatches, "NewtonSiteAPI mismatches vs add_usd:\n  " + "\n  ".join(mismatches)


def test_visual_gprim_without_newton_site_api_is_ignored():
    usda = """#usda 1.0
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Sphere "VisualOnly" {
        double radius = 5
    }
}
"""
    model = _ovstage_model_from_string(usda, "visual-only")
    assert model.shape_count == 0


def test_invisible_collider_remains_physics_geometry():
    usda = """#usda 1.0
def Xform "Hidden" {
    token visibility = "invisible"
    def Cube "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {}
}
"""
    model = _ovstage_model_from_string(usda, "invisible-collider")
    index = model.shape_label.index("/Hidden/Body")
    flags = int(model.shape_flags.numpy()[index])
    assert flags & newton.ShapeFlags.COLLIDE_SHAPES
    assert flags & newton.ShapeFlags.VISIBLE


def test_site_beneath_collider_is_preserved():
    usda = """#usda 1.0
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Cube "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        def Sphere "Site" (prepend apiSchemas = ["NewtonSiteAPI"]) {
            double radius = 0.1
        }
    }
}
"""
    model = _ovstage_model_from_string(usda, "site-beneath-collider")
    site = model.shape_label.index("/Body/Collider/Site")
    collider = model.shape_label.index("/Body/Collider")
    assert int(model.shape_flags.numpy()[site]) & newton.ShapeFlags.SITE
    assert not int(model.shape_flags.numpy()[collider]) & newton.ShapeFlags.SITE


@pytest.mark.parametrize(
    "usda,message,error",
    (
        (
            """def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Sphere "Site" (prepend apiSchemas = ["NewtonSiteAPI", "PhysicsCollisionAPI"]) {}
}
""",
            "both NewtonSiteAPI and PhysicsCollisionAPI",
            ovnewton.InvalidPhysicsError,
        ),
        (
            """def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Mesh "Site" (prepend apiSchemas = ["NewtonSiteAPI"]) {}
}
""",
            "unsupported NewtonSiteAPI prim type",
            ovnewton.UnsupportedPhysicsError,
        ),
        (
            'def Sphere "Site" (prepend apiSchemas = ["NewtonSiteAPI"]) {}',
            "is not beneath a rigid body",
            ovnewton.UnsupportedPhysicsError,
        ),
    ),
)
def test_unsupported_site_semantics_fail(usda, message, error):
    with pytest.raises(error, match=message):
        _ovstage_model_from_string("#usda 1.0\n" + usda, "unsupported-site")


@_REQUIRES_CUDA
def test_invalid_newton_sdf_values_fall_back_with_diagnostics(monkeypatch, import_log):
    usda = """#usda 1.0
def Xform "InvalidMax" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Sphere "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI", "NewtonSDFCollisionAPI"]) {
        int newton:sdfMaxResolution = 63
    }
}
def Xform "InvalidTarget" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Sphere "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI", "NewtonSDFCollisionAPI"]) {
        float newton:sdfTargetVoxelSize = 0
    }
}
def Xform "ResolutionConflict" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Sphere "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI", "NewtonSDFCollisionAPI"]) {
        int newton:sdfMaxResolution = 128
        float newton:sdfTargetVoxelSize = 0.1
    }
}
def Xform "InvalidTexture" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Sphere "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI", "NewtonSDFCollisionAPI"]) {
        token newton:sdfTextureFormat = "bogus"
    }
}
def Xform "InvalidPadding" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Sphere "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI", "NewtonSDFCollisionAPI"]) {
        float newton:sdfPadding = -1
    }
}
def Xform "InvalidStiffness" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Sphere "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI", "NewtonSDFCollisionAPI"]) {
        bool newton:hydroelasticEnabled = true
        float newton:hydroelasticStiffness = 0
    }
}
def Xform "NonFinite" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Sphere "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI", "NewtonSDFCollisionAPI"]) {
        float newton:contactGap = nan
        float newton:sdfTargetVoxelSize = nan
        float newton:sdfNarrowBandInner = nan
        float newton:sdfPadding = inf
        bool newton:hydroelasticEnabled = true
        float newton:hydroelasticStiffness = nan
    }
}
"""

    def check(binding, builder):
        diagnostics = set(_diagnostic_pairs(import_log))
        assert diagnostics == {
            ("sdf-max-resolution-ignored", "/InvalidMax/Collider"),
            ("sdf-padding-ignored", "/InvalidPadding/Collider"),
            ("hydroelastic-stiffness-ignored", "/InvalidStiffness/Collider"),
            ("sdf-target-voxel-size-ignored", "/InvalidTarget/Collider"),
            ("sdf-texture-format-ignored", "/InvalidTexture/Collider"),
            ("sdf-resolution-conflict", "/ResolutionConflict/Collider"),
            ("contact-gap-ignored", "/NonFinite/Collider"),
            ("sdf-target-voxel-size-ignored", "/NonFinite/Collider"),
            ("sdf-narrow-band-ignored", "/NonFinite/Collider"),
            ("sdf-padding-ignored", "/NonFinite/Collider"),
            ("hydroelastic-stiffness-ignored", "/NonFinite/Collider"),
        }
        by_label = {label: i for i, label in enumerate(builder.shape_label)}
        assert builder.shape_sdf_max_resolution[by_label["/InvalidMax/Collider"]] == 64
        assert builder.shape_sdf_max_resolution[by_label["/InvalidTarget/Collider"]] == 64
        conflict = by_label["/ResolutionConflict/Collider"]
        assert builder.shape_sdf_max_resolution[conflict] is None
        assert builder.shape_sdf_target_voxel_size[conflict] == pytest.approx(0.1)
        assert builder.shape_sdf_texture_format[by_label["/InvalidTexture/Collider"]] == "uint16"
        assert builder.shape_sdf_padding[by_label["/InvalidPadding/Collider"]] is None
        assert builder.shape_material_kh[by_label["/InvalidStiffness/Collider"]] == pytest.approx(1e10)
        non_finite = by_label["/NonFinite/Collider"]
        assert builder.shape_gap[non_finite] == pytest.approx(builder.rigid_gap)
        assert builder.shape_sdf_max_resolution[non_finite] == 64
        assert builder.shape_sdf_target_voxel_size[non_finite] is None
        assert builder.shape_sdf_narrow_band_range[non_finite][0] == pytest.approx(-0.1)
        assert builder.shape_sdf_padding[non_finite] is None
        assert builder.shape_material_kh[non_finite] == pytest.approx(1e10)

    _binding_with_builder_from_string(monkeypatch, usda, "invalid-newton-sdf", check)


def test_unapplied_newton_sdf_attributes_are_ignored(import_log):
    usda = """#usda 1.0
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Sphere "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        custom int newton:sdfMaxResolution = 16
        custom bool newton:hydroelasticEnabled = true
        custom float newton:hydroelasticStiffness = 1e7
    }
}
"""

    def check(binding):
        model = binding.model
        assert int(model._shape_sdf_index.numpy()[0]) == -1
        assert not int(model.shape_flags.numpy()[0]) & newton.ShapeFlags.HYDROELASTIC
        assert model.shape_material_kh.numpy()[0] == pytest.approx(1e10)
        assert not _diagnostics(import_log)

    _binding_from_string(usda, "unapplied-newton-sdf", check)


@_REQUIRES_CUDA
def test_sdf_wins_over_mesh_collision_configuration(import_log):
    usda = """#usda 1.0
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Mesh "Collider" (
        prepend apiSchemas = [
            "PhysicsCollisionAPI", "PhysicsMeshCollisionAPI",
            "NewtonMeshCollisionAPI", "NewtonSDFCollisionAPI"
        ]
    ) {
        point3f[] points = [
            (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5),
            (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5),
            (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5),
            (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)
        ]
        int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
        int[] faceVertexIndices = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 5, 4, 2, 3, 7, 6, 0, 3, 7, 4, 1, 2, 6, 5]
        token physics:approximation = "convexHull"
        int newton:sdfMaxResolution = 16
    }
}
"""

    def check(binding):
        assert int(binding.model._shape_sdf_index.numpy()[0]) >= 0
        assert _diagnostic_pairs(import_log) == [
            ("sdf-mesh-collision-api-ignored", "/Body/Collider"),
            ("sdf-mesh-approximation-ignored", "/Body/Collider"),
        ]

    with pytest.warns(UserWarning, match="Inertia validation corrected"):
        _binding_from_string(usda, "sdf-mesh-precedence", check)


def test_disabled_rigid_body_is_imported_with_diagnostic(import_log):
    usda = """#usda 1.0
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    bool physics:rigidBodyEnabled = 0
}
"""

    def check(binding):
        assert binding.model.body_count == 1
        assert _diagnostic_pairs(import_log) == [("disabled-rigid-body-imported", "/Body")]

    _binding_from_string(usda, "disabled-body-policy", check)


def test_disabled_collider_retains_geometry_and_mass_but_not_collision():
    usda = """#usda 1.0
def Cube "Collider" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {
    bool physics:collisionEnabled = 0
}
"""

    def check(binding):
        assert binding.model.shape_count == 1
        assert float(binding.model.body_mass.numpy()[0]) > 0.0
        flags = int(binding.model.shape_flags.numpy()[0])
        assert not flags & int(newton.ShapeFlags.COLLIDE_SHAPES)

    _binding_from_string(usda, "disabled-collider-policy", check)


def test_collision_api_container_is_nonfatal_and_reported(import_log):
    usda = """#usda 1.0
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {
    def Cube "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {}
}
"""

    def check(binding):
        assert binding.model.shape_count == 1
        assert [
            (record.diagnostic_code, record.prim_path, record.levelno)
            for record in _diagnostics(import_log)
        ] == [
            ("collision-container-ignored", "/Body", logging.INFO)
        ]

    _binding_from_string(usda, "collision-container-policy", check)


def test_logging_configuration_remains_application_owned(import_log):
    usda = """#usda 1.0
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {
    def Cube "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {}
}
"""
    logger = logging.getLogger("ovnewton")
    configuration = (tuple(logger.handlers), tuple(logger.filters), logger.level, logger.disabled, logger.propagate)

    _binding_from_string(usda, "application-owned-logging", lambda binding: None)

    assert (
        tuple(logger.handlers),
        tuple(logger.filters),
        logger.level,
        logger.disabled,
        logger.propagate,
    ) == configuration


def test_application_logging_level_controls_diagnostics(caplog):
    usda = """#usda 1.0
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {
    def Cube "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {}
}
"""
    caplog.set_level(logging.ERROR, logger="ovnewton")

    _binding_from_string(usda, "application-logging-level", lambda binding: None)

    assert not _diagnostics(caplog)


def test_application_logging_filter_controls_diagnostics(import_log):
    usda = """#usda 1.0
def Cube "Body" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]
) {
    bool physics:rigidBodyEnabled = false
    bool physics:startsAsleep = true
}
"""

    class DiagnosticFilter(logging.Filter):
        def filter(self, record):
            return getattr(record, "diagnostic_code", None) == "starts-asleep-ignored"

    diagnostic_filter = DiagnosticFilter()
    import_log.handler.addFilter(diagnostic_filter)
    try:
        _binding_from_string(usda, "application-filtered-logging", lambda binding: None)
    finally:
        import_log.handler.removeFilter(diagnostic_filter)

    assert _diagnostic_pairs(import_log) == [("starts-asleep-ignored", "/Body")]


def test_disabled_joint_is_omitted_with_diagnostic(import_log):
    usda = """#usda 1.0
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
def PhysicsFixedJoint "Joint" {
    rel physics:body1 = </Body>
    bool physics:jointEnabled = 0
}
"""

    def check(binding):
        assert "/Joint" not in binding.model.joint_label
        assert _diagnostic_pairs(import_log) == [("disabled-joint-skipped", "/Joint")]

    _binding_from_string(usda, "disabled-joint-policy", check)


@pytest.mark.parametrize("collision_enabled,filtered", [(False, True), (True, False)])
def test_joint_collision_enabled_controls_only_connected_body_filter(collision_enabled, filtered):
    value = "true" if collision_enabled else "false"
    usda = f"""#usda 1.0
def Cube "Parent" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {{}}
def Cube "Child" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {{}}
def PhysicsFixedJoint "Joint" {{
    rel physics:body0 = </Parent>
    rel physics:body1 = </Child>
    bool physics:collisionEnabled = {value}
}}
"""

    def check(binding):
        pair = (0, 1)
        assert (pair in binding.model.shape_collision_filter_pairs) is filtered

    _binding_from_string(usda, f"joint-collision-{value}", check)


def test_merged_joint_collision_filter_uses_any_disabling_joint():
    usda = """#usda 1.0
def Cube "Parent" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {}
def Cube "Child" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {}
def PhysicsRevoluteJoint "Angular" {
    rel physics:body0 = </Parent>
    rel physics:body1 = </Child>
    bool physics:collisionEnabled = true
}
def PhysicsPrismaticJoint "Linear" {
    rel physics:body0 = </Parent>
    rel physics:body1 = </Child>
    bool physics:collisionEnabled = false
}
"""

    def check(binding):
        assert binding.model.joint_count == 1
        assert (0, 1) in binding.model.shape_collision_filter_pairs

    _binding_from_string(usda, "merged-joint-collision-filter", check)


@pytest.mark.parametrize("self_collisions,filtered", [(False, True), (True, False)])
def test_newton_articulation_self_collision_filters_all_internal_body_pairs(self_collisions, filtered):
    value = "true" if self_collisions else "false"
    usda = f"""#usda 1.0
def Xform "Robot" (prepend apiSchemas = ["NewtonArticulationRootAPI"]) {{
    bool newton:selfCollisionEnabled = {value}
    def Cube "A" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {{}}
    def Cube "B" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {{}}
    def Cube "C" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {{}}
    def PhysicsFixedJoint "AB" {{
        rel physics:body0 = </Robot/A>
        rel physics:body1 = </Robot/B>
        bool physics:collisionEnabled = true
    }}
    def PhysicsFixedJoint "BC" {{
        rel physics:body0 = </Robot/B>
        rel physics:body1 = </Robot/C>
        bool physics:collisionEnabled = true
    }}
}}
"""

    def check(binding):
        shape_by_path = {path: i for i, path in enumerate(binding.model.shape_label)}
        expected = {
            tuple(sorted((shape_by_path[a], shape_by_path[b])))
            for a, b in (
                ("/Robot/A", "/Robot/B"),
                ("/Robot/A", "/Robot/C"),
                ("/Robot/B", "/Robot/C"),
            )
        }
        actual = {tuple(sorted(pair)) for pair in binding.model.shape_collision_filter_pairs}
        assert expected.issubset(actual) is filtered

    _binding_from_string(usda, f"articulation-self-collision-{value}", check)


def test_newton_articulation_root_rejects_ambiguous_disconnected_components():
    usda = """#usda 1.0
def Xform "Robot" (prepend apiSchemas = ["NewtonArticulationRootAPI"]) {
    bool newton:selfCollisionEnabled = false
    def Xform "A" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def Xform "B" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def Xform "C" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def Xform "D" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def PhysicsFixedJoint "AB" {
        rel physics:body0 = </Robot/A>
        rel physics:body1 = </Robot/B>
    }
    def PhysicsFixedJoint "CD" {
        rel physics:body0 = </Robot/C>
        rel physics:body1 = </Robot/D>
    }
}
"""
    with pytest.raises(ovnewton.InvalidPhysicsError, match="multiple disconnected joint components"):
        _ovstage_model_from_string(usda, "ambiguous-articulation-root")


def test_authored_body_velocity_initializes_newton_com_twist():
    usda = """#usda 1.0
def Cube "Body" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "PhysicsMassAPI"]
) {
    double3 xformOp:rotateXYZ = (0, 90, 0)
    uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    vector3f physics:velocity = (1, 2, 3)
    vector3f physics:angularVelocity = (0, 0, 90)
    point3f physics:centerOfMass = (1, 0, 0)
    float physics:mass = 1
    float3 physics:diagonalInertia = (1, 1, 1)
}
"""

    def check(binding):
        qd = binding.model.body_qd.numpy()[0]
        expected = np.asarray([3.0, 2.0, -1.0, np.pi / 2.0, 0.0, 0.0])
        np.testing.assert_allclose(qd, expected, atol=1.0e-6)
        np.testing.assert_allclose(binding.model.joint_qd.numpy(), expected, atol=1.0e-6)
        state = binding.model.state()
        newton.eval_fk(binding.model, binding.model.joint_q, binding.model.joint_qd, state)
        np.testing.assert_allclose(state.body_qd.numpy()[0], expected, atol=1.0e-6)

    _binding_from_string(usda, "authored-body-velocity", check)


def test_starts_asleep_is_nonfatal_and_reported(import_log):
    usda = """#usda 1.0
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    bool physics:startsAsleep = true
}
"""

    def check(binding):
        assert binding.model.body_count == 1
        assert _diagnostic_pairs(import_log) == [("starts-asleep-ignored", "/Body")]

    _binding_from_string(usda, "starts-asleep-policy", check)


def test_excluded_loop_joint_stays_outside_articulation():
    usda = """#usda 1.0
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"]) {
    def Xform "A" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def Xform "B" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def PhysicsFixedJoint "Tree" {
        rel physics:body0 = </Robot/A>
        rel physics:body1 = </Robot/B>
    }
    def PhysicsFixedJoint "Loop" {
        rel physics:body0 = </Robot/A>
        rel physics:body1 = </Robot/B>
        bool physics:excludeFromArticulation = true
    }
}
"""

    def check(binding):
        membership = binding.model.joint_articulation.numpy()
        assert membership[binding.model.joint_label.index("/Robot/Tree")] >= 0
        assert membership[binding.model.joint_label.index("/Robot/Loop")] == -1

    _binding_from_string(usda, "excluded-loop-joint", check)


@pytest.mark.parametrize(
    "groups,blocked",
    [
        (
            '''def PhysicsCollisionGroup "GA" {
    rel collection:colliders:includes = </A>
}
def PhysicsCollisionGroup "GB" {
    rel collection:colliders:includes = </B>
}''',
            set(),
        ),
        (
            '''def PhysicsCollisionGroup "GA" {
    rel collection:colliders:includes = </A>
    rel physics:filteredGroups = </GB>
}
def PhysicsCollisionGroup "GB" {
    rel collection:colliders:includes = </B>
}
def PhysicsCollisionGroup "GC" {
    rel collection:colliders:includes = </C>
}''',
            {("/A", "/B")},
        ),
        (
            '''def PhysicsCollisionGroup "GA" {
    rel collection:colliders:includes = [</A>, </C>]
    rel physics:filteredGroups = [</GA>, </GB>]
}
def PhysicsCollisionGroup "GB" {
    rel collection:colliders:includes = </B>
}
def PhysicsCollisionGroup "GD" {
    rel collection:colliders:includes = </D>
    bool physics:invertFilteredGroups = true
    rel physics:filteredGroups = </GB>
}''',
            {("/A", "/B"), ("/A", "/C"), ("/B", "/C"), ("/A", "/D"), ("/C", "/D")},
        ),
        (
            '''def PhysicsCollisionGroup "GA" {
    rel collection:colliders:includes = </A>
    bool physics:invertFilteredGroups = true
    rel physics:filteredGroups = </GB>
}
def PhysicsCollisionGroup "GB" {
    rel collection:colliders:includes = </B>
}''',
            {("/A", "/C"), ("/A", "/D")},
        ),
        (
            '''def PhysicsCollisionGroup "GA" {
    rel collection:colliders:includes = </A>
    string physics:mergeGroup = "shared"
    rel physics:filteredGroups = </GC>
}
def PhysicsCollisionGroup "GB" {
    rel collection:colliders:includes = </B>
    string physics:mergeGroup = "shared"
}
def PhysicsCollisionGroup "GC" {
    rel collection:colliders:includes = </C>
}''',
            {("/A", "/C"), ("/B", "/C")},
        ),
        (
            '''def PhysicsCollisionGroup "GA" {
    rel collection:colliders:includes = </A>
    string physics:mergeGroup = "shared"
    bool physics:invertFilteredGroups = true
    rel physics:filteredGroups = [</GA>, </GC>]
}
def PhysicsCollisionGroup "GB" {
    rel collection:colliders:includes = </B>
    string physics:mergeGroup = "shared"
}
def PhysicsCollisionGroup "GC" {
    rel collection:colliders:includes = </C>
}''',
            {("/A", "/D"), ("/B", "/D")},
        ),
        (
            '''def PhysicsCollisionGroup "GA" {
    rel collection:colliders:includes = [</A>, </B>]
}
def PhysicsCollisionGroup "GB" {
    rel collection:colliders:includes = </A>
    rel physics:filteredGroups = </GC>
}
def PhysicsCollisionGroup "GC" {
    rel collection:colliders:includes = </C>
}''',
            {("/A", "/C")},
        ),
        (
            '''def PhysicsCollisionGroup "GA" {
    rel collection:colliders:includes = </A>
    rel physics:filteredGroups = </GB>
}
def PhysicsCollisionGroup "GB" {
    rel collection:colliders:includes = </B>
}
def Cube "E" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "PhysicsFilteredPairsAPI"]) {
    rel physics:filteredPairs = </C>
}''',
            {("/A", "/B"), ("/C", "/E")},
        ),
        (
            '''def PhysicsCollisionGroup "Unused" {
    bool physics:invertFilteredGroups = true
}''',
            set(),
        ),
    ],
    ids=[
        "unfiltered", "selective", "self-and-inverted", "ungrouped", "merged", "merged-inverted",
        "multiple", "pairs", "unused",
    ],
)
def test_collision_groups_preserve_usd_filter_rules(groups, blocked):
    # Test USD semantics directly: Newton 1.6's importer isolated all distinct
    # groups, so it is not a valid oracle for these filtering rules.
    usda = '#usda 1.0\n' + '\n'.join(
        f'def Cube "{name}" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {{}}'
        for name in "ABCD"
    ) + '\n' + groups
    model = _ovstage_model_from_string(usda, "collision-group-rules")
    assert set(model.shape_collision_group.numpy()) == {newton.ModelBuilder().default_shape_cfg.collision_group}
    labels = model.shape_label
    actual = {tuple(sorted((labels[a], labels[b]))) for a, b in model.shape_collision_filter_pairs}
    assert actual == blocked


@pytest.mark.parametrize("default_group", [0, -1, 5])
def test_collision_groups_preserve_builder_default(default_group):
    usda = '''#usda 1.0
def Cube "A" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {}
def PhysicsCollisionGroup "GA" {
    rel collection:colliders:includes = </A>
}
'''
    with ovstage.Stage("collision-group-default") as stage, ovstage.PathDictionary(stage) as pd:
        population.open_usd_from_string(stage, usda, ordinal=1, domains=PopulationDomain.ALL)
        stage.advance_write_floor(ordinal=1).wait()
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.collision_group = default_group
        _build.add_ovstage(builder, stage, pd)
        assert builder.shape_collision_group == [default_group]


@pytest.mark.parametrize("filtered", [False, True])
def test_collision_groups_control_contact_generation(filtered):
    filter_rule = "rel physics:filteredGroups = </GB>" if filtered else ""
    usda = f'''#usda 1.0
def Sphere "A" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {{}}
def Sphere "B" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {{
    double3 xformOp:translate = (1, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]
}}
def PhysicsCollisionGroup "GA" {{
    rel collection:colliders:includes = </A>
    {filter_rule}
}}
def PhysicsCollisionGroup "GB" {{
    rel collection:colliders:includes = </B>
}}
'''
    model = _ovstage_model_from_string(usda, "collision-group-contacts")
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)
    assert (int(contacts.rigid_contact_count.numpy()[0]) > 0) == (not filtered)


@pytest.mark.parametrize("filtered", [False, True])
def test_collision_groups_filter_all_convex_parts(filtered):
    filter_rule = "rel physics:filteredGroups = </GB>" if filtered else ""
    usda = f'''#usda 1.0
def Mesh "A" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]) {{
    uniform token subdivisionScheme = "none"
    uniform token physics:approximation = "convexDecomposition"
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                        (3, 0, 0), (4, 0, 0), (3, 1, 0), (3, 0, 1)]
    int[] faceVertexCounts = [3, 3, 3, 3, 3, 3, 3, 3]
    int[] faceVertexIndices = [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3,
                              4, 6, 5, 4, 5, 7, 4, 7, 6, 5, 6, 7]
}}
def Sphere "B" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {{}}
def PhysicsCollisionGroup "GA" {{
    rel collection:colliders:includes = </A>
    {filter_rule}
}}
def PhysicsCollisionGroup "GB" {{
    rel collection:colliders:includes = </B>
}}
'''
    model = _ovstage_model_from_string(usda, "collision-group-convex-parts")
    body = list(model.body_label).index("/A")
    parts = np.flatnonzero(model.shape_body.numpy() == body)
    # Two disconnected tetrahedra must produce multiple collision shapes.
    assert len(parts) >= 2
    target = list(model.shape_label).index("/B")
    pairs = {tuple(sorted(pair)) for pair in model.shape_collision_filter_pairs}
    for part in parts:
        assert (tuple(sorted((int(part), target))) in pairs) == filtered


def test_collision_groups_resolve_subtree_includes_and_excludes():
    usda = '''#usda 1.0
def Xform "Parent" {
    def Cube "A" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {}
    def Cube "B" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {}
}
def Cube "ParentSibling" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {}
def Cube "C" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {}
def PhysicsCollisionGroup "Group" {
    rel collection:colliders:includes = </Parent>
    rel collection:colliders:excludes = </Parent/B>
    rel physics:filteredGroups = </Other>
}
def PhysicsCollisionGroup "Other" {
    rel collection:colliders:includes = </C>
}
'''
    model = _ovstage_model_from_string(usda, "collision-group-subtree")
    labels = model.shape_label
    assert {tuple(sorted((labels[a], labels[b]))) for a, b in model.shape_collision_filter_pairs} == {
        ("/C", "/Parent/A")
    }


def test_collision_groups_match_add_usd(tmp_path):
    asset = tmp_path / "collision-groups.usda"
    asset.write_text(
        """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "World" {
    def Cube "A" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {}
    def Cube "B" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {}
    def Cube "C" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {}
    def PhysicsCollisionGroup "GroupA" {
        rel physics:filteredGroups = </World/GroupB>
        rel collection:colliders:includes = [</World/A>, </World/B>]
    }
    def PhysicsCollisionGroup "GroupB" {
        rel collection:colliders:includes = </World/C>
    }
}
""",
        encoding="utf-8",
    )
    ref = dh.reference_model(str(asset))
    model = _ovstage_model(str(asset), "collision-groups")
    mismatches = dh.compare_models(ref, dh._model_arrays(model), check_joints=False, check_shapes=True)
    assert not mismatches, "collision-group mismatches vs add_usd:\n  " + "\n  ".join(mismatches)


def test_collision_group_collection_resolution_requires_population():
    usda = """#usda 1.0
def Xform "World" {
    def Xform "ParentA" {
        def Cube "A" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {}
    }
    def Cube "B" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {}
    def Cube "C" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {}
    def PhysicsCollisionGroup "Group" {
        uniform token collection:colliders:expansionRule = "explicitOnly"
        rel collection:colliders:includes = [</World/ParentA>, </World/B>]
        rel physics:filteredGroups = </World/Other>
    }
    def PhysicsCollisionGroup "Other" {
        rel collection:colliders:includes = </World/C>
    }
}
"""
    model = _ovstage_model_from_string(usda, "collision-group-explicit-only")
    labels = model.shape_label
    pairs = {tuple(sorted((labels[a], labels[b]))) for a, b in model.shape_collision_filter_pairs}
    if pairs == {("/World/B", "/World/C"), ("/World/C", "/World/ParentA/A")}:
        pytest.xfail("ovpopulation does not provide resolved collision-group collection membership")
    assert pairs == {("/World/B", "/World/C")}


def test_filtered_pairs_match_add_usd(tmp_path):
    asset = tmp_path / "filtered-pairs.usda"
    asset.write_text(
        """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "World" {
    def Cube "A" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "PhysicsFilteredPairsAPI"]
    ) {
        rel physics:filteredPairs = </World/B>
    }
    def Cube "B" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {}
}
""",
        encoding="utf-8",
    )
    ref = dh.reference_model(str(asset))
    model = _ovstage_model(str(asset), "filtered-pairs")
    mismatches = dh.compare_models(ref, dh._model_arrays(model), check_joints=False, check_shapes=True)
    assert not mismatches, "filtered-pair mismatches vs add_usd:\n  " + "\n  ".join(mismatches)


@pytest.mark.parametrize("attribute", ["breakForce", "breakTorque"])
def test_joint_break_threshold_is_ignored_with_diagnostic(attribute, import_log):
    usda = f"""#usda 1.0
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {{}}
def PhysicsFixedJoint "Joint" {{
    rel physics:body1 = </Body>
    float physics:{attribute} = 10
}}
"""

    def check(binding):
        assert binding.model.joint_count == 1
        assert _diagnostic_pairs(import_log) == [
            ("joint-break-threshold-ignored", "/Joint")
        ]

    _binding_from_string(usda, f"ignored-{attribute}", check)


def test_unknown_mesh_approximation_retains_raw_mesh_with_diagnostic(import_log):
    usda = """#usda 1.0
def Mesh "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]) {
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    token physics:approximation = "futureApproximation"
}
"""

    def check(binding):
        assert binding.model.shape_count == 1
        assert _diagnostic_pairs(import_log) == [
            ("mesh-approximation-ignored", "/Collider")
        ]

    _binding_from_string(usda, "ignored-mesh-approximation", check)
