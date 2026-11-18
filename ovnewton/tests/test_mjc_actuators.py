# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo scalar position and velocity actuators compared with Newton's USD importer."""

import newton
import numpy as np
import ovstage
import pytest
import warp as wp
from newton.usd import SchemaResolverMjc, SchemaResolverNewton

import ovnewton

from . import diff_harness as dh
from .test_mjc_joints import JOINT, _populate

ACTUATOR = """
def MjcActuator "Motor" {
    double[] mjc:gainPrm = [75, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    double[] mjc:biasPrm = [0, -75, -5, 0, 0, 0, 0, 0, 0, 0]
    token mjc:biasType = "affine"
    double mjc:inheritRange = 1
    rel mjc:target = </Joint>
}
"""


def _import_builder(scene):
    with wp.ScopedDevice("cpu"), ovstage.Stage("mjc-actuator") as stage:
        _populate(stage, scene)
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        ovnewton.add_ovstage(builder, stage)
    return builder


def _models(tmp_path, scene, device="cpu"):
    asset = tmp_path / "actuator.usda"
    asset.write_text(scene)
    reference = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(reference)
    reference.add_usd(str(asset), schema_resolvers=[SchemaResolverNewton(), SchemaResolverMjc()])
    return [builder.finalize(device=device) for builder in (reference, _import_builder(scene))]


@pytest.mark.parametrize("joint_type", ["PhysicsRevoluteJoint", "PhysicsPrismaticJoint"])
@pytest.mark.parametrize("kind", ["position", "velocity", "combined", "dampratio"])
@pytest.mark.parametrize("limits", ["inherited", "explicit", "disabled"])
def test_scalar_mjc_actuators_match_add_usd(tmp_path, joint_type, kind, limits, warp_device):
    actuator = ACTUATOR
    velocity = ACTUATOR.replace('"Motor"', '"Velocity"').replace("[0, -75, -5,", "[0, 0, -75,")
    if kind == "dampratio":
        actuator = actuator.replace("[0, -75, -5,", "[0, -75, 1,")
    elif kind == "velocity":
        actuator = velocity
    elif kind == "combined":
        actuator += velocity
    if limits != "inherited":
        actuator = actuator.replace(
            "double mjc:inheritRange = 1",
            """double mjc:ctrlRange:min = -0.3
    double mjc:ctrlRange:max = 0.4
    double mjc:forceRange:min = -7
    double mjc:forceRange:max = 9""",
        )
        if limits == "disabled":
            actuator = actuator.replace(
                "token mjc:biasType",
                'token mjc:ctrlLimited = "false"\n    token mjc:forceLimited = "false"\n    token mjc:biasType',
            )
    scene = JOINT.replace("PhysicsRevoluteJoint", joint_type) + actuator
    with wp.ScopedDevice(warp_device):
        reference, actual = _models(tmp_path, scene, device=warp_device)
        mismatches = dh.compare_models(
            dh._model_arrays(reference), dh._model_arrays(actual), atol=0, rtol=1e-7
        )
        assert not mismatches, "\n".join(mismatches)
        assert actual.custom_frequency_counts["mujoco:actuator"] == reference.custom_frequency_counts["mujoco:actuator"]
        np.testing.assert_allclose(actual.mujoco.actuator_trnid.numpy(), reference.mujoco.actuator_trnid.numpy())
        pytest.importorskip("mujoco")
        pytest.importorskip("mujoco_warp")
        solvers = [newton.solvers.SolverMuJoCo(model) for model in (reference, actual)]
        for field in (
            "actuator_gainprm",
            "actuator_biasprm",
            "actuator_ctrlrange",
            "actuator_forcerange",
            "actuator_ctrllimited",
            "actuator_forcelimited",
        ):
            np.testing.assert_allclose(
                getattr(solvers[1].mjw_model, field).numpy(),
                getattr(solvers[0].mjw_model, field).numpy(),
                err_msg=field,
            )


@pytest.mark.parametrize(
    "replacement",
    [
        ('token mjc:biasType = "affine"', 'token mjc:biasType = "affine"\n    token mjc:dynType = "filter"'),
        ("double mjc:inheritRange = 1", "double[] mjc:gear = [2, 0, 0, 0, 0, 0]"),
        ("[0, -75, -5,", "[1, -75, -5,"),
    ],
)
def test_unsupported_actuator_is_rejected(replacement):
    with pytest.raises(ovnewton.UnsupportedPhysicsError):
        _import_builder(JOINT + ACTUATOR.replace(*replacement))


def test_actuator_requires_solver_registration():
    with ovstage.Stage("unregistered-actuator") as stage:
        _populate(stage, JOINT + ACTUATOR)
        with pytest.raises(ovnewton.UnsupportedPhysicsError, match="register_custom_attributes"):
            ovnewton.add_ovstage(newton.ModelBuilder(), stage)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("joint_type", ["PhysicsRevoluteJoint", "PhysicsPrismaticJoint"])
def test_inherited_range_includes_joint_reference(tmp_path, reverse, joint_type):
    scene = (
        JOINT.replace("PhysicsRevoluteJoint", joint_type).replace(
            "double mjc:armature", "double mjc:ref = 0.2\n    double mjc:armature"
        )
        + ACTUATOR
    )
    if reverse:
        scene = scene.replace("physics:body1", "physics:body0")
    with wp.ScopedDevice("cpu"):
        reference, actual = _models(tmp_path, scene)
        np.testing.assert_allclose(
            actual.mujoco.actuator_ctrlrange.numpy(), reference.mujoco.actuator_ctrlrange.numpy()
        )
        np.testing.assert_allclose(actual.joint_target_ke.numpy(), reference.joint_target_ke.numpy())


def test_duplicate_position_actuators_are_rejected():
    scene = JOINT + ACTUATOR + ACTUATOR.replace('"Motor"', '"OtherMotor"')
    with pytest.raises(ovnewton.UnsupportedPhysicsError, match="same kind"):
        _import_builder(scene)


def test_corpus_comparison_detects_missing_actuators_and_changed_limits(tmp_path):
    with wp.ScopedDevice("cpu"):
        reference, actual = _models(tmp_path, JOINT + ACTUATOR)
    ref, ours = dh._model_arrays(reference), dh._model_arrays(actual)
    assert not dh.compare_models(ref, ours, check_joints=True)
    ours["mjc_actuators"]["/Motor"]["actuator_ctrlrange"] = np.asarray([-10, 10])
    assert any("actuator_ctrlrange" in issue for issue in dh.compare_models(ref, ours, check_joints=True))
    ours["mjc_actuators"] = {}
    assert any("actuator labels" in issue for issue in dh.compare_models(ref, ours, check_joints=True))


@pytest.mark.parametrize("dampratio", [False, True])
def test_actuator_activation_dimensions_are_rejected(dampratio):
    actuator = ACTUATOR.replace("double mjc:inheritRange = 1", "int mjc:actDim = 1")
    if dampratio:
        actuator = actuator.replace("[0, -75, -5,", "[0, -75, 1,")
    with pytest.raises(ovnewton.UnsupportedPhysicsError, match="unsupported mjc:actDim"):
        _import_builder(JOINT + actuator)


@pytest.mark.parametrize("instance_count", [1, 2])
def test_native_actuator_instances_are_rejected(instance_count):
    scene = JOINT + 'class Xform "Template" {\n' + ACTUATOR + "}\n"
    for index in range(instance_count):
        scene += f"""def Xform "Instance{index}" (
    instanceable = true
    prepend references = </Template>
) {{}}
"""
    with pytest.raises(ovnewton.UnsupportedPhysicsError, match="native USD actuator instancing"):
        _import_builder(scene)


@pytest.mark.parametrize(
    "target,disabled,error,message",
    [
        ("/Missing", False, ovnewton.InvalidPhysicsError, "missing target /Missing"),
        ("/Body", False, ovnewton.UnsupportedPhysicsError, "unmerged scalar joint"),
        ("/Joint", True, ovnewton.UnsupportedPhysicsError, "unmerged scalar joint"),
    ],
)
def test_actuator_target_errors_are_distinct(target, disabled, error, message):
    scene = JOINT + ACTUATOR.replace("</Joint>", f"<{target}>")
    if disabled:
        scene = scene.replace("rel physics:body1", "bool physics:jointEnabled = false\n    rel physics:body1")
    with pytest.raises(error, match=message):
        _import_builder(scene)


@pytest.mark.parametrize("dampratio,target", [(False, "other"), (True, "other"), (False, "unresolved")])
def test_corpus_comparison_detects_wrong_actuator_binding(tmp_path, dampratio, target):
    other_joint = (
        JOINT[JOINT.index("def Xform") :]
        .replace('"Body"', '"OtherBody"')
        .replace('"Joint"', '"OtherJoint"')
        .replace("</Body>", "</OtherBody>")
    )
    actuator = ACTUATOR.replace("[0, -75, -5,", "[0, -75, 1,") if dampratio else ACTUATOR
    with wp.ScopedDevice("cpu"):
        reference, actual = _models(tmp_path, JOINT + other_joint + actuator)
    ref = dh._model_arrays(reference)
    assert not dh.compare_models(ref, dh._model_arrays(actual), check_joints=True)
    transmission = actual.mujoco.actuator_trnid.numpy().copy()
    transmission[0, 0] = (
        int(actual.joint_qd_start.numpy()[actual.joint_label.index("/OtherJoint")]) if target == "other" else -1
    )
    actual.mujoco.actuator_trnid.assign(transmission)
    assert any(
        "/Motor.binding" in issue for issue in dh.compare_models(ref, dh._model_arrays(actual), check_joints=True)
    )


@pytest.mark.parametrize("deferred", [False, True])
def test_corpus_actuator_binding_ignores_joint_order(deferred):
    models = []
    with wp.ScopedDevice("cpu"):
        for index, names in enumerate((("Joint", "OtherJoint"), ("OtherJoint", "Joint"))):
            builder = newton.ModelBuilder()
            newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
            for name in names:
                body = builder.add_link(label=f"/{name}Body")
                joint = builder.add_joint_revolute(parent=-1, child=body, label=f"/{name}")
                dof = builder.joint_qd_start[joint]
                builder.custom_attributes["mujoco:joint_dof_label"].values[dof] = f"/{name}"
            target_dof = builder.joint_qd_start[builder.joint_label.index("/Joint")]
            builder.add_custom_values(
                **{
                    "mujoco:actuator_label": "/Motor",
                    "mujoco:actuator_target_label": "/Joint",
                    "mujoco:actuator_trnid": wp.vec2i(-1, -1) if deferred and index else wp.vec2i(target_dof, 0),
                }
            )
            models.append(builder.finalize())
    assert models[0].joint_label != models[1].joint_label
    reference, actual = [dh._model_arrays(model) for model in models]
    assert reference["mjc_actuators"]["/Motor"]["binding"] == (0, "/Joint", 0)
    assert not dh.compare_models(reference, actual, check_joints=True)


@pytest.mark.parametrize("kind", ["position", "velocity", "dampratio"])
@pytest.mark.parametrize("joint_type", ["PhysicsRevoluteJoint", "PhysicsPrismaticJoint"])
def test_actuators_import_from_populated_ovstage_without_usd_fallback(monkeypatch, kind, joint_type):
    from pxr import Usd

    def unexpected_usd_import(*args, **kwargs):
        raise AssertionError("add_ovstage must consume the populated stage")

    actuator = ACTUATOR
    if kind == "velocity":
        actuator = actuator.replace("[0, -75, -5,", "[0, 0, -75,")
    elif kind == "dampratio":
        actuator = actuator.replace("[0, -75, -5,", "[0, -75, 1,")
    with wp.ScopedDevice("cpu"), ovstage.Stage("ovstage-actuator-import") as stage:
        _populate(stage, JOINT.replace("PhysicsRevoluteJoint", joint_type) + actuator)
        monkeypatch.setattr(newton.ModelBuilder, "add_usd", unexpected_usd_import)
        monkeypatch.setattr(Usd.Stage, "Open", unexpected_usd_import)
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        ovnewton.add_ovstage(builder, stage)
        model = builder.finalize()
        assert model.custom_frequency_counts["mujoco:actuator"] == 1
        assert model.mujoco.actuator_label == ["/Motor"]
        assert ovnewton.attach_ovstage(stage, model=model).model is model


@pytest.mark.parametrize("explicit_dynamics", [False, True])
def test_mixed_joint_defaults_import_without_usd_fallback(monkeypatch, explicit_dynamics):
    def unexpected_usd_import(*args, **kwargs):
        raise AssertionError("mixed joint defaults must not switch importers")

    scene = JOINT.replace('"MjcJointAPI"', '"MjcJointAPI", "NewtonJointAPI"')
    if explicit_dynamics:
        scene = scene.replace(
            "double mjc:armature",
            "float newton:armature = 1\n    float newton:damping = 2\n"
            "    float newton:friction = 3\n    double mjc:armature",
        )
    with wp.ScopedDevice("cpu"), ovstage.Stage("mixed-ovstage-actuator") as stage:
        _populate(stage, scene + ACTUATOR)
        monkeypatch.setattr(newton.ModelBuilder, "add_usd", unexpected_usd_import)
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        ovnewton.add_ovstage(builder, stage)
        model = builder.finalize()
        np.testing.assert_allclose(model.joint_armature.numpy(), [1 if explicit_dynamics else 0.2])
        np.testing.assert_allclose(model.joint_damping.numpy(), [2 * 180 / np.pi if explicit_dynamics else 3])
        np.testing.assert_allclose(model.joint_friction.numpy(), [3 if explicit_dynamics else 0.4])
        np.testing.assert_array_equal(model.mujoco.solreflimit_mode.numpy(), [2])
        assert model.mujoco.actuator_label == ["/Motor"]
