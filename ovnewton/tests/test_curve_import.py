# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import newton
import numpy as np
import ovstage
import pytest
import warp as wp

import ovnewton
from ovnewton._src import _build, _curves, _deformables

from .deformable_helpers import CABLE, populate, read_points


@pytest.mark.parametrize("collision_api", [False, True])
@pytest.mark.parametrize("caller_owned", [False, True])
def test_cable_load_step_and_local_publication(warp_device, caller_owned, collision_api):
    scene = CABLE
    if collision_api:
        scene = scene.replace(
            '"PhysicsCurvesDeformableSimAPI"', '"PhysicsCurvesDeformableSimAPI", "PhysicsCollisionAPI"'
        )
    with wp.ScopedDevice(warp_device), ovstage.Stage("cable-import") as stage:
        populate(stage, scene)
        if caller_owned:
            builder = newton.ModelBuilder()
            ovnewton.add_ovstage(builder, stage)
            builder.color()
            model = builder.finalize()
            del builder
            binding = ovnewton.attach_ovstage(stage, model=model)
        else:
            binding = ovnewton.attach_ovstage(stage)
            model = binding.model
        curves = _curves.model_curve_ranges(model)
        assert len(curves) == 1
        curve = curves[0]
        assert curve.body_end - curve.body_start == 2
        assert model.particle_count == 0
        assert model.shape_count == 3  # One rigid cube and two cable segments.
        np.testing.assert_allclose(model.body_com.numpy()[curve.body_start : curve.body_end], 0)
        np.testing.assert_allclose(model.body_qd.numpy()[curve.body_start : curve.body_end, :3], [[0, 1, 0]] * 2)
        # Canonical EA is divided by the joint's dual length, not treated as a per-joint gain.
        rod_joints = np.flatnonzero(model.joint_type.numpy() == int(newton.JointType.ROD))
        assert len(rod_joints) == 1
        dof = model.joint_qd_start.numpy()[rod_joints[0]]
        assert model.joint_target_ke.numpy()[dof + 2] == pytest.approx(60 / 1.5)
        state, output = model.state(), model.state()
        solver = newton.solvers.SolverVBD(model, iterations=2)
        initial = state.body_q.numpy().copy()
        solver.step(state, output, model.control(), None, 1e-3)
        assert np.all(np.isfinite(output.body_q.numpy()))
        assert np.any(output.body_q.numpy() != initial)
        # A stage resync must not overwrite generated rod bodies or joint coordinates.
        body_before = output.body_q.numpy().copy()
        velocity_before = output.body_qd.numpy().copy()
        q_before = output.joint_q.numpy().copy()
        binding.update_from_ovstage(output, model.control(), ordinal=1)
        np.testing.assert_array_equal(
            output.body_q.numpy()[curve.body_start : curve.body_end], body_before[curve.body_start : curve.body_end]
        )
        np.testing.assert_array_equal(
            output.body_qd.numpy()[curve.body_start : curve.body_end],
            velocity_before[curve.body_start : curve.body_end],
        )
        q_start = model.joint_q_start.numpy()
        np.testing.assert_array_equal(
            output.joint_q.numpy()[q_start[curve.joint_start] : q_start[curve.joint_end]],
            q_before[q_start[curve.joint_start] : q_start[curve.joint_end]],
        )
        binding.update_to_ovstage(output, ordinal=2)
        stage.advance_write_floor(2).wait()
        assert np.all(np.isfinite(read_points(stage, "/Cable/Sim", 2)))
        state = model.state()
        for ordinal in (3, 4):
            moved = state.body_q.numpy()
            moved[curve.body_start : curve.body_end, 2] += 1
            state.body_q.assign(moved)
            binding.update_to_ovstage(state, ordinal=ordinal)
            stage.advance_write_floor(ordinal).wait()
        np.testing.assert_allclose(
            read_points(stage, "/Cable/Sim", 4), [[-2, 0, 0], [-2, 0, 1], [-2, 0, 3]], atol=1e-6
        )


def test_multiple_curves_in_one_prim_publish_independent_endpoints(warp_device):
    scene = CABLE.replace("int[] curveVertexCounts = [3]", "int[] curveVertexCounts = [3, 3]")
    scene = scene.replace(
        "[(0, 0, 0), (0, 0, 1), (0, 0, 3)]", "[(0, 0, 0), (0, 0, 1), (0, 0, 3), (0, 2, 0), (0, 2, 2), (0, 2, 3)]"
    )
    scene = "\n".join(line for line in scene.splitlines() if "normals =" not in line and "velocities =" not in line)
    with wp.ScopedDevice(warp_device), ovstage.Stage("multiple-curves") as stage:
        populate(stage, scene)
        expected = read_points(stage, "/Cable/Sim", 1).copy()
        binding = ovnewton.attach_ovstage(stage)
        assert len(_curves.model_curve_ranges(binding.model)) == 2
        binding.update_to_ovstage(binding.model.state(), ordinal=2)
        stage.advance_write_floor(2).wait()
        np.testing.assert_allclose(read_points(stage, "/Cable/Sim", 2), expected, atol=1e-6)


@pytest.mark.parametrize(
    "problem", ["mass", "kinematic", "scale", "zero-segment", "normals", "velocity", "density", "stiffness"]
)
def test_cable_rejects_unsupported_or_invalid_inputs_before_mutation(problem):
    with ovstage.Stage("invalid-cable") as stage, ovstage.PathDictionary(stage) as pd:
        populate(stage, CABLE)
        value = _deformables.read_deformables(stage, pd, 1)[0]
    if problem == "mass":
        value = replace(value, mass=1)
    elif problem == "kinematic":
        value = replace(value, kinematic_enabled=True)
    elif problem == "scale":
        matrix = value.world_matrix.copy()
        matrix[:3, :3] *= 2
        value = replace(value, world_matrix=matrix)
    elif problem == "zero-segment":
        value = replace(value, points=np.zeros_like(value.points))
    elif problem == "normals":
        value = replace(value, payload=replace(value.payload, normals=np.tile([0, 0, 1], (3, 1))))
    elif problem == "velocity":
        value = replace(value, velocities=np.eye(3))
    elif problem == "density":
        value = replace(value, density=-1)
    else:
        value = replace(value, material={**value.material, "curvesStretchStiffness": float("nan")})
    builder = newton.ModelBuilder()
    with pytest.raises((ovnewton.InvalidPhysicsError, ovnewton.UnsupportedPhysicsError)):
        _build.build_deformables(builder, [value])
    assert builder.body_count == builder.joint_count == builder.particle_count == 0


def test_separate_rest_shape_is_rejected():
    scene = CABLE.replace(
        "point3f[] points =",
        "point3f[] physics:restShapePoints = [(0, 0, 0), (0, 0, 1), (0, 0, 2)]\n        point3f[] points =",
    )
    with ovstage.Stage("cable-rest-shape") as stage:
        populate(stage, scene)
        with pytest.raises(ovnewton.UnsupportedPhysicsError, match="separate deformable rest geometry"):
            ovnewton.attach_ovstage(stage)


def test_cable_collision_disabling_is_rejected():
    scene = CABLE.replace(
        'uniform token type = "linear"', 'bool physics:collisionEnabled = false\n        uniform token type = "linear"'
    )
    with ovstage.Stage("cable-collision-mode") as stage:
        populate(stage, scene)
        with pytest.raises(ovnewton.UnsupportedPhysicsError, match="collision disabling"):
            ovnewton.attach_ovstage(stage)


@pytest.mark.parametrize("point_count", [3, 5])
def test_cable_density_mass_does_not_change_with_subdivision(point_count):
    with ovstage.Stage("cable-mass") as stage, ovstage.PathDictionary(stage) as pd:
        populate(stage, CABLE)
        value = _deformables.read_deformables(stage, pd, 1)[0]
    points = np.column_stack((np.zeros(point_count), np.zeros(point_count), np.linspace(0, 1, point_count)))
    value = replace(
        value,
        points=points,
        velocities=np.empty((0, 3)),
        payload=replace(value.payload, curve_vertex_counts=np.array([point_count]), normals=np.empty((0, 3))),
    )
    builder = newton.ModelBuilder()
    _build.build_deformables(builder, [value])
    expected = 80 * np.pi * 0.02**2
    assert sum(builder.body_mass) == pytest.approx(expected)
    model = builder.finalize(device="cpu")
    np.testing.assert_allclose(model.body_inv_mass.numpy(), 1 / model.body_mass.numpy())
    np.testing.assert_allclose(
        model.body_inertia.numpy() @ model.body_inv_inertia.numpy(),
        np.broadcast_to(np.eye(3), (point_count - 1, 3, 3)),
        atol=1e-6,
    )


@pytest.mark.parametrize("meters", [1, 0.01])
def test_cable_defaults_follow_stage_distance_units(meters):
    scene = CABLE.replace("metersPerUnit = 1", f"metersPerUnit = {meters}")
    scene = "\n".join(
        line
        for line in scene.splitlines()
        if not any(name in line for name in ("physics:density", "youngsModulus", "curvesStretchStiffness"))
    )
    with ovstage.Stage("cable-units") as stage:
        populate(stage, scene)
        model = ovnewton.attach_ovstage(stage).model
    curve = _curves.model_curve_ranges(model)[0]
    masses = model.body_mass.numpy()[curve.body_start : curve.body_end]
    np.testing.assert_allclose(masses, 1000 * meters**3 * np.pi * 0.02**2 * np.array([1, 2]), rtol=1e-6)
    joint = np.flatnonzero(model.joint_type.numpy() == int(newton.JointType.ROD))[0]
    axial = model.joint_qd_start.numpy()[joint] + 2
    assert model.joint_target_ke.numpy()[axial] == pytest.approx(1e6 * meters * np.pi * 0.02**2 / 1.5)
