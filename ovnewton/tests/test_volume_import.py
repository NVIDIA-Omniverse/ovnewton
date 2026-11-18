# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import newton
import numpy as np
import ovstage
import pytest
import warp as wp

import ovnewton
from ovnewton._src import _build, _deformables

from .deformable_helpers import VOLUME, populate, read_points


@pytest.mark.parametrize("collision_api", [False, True])
@pytest.mark.parametrize("caller_owned", [False, True])
def test_volume_import_native_parameters_step_and_local_publication(warp_device, caller_owned, collision_api):
    second = (
        VOLUME[VOLUME.index('def Xform "Volume"') :].replace('"Volume"', '"Other"').replace("(2, 3, 4)", "(0, 0, 0)")
    )
    scene = VOLUME + second
    if collision_api:
        scene = scene.replace(
            '"PhysicsVolumeDeformableSimAPI"', '"PhysicsVolumeDeformableSimAPI", "PhysicsCollisionAPI"'
        )
    with wp.ScopedDevice(warp_device), ovstage.Stage("volumes") as stage:
        populate(stage, scene)
        if caller_owned:
            builder = newton.ModelBuilder()
            ovnewton.add_ovstage(builder, stage)
            builder.color(include_bending=True)
            model = builder.finalize()
            del builder
            binding = ovnewton.attach_ovstage(stage, model=model)
        else:
            binding = ovnewton.attach_ovstage(stage)
        model = binding.model
        assert model.particle_count == 8 and model.tet_count == 2
        assert model.body_count == model.shape_count == 0
        np.testing.assert_allclose(model.particle_mass.numpy(), 2.5)
        np.testing.assert_allclose(model.tet_materials.numpy()[:, :2], [[480, 480]] * 2)
        state, output = model.state(), model.state()
        solver = newton.solvers.SolverXPBD(model)
        solver.step(state, output, model.control(), None, 1 / 1000)
        assert np.all(np.isfinite(output.particle_q.numpy()))
        state = model.state()
        state.particle_q.assign(state.particle_q.numpy() + [0, 0, 2])
        expected = [[0, 0, 2], [1, 0, 2], [0, 1, 2], [0, 0, 3]]
        binding.update_to_ovstage(state, ordinal=2)
        stage.advance_write_floor(2).wait()
        # Transformed and identity volumes share one publication batch.
        for path in ("/Volume/Sim", "/Other/Sim"):
            np.testing.assert_allclose(read_points(stage, path, 2), expected)


@pytest.mark.parametrize(
    "problem", ["inverted", "degenerate", "duplicate", "material", "mass", "scale", "precision", "elastic-overflow"]
)
def test_volume_rejects_invalid_or_unsupported_inputs_before_mutation(problem):
    with ovstage.Stage("volume-rejections") as stage, ovstage.PathDictionary(stage) as pd:
        populate(stage, VOLUME)
        value = _deformables.read_deformables(stage, pd, 1)[0]
    if problem in ("inverted", "degenerate", "duplicate"):
        tets = {"inverted": [[0, 2, 1, 3]], "degenerate": [[0, 1, 1, 3]], "duplicate": [[0, 1, 2, 3], [0, 1, 2, 3]]}[
            problem
        ]
        value = replace(value, payload=replace(value.payload, tet_vertex_indices=np.asarray(tets)))
    elif problem == "material":
        value = replace(value, material={**value.material, "poissonsRatio": 0.5})
    elif problem == "elastic-overflow":
        value = replace(value, material={**value.material, "youngsModulus": 3e38, "poissonsRatio": 0.49})
    elif problem == "precision":
        matrix = value.world_matrix.copy()
        matrix[3, :3] = 1e8
        value = replace(value, world_matrix=matrix)
    elif problem == "mass":
        value = replace(value, mass=1)
    else:
        matrix = value.world_matrix.copy()
        matrix[0, 0] = 2
        value = replace(value, world_matrix=matrix)
    builder = newton.ModelBuilder()
    with pytest.raises((ovnewton.InvalidPhysicsError, ovnewton.UnsupportedPhysicsError)):
        _build.build_deformables(builder, [value])
    assert builder.particle_count == builder.tet_count == 0


@pytest.mark.parametrize("attribute", ["points", "velocities"])
def test_nonfinite_volume_vectors_are_rejected(attribute):
    scene = VOLUME.replace(
        "        int4[] tetVertexIndices",
        "        vector3f[] velocities = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]\n        int4[] tetVertexIndices",
    )
    if attribute == "points":
        scene = scene.replace("points = [(0, 0, 0)", "points = [(nan, 0, 0)")
    else:
        scene = scene.replace("velocities = [(0, 0, 0)", "velocities = [(nan, 0, 0)")
    with ovstage.Stage("nonfinite-volume") as stage:
        populate(stage, scene)
        with pytest.raises(ovnewton.InvalidPhysicsError, match="must be finite"):
            ovnewton.attach_ovstage(stage)


@pytest.mark.parametrize("field", ["points", "velocities", "mass"])
def test_volume_storage_overflow_is_rejected_before_adding_any_geometry(field):
    with ovstage.Stage("volume-storage") as stage, ovstage.PathDictionary(stage) as pd:
        populate(stage, VOLUME)
        value = _deformables.read_deformables(stage, pd, 1)[0]
    matrix = np.eye(4)
    if field == "points":
        matrix[3, 0] = 1e40
        invalid = replace(value, world_matrix=matrix)
    elif field == "velocities":
        angle = np.pi / 4
        matrix[:2, :2] = [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        invalid = replace(value, world_matrix=matrix, velocities=np.tile([3e38, 3e38, 0], (len(value.points), 1)))
    else:
        invalid = replace(value, density=3e38, points=value.points * 10)
    builder = newton.ModelBuilder()
    with pytest.raises(ovnewton.InvalidPhysicsError, match="finite float32 storage"):
        _build.build_deformables(builder, [value, invalid])
    assert builder.particle_count == builder.tet_count == 0


@pytest.mark.parametrize("meters", [1, 0.01])
def test_volume_default_density_follows_stage_distance_units(meters):
    scene = VOLUME.replace("#usda 1.0", f"#usda 1.0\n( metersPerUnit = {meters} )")
    scene = scene.replace("    float physics:density = 60\n", "")
    with ovstage.Stage("volume-units") as stage:
        populate(stage, scene)
        model = ovnewton.attach_ovstage(stage).model
    assert sum(model.particle_mass.numpy()) == pytest.approx(newton.ModelBuilder().default_tet_density * meters**3 / 6)
