# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import numpy as np
import pytest

from ovnewton._src import _build, _deformables

from .deformable_helpers import CABLE, VOLUME, populate, read_points, read_test_deformables, supported_surface


@pytest.mark.parametrize("family", ["cable", "cloth", "volume"])
def test_imported_deformable_builder_outlives_stage(family, warp_device):
    import newton
    import ovstage

    import ovnewton

    cloth = """#usda 1.0
def Xform "Cloth" (prepend apiSchemas = ["PhysicsDeformableBodyAPI"]) {
    float physics:density = 80
    double3 xformOp:translate = (2, 3, 4)
    uniform token[] xformOpOrder = ["xformOp:translate"]
    def Mesh "Sim" (prepend apiSchemas = ["PhysicsSurfaceDeformableSimAPI"]) {
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        vector3f[] velocities = [(0, 0, 1), (0, 0, 2), (0, 0, 3)]
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        float[] physics:thicknesses = [0.01]
        uniform token physics:thicknesses:elementType = "constant"
    }
}
"""
    builder = newton.ModelBuilder()
    with ovstage.Stage("deformable-snapshot-lifetime") as stage:
        populate(stage, {"cable": CABLE, "cloth": cloth, "volume": VOLUME}[family])
        ovnewton.add_ovstage(builder, stage)

    # Finalization must not need the now-closed stage or its borrowed columns.
    builder.color(include_bending=True)
    model = builder.finalize(device=warp_device)
    if family == "cable":
        rods = np.flatnonzero(model.joint_type.numpy() == int(newton.JointType.ROD))
        assert len(rods) == 1
        assert model.body_count == 3
        assert model.particle_count == 0
        # Native rod body frames are at the transformed segment midpoints.
        bodies = np.flatnonzero(np.any(model.body_qd.numpy()[:, :3] != 0, axis=1))
        np.testing.assert_allclose(model.body_q.numpy()[bodies, :3], [[2.5, 3, 4], [4, 3, 4]], atol=1e-6)
        np.testing.assert_allclose(model.body_qd.numpy()[bodies, :3], [[0, 1, 0]] * 2)
    elif family == "cloth":
        assert model.tri_count == 1
        np.testing.assert_allclose(model.particle_q.numpy(), [[2, 3, 4], [3, 3, 4], [2, 4, 4]])
        np.testing.assert_allclose(model.particle_qd.numpy(), [[0, 0, 1], [0, 0, 2], [0, 0, 3]])
        assert sum(model.particle_mass.numpy()) == pytest.approx(0.4)
    else:
        assert model.tet_count == 1
        np.testing.assert_allclose(model.particle_q.numpy(), [[2, 3, 4], [3, 3, 4], [2, 4, 4], [2, 3, 5]])
        np.testing.assert_allclose(model.particle_mass.numpy(), [2.5] * 4)


def test_reads_all_canonical_deformable_families():
    values = read_test_deformables()

    assert [value.family.value for value in values] == ["curves", "surface", "volume"]
    assert values[0].body_path == "/World/Cable"
    assert values[0].material_path == "/World/Material"
    assert values[0].masses.element_type == "point"
    assert values[0].thicknesses.element_type == "constant"
    assert values[0].geometry_path == "/World/Cable/Sim"
    assert values[1].payload.face_vertex_indices.tolist() == [0, 1, 2]
    assert values[2].payload.tet_vertex_indices.tolist() == [[0, 1, 2, 3]]
    assert values[2].velocities.tolist()[-1] == [0.0, 0.0, 4.0]


def test_rejects_wrong_element_frequency_length():
    with pytest.raises(ValueError, match="point count 3"):
        _deformables._element_field("masses", [1.0, 2.0], "point", {"point": 3}, path="/World/Surface/Sim")


@pytest.mark.parametrize("problem", ["degenerate", "body-density", "material-density"])
def test_invalid_surface_does_not_append_an_earlier_cable(problem):
    import newton

    cable, surface, _ = read_test_deformables()
    cable = replace(cable, material={}, masses=_deformables.ElementField(np.empty(0, dtype=np.float32), ""))
    surface = supported_surface(surface)
    if problem == "degenerate":
        surface = replace(surface, points=np.zeros_like(surface.points))
        error, message = _deformables.OvstageContractError, "zero-area triangle"
    else:
        if problem == "body-density":
            surface = replace(surface, density=0)
        else:
            surface = replace(surface, material={"density": -1})
        error, message = _deformables.InvalidPhysicsError, "density must be finite and positive"
    builder = newton.ModelBuilder()
    with pytest.raises(error, match=message):
        _build.build_deformables(builder, [cable, surface])
    assert builder.body_count == builder.joint_count == builder.particle_count == 0
    assert builder.curve_label == builder.surface_label == []


@pytest.mark.parametrize("compose_builder", [False, True])
def test_cables_and_volumes_share_a_binding(warp_device, compose_builder):
    import newton
    import ovstage
    import warp as wp

    import ovnewton

    volume_scene = VOLUME.removeprefix("#usda 1.0\n").replace('"Material"', '"VolumeMaterial"')
    volume_scene = volume_scene.replace("</Material>", "</VolumeMaterial>")
    with wp.ScopedDevice(warp_device), ovstage.Stage("mixed-deformables") as stage:
        populate(stage, CABLE + volume_scene)
        cable_points = read_points(stage, "/Cable/Sim", 1).copy()
        volume_points = read_points(stage, "/Volume/Sim", 1).copy()
        if compose_builder:
            source = newton.ModelBuilder()
            with ovstage.PathDictionary(stage) as pd:
                values = _deformables.read_deformables(stage, pd, 1)
            _build.build_deformables(source, values)
            combined = newton.ModelBuilder()
            combined.add_particle(pos=wp.vec3(-5), vel=wp.vec3(0), mass=1)
            combined.add_builder(source)
            # Put a USD rigid body after the generated rod bodies to exercise nonzero row mapping.
            combined.add_body(label="/Rigid")
            combined.color()
            model = combined.finalize()
            del source, combined
            binding = ovnewton.attach_ovstage(stage, model=model)
        else:
            binding = ovnewton.attach_ovstage(stage)
            model = binding.model
        state = model.state()
        binding.update_to_ovstage(state, ordinal=2)
        stage.advance_write_floor(2).wait()
        np.testing.assert_allclose(read_points(stage, "/Cable/Sim", 2), cable_points, atol=1e-6)
        np.testing.assert_allclose(read_points(stage, "/Volume/Sim", 2), volume_points, atol=1e-6)
        before = state.body_q.numpy().copy()
        binding.update_from_ovstage(state, model.control(), ordinal=2)
        np.testing.assert_allclose(state.body_q.numpy(), before, atol=1e-6)
        if not compose_builder:
            solver = newton.solvers.SolverVBD(model, iterations=2)
            output = model.state()
            solver.step(state, output, model.control(), None, 1e-3)
            binding.update_to_ovstage(output, ordinal=3)
            stage.advance_write_floor(3).wait()
            assert np.all(np.isfinite(read_points(stage, "/Cable/Sim", 3)))
            assert np.all(np.isfinite(read_points(stage, "/Volume/Sim", 3)))


@pytest.mark.parametrize("rigid_location", ["ancestor", "owner", "geometry"])
def test_rejects_authored_rigid_ancestry(rigid_location):
    import ovstage

    scene = VOLUME
    if rigid_location == "ancestor":
        scene = scene.replace(
            'def Xform "Volume"',
            'def Xform "Rigid" (apiSchemas = ["PhysicsRigidBodyAPI"]) {\n    def Xform "Volume"',
        ) + "\n}\n"
        path = "/Rigid/Volume/Sim"
    else:
        schema = "PhysicsDeformableBodyAPI" if rigid_location == "owner" else "PhysicsVolumeDeformableSimAPI"
        scene = scene.replace(f'"{schema}"', f'"{schema}", "PhysicsRigidBodyAPI"')
        path = "/Volume/Sim"
    with ovstage.Stage("rigid-deformable-owner") as stage, ovstage.PathDictionary(stage) as pd:
        populate(stage, scene)
        with pytest.raises(_deformables.UnsupportedPhysicsError, match=f"rigid body is unsupported at {path}"):
            _deformables.read_deformables(stage, pd, 1)


def test_caller_body_labels_do_not_define_usd_ancestry():
    import newton
    import ovstage

    with ovstage.Stage("caller-body-label") as stage, ovstage.PathDictionary(stage) as pd:
        populate(stage, VOLUME)
        values = _deformables.read_deformables(stage, pd, 1)
        builder = newton.ModelBuilder()
        builder.add_body(label="/Volume")
        _build.build_deformables(builder, values)
        assert builder.body_count == 1
        assert builder.particle_count == 4
        assert builder.tet_count == 1
