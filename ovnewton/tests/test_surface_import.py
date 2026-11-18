# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import replace

import numpy as np
import pytest

from ovnewton._src import _build, _deformables

from .deformable_helpers import read_points, read_test_deformables, supported_surface


def test_builds_supported_identity_transform_surface():
    import newton

    value = supported_surface(read_test_deformables()[1])
    builder = newton.ModelBuilder()

    _build.build_deformables(builder, [value])

    assert builder.particle_count == 3
    assert builder.tri_count == 1


def test_surface_ranges_keep_usd_ownership_when_builders_are_combined(warp_device):
    import newton
    import warp as wp

    value = supported_surface(read_test_deformables()[1])
    builder = newton.ModelBuilder()
    transformed = np.array([[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 2, 3, 1]], dtype=np.float64)
    frames = (np.eye(4), transformed)
    # A matching label does not make caller-added cloth part of the USD binding.
    for index in range(2):
        builder.add_cloth_mesh(
            pos=wp.vec3(10, 0, 0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0),
            vertices=value.points.tolist(),
            indices=[0, 1, 2],
            density=1.0,
            label=f"/World/Surface{index}/Sim",
        )
        _build.build_deformables(
            builder,
            [replace(value, geometry_path=f"/World/Surface{index}/Sim", world_matrix=frames[index])],
        )

    combined = newton.ModelBuilder()
    combined.add_particle(pos=wp.vec3(-10, 0, 0), vel=wp.vec3(0), mass=1.0)
    combined.add_builder(builder)
    model = combined.finalize(device=warp_device)
    del builder, combined

    surfaces = _deformables.model_particle_ranges(model, "surface")
    assert [surface.geometry_path for surface in surfaces] == ["/World/Surface0/Sim", "/World/Surface1/Sim"]
    assert [(surface.particle_start, surface.particle_end) for surface in surfaces] == [(4, 7), (10, 13)]
    points = model.particle_q.numpy()
    expected_frames = ((0, 0, 0, 0, 0, 0, 1), (1, 2, 3, 0, math.sqrt(0.5), 0, math.sqrt(0.5)))
    expected_points = (value.points, [(1, 2, 3), (1, 2, 2), (1, 3, 3)])
    for surface, frame, expected in zip(surfaces, expected_frames, expected_points):
        np.testing.assert_allclose(surface.world_transform, frame, atol=1e-6)
        np.testing.assert_allclose(points[surface.particle_start : surface.particle_end], expected)


@pytest.mark.parametrize("build_path", ["build_model", "attach_ovstage", "caller_owned"])
def test_surface_points_publish_after_model_finalization(build_path, warp_device, monkeypatch):
    import newton
    import ovstage
    import warp as wp
    from ovstage import PopulationDomain, population

    import ovnewton

    usda = """#usda 1.0
def Xform "World" {
    def Xform "Surface" (
        prepend apiSchemas = ["PhysicsDeformableBodyAPI"]
    ) {
        float physics:density = 80
        def Mesh "Sim" (
            prepend apiSchemas = ["PhysicsSurfaceDeformableSimAPI", "PhysicsCollisionAPI"]
        ) {
            point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
            int[] faceVertexCounts = [3]
            int[] faceVertexIndices = [0, 1, 2]
            float[] physics:thicknesses = [0.01]
            uniform token physics:thicknesses:elementType = "constant"
        }
    }
    def Xform "SecondSurface" (
        prepend apiSchemas = ["PhysicsDeformableBodyAPI"]
    ) {
        double3 xformOp:translate = (2, 3, 4)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        float physics:density = 80
        def Mesh "Sim" (
            prepend apiSchemas = ["PhysicsSurfaceDeformableSimAPI"]
        ) {
            float xformOp:rotateY = 90
            uniform token[] xformOpOrder = ["xformOp:rotateY"]
            point3f[] points = [(2, 0, 0), (3, 0, 0), (2, 1, 0)]
            vector3f[] velocities = [(0, 0, 1), (0, 0, 2), (0, 0, 3)]
            int[] faceVertexCounts = [3]
            int[] faceVertexIndices = [0, 1, 2]
            float[] physics:thicknesses = [0.01]
            uniform token physics:thicknesses:elementType = "constant"
        }
    }
}
"""

    with wp.ScopedDevice(warp_device), ovstage.Stage("ovnewton-deformable-binding") as stage:
        population.open_usd_from_string_with_desc(
            stage,
            usda,
            ordinal=1,
            time=math.nan,
            descs=(
                population.Desc(domains=PopulationDomain.RENDERING),
                ovnewton.population_desc(),
            ),
        )
        stage.advance_write_floor(ordinal=1).wait()

        if build_path == "caller_owned":
            builder = newton.ModelBuilder()
            result = ovnewton.add_ovstage(builder, stage)
            builder.color(include_bending=True)
            model = builder.finalize(skip_validation_joints=result.has_orphan_joints)
            del builder
            binding = ovnewton.attach_ovstage(stage, model=model)
        elif build_path == "build_model":
            with ovstage.PathDictionary(stage) as pd:
                model = _build.build_model(stage, pd)
            binding = ovnewton.StageBinding(stage, model)
        else:
            binding = ovnewton.attach_ovstage(stage)

        assert binding.model.shape_count == 0  # Cloth must not also become a static collider.
        state = binding.model.state()
        # The second cloth combines an ancestor translation with a geometry rotation.
        surfaces = {
            surface.geometry_path: surface
            for surface in _deformables.model_particle_ranges(binding.model, "surface")
        }
        surface = surfaces["/World/SecondSurface/Sim"]
        particle_range = slice(surface.particle_start, surface.particle_end)
        np.testing.assert_allclose(state.particle_q.numpy()[particle_range], [(2, 3, 2), (2, 3, 1), (2, 4, 2)])
        np.testing.assert_allclose(
            state.particle_qd.numpy()[particle_range], [(1, 0, 0), (2, 0, 0), (3, 0, 0)], atol=1e-6
        )

        def unexpected_query(*_args):
            raise AssertionError("surface publication must reuse its fixed query")

        for ordinal in (2, 3):
            moved_points = state.particle_q.numpy() + np.asarray((0, 0, 1), dtype=np.float32)
            state = binding.model.state()
            state.particle_q.assign(moved_points)
            with monkeypatch.context() as patch:
                patch.setattr(stage, "query_from_path_list", unexpected_query)
                binding.update_to_ovstage(state, ordinal=ordinal)
            stage.advance_write_floor(ordinal=ordinal).wait()

        expected_points = {
            "/World/Surface/Sim": [(0, 0, 2), (1, 0, 2), (0, 1, 2)],
            "/World/SecondSurface/Sim": [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
        }
        for path, expected in expected_points.items():
            np.testing.assert_allclose(read_points(stage, path, 3), expected, atol=1e-6)


@pytest.mark.parametrize("meters_per_unit", [1.0, 0.01])
def test_uses_newton_default_density_when_density_is_not_authored(meters_per_unit):
    import newton

    value = supported_surface(read_test_deformables()[1])
    value = replace(value, density=None, meters_per_unit=meters_per_unit)
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.density = 123.0

    _build.build_deformables(builder, [value])

    assert sum(builder.particle_mass) == pytest.approx(0.5 * 0.01 * 123.0 * meters_per_unit**3)


@pytest.mark.parametrize("thickness", [0.01, 0.05])
@pytest.mark.parametrize("stiffness", [None, (0.0, 0.0), (1000.0, 5.0)], ids=["defaults", "zero", "authored"])
def test_surface_materials_preserve_structural_stiffness_and_builder_damping(thickness, stiffness, warp_device):
    import newton
    import ovstage
    import warp as wp
    from ovstage import population

    import ovnewton

    material = "" if stiffness is None else (
        f"float physics:surfaceStretchStiffness = {stiffness[0]}\n"
        f"float physics:surfaceBendStiffness = {stiffness[1]}\n"
    )
    usda = f'''#usda 1.0
(metersPerUnit = 1)
def Mesh "Cloth" (
    prepend apiSchemas = ["PhysicsDeformableBodyAPI", "PhysicsSurfaceDeformableSimAPI", "MaterialBindingAPI"]
) {{
    point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    int[] faceVertexCounts = [4]
    int[] faceVertexIndices = [0, 1, 2, 3]
    float physics:density = 80
    float[] physics:thicknesses = [{thickness}]
    uniform token physics:thicknesses:elementType = "constant"
    rel material:binding:physics = </Material>
}}
def Material "Material" (prepend apiSchemas = ["PhysicsSurfaceDeformableMaterialAPI"]) {{
    {material}
}}
'''
    with wp.ScopedDevice(warp_device), ovstage.Stage("cloth-materials") as stage:
        population.open_usd_from_string_with_desc(
            stage, usda, ordinal=1, time=population.DEFAULT_TIME, descs=ovnewton.population_desc()
        )
        stage.advance_write_floor(1).wait()
        builder = newton.ModelBuilder()
        builder.default_tri_ke = 73.0
        builder.default_edge_ke = 0.75
        builder.default_tri_kd = 2.5
        ovnewton.add_ovstage(builder, stage)
        builder.color(include_bending=True)
        model = builder.finalize()
        binding = ovnewton.attach_ovstage(stage, model=model)
        expected_stretch, expected_bend = (73.0, 0.75) if stiffness is None else stiffness
        np.testing.assert_allclose(binding.model.tri_materials.numpy()[:, 0], expected_stretch)
        np.testing.assert_allclose(binding.model.tri_materials.numpy()[:, 2], 2.5)
        np.testing.assert_allclose(binding.model.edge_bending_properties.numpy()[:, 0], expected_bend)


def test_rejects_unsupported_surface_authoring_before_builder_mutation():
    import newton

    value = supported_surface(read_test_deformables()[1])
    scaled = np.diag([2.0, 1.0, 1.0, 1.0])
    cases = (
        (replace(value, world_matrix=scaled), "rigid world transform without scale"),
        (replace(value, body_enabled=False), "disabled bodies"),
        (replace(value, kinematic_enabled=True), "kinematic bodies"),
        (replace(value, mass=1.0), "authored masses"),
        (
            replace(
                value,
                masses=_deformables.ElementField(np.ones(3, dtype=np.float32), "point"),
            ),
            "authored masses",
        ),
        (replace(value, thicknesses=None), "one constant thickness"),
        (
            replace(
                value,
                thicknesses=_deformables.ElementField(np.full(3, 0.01, dtype=np.float32), "point"),
            ),
            "one constant thickness",
        ),
        (
            replace(value, material={**value.material, "surfaceShearStiffness": 3.0}),
            "physics:surfaceShearStiffness",
        ),
    )

    for unsupported, message in cases:
        builder = newton.ModelBuilder()
        with pytest.raises(_deformables.UnsupportedPhysicsError, match=message):
            _build.build_deformables(builder, [unsupported])
        assert builder.particle_count == 0


def test_surface_rejects_authored_rest_points():
    with pytest.raises(_deformables.UnsupportedPhysicsError, match="restShapePoints at /World/Surface/Sim"):
        read_test_deformables("point3f[] physics:restShapePoints = [(0, 0, 0), (2, 0, 0), (0, 2, 0)]")
