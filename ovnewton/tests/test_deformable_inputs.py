# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import replace

import numpy as np
import pytest

from ovnewton._src import _deformables


def _read_test_deformables():
    import ovstage
    from ovstage import PopulationDomain, population

    usda = """#usda 1.0
def Xform "World" {
    def Material "Material" (
        prepend apiSchemas = [
            "PhysicsCurvesDeformableMaterialAPI",
            "PhysicsSurfaceDeformableMaterialAPI",
            "PhysicsVolumeDeformableMaterialAPI"
        ]
    ) {
        float physics:youngsModulus = 1200
        float physics:curvesTwistStiffness = 13
        float physics:surfaceStretchStiffness = 11
    }
    def Xform "Cable" (prepend apiSchemas = ["PhysicsDeformableBodyAPI", "MaterialBindingAPI"]) {
        rel material:binding:physics = </World/Material>
        def BasisCurves "Sim" (
            prepend apiSchemas = ["PhysicsCurvesDeformableSimAPI"]
        ) {
            uniform token type = "linear"
            uniform token wrap = "nonperiodic"
            int[] curveVertexCounts = [3]
            point3f[] points = [(0, 0, 2), (0, 0, 1), (0, 0, 0)]
            normal3f[] normals = [(1, 0, 0), (1, 0, 0), (1, 0, 0)]
            vector3f[] velocities = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
            float[] physics:masses = [1, 1, 1]
            uniform token physics:masses:elementType = "point"
            float[] physics:thicknesses = [0.02]
            uniform token physics:thicknesses:elementType = "constant"
        }
    }
    def Xform "Surface" (prepend apiSchemas = ["PhysicsDeformableBodyAPI", "MaterialBindingAPI"]) {
        rel material:binding:physics = </World/Material>
        def Mesh "Sim" (
            prepend apiSchemas = ["PhysicsSurfaceDeformableSimAPI"]
        ) {
            point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
            int[] faceVertexCounts = [3]
            int[] faceVertexIndices = [0, 1, 2]
            vector3f[] velocities = [(0, 1, 0), (0, 2, 0), (0, 3, 0)]
            float[] physics:masses = [0.5]
            uniform token physics:masses:elementType = "face"
            float[] physics:thicknesses = [0.01, 0.01, 0.01]
            uniform token physics:thicknesses:elementType = "point"
        }
    }
    def Xform "Volume" (prepend apiSchemas = ["PhysicsDeformableBodyAPI", "MaterialBindingAPI"]) {
        rel material:binding:physics = </World/Material>
        def TetMesh "Sim" (
            prepend apiSchemas = ["PhysicsVolumeDeformableSimAPI"]
        ) {
            point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
            int4[] tetVertexIndices = [(0, 1, 2, 3)]
            vector3f[] velocities = [(0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4)]
            float[] physics:masses = [4]
            uniform token physics:masses:elementType = "tetrahedron"
        }
    }
}
"""

    with ovstage.Stage("ovnewton-deformable-inputs") as stage, ovstage.PathDictionary(stage) as pd:
        population.open_usd_from_string_with_desc(
            stage,
            usda,
            ordinal=1,
            time_code=math.nan,
            descs=(
                population.Desc(domains=PopulationDomain.PHYSICS),
                _deformables.population_desc(),
            ),
        )
        stage.advance_write_floor(ordinal=1).wait()
        values = _deformables.read_deformables(stage, pd, 1)

    return values


def test_reads_all_canonical_deformable_families():
    values = _read_test_deformables()

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


def _supported_surface(value: _deformables.DeformableInput) -> _deformables.DeformableInput:
    return replace(
        value,
        mass=None,
        density=80.0,
        material={
            "density": None,
            "surfaceStretchStiffness": 11.0,
            "surfaceBendStiffness": 5.0,
        },
        masses=_deformables.ElementField(np.empty(0, dtype=np.float32), ""),
        thicknesses=_deformables.ElementField(np.asarray([0.01], dtype=np.float32), "constant"),
    )


def test_builds_supported_identity_transform_surface():
    import newton

    value = _supported_surface(_read_test_deformables()[1])
    builder = newton.ModelBuilder()

    build_info = _deformables.build_deformables(builder, [value])

    assert builder.particle_count == 3
    assert builder.tri_count == 1
    assert len(build_info.surfaces) == 1


@pytest.mark.parametrize("build_path", ["build_model", "attach_ovstage", "caller_owned"])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_surface_points_publish_after_model_finalization(build_path, device):
    import newton
    import ovstage
    import warp as wp
    from ovstage import PopulationDomain, population

    import ovnewton
    from ovnewton._src import _build

    if device == "cuda:0" and not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")

    usda = """#usda 1.0
def Xform "World" {
    def Xform "Surface" (
        prepend apiSchemas = ["PhysicsDeformableBodyAPI"]
    ) {
        float physics:density = 80
        def Mesh "Sim" (
            prepend apiSchemas = ["PhysicsSurfaceDeformableSimAPI"]
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
        float physics:density = 80
        def Mesh "Sim" (
            prepend apiSchemas = ["PhysicsSurfaceDeformableSimAPI"]
        ) {
            point3f[] points = [(2, 0, 0), (3, 0, 0), (2, 1, 0)]
            int[] faceVertexCounts = [3]
            int[] faceVertexIndices = [0, 1, 2]
            float[] physics:thicknesses = [0.01]
            uniform token physics:thicknesses:elementType = "constant"
        }
    }
}
"""

    with wp.ScopedDevice(device), ovstage.Stage("ovnewton-deformable-binding") as stage:
        population.open_usd_from_string_with_desc(
            stage,
            usda,
            ordinal=1,
            time_code=math.nan,
            descs=(
                population.Desc(domains=PopulationDomain.ALL),
                _deformables.population_desc(),
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

        state = binding.model.state()
        moved_points = state.particle_q.numpy() + np.asarray((0, 0, 1), dtype=np.float32)
        state.particle_q.assign(moved_points)
        binding.update_to_ovstage(state, ordinal=2)
        stage.advance_write_floor(ordinal=2).wait()

        expected_points = {
            "/World/Surface/Sim": [(0, 0, 1), (1, 0, 1), (0, 1, 1)],
            "/World/SecondSurface/Sim": [(2, 0, 1), (3, 0, 1), (2, 1, 1)],
        }
        with ovstage.PathDictionary(stage) as pd:
            for path, expected in expected_points.items():
                path_list = pd.create_path_list_from_strings([path])
                try:
                    with stage.query_from_path_list(path_list) as query:
                        with stage.read_attributes(
                            query, [pd.intern_token("points")], ovstage.OrdinalRange(2, 2)
                        ) as read:
                            groups = 0
                            for group in read.groups():
                                with group:
                                    np.testing.assert_array_equal(group.array(0).reshape(-1, 3), expected)
                                    groups += 1
                            assert groups == 1
                finally:
                    pd.destroy_path_list(path_list)


def test_uses_newton_default_density_when_density_is_not_authored():
    import newton

    value = _supported_surface(_read_test_deformables()[1])
    value = replace(value, density=None)
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.density = 123.0

    _deformables.build_deformables(builder, [value])

    assert sum(builder.particle_mass) == pytest.approx(0.5 * 0.01 * 123.0)


def test_rejects_unsupported_surface_authoring_before_builder_mutation():
    import newton

    value = _supported_surface(_read_test_deformables()[1])
    translated = np.eye(4, dtype=np.float64)
    translated[3, :3] = (1.0, 2.0, 3.0)
    cases = (
        (replace(value, world_matrix=translated), "identity world transform"),
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
            _deformables.build_deformables(builder, [unsupported])
        assert builder.particle_count == 0
