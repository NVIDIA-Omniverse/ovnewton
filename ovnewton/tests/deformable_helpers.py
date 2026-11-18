# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared scenes and stage helpers for deformable import tests."""

from dataclasses import replace

import numpy as np
import ovstage
from ovstage import population

import ovnewton
from ovnewton._src import _deformables, _stage

CABLE = """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Material "Material" (prepend apiSchemas = ["PhysicsCurvesDeformableMaterialAPI"]) {
    float physics:youngsModulus = 1200
    float physics:poissonsRatio = 0.25
    float physics:curvesStretchStiffness = 60
}
def Xform "Cable" (prepend apiSchemas = ["PhysicsDeformableBodyAPI", "MaterialBindingAPI"]) {
    double3 xformOp:translate = (2, 3, 4)
    float xformOp:rotateY = 90
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateY"]
    float physics:density = 80
    rel material:binding:physics = </Material>
    def BasisCurves "Sim" (prepend apiSchemas = ["PhysicsCurvesDeformableSimAPI"]) {
        uniform token type = "linear"
        uniform token wrap = "nonperiodic"
        int[] curveVertexCounts = [3]
        point3f[] points = [(0, 0, 0), (0, 0, 1), (0, 0, 3)]
        normal3f[] normals = [(1, 0, 0), (1, 0, 0), (1, 0, 0)]
        vector3f[] velocities = [(0, 1, 0), (0, 1, 0), (0, 1, 0)]
        float[] physics:thicknesses = [0.04]
        uniform token physics:thicknesses:elementType = "constant"
    }
}
def Cube "Rigid" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {
    double size = 0.1
}
"""


VOLUME = """#usda 1.0
def Material "Material" (apiSchemas = ["PhysicsVolumeDeformableMaterialAPI"]) {
    float physics:youngsModulus = 1200
    float physics:poissonsRatio = 0.25
}
def Xform "Volume" (apiSchemas = ["PhysicsDeformableBodyAPI", "MaterialBindingAPI"]) {
    float physics:density = 60
    rel material:binding:physics = </Material>
    double3 xformOp:translate = (2, 3, 4)
    uniform token[] xformOpOrder = ["xformOp:translate"]
    def TetMesh "Sim" (apiSchemas = ["PhysicsVolumeDeformableSimAPI"]) {
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        int4[] tetVertexIndices = [(0, 1, 2, 3)]
    }
}
"""


def populate(stage, scene):
    population.open_usd_from_string_with_desc(
        stage, scene, ordinal=1, time=population.DEFAULT_TIME, descs=ovnewton.population_desc()
    )
    stage.advance_write_floor(1).wait()


def read_points(stage, path, ordinal):
    with ovstage.PathDictionary(stage) as pd, _stage._path_list_query(stage, pd, [path]) as query:
        rows = _stage.read_columns(stage, pd, query, ["points"], ordinal, ragged=["points"])["points"]
    return rows[0].reshape(-1, 3)


def read_test_deformables(surface_attributes=""):
    from ovstage import population

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
            $SURFACE_ATTRIBUTES
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
            usda.replace("$SURFACE_ATTRIBUTES", surface_attributes),
            ordinal=1,
            time=population.DEFAULT_TIME,
            descs=(ovnewton.population_desc(),),
        )
        stage.advance_write_floor(ordinal=1).wait()
        values = _deformables.read_deformables(stage, pd, 1)

    return values


def supported_surface(value: _deformables.DeformableInput) -> _deformables.DeformableInput:
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
