# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager

import numpy as np
import ovstage
import pytest
from ovstage import PopulationDomain, population

from ovnewton._src._parse import _api_paths, _type_paths
from ovnewton._src._stage import _path_list_query, read_columns


@contextmanager
def _populated_stage(usda, name, domains):
    with ovstage.Stage(name) as stage, ovstage.PathDictionary(stage) as pd:
        population.open_usd_from_string(stage, usda, ordinal=1, domains=domains)
        stage.advance_write_floor(ordinal=1).wait()
        yield stage, pd


def _physics_stage(usda, name):
    return _populated_stage(usda, name, PopulationDomain.PHYSICS)


def _require_columns(columns, attributes, *, index=0, reason):
    missing = [attribute for attribute in attributes if index not in columns[attribute]]
    if missing:
        pytest.xfail(f"{reason}: {', '.join(missing)}")


def test_newton_joint_api_04_is_mirrored():
    usda = '''#usda 1.0
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
def PhysicsRevoluteJoint "Defaults" (prepend apiSchemas = ["NewtonJointAPI"]) {
    rel physics:body1 = </Body>
}
def PhysicsRevoluteJoint "Authored" (prepend apiSchemas = ["NewtonJointAPI"]) {
    rel physics:body1 = </Body>
    float newton:armature = 1
    float newton:damping = 2
    float newton:friction = 3
    float newton:velocityLimit = 4
    float newton:limitStiffness = 5
    float newton:limitDamping = 6
}
'''
    attributes = (
        "newton:armature",
        "newton:damping",
        "newton:friction",
        "newton:velocityLimit",
        "newton:limitStiffness",
        "newton:limitDamping",
    )
    with _physics_stage(usda, "newton-joint-api-contract") as (stage, pd):
        paths = ["/Defaults", "/Authored"]
        if not set(paths).issubset(_api_paths(stage, pd, 1, "NewtonJointAPI")):
            pytest.xfail("ovpopulation does not register NewtonJointAPI 0.4.0")
        with _path_list_query(stage, pd, paths) as query:
            columns = read_columns(stage, pd, query, attributes, 1)
        for index in range(2):
            _require_columns(
                columns,
                attributes,
                index=index,
                reason="ovpopulation does not mirror NewtonJointAPI values",
            )

    defaults = [float(columns[attribute][0][0]) for attribute in attributes]
    authored = [float(columns[attribute][1][0]) for attribute in attributes]
    assert defaults[:3] == [0.0, 0.0, 0.0]
    assert np.isposinf(defaults[3])
    assert np.isneginf(defaults[4:]).all()
    assert authored == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_mesh_orientation_is_mirrored_under_physics():
    usda = '''#usda 1.0
def Mesh "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]) {
    uniform token orientation = "leftHanded"
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
}
'''
    attributes = ("orientation", "omni:fabric:worldMatrix")
    with _physics_stage(usda, "mesh-orientation-contract") as (stage, pd):
        if _type_paths(stage, pd, "Mesh", 1) != ["/Collider"]:
            pytest.xfail("ovpopulation does not select collider meshes under PHYSICS")
        with _path_list_query(stage, pd, ["/Collider"]) as query:
            columns = read_columns(stage, pd, query, attributes, 1)
        _require_columns(columns, attributes, reason="PHYSICS collider geometry is incomplete")
        assert pd.token_to_string(int(columns["orientation"][0][0])) == "leftHanded"
        assert columns["omni:fabric:worldMatrix"][0].size == 16


def test_newton_site_is_mirrored_under_physics():
    usda = '''#usda 1.0
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Sphere "Site" (prepend apiSchemas = ["NewtonSiteAPI"]) {
        double radius = 0.25
    }
}
'''
    attributes = ("usd-prim-type", "radius", "omni:fabric:worldMatrix")
    with _physics_stage(usda, "newton-site-contract") as (stage, pd):
        if "/Body/Site" not in _api_paths(stage, pd, 1, "NewtonSiteAPI"):
            pytest.xfail("ovpopulation does not select NewtonSiteAPI prims under PHYSICS")
        with _path_list_query(stage, pd, ["/Body/Site"]) as query:
            columns = read_columns(stage, pd, query, attributes, 1)
        _require_columns(columns, attributes, reason="PHYSICS NewtonSiteAPI geometry is incomplete")
        assert pd.token_to_string(int(columns["usd-prim-type"][0][0])) == "Sphere"
        assert columns["radius"][0].tolist() == [0.25]
        assert columns["omni:fabric:worldMatrix"][0].size == 16


def test_newton_actuator_columns_are_mirrored_under_physics():
    usda = '''#usda 1.0
def Xform "World" {
    def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def PhysicsRevoluteJoint "Joint" {
        rel physics:body1 = </World/Body>
    }
    def NewtonActuator "PDActuator" (
        prepend apiSchemas = ["NewtonPDControlAPI", "NewtonActuatorDelayAPI", "NewtonMaxEffortClampingAPI"]
    ) {
        rel newton:targets = </World/Joint>
        float newton:constEffort = 1
        float newton:kp = 12
        float newton:kd = 3
        int newton:delaySteps = 2
        float newton:maxEffort = 8
    }
    def NewtonActuator "PIDActuator" (
        prepend apiSchemas = ["NewtonPIDControlAPI", "NewtonDCMotorClampingAPI"]
    ) {
        rel newton:targets = </World/Joint>
        float newton:constEffort = 2
        float newton:kp = 13
        float newton:kd = 4
        float newton:ki = 5
        float newton:integralMax = 6
        float newton:maxMotorEffort = 7
        float newton:saturationEffort = 9
        float newton:velocityLimit = 10
    }
    def NewtonActuator "NeuralActuator" (
        prepend apiSchemas = ["NewtonNeuralControlAPI", "NewtonPositionBasedClampingAPI"]
    ) {
        rel newton:targets = </World/Joint>
        asset newton:modelPath = @controller.json@
        float[] newton:lookupPositions = [-1, 0, 1]
        float[] newton:lookupEfforts = [-2, 0, 2]
    }
}
'''
    contracts = {
        "/World/PDActuator": {
            "newton:targets": ["/World/Joint"],
            "newton:constEffort": [1.0],
            "newton:kp": [12.0],
            "newton:kd": [3.0],
            "newton:delaySteps": [2],
            "newton:maxEffort": [8.0],
        },
        "/World/PIDActuator": {
            "newton:targets": ["/World/Joint"],
            "newton:constEffort": [2.0],
            "newton:kp": [13.0],
            "newton:kd": [4.0],
            "newton:ki": [5.0],
            "newton:integralMax": [6.0],
            "newton:maxMotorEffort": [7.0],
            "newton:saturationEffort": [9.0],
            "newton:velocityLimit": [10.0],
        },
        "/World/NeuralActuator": {
            "newton:targets": ["/World/Joint"],
            "newton:modelPath": ["controller.json"],
            "newton:lookupPositions": [-1.0, 0.0, 1.0],
            "newton:lookupEfforts": [-2.0, 0.0, 2.0],
        },
    }
    schema_paths = {
        "NewtonPDControlAPI": {"/World/PDActuator"},
        "NewtonActuatorDelayAPI": {"/World/PDActuator"},
        "NewtonMaxEffortClampingAPI": {"/World/PDActuator"},
        "NewtonPIDControlAPI": {"/World/PIDActuator"},
        "NewtonDCMotorClampingAPI": {"/World/PIDActuator"},
        "NewtonNeuralControlAPI": {"/World/NeuralActuator"},
        "NewtonPositionBasedClampingAPI": {"/World/NeuralActuator"},
    }
    ragged = {"newton:targets", "newton:lookupPositions", "newton:lookupEfforts"}
    with _physics_stage(usda, "actuator-contract") as (stage, pd):
        missing_actuators = set(contracts).difference(_type_paths(stage, pd, "NewtonActuator", 1))
        if missing_actuators:
            pytest.xfail(
                "ovpopulation does not select NewtonActuator prims under PHYSICS: "
                + ", ".join(sorted(missing_actuators))
            )
        missing_schemas = [
            schema for schema, paths in schema_paths.items() if not paths.issubset(_api_paths(stage, pd, 1, schema))
        ]
        if missing_schemas:
            pytest.xfail("ovpopulation does not select Newton actuator schemas: " + ", ".join(missing_schemas))

        for path, expected in contracts.items():
            attributes = tuple(expected)
            with _path_list_query(stage, pd, [path]) as query:
                columns = read_columns(stage, pd, query, attributes, 1, ragged=ragged.intersection(attributes))
            _require_columns(columns, attributes, reason="ovpopulation does not mirror NewtonActuator values")
            for attribute, value in expected.items():
                row = columns[attribute][0]
                if attribute == "newton:targets":
                    actual = [pd.path_to_string(int(item)) for item in row]
                elif attribute == "newton:modelPath":
                    actual = [pd.token_to_string(int(item)) for item in row]
                else:
                    actual = row.tolist()
                assert actual == value


def test_effective_physics_material_binding_is_mirrored():
    usda = '''#usda 1.0
def Material "PhysicsMaterial" (prepend apiSchemas = ["PhysicsMaterialAPI"]) {
    float physics:density = 250
}
def Xform "Parent" (prepend apiSchemas = ["MaterialBindingAPI"]) {
    rel material:binding:physics = </PhysicsMaterial>
    def Cube "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {}
}
'''
    with _physics_stage(usda, "material-binding-contract") as (stage, pd):
        with _path_list_query(stage, pd, ["/Parent/Collider"]) as query:
            columns = read_columns(
                stage,
                pd,
                query,
                ("material:binding:physics",),
                1,
                ragged=("material:binding:physics",),
            )
        _require_columns(
            columns,
            ("material:binding:physics",),
            reason="ovpopulation does not mirror effective inherited physics-material binding",
        )
        assert pd.path_to_string(int(columns["material:binding:physics"][0][0])) == "/PhysicsMaterial"


def test_collection_material_binding_strength_is_resolved():
    usda = '''#usda 1.0
def Material "WeakMaterial" (prepend apiSchemas = ["PhysicsMaterialAPI"]) {}
def Material "StrongMaterial" (prepend apiSchemas = ["PhysicsMaterialAPI"]) {}
def Xform "World" (
    prepend apiSchemas = ["CollectionAPI:colliders", "MaterialBindingAPI"]
) {
    prepend rel collection:colliders:includes = </World/Collider>
    rel material:binding:collection:physics:colliders = [
        </World.collection:colliders>,
        </StrongMaterial>
    ] (
        bindMaterialAs = "strongerThanDescendants"
    )
    def Cube "Collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"]
    ) {
        rel material:binding:physics = </WeakMaterial>
    }
}
'''
    with _physics_stage(usda, "collection-material-binding-contract") as (stage, pd):
        with _path_list_query(stage, pd, ["/World/Collider"]) as query:
            columns = read_columns(
                stage,
                pd,
                query,
                ("material:binding:physics",),
                1,
                ragged=("material:binding:physics",),
            )
        _require_columns(
            columns,
            ("material:binding:physics",),
            reason="ovpopulation does not resolve collection physics-material binding strength",
        )
        material = pd.path_to_string(int(columns["material:binding:physics"][0][0]))
        if material != "/StrongMaterial":
            pytest.xfail("ovpopulation does not resolve collection physics-material binding strength")


_NATIVE_INSTANCE_USDA = '''#usda 1.0
def Xform "Prototype" (instanceable = true) {
    def Cube "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {}
}
def Xform "Instance" (prepend references = </Prototype>) {}
'''


def test_native_instancing_is_exposed_to_python_under_all():
    assert ovstage.instancing.available()
    with _populated_stage(
        _NATIVE_INSTANCE_USDA,
        "native-instancing-all-contract",
        PopulationDomain.ALL,
    ) as (stage, _):
        prototypes = ovstage.instancing.get_prototype_roots(stage)
        assert prototypes
        prototype = ovstage.instancing.get_prototype_root(stage, "/Instance")
        assert prototype in prototypes
        assert "/Instance" in ovstage.instancing.get_instance_roots(stage, prototype)


def test_native_instancing_is_exposed_under_physics():
    with _physics_stage(_NATIVE_INSTANCE_USDA, "native-instancing-physics-contract") as (stage, _):
        prototypes = ovstage.instancing.get_prototype_roots(stage)
        if not prototypes:
            pytest.xfail("PHYSICS population does not expose native prototype mappings")
        prototype = ovstage.instancing.get_prototype_root(stage, "/Instance")
        assert prototype in prototypes
        instances = ovstage.instancing.get_instance_roots(stage, prototype)
        assert "/Instance" in instances


def test_bare_tetmesh_columns_are_mirrored_under_physics():
    usda = '''#usda 1.0
def TetMesh "SoftBody" {
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    int4[] tetVertexIndices = [(0, 1, 2, 3)]
}
'''
    attributes = ("points", "tetVertexIndices", "orientation", "omni:fabric:worldMatrix")
    with _physics_stage(usda, "tetmesh-contract") as (stage, pd):
        paths = _type_paths(stage, pd, "TetMesh", 1)
        if not paths:
            pytest.xfail("ovpopulation does not select bare UsdGeom.TetMesh prims under PHYSICS")
        with _path_list_query(stage, pd, paths) as query:
            columns = read_columns(stage, pd, query, attributes, 1, ragged=("points", "tetVertexIndices"))
        _require_columns(columns, attributes, reason="PHYSICS TetMesh geometry is incomplete")
        assert columns["points"][0].size == 12
        assert columns["tetVertexIndices"][0].tolist() == [0, 1, 2, 3]
        assert pd.token_to_string(int(columns["orientation"][0][0])) == "rightHanded"
        assert columns["omni:fabric:worldMatrix"][0].size == 16


def test_aousd_deformable_schema_surface_is_mirrored_under_physics():
    usda = '''#usda 1.0
def Xform "Cable" (prepend apiSchemas = ["PhysicsDeformableBodyAPI"]) {
    bool physics:bodyEnabled = true
    bool physics:kinematicEnabled = false
    float physics:mass = 9
    float physics:density = 900
    def BasisCurves "Sim" (prepend apiSchemas = ["PhysicsCurvesDeformableSimAPI"]) {
        uniform token type = "linear"
        uniform token wrap = "nonperiodic"
        int[] curveVertexCounts = [2]
        point3f[] points = [(0, 0, 0), (1, 0, 0)]
        normal3f[] normals = [(0, 1, 0), (0, 1, 0)]
        vector3f[] velocities = [(1, 0, 0), (2, 0, 0)]
        float[] physics:masses = [1, 2]
        point3f[] physics:restShapePoints = [(0, 0, 0), (1.1, 0, 0)]
        normal3f[] physics:restNormals = [(0, 1, 0), (0, 1, 0)]
    }
}
def Xform "Cloth" (prepend apiSchemas = ["PhysicsDeformableBodyAPI"]) {
    bool physics:bodyEnabled = true
    bool physics:kinematicEnabled = false
    float physics:mass = 8
    float physics:density = 800
    def Mesh "Sim" (prepend apiSchemas = ["PhysicsSurfaceDeformableSimAPI"]) {
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        vector3f[] velocities = [(0, 1, 0), (0, 2, 0), (0, 3, 0)]
        float[] physics:masses = [3, 4, 5]
        point3f[] physics:restShapePoints = [(0, 0, 0), (1.1, 0, 0), (0, 1.1, 0)]
        float[] physics:restBendAngles = [0.25]
        int2[] physics:restAdjTriPairs = [(0, 0)]
        uniform token physics:restBendAnglesDefault = "flatDefault"
    }
}
def Xform "Volume" (prepend apiSchemas = ["PhysicsDeformableBodyAPI"]) {
    bool physics:bodyEnabled = true
    bool physics:kinematicEnabled = false
    float physics:mass = 7
    float physics:density = 700
    def TetMesh "Sim" (prepend apiSchemas = ["PhysicsVolumeDeformableSimAPI"]) {
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        int4[] tetVertexIndices = [(0, 1, 2, 3)]
        vector3f[] velocities = [(0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4)]
        float[] physics:masses = [6, 7, 8, 9]
        point3f[] physics:restShapePoints = [(0, 0, 0), (1.1, 0, 0), (0, 1.1, 0), (0, 0, 1.1)]
    }
}
def Material "CableMaterial" (prepend apiSchemas = ["PhysicsCurvesDeformableMaterialAPI"]) {
    float physics:thickness = 0.02
    float physics:stretchStiffness = 11
    float physics:shearStiffness = 12
    float physics:bendStiffness = 13
    float physics:twistStiffness = 14
    float physics:density = 1000
}
def Material "ClothMaterial" (prepend apiSchemas = ["PhysicsSurfaceDeformableMaterialAPI"]) {
    float physics:thickness = 0.01
    float physics:stretchStiffness = 21
    float physics:shearStiffness = 22
    float physics:bendStiffness = 23
    float physics:density = 1100
}
def Material "VolumeMaterial" (prepend apiSchemas = ["PhysicsVolumeDeformableMaterialAPI"]) {
    float physics:youngsModulus = 1000
    float physics:poissonsRatio = 0.3
    float physics:density = 1200
}
def PhysicsAttachment "Attachment" {
    rel physics:src0 = </Cable/Sim>
    rel physics:src1 = </Cloth/Sim>
    token physics:type0 = "body"
    token physics:type1 = "tri"
    int[] physics:indices0 = [0]
    int[] physics:indices1 = [1]
    vector3f[] physics:coords0 = [(0.25, 0, 0)]
    vector3f[] physics:coords1 = [(0.5, 0, 0)]
    bool physics:attachmentEnabled = true
    float physics:stiffness = 31
    float physics:damping = 32
}
def PhysicsElementCollisionFilter "ElementFilter" {
    rel physics:src0 = </Cloth/Sim>
    rel physics:src1 = </Volume/Sim>
    int[] physics:groupElemIndices0 = [0]
    int[] physics:groupElemIndices1 = [0]
    int[] physics:groupElemCounts0 = [1]
    int[] physics:groupElemCounts1 = [1]
    bool physics:filterEnabled = true
}
'''
    api_paths = {
        "PhysicsDeformableBodyAPI": {"/Cable", "/Cloth", "/Volume"},
        "PhysicsCurvesDeformableSimAPI": {"/Cable/Sim"},
        "PhysicsSurfaceDeformableSimAPI": {"/Cloth/Sim"},
        "PhysicsVolumeDeformableSimAPI": {"/Volume/Sim"},
        "PhysicsCurvesDeformableMaterialAPI": {"/CableMaterial"},
        "PhysicsSurfaceDeformableMaterialAPI": {"/ClothMaterial"},
        "PhysicsVolumeDeformableMaterialAPI": {"/VolumeMaterial"},
    }
    type_paths = {
        "PhysicsAttachment": ["/Attachment"],
        "PhysicsElementCollisionFilter": ["/ElementFilter"],
    }
    contracts = {
        "/Cable": (
            {
                "physics:bodyEnabled": [True],
                "physics:kinematicEnabled": [False],
                "physics:mass": [9.0],
                "physics:density": [900.0],
            },
            (),
        ),
        "/Cloth": (
            {
                "physics:bodyEnabled": [True],
                "physics:kinematicEnabled": [False],
                "physics:mass": [8.0],
                "physics:density": [800.0],
            },
            (),
        ),
        "/Volume": (
            {
                "physics:bodyEnabled": [True],
                "physics:kinematicEnabled": [False],
                "physics:mass": [7.0],
                "physics:density": [700.0],
            },
            (),
        ),
        "/Cable/Sim": (
            {
                "points": [0, 0, 0, 1, 0, 0],
                "curveVertexCounts": [2],
                "type": "linear",
                "wrap": "nonperiodic",
                "normals": [0, 1, 0, 0, 1, 0],
                "velocities": [1, 0, 0, 2, 0, 0],
                "physics:masses": [1, 2],
                "physics:restShapePoints": [0, 0, 0, 1.1, 0, 0],
                "physics:restNormals": [0, 1, 0, 0, 1, 0],
                "omni:fabric:worldMatrix": None,
            },
            (
                "points",
                "curveVertexCounts",
                "normals",
                "velocities",
                "physics:masses",
                "physics:restShapePoints",
                "physics:restNormals",
            ),
        ),
        "/Cloth/Sim": (
            {
                "points": [0, 0, 0, 1, 0, 0, 0, 1, 0],
                "faceVertexCounts": [3],
                "faceVertexIndices": [0, 1, 2],
                "orientation": "rightHanded",
                "velocities": [0, 1, 0, 0, 2, 0, 0, 3, 0],
                "physics:masses": [3, 4, 5],
                "physics:restShapePoints": [0, 0, 0, 1.1, 0, 0, 0, 1.1, 0],
                "physics:restBendAngles": [0.25],
                "physics:restAdjTriPairs": [0, 0],
                "physics:restBendAnglesDefault": "flatDefault",
                "omni:fabric:worldMatrix": None,
            },
            (
                "points",
                "faceVertexCounts",
                "faceVertexIndices",
                "velocities",
                "physics:masses",
                "physics:restShapePoints",
                "physics:restBendAngles",
                "physics:restAdjTriPairs",
            ),
        ),
        "/Volume/Sim": (
            {
                "points": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                "tetVertexIndices": [0, 1, 2, 3],
                "orientation": "rightHanded",
                "velocities": [0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4],
                "physics:masses": [6, 7, 8, 9],
                "physics:restShapePoints": [0, 0, 0, 1.1, 0, 0, 0, 1.1, 0, 0, 0, 1.1],
                "omni:fabric:worldMatrix": None,
            },
            ("points", "tetVertexIndices", "velocities", "physics:masses", "physics:restShapePoints"),
        ),
        "/CableMaterial": (
            {
                "physics:thickness": [0.02],
                "physics:stretchStiffness": [11.0],
                "physics:shearStiffness": [12.0],
                "physics:bendStiffness": [13.0],
                "physics:twistStiffness": [14.0],
                "physics:density": [1000.0],
            },
            (),
        ),
        "/ClothMaterial": (
            {
                "physics:thickness": [0.01],
                "physics:stretchStiffness": [21.0],
                "physics:shearStiffness": [22.0],
                "physics:bendStiffness": [23.0],
                "physics:density": [1100.0],
            },
            (),
        ),
        "/VolumeMaterial": (
            {
                "physics:youngsModulus": [1000.0],
                "physics:poissonsRatio": [0.3],
                "physics:density": [1200.0],
            },
            (),
        ),
        "/Attachment": (
            {
                "physics:src0": ["/Cable/Sim"],
                "physics:src1": ["/Cloth/Sim"],
                "physics:type0": "body",
                "physics:type1": "tri",
                "physics:indices0": [0],
                "physics:indices1": [1],
                "physics:coords0": [0.25, 0, 0],
                "physics:coords1": [0.5, 0, 0],
                "physics:attachmentEnabled": [True],
                "physics:stiffness": [31.0],
                "physics:damping": [32.0],
            },
            (
                "physics:src0",
                "physics:src1",
                "physics:indices0",
                "physics:indices1",
                "physics:coords0",
                "physics:coords1",
            ),
        ),
        "/ElementFilter": (
            {
                "physics:src0": ["/Cloth/Sim"],
                "physics:src1": ["/Volume/Sim"],
                "physics:groupElemIndices0": [0],
                "physics:groupElemIndices1": [0],
                "physics:groupElemCounts0": [1],
                "physics:groupElemCounts1": [1],
                "physics:filterEnabled": [True],
            },
            (
                "physics:src0",
                "physics:src1",
                "physics:groupElemIndices0",
                "physics:groupElemIndices1",
                "physics:groupElemCounts0",
                "physics:groupElemCounts1",
            ),
        ),
    }
    path_attributes = {"physics:src0", "physics:src1"}
    token_attributes = {
        "type",
        "wrap",
        "orientation",
        "physics:type0",
        "physics:type1",
        "physics:restBendAnglesDefault",
    }

    with _physics_stage(usda, "aousd-deformable-contract") as (stage, pd):
        missing_schemas = []
        for schema, expected in api_paths.items():
            if not expected.issubset(_api_paths(stage, pd, 1, schema)):
                missing_schemas.append(schema)
        for prim_type, expected in type_paths.items():
            if _type_paths(stage, pd, prim_type, 1) != expected:
                missing_schemas.append(prim_type)
        if missing_schemas:
            pytest.xfail("ovpopulation does not mirror AOUSD deformable schemas: " + ", ".join(missing_schemas))

        missing_columns = []
        values = {}
        for path, (expected, ragged) in contracts.items():
            attributes = tuple(expected)
            with _path_list_query(stage, pd, [path]) as query:
                columns = read_columns(stage, pd, query, attributes, 1, ragged=ragged)
            values[path] = columns
            missing_columns.extend(f"{path}:{attribute}" for attribute in attributes if 0 not in columns[attribute])
        if missing_columns:
            pytest.xfail("ovpopulation does not mirror AOUSD deformable values: " + ", ".join(missing_columns))

        for path, (expected, _ragged) in contracts.items():
            for attribute, expected_value in expected.items():
                row = values[path][attribute][0]
                if attribute in path_attributes:
                    actual = [pd.path_to_string(int(item)) for item in row]
                    assert actual == expected_value
                elif attribute in token_attributes:
                    assert pd.token_to_string(int(row[0])) == expected_value
                elif expected_value is None:
                    assert row.size == 16
                else:
                    np.testing.assert_allclose(row, expected_value)
