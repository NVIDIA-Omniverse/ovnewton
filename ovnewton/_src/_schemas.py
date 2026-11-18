# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register USD schemas and select the population required by ovnewton."""

from __future__ import annotations

from importlib.metadata import distribution
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ovstage import population

# Keep missing joint overrides distinct from authored zero/-inf values.
_NEWTON_JOINT_AUTHORED_PROPERTIES = (
    "newton:armature",
    "newton:damping",
    "newton:friction",
    "newton:limitStiffness",
    "newton:limitDamping",
)

_MJC_EQUALITY_PROPERTIES = (
    "mjc:solref",
    "mjc:solimp",
    "mjc:torqueScale",
    "newton:mimicJoint",
    "newton:mimicEnabled",
    "newton:mimicCoef0",
    "newton:mimicCoef1",
    "mjc:target",
    "mjc:coef0",
    "mjc:coef1",
    "mjc:coef2",
    "mjc:coef3",
    "mjc:coef4",
)

_MJC_JOINT_DYNAMICS = {
    "armature": "mjc:armature",
    "damping": "mjc:damping",
    "friction": "mjc:frictionloss",
}
_MJC_JOINT_CUSTOM_ATTRIBUTES = {
    "mjc:margin": ("limit_margin", 1),
    "mjc:solreflimit": ("solreflimit", 2),
    "mjc:solimplimit": ("solimplimit", 5),
    "mjc:solreffriction": ("solreffriction", 2),
    "mjc:solimpfriction": ("solimpfriction", 5),
    "mjc:stiffness": ("dof_passive_stiffness", 1),
    "mjc:springref": ("dof_springref", 1),
    "mjc:ref": ("dof_ref", 1),
    "mjc:actuatorgravcomp": ("jnt_actgravcomp", 1),
}
_MJC_JOINT_UNSUPPORTED_PROPERTIES = ("mjc:springdamper",)
_MJC_JOINT_PROPERTIES = (
    *_MJC_JOINT_DYNAMICS.values(),
    *_MJC_JOINT_CUSTOM_ATTRIBUTES,
    "mjc:actuatorfrcrange:min",
    "mjc:actuatorfrcrange:max",
    "mjc:actuatorfrclimited",
    *_MJC_JOINT_UNSUPPORTED_PROPERTIES,
)


_MJC_ACTUATOR_UNSUPPORTED_PROPERTIES = (
    "mjc:actDim",
    "mjc:actLimited",
    "mjc:actRange:min",
    "mjc:actRange:max",
    "mjc:actEarly",
    "mjc:lengthRange:min",
    "mjc:lengthRange:max",
    "mjc:dynPrm",
)
_MJC_ACTUATOR_PROPERTIES = (
    "mjc:target",
    "mjc:gainPrm",
    "mjc:biasPrm",
    "mjc:gainType",
    "mjc:biasType",
    "mjc:dynType",
    "mjc:gear",
    "mjc:inheritRange",
    "mjc:ctrlLimited",
    "mjc:ctrlRange:min",
    "mjc:ctrlRange:max",
    "mjc:forceLimited",
    "mjc:forceRange:min",
    "mjc:forceRange:max",
    *_MJC_ACTUATOR_UNSUPPORTED_PROPERTIES,
)


def population_desc() -> population.Desc:
    """Select physics data and composed API-schema names for ovnewton imports.

    Use with ``population.open_usd_with_desc`` (or its string variant), alongside
    any rendering descriptors. This preserves unregistered schema names and
    supported MuJoCo joint and equality parameters, allowing the importer to identify
    equalities and detect unsupported physics. It also resolves physics materials
    and collision-group collections while the USD stage is available. It does not
    register plugins or repopulate a stage; call :func:`ovnewton.register_usd_schemas`
    before the first population operation.
    """
    from ovstage import population  # noqa: PLC0415

    from . import _deformables  # noqa: PLC0415

    return population.Desc(
        selectors=(
            *_physics_selectors(),
            *_deformables.population_desc().selectors,
            population.Selector(
                prim_predicate=population.PrimPredicate.has_type("MjcActuator"),
                property_predicate=population.PropertyPredicate.and_(
                    population.PropertyPredicate.has_name(*_MJC_ACTUATOR_PROPERTIES),
                    population.PropertyPredicate.is_authored(),
                ),
            ),
            population.Selector(
                prim_predicate=population.PrimPredicate.has_metadata("apiSchemas"),
                property_predicate=population.PropertyPredicate.has_name(
                    *_MJC_EQUALITY_PROPERTIES, *_MJC_JOINT_PROPERTIES
                ),
                prim_metadata_paths=("apiSchemas",),
            ),
            population.Selector(
                prim_predicate=population.PrimPredicate.has_type("PhysicsScene"),
                property_predicate=population.PropertyPredicate.has_name("mjc:compiler:angle"),
            ),
        ),
        stage_metadata_paths=("metersPerUnit", "upAxis"),
    )


def _physics_selectors():
    """Select simulation inputs and resolve physics materials and collections."""
    import numpy as np  # noqa: PLC0415
    from ovstage import AttributeSemantic, numpy_to_dldatatype, population  # noqa: PLC0415

    prim = population.PrimPredicate
    prop = population.PropertyPredicate
    return (
        population.Selector(
            prim_predicate=prim.all(),
            property_predicate=prop.and_(
                prop.in_namespace("physics", "newton", "state", "drive", "limit"),
                prop.or_(prop.not_(prop.has_name(*_NEWTON_JOINT_AUTHORED_PROPERTIES)), prop.is_authored()),
            ),
        ),
        population.Selector(
            prim_predicate=prim.is_a_type("Gprim"),
            property_predicate=prop.has_name(
                "extent", "size", "radius", "height", "axis",
                "points", "faceVertexCounts", "faceVertexIndices", "orientation"
            ),
        ),
        population.Selector(
            prim_predicate=prim.has_type("PointInstancer"),
            property_predicate=prop.has_name("prototypes"),
        ),
        population.Selector(
            prim_predicate=prim.has_type("PhysicsCollisionGroup"),
            derived_properties=(
                population.DerivedProperty(
                    "collection:colliders:members", population.Expression.resolved_collection_members("colliders"),
                    dtype=numpy_to_dldatatype(np.dtype(np.uint64)),
                    is_array=True, semantic=AttributeSemantic.RELATIONSHIP_PATH_ID,
                ),
            ),
        ),
        population.Selector(
            prim_predicate=prim.has_schema("PhysicsCollisionAPI"),
            derived_properties=(
                population.DerivedProperty(
                    "material:binding:physics", population.Expression.resolved_bound_material("physics"),
                    dtype=numpy_to_dldatatype(np.dtype(np.uint64)),
                    is_array=True, semantic=AttributeSemantic.RELATIONSHIP_PATH_ID,
                ),
            ),
        ),
    )


def register_usd_schemas() -> None:
    """Register ovnewton's USD schemas before the first USD schema read.

    Registration is process-wide, irreversible, and safe to repeat. It must run
    before any USD schema definitions are read in the process, including by an
    ovstage population or export call.
    """
    from ovstage import population  # noqa: PLC0415

    descriptors = [
        distribution("newton-usd-schemas").locate_file("newton_usd_schemas/plugInfo.json"),
        distribution("physx-usd-schemas").locate_file("physx_usd_schemas/PhysxSchema/resources/plugInfo.json"),
    ]
    population.register_usd_schemas(descriptors)
