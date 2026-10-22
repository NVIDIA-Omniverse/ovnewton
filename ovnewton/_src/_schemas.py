# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register USD schemas and select the population required by ovnewton."""

from __future__ import annotations

from importlib.metadata import distribution
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ovstage import population

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
_MJC_JOINT_UNSUPPORTED_PROPERTIES = (
    "mjc:actuatorfrcrange:min",
    "mjc:actuatorfrcrange:max",
    "mjc:actuatorfrclimited",
    "mjc:springdamper",
)
_MJC_JOINT_PROPERTIES = (
    *_MJC_JOINT_DYNAMICS.values(),
    *_MJC_JOINT_CUSTOM_ATTRIBUTES,
    *_MJC_JOINT_UNSUPPORTED_PROPERTIES,
)


def population_desc() -> population.Desc:
    """Select physics data and composed API-schema names for ovnewton imports.

    Use with ``population.open_usd_with_desc`` (or its string variant), alongside
    any rendering descriptors. This preserves unregistered schema names and
    supported MuJoCo joint and equality parameters, allowing the importer to identify
    equalities and detect unsupported physics. It does not register plugins or
    repopulate a stage; call :func:`ovnewton.register_usd_schemas` before the first
    population operation.
    """
    from ovstage import PopulationDomain, population  # noqa: PLC0415

    return population.Desc(
        domains=PopulationDomain.PHYSICS,
        selectors=(
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
