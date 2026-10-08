# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for the ovstage backend.

Proves the installed ovstage package or an explicit source build can
populate ovnewton's physics inputs end-to-end:
``ovstage.population.open_usd`` mirrors a USD scene into an ovstage, and the
v2 query surface sees the expected rigid bodies / colliders / joints. This is the
package reproducibility check. It does NOT exercise the full reader path — for
that, see ``test_diff.py`` and ``test_byo.py``.

Run it:
    ovnewton/tests/run.sh ovnewton/tests/test_smoke.py
"""

from importlib.metadata import version

import ovstage
from ovstage import Filter, FilterOp, PopulationDomain, Predicate, population

import ovnewton
from ovnewton.examples import get_asset

CARTPOLE = get_asset("scene_cartpole.usda")


def test_public_version_matches_installed_distribution():
    assert ovnewton.__version__ == version("ovnewton")


def _prim_count(stage, predicates):
    query = stage.query(filter=Filter(predicates))
    query.wait()
    count = stage.fetch_query_result(query).total_prim_count
    stage.release_query(query).wait()
    return count


def test_populates_cartpole_physics():
    assert population.available(), (
        "ovstage.population bridge unavailable; install a compatible ovstage package"
    )
    with ovstage.Stage("ovnewton-smoke") as stage:
        population.open_usd(stage, CARTPOLE, ordinal=1, domains=PopulationDomain.ALL)
        stage.advance_write_floor(ordinal=1).wait()
        rigid_bodies = _prim_count(stage, [Predicate("usd-schemas", FilterOp.CONTAINS, ["PhysicsRigidBodyAPI"])])
        colliders = _prim_count(stage, [Predicate("usd-schemas", FilterOp.CONTAINS, ["PhysicsCollisionAPI"])])
        articulation_roots = _prim_count(
            stage,
            [Predicate("usd-schemas", FilterOp.CONTAINS, ["PhysicsArticulationRootAPI"])],
        )
        joints = _prim_count(stage, [Predicate("usd-prim-type", FilterOp.IN,
                                               ["PhysicsRevoluteJoint", "PhysicsPrismaticJoint",
                                                "PhysicsFixedJoint"])])
    # scene_cartpole.usda: cart + 2 poles + rail; 4 colliders; 4 joints
    # (rootJoint fixed, railCartJoint prismatic, cart/pole + pole/pole revolute).
    assert rigid_bodies == 4, f"expected 4 PhysicsRigidBodyAPI prims, got {rigid_bodies}"
    assert colliders == 4, f"expected 4 PhysicsCollisionAPI prims, got {colliders}"
    assert articulation_roots == 1, f"expected 1 PhysicsArticulationRootAPI prim, got {articulation_roots}"
    assert joints == 4, f"expected 4 joint prims, got {joints}"
