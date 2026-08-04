# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assert ovnewton's read layer reconstructs cartpole's rigid-body and joint
topology from a populated ovstage, using only the readable surface
(relationships + filter queries).

Run:
    ovnewton/tests/run.sh ovnewton/tests/test_topology.py
"""

import numpy as np
import ovstage
import pytest
from ovstage import Filter, FilterOp, PopulationDomain, Predicate, population

import ovnewton
from ovnewton._src import _parse, _stage
from ovnewton._src._errors import OvstageContractError
from ovnewton.examples import get_asset

CARTPOLE = get_asset("scene_cartpole.usda")

COLLISION_GROUPS = """#usda 1.0
def PhysicsCollisionGroup "GroupA" {
    rel collection:colliders:includes = </BodyX>
    rel physics:filteredGroups = </GroupB>
}
def PhysicsCollisionGroup "GroupB" {
    rel collection:colliders:includes = </BodyY>
    rel collection:colliders:excludes = </BodyX>
}
def Cube "BodyX" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {}
def Cube "BodyY" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {}
"""


def test_cartpole_topology_from_ovstage():
    with ovstage.Stage("ovnewton-topology") as stage, ovstage.PathDictionary(stage) as pd:
        population.open_usd(stage, CARTPOLE, ordinal=1, domains=PopulationDomain.ALL)
        stage.advance_write_floor(ordinal=1).wait()

        hierarchy = _parse.read_hierarchy(stage, pd, ordinal=1)
        body_ids, body_paths = _parse.read_bodies(stage, pd, ordinal=1, hierarchy=hierarchy)
        joints = _parse.read_joints(stage, pd, ordinal=1, hierarchy=hierarchy)

        def body_name(path):
            return path.rsplit("/", 1)[-1] if path else None

        topology = {(body_name(joint.body0), body_name(joint.body1)) for _, joint in joints}

        assert all(pd.path_to_string(body_id) == path for body_id, path in zip(body_ids, body_paths, strict=True))
        assert {body_name(path) for path in body_paths} == {"rail", "cart", "pole1", "pole2"}
        # 4 joints forming world->rail->cart->pole1->pole2 (parent=None is world).
        assert len(joints) == 4
        assert topology == {(None, "rail"), ("rail", "cart"), ("cart", "pole1"), ("pole1", "pole2")}
        # Each body carries exactly one Cube collider, assigned from materialized
        # collider paths rather than a separate prefix query per body.
        colliders = _parse.read_colliders(stage, pd, ordinal=1)
        for body_path in body_paths:
            cols = [c for c in colliders if c["path"] == body_path or c["path"].startswith(body_path + "/")]
            assert [c["type"] for c in cols] == ["Cube"], body_path


def test_query_paths_uses_readable_usd_path():
    with ovstage.Stage("ovnewton-query-paths") as stage, ovstage.PathDictionary(stage) as pd:
        population.open_usd_from_string(stage, COLLISION_GROUPS, ordinal=1, domains=PopulationDomain.ALL)
        stage.advance_write_floor(ordinal=1).wait()

        with stage.query(
            filter=Filter([Predicate("usd-prim-type", FilterOp.IN, ["PhysicsCollisionGroup"])])
        ) as query:
            count = _stage._count(stage, query)
            paths = _stage._query_paths(stage, pd, query, ordinal=1, expected_count=count)

    assert count == 2
    assert paths == ["/GroupA", "/GroupB"]


@pytest.mark.parametrize("domains", [PopulationDomain.ALL, PopulationDomain.PHYSICS])
def test_readable_parent_builds_authoritative_hierarchy(domains):
    usda = """#usda 1.0
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Cube "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {}
}
def Xform "BodyExtra" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
"""
    with ovstage.Stage("ovnewton-readable-parent") as stage, ovstage.PathDictionary(stage) as pd:
        population.open_usd_from_string(stage, usda, ordinal=1, domains=domains)
        stage.advance_write_floor(ordinal=1).wait()
        hierarchy = _parse.read_hierarchy(stage, pd, ordinal=1)

    assert hierarchy.parents["/Body/Collider"] == "/Body"
    assert hierarchy.contains("/Body", "/Body/Collider")
    assert not hierarchy.contains("/Body", "/BodyExtra")
    assert hierarchy.nearest("/Body/Collider", {"/Body"}) == "/Body"


def test_readable_prim_type_groups_materialized_paths():
    with ovstage.Stage("ovnewton-readable-prim-type") as stage, ovstage.PathDictionary(stage) as pd:
        population.open_usd_from_string(stage, COLLISION_GROUPS, ordinal=1, domains=PopulationDomain.ALL)
        stage.advance_write_floor(ordinal=1).wait()
        paths_by_type = _parse._paths_by_prim_type(stage, pd, ["/GroupA", "/BodyX"], ordinal=1)

    assert paths_by_type == {"PhysicsCollisionGroup": ["/GroupA"], "Cube": ["/BodyX"]}


def test_filtered_paths_materializes_untyped_prims():
    usda = '''#usda 1.0
def "Untyped" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
def Xform "Typed" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
'''
    with ovstage.Stage("ovnewton-untyped-path") as stage, ovstage.PathDictionary(stage) as pd:
        population.open_usd_from_string(stage, usda, ordinal=1, domains=PopulationDomain.ALL)
        stage.advance_write_floor(ordinal=1).wait()
        paths = _parse._filtered_paths(
            stage,
            pd,
            [Predicate("usd-schemas", FilterOp.CONTAINS, ["PhysicsRigidBodyAPI"])],
            ordinal=1,
        )

    assert paths == ["/Typed", "/Untyped"]


def test_batched_site_read_preserves_type_order():
    usda = '''#usda 1.0
def Sphere "ASphere" (prepend apiSchemas = ["NewtonSiteAPI"]) {}
def Cube "ZCube" (prepend apiSchemas = ["NewtonSiteAPI"]) {}
'''
    with ovstage.Stage("ovnewton-site-order") as stage, ovstage.PathDictionary(stage) as pd:
        population.open_usd_from_string(stage, usda, ordinal=1, domains=PopulationDomain.ALL)
        stage.advance_write_floor(ordinal=1).wait()
        sites = _parse.read_sites(stage, pd, ordinal=1)

    assert [(site["type"], site["path"]) for site in sites] == [
        ("Cube", "/ZCube"),
        ("Sphere", "/ASphere"),
    ]


def test_model_import_batches_initial_columns():
    with ovstage.Stage("ovnewton-import-batches") as stage, ovstage.PathDictionary(stage) as pd:
        population.open_usd(stage, CARTPOLE, ordinal=1, domains=PopulationDomain.ALL)
        stage.advance_write_floor(ordinal=1).wait()

        batches = []
        original_read = stage.read_attributes

        def tracked_read(query, attributes, ordinal_range):
            batches.append({pd.token_to_string(token) for token in attributes})
            return original_read(query, attributes, ordinal_range)

        stage.read_attributes = tracked_read
        ovnewton.attach_ovstage(stage)

    assert any({"usd-path", "usd-parent"}.issubset(batch) for batch in batches)
    assert any(
        {
            "omni:fabric:worldMatrix",
            "physics:mass",
            "physics:diagonalInertia",
            "physics:velocity",
            "physics:angularVelocity",
        }.issubset(batch)
        for batch in batches
    )
    assert any(
        {
            "physics:body0",
            "physics:body1",
            "physics:localPos0",
            "physics:collisionEnabled",
            "newton:damping",
            "newton:limitStiffness",
            "newton:limitDamping",
        }.issubset(batch)
        for batch in batches
    )
    assert any(
        {
            "omni:fabric:worldMatrix",
            "material:binding:physics",
            "physics:collisionEnabled",
            "extent",
        }.issubset(batch)
        for batch in batches
    )


def test_fixed_read_rejects_malformed_row_width_instead_of_defaulting():
    class Operation:
        def wait(self):
            return None

    class Group:
        attribute = 1
        ordinal = 1
        is_delete = False
        tensor_count = 1
        prim_count = 2
        has_data_index_map = False

        def array(self, _index):
            return np.asarray([1.0, 2.0, 3.0], dtype=np.float32)

        def data_row_index(self, local):
            return local

        def prim_index(self, local):
            return local

    class Read:
        def __init__(self):
            self.group = Group()

        def wait(self):
            return None

        def fetch_next(self):
            group, self.group = self.group, None
            return group

        def release(self):
            return Operation()

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _tb):
            self.release().wait()

    class Stage:
        def read_attributes(self, *_args):
            return Read()

        def release_group(self, _group):
            return None

    class PathDictionary:
        def intern_token(self, _attribute):
            return 1

    with pytest.raises(OvstageContractError, match="3 values for 2 logical rows"):
        _stage.read_fixed(Stage(), PathDictionary(), object(), "physics:mass", ordinal=1)
