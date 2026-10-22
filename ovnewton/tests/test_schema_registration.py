# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import textwrap

import newton
import numpy as np
import ovstage
import pytest
from ovstage import PopulationDomain, population

import ovnewton


@pytest.mark.parametrize("domains", (PopulationDomain.PHYSICS, PopulationDomain.ALL))
@pytest.mark.parametrize("import_mode", ("add", "attach"))
def test_population_desc_preserves_unregistered_schema(domains, import_mode):
    usda = '''#usda 1.0
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
def PhysicsFixedJoint "Joint" (prepend apiSchemas = ["MjcJointAPI"]) {
    rel physics:body1 = </Body>
}
'''
    with ovstage.Stage("unsupported-authored-schema") as stage:
        population.open_usd_from_string_with_desc(
            stage,
            usda,
            ordinal=1,
            time=population.DEFAULT_TIME,
            descs=(population.Desc(domains=domains), ovnewton.population_desc()),
        )
        stage.advance_write_floor(1).wait()
        with pytest.raises(ovnewton.UnsupportedPhysicsError, match="unsupported MjcJointAPI properties at /Joint"):
            if import_mode == "add":
                ovnewton.add_ovstage(newton.ModelBuilder(), stage)
            else:
                ovnewton.attach_ovstage(stage)


@pytest.mark.parametrize("schema", ("", 'prepend apiSchemas = ["ApplicationDisplayAPI"]'))
def test_population_desc_keeps_supported_imports(schema):
    usda = f'''#usda 1.0
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {{}}
def PhysicsFixedJoint "Joint" ({schema}) {{
    rel physics:body1 = </Body>
}}
'''
    with ovstage.Stage("supported-authored-schema") as stage:
        population.open_usd_from_string_with_desc(
            stage, usda, ordinal=1, time=population.DEFAULT_TIME, descs=(ovnewton.population_desc(),)
        )
        stage.advance_write_floor(1).wait()
        builder = newton.ModelBuilder()
        ovnewton.add_ovstage(builder, stage)
        assert builder.body_count == 1
        assert builder.joint_count == 1


@pytest.mark.parametrize(
    "schema", ("MjcJointAPI", "MjcEqualityConnectAPI", "MjcEqualityWeldAPI", "MjcEqualityJointAPI")
)
def test_population_desc_uses_composed_schema_list(tmp_path, schema):
    # A stronger layer deletes an API supplied by a sublayer. The importer must
    # neither reject nor build an equality from the removed schema.
    (tmp_path / "base.usda").write_text('''#usda 1.0
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
def PhysicsFixedJoint "Joint" (prepend apiSchemas = ["MjcJointAPI"]) {
    rel physics:body1 = </Body>
}
'''.replace("MjcJointAPI", schema))
    scene = tmp_path / "scene.usda"
    scene.write_text('''#usda 1.0
(subLayers = [@base.usda@])
over "Joint" (delete apiSchemas = ["MjcJointAPI"]) {}
'''.replace("MjcJointAPI", schema))
    with ovstage.Stage("composed-authored-schema") as stage:
        population.open_usd_with_desc(
            stage, str(scene), ordinal=1, time=population.DEFAULT_TIME, descs=(ovnewton.population_desc(),)
        )
        stage.advance_write_floor(1).wait()
        builder = newton.ModelBuilder()
        ovnewton.add_ovstage(builder, stage)
        assert builder.joint_count == 1


def test_import_rejects_schema_column_without_raw_metadata():
    usda = '''#usda 1.0
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
def PhysicsFixedJoint "Joint" {
    rel physics:body1 = </Body>
}
'''
    with ovstage.Stage("schema-column-only") as stage, ovstage.PathDictionary(stage) as pd:
        population.open_usd_from_string(stage, usda, ordinal=1, domains=PopulationDomain.PHYSICS)
        stage.advance_write_floor(1).wait()
        paths = pd.create_path_list_from_strings(["/Joint"])
        try:
            with stage.query_from_path_list(paths) as query:
                stage.write_attribute(
                    query, "usd-schemas", ordinal=2,
                    tensors=np.asarray([pd.intern_token("MjcJointAPI")], dtype=np.uint64),
                    is_array=True,
                ).wait()
            stage.advance_write_floor(2).wait()
            with pytest.raises(
                ovnewton.OvstageContractError, match="requires population with ovnewton.population_desc"
            ):
                ovnewton.add_ovstage(newton.ModelBuilder(), stage, ordinal=2)
        finally:
            pd.destroy_path_list(paths)


def test_register_required_usd_schemas_before_population():
    script = textwrap.dedent(
        '''
        import ovnewton
        import ovstage
        from ovstage import PopulationDomain, population

        from ovnewton._src._parse import _api_paths, read_stage_units
        from ovnewton._src._stage import _path_list_query, read_columns


        ovnewton.register_usd_schemas()
        ovnewton.register_usd_schemas()

        # Exercise the PhysX schema used for initial joint state. Leave values
        # unauthored so their defaults prove the schema was registered.
        usda = """#usda 1.0
        (
            metersPerUnit = 0.5
            upAxis = "Y"
        )

        def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
        def PhysicsRevoluteJoint "Joint" (
            prepend apiSchemas = ["NewtonJointAPI", "PhysicsJointStateAPI:angular"]
        ) {
            rel physics:body1 = </Body>
        }
        """

        with ovstage.Stage("ovnewton-schema-registration") as stage, ovstage.PathDictionary(stage) as pd:
            population.open_usd_from_string(stage, usda, ordinal=1, domains=PopulationDomain.PHYSICS)
            stage.advance_write_floor(ordinal=1).wait()

            assert read_stage_units(stage, pd, 1) == (0.5, "Y")
            assert _api_paths(stage, pd, 1, "NewtonJointAPI") == {"/Joint"}
            assert _api_paths(stage, pd, 1, "PhysicsJointStateAPI:angular") == {"/Joint"}
            with _path_list_query(stage, pd, ["/Joint"]) as query:
                columns = read_columns(
                    stage, pd, query,
                    ("newton:armature", "state:angular:physics:position", "state:angular:physics:velocity"),
                    1,
                )
            assert columns["newton:armature"][0].tolist() == [0.0]
            assert columns["state:angular:physics:position"][0].tolist() == [0.0]
            assert columns["state:angular:physics:velocity"][0].tolist() == [0.0]

            binding = ovnewton.attach_ovstage(stage)
            assert binding.model.joint_count == 1
        '''
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
