# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import textwrap


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
