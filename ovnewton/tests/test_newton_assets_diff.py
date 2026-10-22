# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys

import pytest

from . import newton_assets_diff as nad


def test_cli_registers_newton_schemas_before_population(tmp_path):
    asset = tmp_path / "disneyresearch/dr_legs/usd/dr_legs.usda"
    asset.parent.mkdir(parents=True)
    asset.write_text('''#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Cube "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI", "NewtonCollisionAPI"]) {
        float newton:contactGap = 0
    }
    def Sphere "Site" (prepend apiSchemas = ["NewtonSiteAPI"]) {}
}
''', encoding="utf-8")
    result = subprocess.run(
        [sys.executable, nad.__file__, "--asset-root", str(tmp_path), "--asset", "dr_legs", "--strict"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "summary: match=1" in result.stdout


def _case(expectation):
    return nad.AssetCase("case", "folder", "scene.usda", expectation)


def test_cli_records_mujoco_solver_registration_requirement(tmp_path):
    pytest.importorskip("mujoco")
    asset = tmp_path / "robotiq_2f85_v4/usd_structured/Dual_wrist_camera.usda"
    asset.parent.mkdir(parents=True)
    asset.write_text('''#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Cube "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {}
def PhysicsRevoluteJoint "Joint" (prepend apiSchemas = ["MjcJointAPI"]) {
    rel physics:body1 = </Body>
    double[] mjc:solreflimit = [0.04, 0.8]
    double mjc:armature = 0.2
    double mjc:damping = 3
}
''', encoding="utf-8")
    result = subprocess.run(
        [sys.executable, nad.__file__, "--asset-root", str(tmp_path), "--asset", "robotiq_2f85", "--strict"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "summary: ovnewton-error=1" in result.stdout
    assert "requires SolverMuJoCo.register_custom_attributes(builder) before add_ovstage" in result.stdout


def test_source_environment_limits_pxr_workers(monkeypatch):
    monkeypatch.delenv("PXR_WORK_THREAD_LIMIT", raising=False)
    assert nad._source_environment()["PXR_WORK_THREAD_LIMIT"] == "1"

    monkeypatch.setenv("PXR_WORK_THREAD_LIMIT", "4")
    assert nad._source_environment()["PXR_WORK_THREAD_LIMIT"] == "4"


def test_worker_nonzero_exit_rejects_result_sentinel(monkeypatch, tmp_path):
    payload = {"name": "case", "asset": str(tmp_path / "scene.usda"), "outcome": "match"}
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=1,
        stdout=nad._RESULT_PREFIX + json.dumps(payload),
        stderr="native teardown failed",
    )
    monkeypatch.setattr(nad.subprocess, "run", lambda *args, **kwargs: completed)

    result = nad._run_worker(tmp_path / "scene.usda", _case("match"), timeout=1.0, verbose=False)
    assert result["outcome"] == "worker-crash"
    assert result["returncode"] == 1


def test_latest_manifest_extends_pinned_corpus():
    pinned = {case.name: case for case in nad.PINNED_ASSET_CASES}
    latest = {case.name: case for case in nad.LATEST_ASSET_CASES}

    assert len(pinned) == 22
    assert len(latest) == 24
    assert {"dr_legs_no_meshes", "fourbar_pole"} <= latest.keys()
    assert nad._select_cases(["fourbar*"], nad.LATEST_ASSET_CASES) == [latest["fourbar_pole"]]
    with pytest.raises(ValueError, match="matched nothing"):
        nad._select_cases(["fourbar*"], nad.PINNED_ASSET_CASES)


def test_native_instancing_requires_successful_reference():
    ours = {
        "type": "UnsupportedPhysicsError",
        "message": "native USD physics instancing cannot preserve composed collider properties at /Robot/Collider",
    }
    assert nad._matches_expectation(
        _case("native-instancing"),
        {"outcome": "ovnewton-error", "ovnewton_error": ours},
    )
    assert not nad._matches_expectation(
        _case("native-instancing"),
        {"outcome": "both-error", "ovnewton_error": ours, "reference_error": {"message": "unexpected"}},
    )


@pytest.mark.parametrize("property_line,expected", [
    ("double mjc:actuatorfrcrange:min = -1", True),
    ("double[] mjc:springdamper = [0.1, 1]", False),
    ("double mjc:armature = -1", False),
])
def test_cli_accepts_only_the_recorded_mjc_limitation(tmp_path, property_line, expected):
    pytest.importorskip("mujoco")
    asset = tmp_path / "apptronik_apollo/usd_structured/apptronik_apollo.usda"
    asset.parent.mkdir(parents=True)
    asset.write_text(f'''#usda 1.0
def Cube "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {{}}
def PhysicsRevoluteJoint "Joint" (prepend apiSchemas = ["MjcJointAPI"]) {{
    rel physics:body1 = </Body>
    {property_line}
}}
''', encoding="utf-8")
    result = subprocess.run(
        [sys.executable, nad.__file__, "--asset-root", str(tmp_path), "--asset", "apptronik_apollo", "--strict"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == (0 if expected else 1), result.stdout + result.stderr
    assert "summary: ovnewton-error=1" in result.stdout
