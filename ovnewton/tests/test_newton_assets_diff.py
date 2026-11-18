# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path

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
    report_path = tmp_path / "reports" / "corpus.json"
    result = subprocess.run(
        [sys.executable, nad.__file__, "--asset-root", str(tmp_path), "--asset", "dr_legs", "--strict",
         "--report", str(report_path)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "summary: match=1" in result.stdout
    assert "supported and matching: 1/1" in result.stdout
    report = json.loads(report_path.read_text())
    assert report["schema_version"] == 1
    assert report["summary"] == {"outcomes": {"match": 1}, "total": 1, "matching": 1, "expected": 1, "unexpected": 0}
    assert report["metadata"]["packages"]["newton"]
    assert report["metadata"]["ovnewton_source"] == str(Path(nad.__file__).resolve().parents[2])
    assert report["assets"]["disneyresearch"]["commit"] is None
    assert report["results"][0]["mismatches"] == []
    assert report["results"][0]["worker_seconds"] > 0
    assert "OVNEWTON_ASSET_RESULT" in report["results"][0]["stdout"]


def test_cli_preserves_native_collision_instance_detection(tmp_path):
    asset = tmp_path / "anybotics_anymal_c/usd/anymal_c.usda"
    asset.parent.mkdir(parents=True)
    asset.write_text('''#usda 1.0
def Xform "Prototype" {
    def Sphere "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        double radius = 0.4
    }
}
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def "Collisions" (
        instanceable = true
        prepend references = </Prototype>
    ) {}
}
''', encoding="utf-8")
    result = subprocess.run(
        [sys.executable, nad.__file__, "--asset-root", str(tmp_path), "--asset", "anymal_c", "--strict"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    expected = (
        "native USD physics instancing cannot preserve composed collider properties "
        "at /Body/Collisions/Collider"
    )
    assert expected in result.stdout
    assert "manifest: expected=1, unexpected=0" in result.stdout


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
        [sys.executable, nad.__file__, "--asset-root", str(tmp_path), "--asset", "robotiq_2f85",
         "--profile", "core", "--strict"],
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


@pytest.mark.parametrize("expectation,message", [
    ("native-instancing",
     "native USD physics instancing cannot preserve composed collider properties at /Robot/Collider"),
    ("physics-point-instancer", "physics PointInstancer is unsupported at /Visuals/goal_marker"),
    ("mjc-solver-registration",
     "MjcActuator requires SolverMuJoCo.register_custom_attributes(builder) before add_ovstage"),
    ("mjc-solver-registration",
     "mjc:solreflimit at /Joint requires SolverMuJoCo.register_custom_attributes(builder) before add_ovstage"),
])
def test_recorded_rejections_require_successful_reference(expectation, message):
    ours = {"type": "UnsupportedPhysicsError", "message": message}
    assert nad._matches_expectation(
        _case(expectation),
        {"outcome": "ovnewton-error", "ovnewton_error": ours},
    )
    assert not nad._matches_expectation(
        _case(expectation),
        {"outcome": "both-error", "ovnewton_error": ours, "reference_error": {"message": "unexpected"}},
    )


@pytest.mark.parametrize("property_line", [
    "double[] mjc:springdamper = [0.1, 1]",
    "double mjc:armature = -1",
])
def test_cli_rejects_unexpected_mjc_errors(tmp_path, property_line):
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
    assert result.returncode == 1, result.stdout + result.stderr
    assert "summary: ovnewton-error=1" in result.stdout


def test_cli_reports_missing_assets_without_counting_them_as_supported(tmp_path):
    report_path = tmp_path / "corpus.json"
    result = subprocess.run(
        [sys.executable, nad.__file__, "--asset-root", str(tmp_path), "--asset", "dr_testmech", "--strict",
         "--report", str(report_path)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    report = json.loads(report_path.read_text())
    assert report["summary"] == {"outcomes": {"missing": 1}, "total": 1, "matching": 0, "expected": 0, "unexpected": 1}
    assert report["results"][0]["expectation"] == "match"
    assert report["results"][0]["outcome"] == "missing"
    assert report["assets"]["disneyresearch"]["commit"] is None


@pytest.mark.parametrize("payload", ['{"outcome":', "[]", "{}"])
def test_malformed_worker_result_keeps_failure_logs(monkeypatch, tmp_path, payload):
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=nad._RESULT_PREFIX + payload, stderr="worker diagnostic",
    )
    monkeypatch.setattr(nad.subprocess, "run", lambda *args, **kwargs: completed)
    result = nad._run_worker(tmp_path / "scene.usda", _case("match"), timeout=1.0, verbose=False)
    assert result["outcome"] == "worker-error"
    assert result["stdout"] == completed.stdout
    assert result["stderr"] == "worker diagnostic"


def test_cli_does_not_attribute_ignored_assets_to_parent_checkout(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    (tmp_path / ".gitignore").write_text("cache/\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", ".gitignore"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Test", "-c", "user.email=test@example.org",
                    "commit", "-m", "Fixture"], check=True, capture_output=True)
    asset = tmp_path / "cache/disneyresearch/dr_testmech/usd/dr_testmech.usda"
    asset.parent.mkdir(parents=True)
    asset.write_text("#usda 1.0\ndef Xform \"Empty\" {}\n")
    report_path = tmp_path / "report.json"
    result = subprocess.run(
        [sys.executable, nad.__file__, "--asset-root", str(tmp_path / "cache"), "--asset", "dr_testmech",
         "--report", str(report_path)], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    provenance = json.loads(report_path.read_text())["assets"]["disneyresearch"]
    assert provenance["commit"] is None
    assert provenance["dirty"] is None


def test_source_provenance_with_and_without_git(tmp_path, monkeypatch):
    source = tmp_path / "ovnewton/tests/newton_assets_diff.py"
    source.parent.mkdir(parents=True)
    source.write_text("# source fixture\n")
    monkeypatch.setattr(nad, "__file__", str(source))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    metadata = nad._report_metadata()
    assert metadata["ovnewton_source"] == str(tmp_path)
    assert metadata["ovnewton_commit"] is None
    assert metadata["ovnewton_dirty"] is None

    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()

    git("init", "-q")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.org", "-c", "commit.gpgsign=false",
        "commit", "-qm", "Source fixture")
    metadata = nad._report_metadata()
    assert metadata["ovnewton_commit"] == git("rev-parse", "HEAD")
    assert metadata["ovnewton_dirty"] is False
    source.write_text("# changed source fixture\n")
    assert nad._report_metadata()["ovnewton_dirty"] is True


def test_cli_mujoco_profile_registers_both_builders(tmp_path):
    pytest.importorskip("mujoco")
    asset = tmp_path / "disneyresearch/dr_legs/usd/dr_legs.usda"
    asset.parent.mkdir(parents=True)
    asset.write_text('''#usda 1.0
    def Cube "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {}
    def PhysicsRevoluteJoint "Joint" (prepend apiSchemas = ["MjcJointAPI"]) {
        rel physics:body1 = </Body>
        double[] mjc:solreflimit = [0.04, 0.8]
        double mjc:armature = 0.2
        double mjc:damping = 3
        double mjc:actuatorfrcrange:min = -2
        double mjc:actuatorfrcrange:max = 4
    }
    ''')
    report_path = tmp_path / "mujoco.json"
    result = subprocess.run(
        [sys.executable, nad.__file__, "--asset-root", str(tmp_path), "--asset", "dr_legs", "--profile", "mujoco",
         "--strict", "--report", str(report_path)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(report_path.read_text())
    assert report["results"][0]["profile"] == "mujoco"
    assert report["results"][0]["outcome"] == "match"


@pytest.mark.parametrize("profile,expectation", [("core", "mjc-solver-registration"), ("auto", "mjc-actuator-joint")])
def test_cli_lists_the_selected_profile_expectation(profile, expectation):
    result = subprocess.run(
        [sys.executable, nad.__file__, "--list", "--asset", "robotiq_2f85", "--profile", profile],
        check=True, capture_output=True, text=True,
    )
    assert expectation in result.stdout



def test_cli_mujoco_profile_preserves_orphan_joint_finalization(tmp_path):
    from .test_add_ovstage import _ROOTLESS_JOINTS

    asset = tmp_path / "disneyresearch/dr_legs/usd/dr_legs.usda"
    asset.parent.mkdir(parents=True)
    asset.write_text(_ROOTLESS_JOINTS)
    report_path = tmp_path / "rootless.json"
    result = subprocess.run(
        [sys.executable, nad.__file__, "--asset-root", str(tmp_path), "--asset", "dr_legs",
         "--profile", "mujoco", "--strict", "--report", str(report_path)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(report_path.read_text())["results"][0]["outcome"] == "match"


def test_cli_requires_solver_reference_for_force_limited_assets():
    result = subprocess.run(
        [sys.executable, nad.__file__, "--asset", "apptronik_apollo", "--profile", "core", "--strict"],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "MuJoCo reference profile required for: apptronik_apollo" in result.stderr


@pytest.mark.parametrize("expectation,message", [
    ("mjc-actuator-joint", "MjcActuator at /Motor requires an unmerged scalar joint: /Joint"),
    ("mjc-actuator-configuration",
     "MjcActuator at /Motor requires fixed gain, affine bias and unit gear without dynamics"),
])
def test_actuator_expectations_require_the_recorded_error(expectation, message):
    result = {"outcome": "ovnewton-error", "ovnewton_error": {"type": "UnsupportedPhysicsError", "message": message}}
    assert nad._matches_expectation(_case(expectation), result)
    result["outcome"] = "both-error"
    assert not nad._matches_expectation(_case(expectation), result)
    result["outcome"] = "ovnewton-error"
    result["ovnewton_error"]["message"] = "MjcActuator at /Motor has an unexpected target"
    assert not nad._matches_expectation(_case(expectation), result)
