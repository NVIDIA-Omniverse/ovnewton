# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess

import pytest

from . import newton_assets_diff as nad


def _case(expectation):
    return nad.AssetCase("case", "folder", "scene.usda", expectation)


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


def test_mjc_allows_only_its_known_reference_dependency_error():
    ours = {"type": "UnsupportedPhysicsError", "message": "unsupported MjcJointAPI properties at /Joint"}
    base = {"outcome": "both-error", "ovnewton_error": ours}
    assert nad._matches_expectation(
        _case("mjc"),
        {**base, "reference_error": {"message": "ModuleNotFoundError: No module named 'mujoco'"}},
    )
    assert not nad._matches_expectation(
        _case("mjc"),
        {**base, "reference_error": {"message": "unexpected reference failure"}},
    )
