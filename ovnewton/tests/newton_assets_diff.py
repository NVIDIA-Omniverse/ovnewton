# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Differential audit of physics scenes from Newton's pinned asset corpus.

This is an opt-in runner rather than a pytest module. Each asset runs in its
own process so a native USD failure cannot stop the rest of the audit; that
worker retains the existing add_usd subprocess boundary from diff_harness.
"""

from __future__ import annotations

import argparse
import fnmatch
import importlib.metadata
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

_RESULT_PREFIX = "OVNEWTON_ASSET_RESULT "


@dataclass(frozen=True)
class AssetCase:
    name: str
    folder: str
    scene: str
    expectation: str | None
    profile: str = "core"
    mujoco_expectation: str | None = None

    def expected_outcome(self, profile: str) -> str | None:
        return self.mujoco_expectation if profile == "mujoco" and self.mujoco_expectation else self.expectation


# Entry scenes with authored physics. Payload, prototype, and visual-only files
# are exercised through their owning scene rather than treated as roots.
PINNED_ASSET_CASES = (
    AssetCase("anymal_c", "anybotics_anymal_c", "usd/anymal_c.usda", "native-instancing"),
    AssetCase("anymal_d", "anybotics_anymal_d", "usd/anymal_d.usda", "native-instancing"),
    AssetCase(
        "apptronik_apollo", "apptronik_apollo", "usd_structured/apptronik_apollo.usda",
        None, "mujoco", "match"
    ),
    AssetCase("booster_t1", "booster_t1", "usd_structured/T1.usda",
              None, "mujoco", "match"),
    AssetCase("dr_legs", "disneyresearch", "dr_legs/usd/dr_legs.usda", "match"),
    AssetCase("dr_legs_boxes", "disneyresearch", "dr_legs/usd/dr_legs_with_boxes.usda", "match"),
    AssetCase(
        "dr_legs_meshes_boxes",
        "disneyresearch",
        "dr_legs/usd/dr_legs_with_meshes_and_boxes.usda",
        "match",
    ),
    AssetCase("dr_testmech", "disneyresearch", "dr_testmech/usd/dr_testmech.usda", "match"),
    AssetCase("robotiq_2f85", "robotiq_2f85_v4", "usd_structured/Dual_wrist_camera.usda",
              "mjc-solver-registration", "mujoco", "mjc-actuator-joint"),
    AssetCase("shadow_hand", "shadow_hand", "usd_structured/left_shadow_hand.usda",
              "mjc-solver-registration", "mujoco", "mjc-actuator-joint"),
    AssetCase("unitree_g1", "unitree_g1", "usd/g1.usd", "native-instancing"),
    AssetCase("unitree_g1_hands", "unitree_g1", "usd/g1_29dof_with_hand_rev_1_0.usd", "match"),
    AssetCase("unitree_g1_isaac", "unitree_g1", "usd/g1_isaac.usd", "native-instancing"),
    AssetCase("unitree_g1_minimal", "unitree_g1", "usd/g1_minimal.usd", "match"),
    AssetCase(
        "unitree_g1_structured",
        "unitree_g1",
        "usd_structured/g1_29dof_with_hand_rev_1_0.usda",
        None, "mujoco", "match",
    ),
    AssetCase("unitree_go2", "unitree_go2", "usd/go2.usda", "native-instancing"),
    AssetCase("unitree_h1_minimal", "unitree_h1", "usd/h1_minimal.usda", "native-instancing"),
    AssetCase("unitree_h1_structured", "unitree_h1", "usd_structured/h1.usda",
              "mjc-solver-registration", "mujoco", "mjc-actuator-configuration"),
    AssetCase("ur10_instanceable", "universal_robots_ur10", "usd/ur10_instanceable.usda", "match"),
    AssetCase("ur5e", "universal_robots_ur5e", "usd_structured/ur5e.usda",
              "mjc-solver-registration", "mujoco", "match"),
    AssetCase(
        "wonik_allegro",
        "wonik_allegro",
        "usd/allegro_left_hand_with_cube.usda",
        "physics-point-instancer",
    ),
    AssetCase("wonik_allegro_structured", "wonik_allegro", "usd_structured/allegro_left.usda",
              "mjc-solver-registration", "mujoco", "match"),
)

LATEST_ASSET_CASES = PINNED_ASSET_CASES + (
    AssetCase(
        "dr_legs_no_meshes",
        "disneyresearch",
        "dr_legs/usd/dr_legs_no_meshes.usda",
        "match",
    ),
    AssetCase("fourbar_pole", "fourbar_pole", "usd/fourbar_pole.usda", "native-instancing"),
)


def _source_environment() -> dict[str, str]:
    env = dict(os.environ)
    # Avoid known OpenUSD crashes in UsdPhysics.LoadUsdPhysicsFromRange.
    env.setdefault("PXR_WORK_THREAD_LIMIT", "1")
    repo = pathlib.Path(__file__).resolve().parents[2]
    paths = [str(repo), str(pathlib.Path(__file__).resolve().parent)]
    newton_source = env.get("OVNEWTON_NEWTON_SOURCE")
    if newton_source:
        paths.insert(0, newton_source)
    existing = env.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def _error(exc: Exception) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def _worker(
    asset: pathlib.Path,
    name: str,
    profile: str,
) -> int:
    import diff_harness as dh
    import ovstage
    from ovstage import PopulationDomain, population

    import ovnewton

    ref = ours = None
    ref_error = ours_error = None
    try:
        ref = dh.reference_model(
            str(asset),
            load_visual_shapes=False,
            include_mjc_resolver=profile == "mujoco",
        )
    except Exception as exc:
        ref_error = _error(exc)

    try:
        ovnewton.register_usd_schemas()
        with ovstage.Stage(f"asset-diff-{name}") as stage:
            # Rendering population also exposes composed collider instances to the audit.
            population.open_usd_with_desc(
                stage, str(asset), ordinal=1, time=population.DEFAULT_TIME,
                descs=(population.Desc(domains=PopulationDomain.RENDERING), ovnewton.population_desc()),
            )
            stage.advance_write_floor(ordinal=1).wait()
            if profile == "mujoco":
                import newton

                builder = newton.ModelBuilder()
                newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
                imported = ovnewton.add_ovstage(builder, stage)
                if builder.particle_count:
                    builder.color(include_bending=True)
                model = builder.finalize(skip_validation_joints=imported.has_orphan_joints)
                ovnewton.attach_ovstage(stage, model=model)
            else:
                model = ovnewton.attach_ovstage(stage).model
            ours = dh._model_arrays(model)
    except Exception as exc:
        ours_error = _error(exc)

    result: dict[str, object] = {"name": name, "asset": str(asset)}
    if ref_error or ours_error:
        if ref_error and ours_error:
            result["outcome"] = "both-error"
        elif ref_error:
            result["outcome"] = "reference-error"
        else:
            result["outcome"] = "ovnewton-error"
        result["reference_error"] = ref_error
        result["ovnewton_error"] = ours_error
    else:
        mismatches = dh.compare_models(
            ref,
            ours,
            check_joints=True,
            check_articulations=True,
            check_shapes=True,
        )
        result["outcome"] = "match" if not mismatches else "mismatch"
        result["mismatches"] = mismatches
        result["reference_counts"] = {
            "bodies": len(ref["body_label"]),
            "shapes": len(ref["shape_label"]),
            "joints": len(ref["joint_label"]),
            "mujoco_actuators": len(ref.get("mjc_actuators", {})),
        }
        result["ovnewton_counts"] = {
            "bodies": len(ours["body_label"]),
            "shapes": len(ours["shape_label"]),
            "joints": len(ours["joint_label"]),
            "mujoco_actuators": len(ours.get("mjc_actuators", {})),
        }

    print(_RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)
    return 0


def _run_worker(
    asset: pathlib.Path, case: AssetCase, timeout: float, verbose: bool, profile: str = "core"
) -> dict[str, object]:
    command = [sys.executable, __file__, "--worker", str(asset), "--worker-name", case.name, "--profile", profile]
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=_source_environment(),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "name": case.name, "asset": str(asset), "outcome": "timeout", "message": str(exc),
            "worker_seconds": time.perf_counter() - started,
            "stdout": exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout or "",
            "stderr": exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr or "",
        }

    output = proc.stdout.splitlines()
    line = next((line for line in reversed(output) if line.startswith(_RESULT_PREFIX)), None)
    if verbose:
        print(proc.stdout, end="")
        print(proc.stderr, end="", file=sys.stderr)
    invalid_result = None
    if proc.returncode == 0 and line is not None:
        try:
            result = json.loads(line[len(_RESULT_PREFIX) :])
            if not isinstance(result, dict) or result.get("name") != case.name or result.get("asset") != str(asset):
                raise ValueError("worker result does not identify the requested asset")
            outcome = result.get("outcome")
            if outcome not in ("match", "mismatch", "reference-error", "ovnewton-error", "both-error"):
                raise ValueError("invalid worker outcome")
            if outcome == "mismatch" and not isinstance(result.get("mismatches"), list):
                raise ValueError("missing mismatch list")
            for side in ("reference", "ovnewton"):
                if outcome in (f"{side}-error", "both-error"):
                    error = result.get(f"{side}_error")
                    if not isinstance(error, dict) or not all(
                        isinstance(error.get(k), str) for k in ("type", "message")
                    ):
                        raise ValueError(f"missing {side} error")
            _format_result(result)  # Check the fields required by the console summary too.
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            invalid_result = f"invalid worker result: {exc}"
        else:
            result.update(stdout=proc.stdout, stderr=proc.stderr, worker_seconds=time.perf_counter() - started)
            return result

    details = "\n".join((proc.stderr or proc.stdout).splitlines()[-12:])
    return {
        "name": case.name,
        "asset": str(asset),
        "outcome": "worker-error" if proc.returncode == 0 else "worker-crash",
        "returncode": proc.returncode,
        "message": invalid_result or details,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "worker_seconds": time.perf_counter() - started,
    }


def _select_cases(patterns: list[str], cases: tuple[AssetCase, ...]) -> list[AssetCase]:
    if not patterns:
        return list(cases)
    selected = [case for case in cases if any(fnmatch.fnmatchcase(case.name, pattern) for pattern in patterns)]
    unknown = [
        pattern
        for pattern in patterns
        if not any(fnmatch.fnmatchcase(case.name, pattern) for case in cases)
    ]
    if unknown:
        raise ValueError("asset pattern matched nothing: " + ", ".join(unknown))
    return selected


def _format_result(result: dict[str, object]) -> str:
    outcome = str(result["outcome"])
    detail = ""
    if outcome == "match":
        counts = result["ovnewton_counts"]
        detail = f"b={counts['bodies']} s={counts['shapes']} j={counts['joints']}"
    elif outcome == "mismatch":
        mismatches = result.get("mismatches", [])
        detail = f"{len(mismatches)} differences; {mismatches[0] if mismatches else ''}"
    elif result.get("ovnewton_error"):
        error = result["ovnewton_error"]
        detail = f"{error['type']}: {error['message']}"
    elif result.get("reference_error"):
        error = result["reference_error"]
        detail = f"{error['type']}: {error['message']}"
    else:
        detail = str(result.get("message", ""))
    expected = "expected" if result.get("as_expected") else "UNEXPECTED"
    return f"[{outcome.upper():15}] {result['name']:30} {expected:10} {detail[:180]}"


def _matches_expectation(case: AssetCase, result: dict[str, object]) -> bool:
    outcome = result["outcome"]
    ours = result.get("ovnewton_error") or {}
    message = str(ours.get("message", ""))
    expectation = result.get("expectation", case.expectation)
    actuator_errors = {
        "mjc-actuator-joint": " requires an unmerged scalar joint: ",
        "mjc-actuator-configuration": " requires fixed gain, affine bias and unit gear without dynamics",
    }
    if expectation in actuator_errors:
        return (
            outcome == "ovnewton-error"
            and ours.get("type") == "UnsupportedPhysicsError"
            and message.startswith("MjcActuator at ")
            and actuator_errors[expectation] in message
        )
    if expectation == "match":
        return outcome == "match"
    if expectation == "native-instancing":
        return (
            outcome == "ovnewton-error"
            and ours.get("type") == "UnsupportedPhysicsError"
            and "native USD" in message
            and "instancing" in message
        )
    if expectation == "physics-point-instancer":
        return (
            outcome == "ovnewton-error"
            and ours.get("type") == "UnsupportedPhysicsError"
            and message.startswith("physics PointInstancer is unsupported at ")
        )
    if expectation == "mjc-solver-registration":
        return (
            outcome == "ovnewton-error"
            and ours.get("type") == "UnsupportedPhysicsError"
            and message.startswith(("mjc:solreflimit at ", "MjcActuator requires "))
            and message.endswith("requires SolverMuJoCo.register_custom_attributes(builder) before add_ovstage")
        )
    raise ValueError(f"unknown expectation: {expectation}")



def _git_revision(path: pathlib.Path) -> str | None:
    """Record the checkout actually used, including symlinked asset caches."""
    try:
        tracked = subprocess.check_output(
            ["git", "-C", str(path.resolve()), "ls-files", "--", "."],
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        )
        if not tracked.strip():
            return None
        return subprocess.check_output(
            ["git", "-C", str(path.resolve()), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None



def _git_dirty(path: pathlib.Path) -> bool | None:
    try:
        return bool(subprocess.check_output(
            ["git", "-C", str(path.resolve()), "status", "--porcelain", "--", "."],
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        ).strip())
    except (OSError, subprocess.SubprocessError):
        return None


def _report_metadata() -> dict[str, object]:
    packages = {}
    for name in ("ovnewton", "newton", "ovstage", "ovrtx", "warp-lang", "usd-core", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    repo = pathlib.Path(__file__).resolve().parents[2]
    source = os.environ.get("OVNEWTON_NEWTON_SOURCE")
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(), "packages": packages,
        "ovnewton_source": str(repo), "ovnewton_commit": _git_revision(repo),
        "ovnewton_dirty": _git_dirty(repo),
        "newton_source": source,
        "newton_source_commit": _git_revision(pathlib.Path(source)) if source else None,
        "newton_source_dirty": _git_dirty(pathlib.Path(source)) if source else None,
        "reference_profiles": {"core": ["SchemaResolverNewton"],
                               "mujoco": ["SchemaResolverNewton", "SchemaResolverMjc"]},
        "reference_options": {"apply_up_axis_from_stage": True, "load_visual_shapes": False},
        "pxr_work_thread_limit": _source_environment()["PXR_WORK_THREAD_LIMIT"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("auto", "core", "mujoco"), default="auto",
                        help="auto uses each case's recorded profile; mujoco registers solver attributes on both paths")
    parser.add_argument("--asset", action="append", default=[], help="case name or glob; repeatable")
    parser.add_argument("--asset-root", type=pathlib.Path, help="existing newton-assets checkout")
    parser.add_argument(
        "--latest-assets",
        action="store_true",
        help="use the current upstream manifest (requires --asset-root unless listing)",
    )
    parser.add_argument("--cache-dir", type=pathlib.Path, help="Newton download cache")
    parser.add_argument("--list", action="store_true", help="list corpus cases without downloading")
    parser.add_argument("--strict", action="store_true", help="return nonzero for outcomes outside the manifest")
    parser.add_argument("--timeout", type=float, default=180.0, help="worker timeout in seconds")
    parser.add_argument("--report", type=pathlib.Path, help="save complete results and provenance as JSON")
    parser.add_argument("--verbose", action="store_true", help="show worker output")
    parser.add_argument("--worker", type=pathlib.Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-name", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.worker:
        return _worker(
            args.worker,
            args.worker_name or args.worker.stem,
            args.profile if args.profile != "auto" else "core",
        )

    if args.latest_assets and args.asset_root is None and not args.list:
        print("error: --latest-assets requires --asset-root", file=sys.stderr)
        return 2
    manifest = LATEST_ASSET_CASES if args.latest_assets else PINNED_ASSET_CASES

    try:
        cases = _select_cases(args.asset, manifest)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.list:
        for case in cases:
            profile = case.profile if args.profile == "auto" else args.profile
            expectation = case.expected_outcome(profile) or "requires-mujoco-profile"
            print(f"{case.name:30} {expectation:22} {profile:8} {case.folder}/{case.scene}")
        return 0

    incompatible = [case.name for case in cases if case.expected_outcome(
        case.profile if args.profile == "auto" else args.profile
    ) is None]
    if incompatible:
        print("error: MuJoCo reference profile required for: " + ", ".join(incompatible), file=sys.stderr)
        return 2

    if args.asset_root is None:
        newton_source = os.environ.get("OVNEWTON_NEWTON_SOURCE")
        if newton_source:
            sys.path.insert(0, newton_source)
        import newton.utils

        print("newton-assets source: Newton package default")
    else:
        print(f"newton-assets checkout: {args.asset_root.resolve()}")

    started = time.perf_counter()
    report = {"schema_version": 1, "metadata": _report_metadata(), "assets": {}}
    roots: dict[str, pathlib.Path | dict[str, str]] = {}
    results = []
    for case in cases:
        profile = case.profile if args.profile == "auto" else args.profile
        if case.folder not in roots:
            acquired = time.perf_counter()
            try:
                if args.asset_root is not None:
                    roots[case.folder] = args.asset_root / case.folder
                else:
                    roots[case.folder] = pathlib.Path(
                        newton.utils.download_asset(
                            case.folder,
                            cache_dir=str(args.cache_dir) if args.cache_dir else None,
                        )
                    )
            except Exception as exc:
                roots[case.folder] = _error(exc)
            root = roots[case.folder]
            commit = _git_revision(root) if isinstance(root, pathlib.Path) and root.is_dir() else None
            report["assets"][case.folder] = {
                "root": str(root.resolve()) if isinstance(root, pathlib.Path) else None,
                "commit": commit,
                "dirty": _git_dirty(root) if commit else None,
                "acquisition_seconds": time.perf_counter() - acquired,
            }
        root = roots[case.folder]
        asset = (root / case.scene) if isinstance(root, pathlib.Path) else pathlib.Path(case.folder) / case.scene
        if isinstance(root, dict):
            result = {
                "name": case.name,
                "asset": str(asset),
                "outcome": "download-error",
                "message": f"{root['type']}: {root['message']}",
            }
        elif not asset.is_file():
            result = {"name": case.name, "asset": str(asset), "outcome": "missing", "message": "file not found"}
        else:
            result = _run_worker(asset, case, args.timeout, args.verbose, profile)
        result["profile"] = profile
        result["expectation"] = case.expected_outcome(profile)
        result["as_expected"] = _matches_expectation(case, result)
        results.append(result)
        print(_format_result(result), flush=True)

    counts: dict[str, int] = {}
    for result in results:
        outcome = str(result["outcome"])
        counts[outcome] = counts.get(outcome, 0) + 1
    print("summary: " + ", ".join(f"{name}={count}" for name, count in sorted(counts.items())))
    unexpected = sum(not result["as_expected"] for result in results)
    print(f"manifest: expected={len(results) - unexpected}, unexpected={unexpected}")
    print(f"supported and matching: {counts.get('match', 0)}/{len(results)}")
    if args.report:
        report.update(
            manifest="latest" if args.latest_assets else "pinned", results=results,
            summary={"outcomes": counts, "total": len(results), "matching": counts.get("match", 0),
                     "expected": len(results) - unexpected, "unexpected": unexpected},
            elapsed_seconds=time.perf_counter() - started,
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return int(args.strict and unexpected != 0)


if __name__ == "__main__":
    raise SystemExit(main())
