from __future__ import annotations

import argparse
from collections import Counter
import ctypes
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, REPO_ROOT / "gpu_backend" / "python"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from flow_matching.collection import replace_json  # noqa: E402
from flow_matching.data import CoilNormalizer, file_sha256  # noqa: E402
from flow_matching.trajectory_dataset import atomic_write_json  # noqa: E402
from scripts.qh_data_space_random_survey import (  # noqa: E402
    CURRENT_REFERENCE_SCORE,
    CURRENT_SCORE_SHA256,
    NORMALIZER_CHECKPOINT_SHA256,
    REFERENCE_CASE_SHA256,
    all_rows,
    case_tokens,
    score_tokens_standalone,
)


FORMAT = "qh_data_space_random_adam_followup_v1"
PROTOCOL_ID = "qh-data-gaussian-global-survey-v1"
SURVEY_MANIFEST_SHA256 = (
    "27779136a94998fcc54b6fd58d684afb6c93511c70f8973e89fd80f4cbda1738"
)
SURVEY_SUMMARY_SHA256 = (
    "066e4d9f41dd33d4ff8b9b68977a1bb9c53e88bb30a41fc95f4752313ae410df"
)
ORIGINAL_SELECTION_SHA256 = (
    "c649c8730499cf6692d3e8362c4d35f9cad7b4d44e3a5357481ad55f4e2ae403"
)
SCORE_CENSUS_THRESHOLD = 10.0
DEFAULT_LOW_OK_QUOTAS = {1: 4, 2: 8, 3: 7, 4: 7, 5: 12}
HISTORICAL_ADAM200_MEAN_WALL_S = 879.5759580791793
ADAM_SETTINGS = {
    "parameter_space": "data",
    "optimizer": "adam",
    "iterations": 200,
    "gradient_mode": "random-orthogonal",
    "directions": 64,
    "difference": "centered",
    "perturbation": 0.0025,
    "learning_rate": 0.01,
    "beta1": 0.7,
    "beta2": 0.999,
    "formal_surface_theta_count": 128,
    "local_surface_theta_count": 64,
}
OPTIMIZER_LIBRARY_SYMBOLS = (
    "sgpu_create_field_batch_f32",
    "sgpu_destroy_field_batch",
    "sgpu_batch_eval_B_f32",
    "sgpu_batch_eval_B_grad_f32",
    "sgpu_batch_trace_period_mixed",
    "sgpu_batch_trace_axis_samples",
    "sgpu_batch_refine_axis_hint",
    "sgpu_score_coils_capture_psi_center",
    "sgpu_fit_psi_batch_pcgls_f32",
    "sgpu_score_coils_local_batch",
    "sgpu_clear_psi_warm_preconditioner",
)


def git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_dirty() -> bool:
    return bool(git_value("status", "--porcelain", "--untracked-files=no"))


def validate_optimizer_library_api(path: Path) -> None:
    try:
        library = ctypes.CDLL(str(path))
    except OSError as exc:
        raise RuntimeError(f"optimizer library cannot be loaded: {path}: {exc}") from exc
    missing = [
        symbol for symbol in OPTIMIZER_LIBRARY_SYMBOLS if not hasattr(library, symbol)
    ]
    if missing:
        raise RuntimeError(
            "optimizer library lacks required batch API symbols: "
            + ", ".join(missing)
        )


def parse_low_ok_quotas(text: str) -> dict[int, int]:
    quotas: dict[int, int] = {}
    for item in text.split(","):
        ncoils_text, quota_text = item.strip().split(":", 1)
        ncoils = int(ncoils_text)
        quota = int(quota_text)
        if ncoils in quotas or ncoils < 1 or quota < 0:
            raise ValueError(f"invalid low-ok quota entry {item!r}")
        quotas[ncoils] = quota
    if not quotas:
        raise ValueError("at least one low-ok quota is required")
    return quotas


def _base_case(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": str(row["sample_id"]),
        "survey_score": float(row["score"]),
        "survey_status": str(row["status"]),
        "nfp": int(row["nfp"]),
        "n_base_coils": int(row["n_base_coils"]),
        "reconstruction": {
            "worker_index": int(row["worker_index"]),
            "condition_index": int(row["condition_index"]),
            "condition_sample_index": int(row["condition_sample_index"]),
        },
        "recorded_reconstruction_diagnostics": {
            "sampled_standard_normal_rms": float(
                row["sampled_standard_normal_rms"]
            ),
            "effective_normalized_rms": float(row["effective_normalized_rms"]),
            "post_projection_clipped_fraction": float(
                row["post_projection_clipped_fraction"]
            ),
        },
        "candidate_file": row.get("candidate_file"),
        "candidate_sha256": row.get("candidate_sha256"),
        "selection_roles": [],
    }


def select_followup_cases(
    rows: list[dict[str, Any]],
    original_selection: dict[str, Any],
    *,
    seed: int,
    low_ok_quotas: dict[int, int],
) -> dict[str, Any]:
    by_id = {str(row["sample_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("survey rows contain duplicate sample IDs")
    selected: dict[str, dict[str, Any]] = {}

    def add_case(row: dict[str, Any], role: dict[str, Any]) -> None:
        sample_id = str(row["sample_id"])
        case = selected.setdefault(sample_id, _base_case(row))
        case["selection_roles"].append(role)

    high_rows = [
        row for row in rows if float(row["score"]) >= SCORE_CENSUS_THRESHOLD
    ]
    for row in high_rows:
        add_case(
            row,
            {
                "name": "score_ge_10_census",
                "analysis_stratum": "score_ge_10",
                "population_count": len(high_rows),
                "selected_count": len(high_rows),
                "selection_probability": 1.0,
                "analysis_weight": 1.0,
            },
        )

    low_strata = []
    for ncoils, quota in sorted(low_ok_quotas.items()):
        eligible = [
            row
            for row in rows
            if int(row["n_base_coils"]) == ncoils
            and float(row["score"]) < SCORE_CENSUS_THRESHOLD
            and str(row["status"]) == "ok"
        ]
        if quota > len(eligible):
            raise ValueError(
                f"low-score ok quota {quota} exceeds nc={ncoils} population {len(eligible)}"
            )
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), ncoils]))
        indices = sorted(
            int(value)
            for value in rng.choice(len(eligible), size=quota, replace=False)
        )
        probability = quota / len(eligible) if eligible else 0.0
        for index in indices:
            add_case(
                eligible[index],
                {
                    "name": "score_lt_10_status_ok_nc_sample",
                    "analysis_stratum": f"score_lt_10_status_ok_nc{ncoils}",
                    "population_count": len(eligible),
                    "selected_count": quota,
                    "selection_probability": probability,
                    "analysis_weight": 1.0 / probability,
                },
            )
        low_strata.append(
            {
                "n_base_coils": ncoils,
                "population_count": len(eligible),
                "selected_count": quota,
                "selection_probability": probability,
            }
        )

    original_ids = []
    for original in original_selection["selected"]:
        sample_id = str(original["sample_id"])
        if sample_id not in by_id:
            raise ValueError(f"original selection sample {sample_id} is absent")
        row = by_id[sample_id]
        for key in ("nfp", "n_base_coils"):
            if int(row[key]) != int(original[key]):
                raise ValueError(f"original selection metadata changed for {sample_id}")
        if str(row["status"]) != str(original["status"]) or not math.isclose(
            float(row["score"]),
            float(original["score"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"original selection score or status changed for {sample_id}")
        original_ids.append(sample_id)
        add_case(
            row,
            {
                "name": "original_preregistered_selection",
                "analysis_stratum": None,
                "population_count": None,
                "selected_count": None,
                "selection_probability": float(
                    original["within_band_selection_probability"]
                ),
                "analysis_weight": 0.0,
                "use": "provenance and reconstruction diagnostic",
            },
        )

    cases = list(selected.values())
    original_set = set(original_ids)
    for case in cases:
        case["original_selection_member"] = case["sample_id"] in original_set
        case["adam_eligible"] = case["survey_status"] == "ok"
        state = np.random.SeedSequence(
            [
                int(seed),
                int(case["reconstruction"]["worker_index"]),
                int(case["reconstruction"]["condition_index"]),
                int(case["reconstruction"]["condition_sample_index"]),
            ]
        ).generate_state(1, dtype=np.uint32)
        case["optimizer_seed"] = int(state[0])
        case["estimated_adam_wall_s"] = (
            HISTORICAL_ADAM200_MEAN_WALL_S if case["adam_eligible"] else 5.0
        )

    cases.sort(
        key=lambda case: (
            not case["original_selection_member"],
            not case["adam_eligible"],
            -case["survey_score"],
            case["sample_id"],
        )
    )
    return {
        "format": "qh_data_space_random_adam_expanded_selection_v1",
        "seed": int(seed),
        "method": (
            "retain the immutable 14-case pre-registration; census every score>=10 "
            "sample; sample score<10,status=ok controls within n_base_coils"
        ),
        "score_census_threshold": SCORE_CENSUS_THRESHOLD,
        "original_selection_count": len(original_ids),
        "original_selection_ids": original_ids,
        "low_ok_strata": low_strata,
        "selected_count": len(cases),
        "adam_eligible_count": sum(bool(case["adam_eligible"]) for case in cases),
        "diagnostic_only_count": sum(not case["adam_eligible"] for case in cases),
        "cases": cases,
    }


def assign_workers(cases: list[dict[str, Any]], worker_count: int) -> list[float]:
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    loads = [0.0] * worker_count
    eligible_counts = [0] * worker_count
    by_worker_nc: list[Counter[int]] = [Counter() for _ in range(worker_count)]
    eligible = sorted(
        (case for case in cases if case["adam_eligible"]),
        key=lambda case: (
            int(case["n_base_coils"]),
            -float(case["survey_score"]),
            str(case["sample_id"]),
        ),
    )
    for case in eligible:
        ncoils = int(case["n_base_coils"])
        worker = min(
            range(worker_count),
            key=lambda index: (
                eligible_counts[index],
                by_worker_nc[index][ncoils],
                loads[index],
                index,
            ),
        )
        case["worker_index"] = worker
        eligible_counts[worker] += 1
        by_worker_nc[worker][ncoils] += 1
        loads[worker] += float(case["estimated_adam_wall_s"])

    diagnostics = sorted(
        (case for case in cases if not case["adam_eligible"]),
        key=lambda case: str(case["sample_id"]),
    )
    for case in diagnostics:
        worker = min(
            range(worker_count),
            key=lambda index: (loads[index], index),
        )
        case["worker_index"] = worker
        loads[worker] += float(case["estimated_adam_wall_s"])
    return loads


def reconstruct_sample(
    normalizer: CoilNormalizer,
    survey_manifest: dict[str, Any],
    case: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    reconstruction = case["reconstruction"]
    condition_index = int(reconstruction["condition_index"])
    conditions = survey_manifest["sampling"]["conditions"]
    condition = conditions[condition_index]
    key = (int(condition["nfp"]), int(condition["n_base_coils"]))
    expected = (int(case["nfp"]), int(case["n_base_coils"]))
    if key != expected:
        raise ValueError(f"condition index mismatch for {case['sample_id']}")
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                int(survey_manifest["sampling"]["seed_base"]),
                int(reconstruction["worker_index"]),
                condition_index,
            ]
        )
    )
    sample_index = int(reconstruction["condition_sample_index"])
    sampled = rng.standard_normal(
        (sample_index + 1, key[1], 100), dtype=np.float32
    )[-1]
    survey_tokens = normalizer.inverse(sampled[None], key)[0]
    effective, clipped_fraction = normalizer.transform(survey_tokens[None], key)
    return sampled, effective[0], survey_tokens, float(clipped_fraction)


def validate_reconstruction(
    case: dict[str, Any],
    sampled: np.ndarray,
    effective: np.ndarray,
    tokens: np.ndarray,
    clipped_fraction: float,
) -> None:
    recorded = case["recorded_reconstruction_diagnostics"]
    sampled_rms = float(np.sqrt(np.mean(sampled.astype(np.float64) ** 2)))
    effective_rms = float(np.sqrt(np.mean(effective.astype(np.float64) ** 2)))
    checks = (
        (sampled_rms, recorded["sampled_standard_normal_rms"], "sampled RMS"),
        (effective_rms, recorded["effective_normalized_rms"], "effective RMS"),
        (
            clipped_fraction,
            recorded["post_projection_clipped_fraction"],
            "clipped fraction",
        ),
    )
    for actual, expected, label in checks:
        if not math.isclose(float(actual), float(expected), abs_tol=1.0e-10):
            raise ValueError(
                f"{label} reconstruction mismatch for {case['sample_id']}: "
                f"{actual} != {expected}"
            )

    candidate_file = case.get("candidate_file")
    if candidate_file is None:
        return
    candidate_path = Path(candidate_file)
    expected_sha = str(case["candidate_sha256"])
    if file_sha256(candidate_path) != expected_sha:
        raise ValueError(f"candidate hash changed for {case['sample_id']}")
    with np.load(candidate_path, allow_pickle=False) as candidate:
        np.testing.assert_array_equal(candidate["sampled_standard_normal"], sampled)
        np.testing.assert_array_equal(candidate["effective_normalized"], effective)
        np.testing.assert_allclose(candidate["decoded_tokens"], tokens, rtol=0.0, atol=0.0)


def make_start_payload(
    case: dict[str, Any],
    effective: np.ndarray,
    clipped_fraction: float,
) -> dict[str, Any]:
    return {
        "format": "qh_data_space_random_adam_start_v1",
        "sample_id": case["sample_id"],
        "nfp": int(case["nfp"]),
        "n_base_coils": int(case["n_base_coils"]),
        "noise": np.asarray(effective, dtype=np.float32).tolist(),
        "data_prior_screening": {
            "format": "qh_data_space_random_global_survey_start_v1",
            "normalized_coil_tokens": np.asarray(
                effective, dtype=np.float32
            ).tolist(),
            "survey_score": float(case["survey_score"]),
            "survey_status": str(case["survey_status"]),
            "post_projection_clipped_fraction": float(clipped_fraction),
            "reconstruction": dict(case["reconstruction"]),
            "axis_hint_policy": "fresh global search; no recorded axis hint is reused",
        },
    }


def prepare(args: argparse.Namespace) -> None:
    import torch

    if args.run_root.exists() or args.run_root.with_name(
        args.run_root.name + ".prepare.partial"
    ).exists():
        raise FileExistsError(f"refusing to overwrite {args.run_root}")
    if git_dirty():
        raise RuntimeError("tracked worktree must be clean before preparing follow-up")

    source_paths = {
        "manifest": args.survey_root / "survey_manifest.json",
        "summary": args.survey_root / "survey_summary.json",
        "original_selection": args.survey_root / "adam_followup_selection.json",
    }
    expected_hashes = {
        "manifest": args.expected_survey_manifest_sha,
        "summary": args.expected_survey_summary_sha,
        "original_selection": args.expected_original_selection_sha,
    }
    for name, path in source_paths.items():
        if file_sha256(path) != expected_hashes[name]:
            raise ValueError(f"survey {name} SHA-256 changed")
    if file_sha256(args.checkpoint) != args.expected_checkpoint_sha:
        raise ValueError("normalizer checkpoint SHA-256 changed")
    if file_sha256(args.lib) != args.expected_lib_sha:
        raise ValueError("score-library SHA-256 changed")
    if file_sha256(args.gradient_lib) != args.expected_gradient_lib_sha:
        raise ValueError("gradient-library SHA-256 changed")
    validate_optimizer_library_api(args.gradient_lib)
    if file_sha256(args.reference_case) != args.expected_reference_case_sha:
        raise ValueError("reference-case SHA-256 changed")

    survey_manifest = json.loads(source_paths["manifest"].read_text(encoding="utf-8"))
    survey_summary = json.loads(source_paths["summary"].read_text(encoding="utf-8"))
    original_selection = json.loads(
        source_paths["original_selection"].read_text(encoding="utf-8")
    )
    if not bool(survey_summary["complete"]):
        raise ValueError("source global survey is incomplete")
    rows = all_rows(args.survey_root, int(survey_manifest["worker_count"]))
    if len(rows) != int(survey_summary["saved_sample_count"]):
        raise ValueError("source survey row count does not match its summary")

    selection = select_followup_cases(
        rows,
        original_selection,
        seed=args.selection_seed,
        low_ok_quotas=parse_low_ok_quotas(args.low_ok_quotas),
    )
    if selection["selected_count"] != args.expected_selected_count:
        raise ValueError("expanded selection count differs from frozen request")
    if selection["adam_eligible_count"] != args.expected_eligible_count:
        raise ValueError("Adam-eligible count differs from frozen request")
    loads = assign_workers(selection["cases"], args.worker_count)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    normalizer = CoilNormalizer.from_dict(checkpoint["normalizer"])
    if int(checkpoint["step"]) != 30000:
        raise RuntimeError("unexpected Flow checkpoint step")

    partial_root = args.run_root.with_name(args.run_root.name + ".prepare.partial")
    partial_root.mkdir(parents=True)
    for name in ("starts", "results", "incomplete", "failures", "workers", "logs"):
        (partial_root / name).mkdir()

    for case in selection["cases"]:
        sampled, effective, tokens, clipped_fraction = reconstruct_sample(
            normalizer, survey_manifest, case
        )
        validate_reconstruction(
            case, sampled, effective, tokens, clipped_fraction
        )
        start_path = partial_root / "starts" / f"{case['sample_id']}.json"
        atomic_write_json(
            start_path,
            make_start_payload(case, effective, clipped_fraction),
        )
        case["initial_case"] = str(
            (args.run_root / "starts" / start_path.name).resolve()
        )
        case["initial_case_sha256"] = file_sha256(start_path)

    selection_path = partial_root / "expanded_selection.json"
    atomic_write_json(selection_path, selection)
    code_commit = git_value("rev-parse", "HEAD")
    manifest = {
        "format": FORMAT,
        "stage": "prepared",
        "run_label": args.run_label,
        "protocol": {
            "id": PROTOCOL_ID,
            "status": "registered-experimental",
            "stage": "score-stratified data-space Adam200 follow-up",
            "relationship_to_default": (
                "This follow-up measures random-start optimizability and does not "
                "change qh-flow-screen32-adam200-64d-v1."
            ),
        },
        "created_unix_s": time.time(),
        "code": {
            "commit": code_commit,
            "branch": git_value("branch", "--show-current"),
            "tracked_dirty": False,
        },
        "source_survey": {
            "root": str(args.survey_root.resolve()),
            "sample_count": len(rows),
            "manifest": str(source_paths["manifest"].resolve()),
            "manifest_sha256": expected_hashes["manifest"],
            "summary": str(source_paths["summary"].resolve()),
            "summary_sha256": expected_hashes["summary"],
            "original_selection": str(source_paths["original_selection"].resolve()),
            "original_selection_sha256": expected_hashes["original_selection"],
        },
        "normalizer": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": args.expected_checkpoint_sha,
        },
        "evaluator": {
            "library": str(args.lib.resolve()),
            "library_sha256": args.expected_lib_sha,
            "library_role": "formal center and survey-reproduction score",
            "gradient_library": str(args.gradient_lib.resolve()),
            "gradient_library_sha256": args.expected_gradient_lib_sha,
            "gradient_library_role": "query-batched local-gradient oracle only",
            "optimizer_required_symbols": list(OPTIMIZER_LIBRARY_SYMBOLS),
            "abi": 10,
            "reference_case": str(args.reference_case.resolve()),
            "reference_case_sha256": args.expected_reference_case_sha,
            "reference_score": float(args.expected_reference_score),
            "reference_score_atol": float(args.reference_score_atol),
            "survey_rescore_overrides": {
                "psi_solver_mode": 2,
                "alpha_solver_mode": 2,
            },
        },
        "selection": {
            "file": str((args.run_root / selection_path.name).resolve()),
            "file_sha256": file_sha256(selection_path),
            "seed": int(args.selection_seed),
            "selected_count": int(selection["selected_count"]),
            "adam_eligible_count": int(selection["adam_eligible_count"]),
            "diagnostic_only_count": int(selection["diagnostic_only_count"]),
            "low_ok_quotas": parse_low_ok_quotas(args.low_ok_quotas),
        },
        "optimizer": dict(ADAM_SETTINGS),
        "runtime_design": {
            "worker_count": int(args.worker_count),
            "historical_mean_adam200_wall_s": HISTORICAL_ADAM200_MEAN_WALL_S,
            "historical_basis": (
                "309-case direct-data rerun optimizer_analysis/summary.json"
            ),
            "estimated_worker_load_s": loads,
            "estimated_serial_gpu_hours": (
                selection["adam_eligible_count"]
                * HISTORICAL_ADAM200_MEAN_WALL_S
                / 3600.0
            ),
            "estimated_parallel_hours": max(loads) / 3600.0,
        },
        "worker_count": int(args.worker_count),
        "cases": selection["cases"],
    }
    atomic_write_json(partial_root / "run_manifest.json", manifest)
    os.replace(partial_root, args.run_root)
    print(
        json.dumps(
            {
                "event": "adam_followup_prepared",
                "run_root": str(args.run_root),
                "selected_count": selection["selected_count"],
                "adam_eligible_count": selection["adam_eligible_count"],
                "worker_eligible_counts": [
                    sum(
                        case["adam_eligible"] and case["worker_index"] == worker
                        for case in selection["cases"]
                    )
                    for worker in range(args.worker_count)
                ],
                "estimated_worker_load_s": loads,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )


def run_logged(command: list[str], log_path: Path) -> tuple[int, float]:
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return completed.returncode, time.perf_counter() - started


def _log_tail(path: Path, line_count: int = 30) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-line_count:])


def validate_optimizer_summary(
    summary: dict[str, Any],
    *,
    expected_formal_library_sha: str | None = None,
    expected_gradient_library_sha: str | None = None,
) -> None:
    if (
        summary.get("status") != "ok"
        or summary.get("stop_reason") != "completed_iterations"
        or int(summary.get("completed_iterations", -1)) != ADAM_SETTINGS["iterations"]
    ):
        raise RuntimeError("optimizer did not complete the frozen 200 iterations")
    manifest = summary["manifest"]
    coordinate = manifest["coordinate_gradient"]
    adam = manifest["adam"]
    checks = {
        "parameter_space": manifest["parameter_space"] == "data",
        "optimizer": manifest["optimizer"] == "adam",
        "iterations": int(manifest["iterations"]) == ADAM_SETTINGS["iterations"],
        "gradient_mode": coordinate["mode"] == ADAM_SETTINGS["gradient_mode"],
        "directions": (
            int(coordinate["random_directions"]) == ADAM_SETTINGS["directions"]
        ),
        "difference": coordinate["difference"] == ADAM_SETTINGS["difference"],
        "formal_surface_theta_count": (
            int(coordinate["formal_surface_theta_count"])
            == ADAM_SETTINGS["formal_surface_theta_count"]
        ),
        "local_surface_theta_count": (
            int(coordinate["local_surface_theta_count"])
            == ADAM_SETTINGS["local_surface_theta_count"]
        ),
        "perturbation": math.isclose(
            float(coordinate["perturbation"]), ADAM_SETTINGS["perturbation"]
        ),
        "learning_rate": math.isclose(
            float(adam["learning_rate"]), ADAM_SETTINGS["learning_rate"]
        ),
        "beta1": math.isclose(float(adam["beta1"]), ADAM_SETTINGS["beta1"]),
        "beta2": math.isclose(float(adam["beta2"]), ADAM_SETTINGS["beta2"]),
    }
    failed = [name for name, valid in checks.items() if not valid]
    if expected_formal_library_sha is not None and str(
        manifest.get("formal_score_library", {}).get("sha256")
    ) != expected_formal_library_sha:
        failed.append("formal_score_library")
    if expected_gradient_library_sha is not None and str(
        manifest.get("gradient_library", {}).get("sha256")
    ) != expected_gradient_library_sha:
        failed.append("gradient_library")
    if failed:
        raise RuntimeError(f"optimizer manifest mismatch: {failed}")


def worker_cases(manifest: dict[str, Any], worker_index: int) -> list[dict[str, Any]]:
    return sorted(
        (
            case
            for case in manifest["cases"]
            if int(case["worker_index"]) == worker_index
        ),
        key=lambda case: (
            bool(case["adam_eligible"]),
            -float(case["survey_score"]),
            str(case["sample_id"]),
        ),
    )


def worker(args: argparse.Namespace) -> None:
    import torch

    manifest_path = args.run_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError("unexpected Adam follow-up manifest format")
    if not 0 <= args.worker_index < int(manifest["worker_count"]):
        raise ValueError("worker index is outside the prepared range")
    if git_value("rev-parse", "HEAD") != manifest["code"]["commit"] or git_dirty():
        raise RuntimeError("worker code does not match the clean prepared commit")

    checkpoint_path = Path(manifest["normalizer"]["checkpoint"])
    library_path = Path(manifest["evaluator"]["library"])
    gradient_library_path = Path(manifest["evaluator"]["gradient_library"])
    reference_path = Path(manifest["evaluator"]["reference_case"])
    source_manifest_path = Path(manifest["source_survey"]["manifest"])
    source_paths = (
        (checkpoint_path, manifest["normalizer"]["checkpoint_sha256"]),
        (library_path, manifest["evaluator"]["library_sha256"]),
        (
            gradient_library_path,
            manifest["evaluator"]["gradient_library_sha256"],
        ),
        (reference_path, manifest["evaluator"]["reference_case_sha256"]),
        (source_manifest_path, manifest["source_survey"]["manifest_sha256"]),
        (
            Path(manifest["source_survey"]["summary"]),
            manifest["source_survey"]["summary_sha256"],
        ),
        (
            Path(manifest["source_survey"]["original_selection"]),
            manifest["source_survey"]["original_selection_sha256"],
        ),
        (
            Path(manifest["selection"]["file"]),
            manifest["selection"]["file_sha256"],
        ),
    )
    for path, expected_sha in source_paths:
        if file_sha256(path) != expected_sha:
            raise ValueError(f"frozen dependency changed: {path}")
    validate_optimizer_library_api(gradient_library_path)

    torch.cuda.set_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    normalizer = CoilNormalizer.from_dict(checkpoint["normalizer"])
    survey_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    reference_tokens, reference_nfp = case_tokens(
        json.loads(reference_path.read_text(encoding="utf-8"))
    )
    reference_result, reference_wall_s = score_tokens_standalone(
        library_path,
        reference_tokens,
        nfp=reference_nfp,
        device=args.device,
    )
    reference_score = float(reference_result.get("score", math.nan))
    reference_abi = int(reference_result.get("diagnostics", {}).get("abi_version", -1))
    if (
        str(reference_result.get("status")) != "ok"
        or reference_abi != 10
        or not math.isfinite(reference_score)
        or abs(reference_score - float(manifest["evaluator"]["reference_score"]))
        > float(manifest["evaluator"]["reference_score_atol"])
    ):
        raise RuntimeError(
            "score reference preflight failed: "
            f"status={reference_result.get('status')} abi={reference_abi} "
            f"score={reference_score}"
        )

    cases = worker_cases(manifest, args.worker_index)
    worker_dir = args.run_root / "workers" / f"worker_{args.worker_index:02d}"
    worker_dir.mkdir(exist_ok=True)
    started = time.perf_counter()
    completed = 0
    skipped = 0
    durations: list[float] = []
    outcomes: Counter[str] = Counter()
    stop_reason = "all_assigned_cases_complete"
    previous_failure_signature: str | None = None
    consecutive_failure_count = 0

    for case in cases:
        result_dir = args.run_root / "results" / str(case["sample_id"])
        if result_dir.is_dir():
            skipped += 1
            saved = json.loads(
                (result_dir / "outcome.json").read_text(encoding="utf-8")
            )
            outcomes[str(saved["outcome_status"])] += 1
            continue
        elapsed = time.perf_counter() - started
        reserve = max(1200.0, 1.25 * max(durations[-3:], default=0.0))
        if elapsed + reserve >= args.max_wall_s:
            stop_reason = "max_wall_s"
            break

        case_started = time.perf_counter()
        partial = args.run_root / "incomplete" / (
            f"{case['sample_id']}.worker{args.worker_index}.{os.getpid()}.partial"
        )
        partial.mkdir()
        try:
            initial_case_path = Path(case["initial_case"])
            if file_sha256(initial_case_path) != case["initial_case_sha256"]:
                raise ValueError(f"initial-case hash changed for {case['sample_id']}")
            sampled, effective, survey_tokens, clipped_fraction = reconstruct_sample(
                normalizer, survey_manifest, case
            )
            validate_reconstruction(
                case, sampled, effective, survey_tokens, clipped_fraction
            )
            global_result, global_wall_s = score_tokens_standalone(
                library_path,
                survey_tokens,
                nfp=int(case["nfp"]),
                device=args.device,
            )
            observed_status = str(global_result.get("status"))
            observed_score = float(global_result.get("score", math.nan))
            if observed_status != str(case["survey_status"]) or not math.isclose(
                observed_score,
                float(case["survey_score"]),
                abs_tol=args.survey_score_atol,
            ):
                raise RuntimeError(
                    "survey reconstruction score gate failed: "
                    f"{observed_status}/{observed_score} != "
                    f"{case['survey_status']}/{case['survey_score']}"
                )

            base_outcome = {
                "format": "qh_data_space_random_adam_case_outcome_v1",
                "sample_id": case["sample_id"],
                "worker_index": args.worker_index,
                "nfp": int(case["nfp"]),
                "n_base_coils": int(case["n_base_coils"]),
                "selection_roles": case["selection_roles"],
                "original_selection_member": bool(case["original_selection_member"]),
                "survey": {
                    "score": float(case["survey_score"]),
                    "status": str(case["survey_status"]),
                    "reproduced_score": observed_score,
                    "reproduced_status": observed_status,
                    "score_wall_s": float(global_wall_s),
                },
            }
            if not bool(case["adam_eligible"]):
                outcome = {
                    **base_outcome,
                    "outcome_status": "ineligible_survey_status",
                    "adam_started": False,
                    "adam_completed": False,
                    "reason": (
                        "The frozen optimizer requires an initial status=ok center; "
                        f"the reproduced survey status is {observed_status}."
                    ),
                    "case_wall_s": time.perf_counter() - case_started,
                }
                atomic_write_json(partial / "outcome.json", outcome)
                os.replace(partial, result_dir)
                completed += 1
                outcomes[outcome["outcome_status"]] += 1
                durations.append(float(outcome["case_wall_s"]))
                previous_failure_signature = None
                consecutive_failure_count = 0
            else:
                optimization_dir = partial / "optimization"
                optimization_log = partial / "optimization.log"
                command = [
                    sys.executable,
                    str(
                        REPO_ROOT
                        / "scripts"
                        / "optimize_flow_prior_local_full_gradient_adam.py"
                    ),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--initial-case",
                    str(initial_case_path),
                    "--lib",
                    str(library_path),
                    "--gradient-lib",
                    str(gradient_library_path),
                    "--out-dir",
                    str(optimization_dir),
                    "--nfp",
                    str(case["nfp"]),
                    "--n-base-coils",
                    str(case["n_base_coils"]),
                    "--iterations",
                    str(ADAM_SETTINGS["iterations"]),
                    "--max-wall-s",
                    str(args.case_max_wall_s),
                    "--flow-steps",
                    "128",
                    "--parameter-space",
                    "data",
                    "--perturbation",
                    str(ADAM_SETTINGS["perturbation"]),
                    "--gradient-mode",
                    str(ADAM_SETTINGS["gradient_mode"]),
                    "--random-directions",
                    str(ADAM_SETTINGS["directions"]),
                    "--seed",
                    str(case["optimizer_seed"]),
                    "--optimizer",
                    "adam",
                    "--learning-rate",
                    str(ADAM_SETTINGS["learning_rate"]),
                    "--beta1",
                    str(ADAM_SETTINGS["beta1"]),
                    "--beta2",
                    str(ADAM_SETTINGS["beta2"]),
                    "--flow-device",
                    str(args.device),
                    "--score-device",
                    str(args.device),
                    "--plot-every",
                    "0",
                    "--progress-every",
                    "20",
                    "--trajectory-every",
                    "0",
                    "--state-every",
                    str(ADAM_SETTINGS["iterations"]),
                ]
                returncode, optimizer_wall_s = run_logged(command, optimization_log)
                if returncode != 0:
                    tail = _log_tail(optimization_log)
                    if "initial center is invalid:" not in tail:
                        raise RuntimeError(
                            f"optimizer exited {returncode}; tail:\n{tail}"
                        )
                    invalid_status = tail.rsplit("initial center is invalid:", 1)[-1]
                    invalid_status = invalid_status.splitlines()[0].strip()
                    outcome = {
                        **base_outcome,
                        "outcome_status": "optimizer_initial_invalid",
                        "adam_started": True,
                        "adam_completed": False,
                        "optimizer_initial_status": invalid_status,
                        "optimizer_process_wall_s": optimizer_wall_s,
                        "case_wall_s": time.perf_counter() - case_started,
                    }
                else:
                    summary = json.loads(
                        (optimization_dir / "summary.json").read_text(encoding="utf-8")
                    )
                    validate_optimizer_summary(
                        summary,
                        expected_formal_library_sha=manifest["evaluator"][
                            "library_sha256"
                        ],
                        expected_gradient_library_sha=manifest["evaluator"][
                            "gradient_library_sha256"
                        ],
                    )
                    outcome = {
                        **base_outcome,
                        "outcome_status": "adam_complete",
                        "adam_started": True,
                        "adam_completed": True,
                        "optimizer": {
                            "initial_score": float(summary["initial_score"]),
                            "final_score": float(summary["final_score"]),
                            "best_score": float(summary["best_score"]),
                            "best_iteration": int(summary["best_iteration"]),
                            "completed_iterations": int(summary["completed_iterations"]),
                            "completed_adam_steps": int(summary["completed_adam_steps"]),
                            "total_wall_s": float(summary["total_wall_s"]),
                        },
                        "optimizer_process_wall_s": optimizer_wall_s,
                        "case_wall_s": time.perf_counter() - case_started,
                    }
                atomic_write_json(partial / "outcome.json", outcome)
                os.replace(partial, result_dir)
                completed += 1
                outcomes[outcome["outcome_status"]] += 1
                durations.append(float(outcome["case_wall_s"]))
                previous_failure_signature = None
                consecutive_failure_count = 0

            replace_json(
                worker_dir / "progress.json",
                {
                    "format": FORMAT,
                    "worker_index": args.worker_index,
                    "stage": "running",
                    "assigned_cases": len(cases),
                    "new_completed": completed,
                    "skipped_existing": skipped,
                    "outcomes": dict(sorted(outcomes.items())),
                    "reference_score": reference_score,
                    "reference_wall_s": float(reference_wall_s),
                    "elapsed_s": time.perf_counter() - started,
                    "mean_case_wall_s": float(np.mean(durations)) if durations else None,
                    "updated_unix_s": time.time(),
                },
            )
            print(
                json.dumps(
                    {
                        "event": "adam_followup_case_complete",
                        "sample_id": case["sample_id"],
                        "outcome_status": outcome["outcome_status"],
                        "case_wall_s": outcome["case_wall_s"],
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
        except Exception as exc:
            failure_signature = f"{type(exc).__name__}: {exc}"
            failure = {
                "format": "qh_data_space_random_adam_runtime_failure_v1",
                "sample_id": case["sample_id"],
                "worker_index": args.worker_index,
                "error": failure_signature,
                "wall_s": time.perf_counter() - case_started,
            }
            atomic_write_json(partial / "failure.json", failure)
            failure_destination = args.run_root / "failures" / (
                f"{case['sample_id']}.worker{args.worker_index}.{int(time.time())}"
            )
            os.replace(partial, failure_destination)
            outcomes["runtime_failure"] += 1
            print(json.dumps({"event": "adam_followup_case_failed", **failure}), flush=True)
            if failure_signature == previous_failure_signature:
                consecutive_failure_count += 1
            else:
                previous_failure_signature = failure_signature
                consecutive_failure_count = 1
            if consecutive_failure_count >= 3:
                stop_reason = "three_identical_consecutive_failures"
                break

    finished_ids = {
        path.name for path in (args.run_root / "results").iterdir() if path.is_dir()
    }
    missing = [
        str(case["sample_id"])
        for case in cases
        if str(case["sample_id"]) not in finished_ids
    ]
    final = {
        "format": FORMAT,
        "worker_index": args.worker_index,
        "stage": "complete" if not missing else "incomplete",
        "stop_reason": stop_reason,
        "assigned_cases": len(cases),
        "new_completed": completed,
        "skipped_existing": skipped,
        "missing_cases": missing,
        "outcomes": dict(sorted(outcomes.items())),
        "reference_score": reference_score,
        "reference_wall_s": float(reference_wall_s),
        "elapsed_s": time.perf_counter() - started,
        "mean_case_wall_s": float(np.mean(durations)) if durations else None,
        "finished_unix_s": time.time(),
    }
    replace_json(worker_dir / "progress.json", final)
    print(json.dumps(final, indent=2), flush=True)
    if outcomes["runtime_failure"]:
        raise RuntimeError(
            f"worker recorded {outcomes['runtime_failure']} runtime failures"
        )
    if missing and not args.allow_partial:
        raise RuntimeError(f"worker left {len(missing)} assigned cases incomplete")
    if missing and stop_reason != "max_wall_s":
        raise RuntimeError(
            f"worker stopped with {len(missing)} missing cases: {stop_reason}"
        )


def _case_weight(case: dict[str, Any]) -> float:
    weights = [
        float(role["analysis_weight"])
        for role in case["selection_roles"]
        if role.get("analysis_stratum") is not None
    ]
    return max(weights, default=0.0)


def summarize(args: argparse.Namespace) -> None:
    manifest = json.loads(
        (args.run_root / "run_manifest.json").read_text(encoding="utf-8")
    )
    outcomes = {}
    for case in manifest["cases"]:
        path = args.run_root / "results" / str(case["sample_id"]) / "outcome.json"
        if path.is_file():
            outcomes[str(case["sample_id"])] = json.loads(path.read_text(encoding="utf-8"))
    missing = [
        str(case["sample_id"])
        for case in manifest["cases"]
        if str(case["sample_id"]) not in outcomes
    ]
    if missing and not args.allow_partial:
        raise RuntimeError(f"follow-up is missing {len(missing)} case outcomes")

    source_summary = json.loads(
        Path(manifest["source_survey"]["summary"]).read_text(encoding="utf-8")
    )
    denominators: Counter[int] = Counter()
    for key, group in source_summary["by_condition"].items():
        ncoils = int(key.rsplit("_nc", 1)[1])
        denominators[ncoils] += int(group["count"])

    thresholds = (20.0, 30.0, 40.0, 50.0)
    estimates = {}
    for threshold in thresholds:
        weighted_by_nc: Counter[int] = Counter()
        for case in manifest["cases"]:
            outcome = outcomes.get(str(case["sample_id"]))
            if outcome is None:
                continue
            weight = _case_weight(case)
            best_score = (
                float(outcome["optimizer"]["best_score"])
                if outcome.get("adam_completed")
                else float("-inf")
            )
            if best_score >= threshold:
                weighted_by_nc[int(case["n_base_coils"])] += weight
        total_weighted = float(sum(weighted_by_nc.values()))
        estimates[str(int(threshold))] = {
            "estimated_population_success_count": total_weighted,
            "estimated_rate": total_weighted / int(source_summary["saved_sample_count"]),
            "by_n_base_coils": {
                str(ncoils): {
                    "estimated_population_success_count": float(weighted_by_nc[ncoils]),
                    "estimated_rate": float(weighted_by_nc[ncoils]) / count,
                }
                for ncoils, count in sorted(denominators.items())
            },
        }

    payload = {
        "format": "qh_data_space_random_adam_followup_summary_v1",
        "run_manifest_sha256": file_sha256(args.run_root / "run_manifest.json"),
        "complete": not missing,
        "selected_count": len(manifest["cases"]),
        "completed_outcome_count": len(outcomes),
        "missing_cases": missing,
        "outcome_counts": dict(
            sorted(Counter(row["outcome_status"] for row in outcomes.values()).items())
        ),
        "adam_complete_count": sum(
            bool(row.get("adam_completed")) for row in outcomes.values()
        ),
        "observed_best_score": {
            "maximum": max(
                (
                    float(row["optimizer"]["best_score"])
                    for row in outcomes.values()
                    if row.get("adam_completed")
                ),
                default=None,
            ),
            "threshold_counts_unweighted": {
                str(int(threshold)): sum(
                    row.get("adam_completed", False)
                    and float(row["optimizer"]["best_score"]) >= threshold
                    for row in outcomes.values()
                )
                for threshold in thresholds
            },
        },
        "weighted_population_estimates": estimates,
        "estimator_scope": (
            "Score>=10 is a census; score<10,status=ok is inverse-probability "
            "weighted within n_base_coils; non-ok survey starts have zero success "
            "under this optimizer because they cannot enter its initial center."
        ),
        "updated_unix_s": time.time(),
    }
    replace_json(args.run_root / "followup_summary.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare, run, and summarize the random-data Adam200 follow-up."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--survey-root", type=Path, required=True)
    prepare_parser.add_argument("--run-root", type=Path, required=True)
    prepare_parser.add_argument("--run-label", required=True)
    prepare_parser.add_argument("--checkpoint", type=Path, required=True)
    prepare_parser.add_argument("--lib", type=Path, required=True)
    prepare_parser.add_argument("--gradient-lib", type=Path, required=True)
    prepare_parser.add_argument("--reference-case", type=Path, required=True)
    prepare_parser.add_argument("--worker-count", type=int, default=6)
    prepare_parser.add_argument("--selection-seed", type=int, default=2026082901)
    prepare_parser.add_argument(
        "--low-ok-quotas",
        default=",".join(f"{key}:{value}" for key, value in DEFAULT_LOW_OK_QUOTAS.items()),
    )
    prepare_parser.add_argument("--expected-selected-count", type=int, default=76)
    prepare_parser.add_argument("--expected-eligible-count", type=int, default=72)
    prepare_parser.add_argument(
        "--expected-survey-manifest-sha", default=SURVEY_MANIFEST_SHA256
    )
    prepare_parser.add_argument(
        "--expected-survey-summary-sha", default=SURVEY_SUMMARY_SHA256
    )
    prepare_parser.add_argument(
        "--expected-original-selection-sha", default=ORIGINAL_SELECTION_SHA256
    )
    prepare_parser.add_argument(
        "--expected-checkpoint-sha", default=NORMALIZER_CHECKPOINT_SHA256
    )
    prepare_parser.add_argument("--expected-lib-sha", default=CURRENT_SCORE_SHA256)
    prepare_parser.add_argument("--expected-gradient-lib-sha", required=True)
    prepare_parser.add_argument(
        "--expected-reference-case-sha", default=REFERENCE_CASE_SHA256
    )
    prepare_parser.add_argument(
        "--expected-reference-score", type=float, default=CURRENT_REFERENCE_SCORE
    )
    prepare_parser.add_argument("--reference-score-atol", type=float, default=1.0e-5)
    prepare_parser.set_defaults(func=prepare)

    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--run-root", type=Path, required=True)
    worker_parser.add_argument("--worker-index", type=int, required=True)
    worker_parser.add_argument("--device", type=int, default=0)
    worker_parser.add_argument("--max-wall-s", type=float, default=13200.0)
    worker_parser.add_argument("--case-max-wall-s", type=float, default=3600.0)
    worker_parser.add_argument("--survey-score-atol", type=float, default=1.0e-5)
    worker_parser.add_argument("--allow-partial", action="store_true")
    worker_parser.set_defaults(func=worker)

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--run-root", type=Path, required=True)
    summary_parser.add_argument("--allow-partial", action="store_true")
    summary_parser.set_defaults(func=summarize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
