from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, REPO_ROOT / "gpu_backend" / "python"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from flow_matching.data import CoilNormalizer, file_sha256  # noqa: E402
from flow_matching.collection import replace_json  # noqa: E402
from flow_matching.trajectory_dataset import (  # noqa: E402
    COMPONENT_KEYS,
    atomic_savez_compressed,
    atomic_write_json,
    atomic_write_jsonl_gzip,
)


FORMAT = "qh_data_space_random_global_survey_v1"
PROTOCOL_ID = "qh-data-gaussian-global-survey-v1"
CURRENT_SCORE_SHA256 = (
    "565c32073b145d97a1f2244705fb06e4b3458ce798cd74d0c97ee4e0129dc729"
)
NORMALIZER_CHECKPOINT_SHA256 = (
    "39a3293a459e248a0d1ec062607a1a467128b14d8ca973aadd82e113532ab99f"
)
REFERENCE_CASE_SHA256 = (
    "6ee6f8e1f0290ec49093596a5f95b7f2aac98c61d51af3cad59410a771b7e8c1"
)
CURRENT_REFERENCE_SCORE = 94.62541477362565
STANDALONE_SCORE_OVERRIDES = {
    "psi_solver_mode": 2,
    "alpha_solver_mode": 2,
}
GROUP_PATTERN = re.compile(r"^nfp(?P<nfp>\d+)_nc(?P<ncoils>\d+)$")
SCORE_THRESHOLDS = (10.0, 20.0, 30.0, 40.0, 50.0)
FOLLOWUP_BANDS = (
    (0.0, 20.0, 4),
    (20.0, 30.0, 8),
    (30.0, 40.0, 8),
    (40.0, 50.0, 12),
    (50.0, math.inf, 24),
)


def score_tokens_standalone(
    library: Path,
    tokens: np.ndarray,
    *,
    nfp: int,
    device: int,
) -> tuple[dict[str, Any], float]:
    from stellarator_gpu import score_coils_native

    values = np.asarray(tokens, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 100:
        raise ValueError("tokens must have shape (n_base_coils, 100)")
    started = time.perf_counter()
    result = score_coils_native(
        library,
        values[:, :33],
        values[:, 33:66],
        values[:, 66:99],
        values[:, 99],
        int(nfp),
        device_id=int(device),
        target_helicity=(1, int(nfp)),
        config_overrides=dict(STANDALONE_SCORE_OVERRIDES),
    )
    return result, time.perf_counter() - started


def case_tokens(payload: dict[str, Any]) -> tuple[np.ndarray, int]:
    raw = payload["raw"]
    unit = str(raw.get("current_unit", "A")).lower()
    if unit in {"a", "amp", "amps"}:
        current_scale = 1.0
    elif unit in {"ma", "megaamp", "megaamps"}:
        current_scale = 1.0e6
    else:
        raise ValueError(f"unsupported current unit {unit!r}")
    x = np.asarray(raw["x"], dtype=np.float64)
    y = np.asarray(raw["y"], dtype=np.float64)
    z = np.asarray(raw["z"], dtype=np.float64)
    current = np.asarray(raw["current"], dtype=np.float64) * current_scale
    if x.ndim != 2 or x.shape != y.shape or x.shape != z.shape or x.shape[1] != 33:
        raise ValueError("reference case has inconsistent Fourier arrays")
    if current.shape != (x.shape[0],):
        raise ValueError("reference case has inconsistent currents")
    tokens = np.concatenate((x, y, z, current[:, None]), axis=1)
    nfp = payload["nfp"] if "nfp" in payload else raw["nfp"]
    return tokens, int(nfp)


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


def parse_worker_counts(text: str, worker_count: int) -> list[int]:
    counts = [
        int(value.strip())
        for value in re.split(r"[,:]", text)
        if value.strip()
    ]
    if len(counts) == 1:
        counts *= worker_count
    if len(counts) != worker_count or any(value < 1 for value in counts):
        raise ValueError(
            "worker samples per condition must be one positive integer or one "
            "comma/colon-separated positive integer per worker"
        )
    return counts


def manifest_conditions(dataset_manifest: dict[str, Any]) -> list[dict[str, int]]:
    conditions = []
    for name, count in dataset_manifest["group_counts"].items():
        match = GROUP_PATTERN.fullmatch(str(name))
        if match is None:
            raise ValueError(f"unrecognized condition group {name!r}")
        conditions.append(
            {
                "nfp": int(match.group("nfp")),
                "n_base_coils": int(match.group("ncoils")),
                "quasr_count": int(count),
            }
        )
    conditions.sort(key=lambda row: (row["nfp"], row["n_base_coils"]))
    if not conditions:
        raise ValueError("dataset manifest contains no condition groups")
    return conditions


def prepare(args: argparse.Namespace) -> None:
    if args.run_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.run_root}")
    if git_dirty():
        raise RuntimeError("tracked worktree must be clean before preparing a survey")
    checkpoint_sha = file_sha256(args.checkpoint)
    library_sha = file_sha256(args.lib)
    reference_case_sha = file_sha256(args.reference_case)
    if checkpoint_sha != args.expected_checkpoint_sha:
        raise ValueError(f"unexpected normalizer checkpoint SHA-256: {checkpoint_sha}")
    if library_sha != args.expected_lib_sha:
        raise ValueError(f"unexpected score-library SHA-256: {library_sha}")
    if reference_case_sha != args.expected_reference_case_sha:
        raise ValueError(f"unexpected reference-case SHA-256: {reference_case_sha}")

    dataset_manifest_path = args.data_dir / "manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if dataset_manifest.get("format") != "quasr_qh_flow_v1":
        raise ValueError("the data manifest is not quasr_qh_flow_v1")
    conditions = manifest_conditions(dataset_manifest)
    worker_samples = parse_worker_counts(
        args.worker_samples_per_condition, args.worker_count
    )
    target_sample_count = len(conditions) * sum(worker_samples)
    if target_sample_count != args.expected_target_sample_count:
        raise ValueError(
            "computed target sample count does not match the required total: "
            f"{target_sample_count} != {args.expected_target_sample_count}"
        )

    args.run_root.mkdir(parents=True)
    for name in ("workers", "candidates", "logs"):
        (args.run_root / name).mkdir()
    manifest = {
        "format": FORMAT,
        "stage": "prepared",
        "run_label": args.run_label,
        "protocol": {
            "id": PROTOCOL_ID,
            "status": "registered-experimental",
            "purpose": (
                "Estimate the condition-balanced probability that an independent "
                "standardized-data Gaussian QH start reaches score tails under the "
                "current global evaluator."
            ),
            "relationship_to_default": (
                "Global initialization survey only; it does not alter the current "
                "Flow-screen32/Adam200/64-direction optimization default."
            ),
        },
        "created_unix_s": time.time(),
        "code": {
            "commit": git_value("rev-parse", "HEAD"),
            "tracked_dirty": False,
            "branch": git_value("branch", "--show-current"),
        },
        "dataset": {
            "root": str(args.data_dir.resolve()),
            "manifest": str(dataset_manifest_path.resolve()),
            "manifest_sha256": file_sha256(dataset_manifest_path),
            "format": dataset_manifest["format"],
            "sample_count": int(dataset_manifest["requested_count"]),
            "group_counts": dataset_manifest["group_counts"],
        },
        "normalizer": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "role": "normalizer only; no Flow model decoding is used",
        },
        "evaluator": {
            "library": str(args.lib.resolve()),
            "library_sha256": library_sha,
            "abi": 10,
            "target": "QH",
            "target_helicity": "(1,nfp)",
            "configuration": (
                "standalone library defaults with psi_solver_mode=2 and "
                "alpha_solver_mode=2 explicitly pinned"
            ),
            "explicit_overrides": dict(STANDALONE_SCORE_OVERRIDES),
            "resolved_key_defaults": {
                "psi_grid": 48,
                "iota_degree": 3,
                "surface_selection_mode": 0,
                "surface_theta_count": 256,
                "surface_trace_steps": 800,
                "surface_confidence_periods": 2,
            },
            "axis_history": "independent global search; no continuation hint",
            "process_reference_preflight": {
                "case": str(args.reference_case.resolve()),
                "case_sha256": reference_case_sha,
                "expected_score": float(args.expected_reference_score),
                "score_atol": float(args.reference_score_atol),
                "evaluations_per_worker_process": 1,
                "reason": "detect evaluator environment or device drift before sampling",
            },
        },
        "sampling": {
            "prior": "independent N(0,1) per standardized coil-data coordinate",
            "current_projection": (
                "condition-specific current L1 normalization and dominant-current sign"
            ),
            "condition_prior": "exactly balanced across supported (nfp,n_base_coils) groups",
            "condition_count": len(conditions),
            "conditions": conditions,
            "seed_base": int(args.seed_base),
            "numpy_generator": "PCG64 via numpy.random.default_rng(SeedSequence)",
            "tail_retention_score": float(args.tail_retention_score),
        },
        "workers": [
            {
                "worker_index": index,
                "samples_per_condition": worker_samples[index],
                "target_count": worker_samples[index] * len(conditions),
            }
            for index in range(args.worker_count)
        ],
        "worker_count": args.worker_count,
        "target_sample_count": target_sample_count,
        "deferred_adam_followup": {
            "reason": (
                "Adam200 is scheduled separately so its candidate-dependent cost "
                "cannot change the wall time or sample count of this global survey."
            ),
            "parameter_space": "standardized coil data",
            "iterations": 200,
            "directions": 64,
            "difference": "centered",
            "perturbation": 0.0025,
            "learning_rate": 0.01,
            "beta": [0.7, 0.999],
            "selection_bands": [
                {
                    "score_low_inclusive": low,
                    "score_high_exclusive": None if math.isinf(high) else high,
                    "maximum_selected": quota,
                }
                for low, high, quota in FOLLOWUP_BANDS
            ],
            "interpretation": (
                "The global-score tail rate and the Adam-validated convergence-basin "
                "rate are distinct quantities."
            ),
        },
    }
    atomic_write_json(args.run_root / "survey_manifest.json", manifest)
    print(
        json.dumps(
            {
                "event": "survey_prepared",
                "run_root": str(args.run_root),
                "target_sample_count": manifest["target_sample_count"],
                "worker_target_counts": [row["target_count"] for row in manifest["workers"]],
            },
            separators=(",", ":"),
        ),
        flush=True,
    )


def iter_chunk_rows(worker_dir: Path) -> Iterable[dict[str, Any]]:
    for path in sorted((worker_dir / "chunks").glob("chunk_*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def load_worker_rows(run_root: Path, worker_index: int) -> list[dict[str, Any]]:
    return list(iter_chunk_rows(run_root / "workers" / f"worker_{worker_index:02d}"))


def condition_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["nfp"]), int(row["n_base_coils"])


def validate_existing_rows(
    rows: list[dict[str, Any]], conditions: list[tuple[int, int]], targets: dict[tuple[int, int], int]
) -> dict[tuple[int, int], int]:
    seen: set[tuple[int, int, int]] = set()
    counts: Counter[tuple[int, int]] = Counter()
    allowed = set(conditions)
    for row in rows:
        key = condition_key(row)
        index = int(row["condition_sample_index"])
        identity = (*key, index)
        if key not in allowed or identity in seen:
            raise ValueError(f"invalid or duplicate saved sample identity {identity}")
        seen.add(identity)
        counts[key] += 1
    for key, count in counts.items():
        if count > targets[key] or {index for nfp, nc, index in seen if (nfp, nc) == key} != set(range(count)):
            raise ValueError(f"saved samples for {key} are not a contiguous prefix")
    return {key: counts[key] for key in conditions}


def candidate_path(run_root: Path, row: dict[str, Any]) -> Path:
    return run_root / "candidates" / f"{row['sample_id']}.npz"


def compact_failure(error: Exception) -> dict[str, Any]:
    return {
        "score": 0.0,
        "status": "python_error",
        "components": {name: 0.0 for name in COMPONENT_KEYS},
        "diagnostics": {},
        "error": f"{type(error).__name__}: {error}",
    }


def worker(args: argparse.Namespace) -> None:
    import torch

    from scripts.optimize_native_score_cem import compact_score_diagnostics

    manifest_path = args.run_root / "survey_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError("unexpected survey manifest format")
    worker_count = int(manifest["worker_count"])
    if not 0 <= args.worker_index < worker_count:
        raise ValueError("worker index is outside the prepared range")
    if git_value("rev-parse", "HEAD") != manifest["code"]["commit"] or git_dirty():
        raise RuntimeError("worker code does not match the clean prepared commit")

    checkpoint_path = Path(manifest["normalizer"]["checkpoint"])
    library_path = Path(manifest["evaluator"]["library"])
    reference_spec = manifest["evaluator"]["process_reference_preflight"]
    reference_case_path = Path(reference_spec["case"])
    if file_sha256(checkpoint_path) != manifest["normalizer"]["checkpoint_sha256"]:
        raise ValueError("normalizer checkpoint hash changed")
    if file_sha256(library_path) != manifest["evaluator"]["library_sha256"]:
        raise ValueError("score-library hash changed")
    if file_sha256(reference_case_path) != reference_spec["case_sha256"]:
        raise ValueError("score reference-case hash changed")

    torch.cuda.set_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    normalizer = CoilNormalizer.from_dict(checkpoint["normalizer"])
    worker_spec = manifest["workers"][args.worker_index]
    samples_per_condition = int(worker_spec["samples_per_condition"])
    conditions = [
        (int(row["nfp"]), int(row["n_base_coils"]))
        for row in manifest["sampling"]["conditions"]
    ]
    targets = {key: samples_per_condition for key in conditions}
    worker_dir = args.run_root / "workers" / f"worker_{args.worker_index:02d}"
    chunks_dir = worker_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    existing_rows = load_worker_rows(args.run_root, args.worker_index)
    completed = validate_existing_rows(existing_rows, conditions, targets)

    generators: dict[tuple[int, int], np.random.Generator] = {}
    for condition_index, key in enumerate(conditions):
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    int(manifest["sampling"]["seed_base"]),
                    args.worker_index,
                    condition_index,
                ]
            )
        )
        if completed[key]:
            rng.standard_normal(
                (completed[key], key[1], 100), dtype=np.float32
            )
        generators[key] = rng

    existing_chunks = sorted(chunks_dir.glob("chunk_*.jsonl.gz"))
    next_chunk = (
        max(int(path.stem.split("_")[-1].split(".")[0]) for path in existing_chunks) + 1
        if existing_chunks
        else 0
    )
    started = time.perf_counter()
    reference_tokens, reference_nfp = case_tokens(
        json.loads(reference_case_path.read_text(encoding="utf-8"))
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
        or abs(reference_score - float(reference_spec["expected_score"]))
        > float(reference_spec["score_atol"])
    ):
        raise RuntimeError(
            "score reference preflight failed: "
            f"status={reference_result.get('status')} abi={reference_abi} "
            f"score={reference_score}"
        )
    new_count = 0
    score_wall_sum = 0.0
    recent_score_wall: list[float] = []
    pending_rows: list[dict[str, Any]] = []
    outcomes: Counter[str] = Counter(str(row["status"]) for row in existing_rows)
    stop_reason = "target_count_complete"

    def write_progress(stage: str) -> None:
        replace_json(
            worker_dir / "progress.json",
            {
                "format": FORMAT,
                "worker_index": args.worker_index,
                "stage": stage,
                "stop_reason": stop_reason if stage != "running" else None,
                "saved_count": int(sum(completed.values())),
                "target_count": int(sum(targets.values())),
                "new_count": new_count,
                "condition_counts": {
                    f"nfp{key[0]}_nc{key[1]}": completed[key] for key in conditions
                },
                "status_counts": dict(sorted(outcomes.items())),
                "new_score_wall_s": score_wall_sum,
                "process_reference_score": reference_score,
                "process_reference_wall_s": float(reference_wall_s),
                "elapsed_s": time.perf_counter() - started,
                "updated_unix_s": time.time(),
            },
        )

    def flush() -> None:
        nonlocal next_chunk, pending_rows
        if not pending_rows:
            return
        atomic_write_jsonl_gzip(
            chunks_dir / f"chunk_{next_chunk:06d}.jsonl.gz", pending_rows
        )
        next_chunk += 1
        pending_rows = []
        write_progress("running")

    while any(completed[key] < targets[key] for key in conditions):
        made_progress = False
        for condition_index, key in enumerate(conditions):
            if completed[key] >= targets[key]:
                continue
            elapsed = time.perf_counter() - started
            reserve = max(30.0, 1.5 * max(recent_score_wall[-8:], default=0.0))
            if elapsed + reserve >= args.max_wall_s:
                stop_reason = "max_wall_s"
                flush()
                write_progress("stopped")
                print(json.dumps(json.loads((worker_dir / "progress.json").read_text())), flush=True)
                return

            nfp, ncoils = key
            sample_index = completed[key]
            sampled = generators[key].standard_normal(
                (ncoils, 100), dtype=np.float32
            )
            tokens = normalizer.inverse(sampled[None], key)[0]
            effective, clipped_fraction = normalizer.transform(tokens[None], key)
            score_started = time.perf_counter()
            error: str | None = None
            try:
                native, score_wall = score_tokens_standalone(
                    library_path,
                    tokens,
                    nfp=nfp,
                    device=args.device,
                )
                compact = compact_score_diagnostics(native)
                valid = str(native.get("status")) == "ok" and math.isfinite(
                    float(native.get("score", math.nan))
                )
            except Exception as exc:
                score_wall = time.perf_counter() - score_started
                compact = compact_failure(exc)
                valid = False
                error = compact.pop("error")
            sample_id = (
                f"w{args.worker_index:02d}_nfp{nfp:02d}_nc{ncoils:02d}_"
                f"s{sample_index:07d}"
            )
            row = {
                "sample_id": sample_id,
                "worker_index": args.worker_index,
                "condition_index": condition_index,
                "condition_sample_index": sample_index,
                "nfp": nfp,
                "n_base_coils": ncoils,
                "score": float(compact["score"]),
                "status": str(compact["status"]),
                "valid": valid,
                "score_wall_s": float(score_wall),
                "sampled_standard_normal_rms": float(
                    np.sqrt(np.mean(sampled.astype(np.float64) ** 2))
                ),
                "effective_normalized_rms": float(
                    np.sqrt(np.mean(effective.astype(np.float64) ** 2))
                ),
                "post_projection_clipped_fraction": float(clipped_fraction),
                "native": compact,
                "error": error,
            }
            if row["score"] >= float(manifest["sampling"]["tail_retention_score"]):
                path = candidate_path(args.run_root, row)
                if not path.exists():
                    candidate_sha = atomic_savez_compressed(
                        path,
                        sampled_standard_normal=sampled,
                        effective_normalized=effective[0],
                        decoded_tokens=tokens,
                        score=np.asarray([row["score"]], dtype=np.float64),
                        status=np.asarray([row["status"]]),
                    )
                    row["candidate_file"] = str(path.resolve())
                    row["candidate_sha256"] = candidate_sha
            pending_rows.append(row)
            completed[key] += 1
            new_count += 1
            score_wall_sum += float(score_wall)
            recent_score_wall.append(float(score_wall))
            outcomes[row["status"]] += 1
            made_progress = True
            print(
                json.dumps(
                    {
                        "event": "global_score_complete",
                        "sample_id": sample_id,
                        "score": row["score"],
                        "status": row["status"],
                        "score_wall_s": score_wall,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            if len(pending_rows) >= args.chunk_size:
                flush()
        if not made_progress:
            break

    flush()
    write_progress("complete")
    print(json.dumps(json.loads((worker_dir / "progress.json").read_text())), flush=True)


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float | None]:
    if trials == 0:
        return [None, None]
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    low = 0.0 if successes == 0 else max(0.0, center - half)
    high = 1.0 if successes == trials else min(1.0, center + half)
    return [low, high]


def score_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    statuses = Counter(str(row["status"]) for row in rows)
    result: dict[str, Any] = {
        "count": len(rows),
        "status_counts": dict(sorted(statuses.items())),
    }
    if not len(rows):
        return result
    result["score_quantiles"] = {
        str(q): float(np.quantile(scores, q))
        for q in (0.5, 0.9, 0.95, 0.99, 0.999)
    }
    result["score_max"] = float(np.max(scores))
    result["exceedance"] = {}
    for threshold in SCORE_THRESHOLDS:
        successes = int(np.sum(scores >= threshold))
        result["exceedance"][str(int(threshold))] = {
            "count": successes,
            "rate": successes / len(rows),
            "wilson_95": wilson_interval(successes, len(rows)),
        }
    return result


def select_adam_followup(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    selected: list[dict[str, Any]] = []
    bands = []
    for low, high, quota in FOLLOWUP_BANDS:
        eligible = [
            row
            for row in rows
            if float(row["score"]) >= low and float(row["score"]) < high
        ]
        count = min(len(eligible), quota)
        indices = (
            sorted(int(value) for value in rng.choice(len(eligible), size=count, replace=False))
            if count and count < len(eligible)
            else list(range(len(eligible)))
        )
        probability = count / len(eligible) if eligible else 0.0
        for index in indices:
            row = eligible[index]
            selected.append(
                {
                    "sample_id": row["sample_id"],
                    "score": float(row["score"]),
                    "status": row["status"],
                    "nfp": int(row["nfp"]),
                    "n_base_coils": int(row["n_base_coils"]),
                    "score_band": [low, None if math.isinf(high) else high],
                    "within_band_selection_probability": probability,
                    "reconstruction": {
                        "worker_index": int(row["worker_index"]),
                        "condition_index": int(row["condition_index"]),
                        "condition_sample_index": int(row["condition_sample_index"]),
                    },
                    "candidate_file": row.get("candidate_file"),
                    "candidate_sha256": row.get("candidate_sha256"),
                }
            )
        bands.append(
            {
                "score_low_inclusive": low,
                "score_high_exclusive": None if math.isinf(high) else high,
                "population_count": len(eligible),
                "selected_count": count,
                "selection_probability": probability,
            }
        )
    selected.sort(key=lambda row: (-row["score"], row["sample_id"]))
    assumed_adam_wall_s = 1200.0
    return {
        "format": "qh_data_space_random_adam_followup_selection_v1",
        "seed": seed,
        "method": "uniform without replacement within pre-registered score bands",
        "bands": bands,
        "selected_count": len(selected),
        "selected": selected,
        "assumed_adam200_wall_s_per_candidate": assumed_adam_wall_s,
        "estimated_serial_gpu_hours": len(selected) * assumed_adam_wall_s / 3600.0,
        "estimator_note": (
            "Estimate the Adam-validated basin rate by combining each score band's "
            "observed prevalence with its weighted conditional success rate."
        ),
    }


def all_rows(run_root: Path, worker_count: int) -> list[dict[str, Any]]:
    rows = []
    for worker_index in range(worker_count):
        rows.extend(load_worker_rows(run_root, worker_index))
    identities = [row["sample_id"] for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate sample IDs across worker chunks")
    return rows


def summarize(args: argparse.Namespace) -> None:
    manifest = json.loads((args.run_root / "survey_manifest.json").read_text())
    rows = all_rows(args.run_root, int(manifest["worker_count"]))
    groups: defaultdict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    workers: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[condition_key(row)].append(row)
        workers[int(row["worker_index"])].append(row)
    followup = select_adam_followup(rows, args.followup_seed)
    payload = {
        "format": FORMAT,
        "manifest_sha256": file_sha256(args.run_root / "survey_manifest.json"),
        "target_sample_count": int(manifest["target_sample_count"]),
        "saved_sample_count": len(rows),
        "complete": len(rows) == int(manifest["target_sample_count"]),
        "aggregate": score_summary(rows),
        "by_condition": {
            f"nfp{key[0]}_nc{key[1]}": score_summary(group_rows)
            for key, group_rows in sorted(groups.items())
        },
        "by_worker": {
            str(index): {
                **score_summary(worker_rows),
                "score_wall_s": float(sum(float(row["score_wall_s"]) for row in worker_rows)),
                "mean_score_wall_s": (
                    float(np.mean([float(row["score_wall_s"]) for row in worker_rows]))
                    if worker_rows
                    else None
                ),
            }
            for index, worker_rows in sorted(workers.items())
        },
        "top_samples": [
            {
                key: row.get(key)
                for key in (
                    "sample_id",
                    "score",
                    "status",
                    "nfp",
                    "n_base_coils",
                    "candidate_file",
                    "candidate_sha256",
                )
            }
            for row in sorted(rows, key=lambda row: (-float(row["score"]), row["sample_id"]))[
                : args.top_count
            ]
        ],
        "adam_followup": {
            "status": "selection_only; Adam200 has not run",
            "selection_file": "adam_followup_selection.json",
            "selected_count": followup["selected_count"],
            "estimated_serial_gpu_hours": followup["estimated_serial_gpu_hours"],
        },
        "updated_unix_s": time.time(),
    }
    replace_json(args.run_root / "survey_summary.json", payload)
    replace_json(args.run_root / "adam_followup_selection.json", followup)
    print(json.dumps(payload, indent=2), flush=True)


def calibrate(args: argparse.Namespace) -> None:
    manifest = json.loads((args.run_root / "survey_manifest.json").read_text())
    worker_count = int(manifest["worker_count"])
    condition_count = int(manifest["sampling"]["condition_count"])
    recommendations = []
    diagnostics = []
    for worker_index in range(worker_count):
        rows = load_worker_rows(args.run_root, worker_index)
        if not rows:
            raise ValueError(f"pilot worker {worker_index} has no saved samples")
        progress_path = args.run_root / "workers" / f"worker_{worker_index:02d}" / "progress.json"
        progress = json.loads(progress_path.read_text())
        score_wall = sum(float(row["score_wall_s"]) for row in rows)
        mean_score_wall = score_wall / len(rows)
        elapsed = float(progress["elapsed_s"])
        fixed_overhead = max(0.0, elapsed - score_wall)
        usable = args.target_wall_s - args.safety_s - fixed_overhead
        per_condition = max(1, math.floor(usable / (mean_score_wall * condition_count)))
        recommendations.append(per_condition)
        diagnostics.append(
            {
                "worker_index": worker_index,
                "pilot_count": len(rows),
                "mean_score_wall_s": mean_score_wall,
                "observed_non_score_overhead_s": fixed_overhead,
                "recommended_samples_per_condition": per_condition,
                "predicted_target_count": per_condition * condition_count,
                "predicted_wall_s": fixed_overhead
                + per_condition * condition_count * mean_score_wall,
            }
        )
    payload = {
        "format": "qh_data_space_random_survey_calibration_v1",
        "pilot_root": str(args.run_root.resolve()),
        "target_wall_s": args.target_wall_s,
        "safety_s": args.safety_s,
        "worker_samples_per_condition": recommendations,
        "target_sample_count": condition_count * sum(recommendations),
        "workers": diagnostics,
        "scope": "global scoring only; deferred Adam200 cost is reported separately",
    }
    if args.output is not None:
        atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare, run, calibrate, and summarize a QH data-space random global survey."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--run-root", type=Path, required=True)
    prepare_parser.add_argument("--run-label", required=True)
    prepare_parser.add_argument("--checkpoint", type=Path, required=True)
    prepare_parser.add_argument("--lib", type=Path, required=True)
    prepare_parser.add_argument("--reference-case", type=Path, required=True)
    prepare_parser.add_argument("--data-dir", type=Path, required=True)
    prepare_parser.add_argument("--worker-count", type=int, default=6)
    prepare_parser.add_argument("--worker-samples-per-condition", default="2")
    prepare_parser.add_argument(
        "--expected-target-sample-count", type=int, required=True
    )
    prepare_parser.add_argument("--seed-base", type=int, default=2026082801)
    prepare_parser.add_argument("--tail-retention-score", type=float, default=20.0)
    prepare_parser.add_argument(
        "--expected-checkpoint-sha", default=NORMALIZER_CHECKPOINT_SHA256
    )
    prepare_parser.add_argument("--expected-lib-sha", default=CURRENT_SCORE_SHA256)
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
    worker_parser.add_argument("--chunk-size", type=int, default=8)
    worker_parser.add_argument("--max-wall-s", type=float, default=39600.0)
    worker_parser.set_defaults(func=worker)

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--run-root", type=Path, required=True)
    summary_parser.add_argument("--followup-seed", type=int, default=2026082802)
    summary_parser.add_argument("--top-count", type=int, default=64)
    summary_parser.set_defaults(func=summarize)

    calibrate_parser = subparsers.add_parser("calibrate")
    calibrate_parser.add_argument("--run-root", type=Path, required=True)
    calibrate_parser.add_argument("--target-wall-s", type=float, default=36000.0)
    calibrate_parser.add_argument("--safety-s", type=float, default=600.0)
    calibrate_parser.add_argument("--output", type=Path)
    calibrate_parser.set_defaults(func=calibrate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
