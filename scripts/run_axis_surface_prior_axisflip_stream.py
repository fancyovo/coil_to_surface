from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "gpu_backend" / "python"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from flow_matching.axis_surface_prior import supported_conditions
from flow_matching.axis_surface_prior_v2 import sample_shaped_prior_prototype
from flow_matching.collection import replace_json
from flow_matching.data import CoilNormalizer, file_sha256
from flow_matching.trajectory_dataset import atomic_write_json
from scripts.native_score_runtime import append_jsonl, token_case, write_json
from scripts.optimize_flow_latent import score_config
from scripts.prepare_axis_surface_prior_adam200 import exact_standardized_start
from scripts.run_axis_surface_prior_adam200 import run_logged
from scripts.sample_axis_surface_prior import compact_result


PROTOCOL_ID = (
    "qh-axis-surface-contour-compact-flexible-axisflip-stream-"
    "adam200-64d-abi11-v4"
)
GENERATOR_FORMAT = "axis_surface_contour_prior_compact_flexible_axis_flip_v4"
PRESET = "compact_flexible"
TARGET_HELICITY_SIGN = 1
ARTIFACT_FORMATS = {
    "screening": "axis_surface_prior_axisflip_stream_screening_v4",
    "start": "axis_surface_prior_axisflip_stream_exact_data_start_v4",
    "trajectory": "axis_surface_prior_axisflip_stream_adam200_trajectory_v4",
    "failure": "axis_surface_prior_axisflip_stream_adam200_failure_v4",
    "worker": "axis_surface_prior_axisflip_stream_worker_v4",
}


def formal_screening_score_config() -> dict[str, Any]:
    return score_config(
        iota_degree=3,
        surface_theta_count=128,
        axis_hint=None,
    )


def case_id_for_worker(worker_index: int, worker_count: int, sequence_index: int) -> int:
    if not 0 <= worker_index < worker_count:
        raise ValueError("worker-index must be in [0, worker-count)")
    if sequence_index < 0:
        raise ValueError("sequence-index must be nonnegative")
    return worker_index + worker_count * sequence_index


def discovery_is_open(elapsed_s: float, discovery_wall_s: float) -> bool:
    return elapsed_s < discovery_wall_s


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def classify_iota_interval(iota_min: Any, iota_max: Any) -> str:
    lower = finite_float(iota_min)
    upper = finite_float(iota_max)
    if lower is None or upper is None:
        return "missing"
    lower, upper = sorted((lower, upper))
    if upper < -1.0e-10:
        return "negative"
    if lower > 1.0e-10:
        return "positive"
    return "crosses_zero"


def native_endpoint(native: dict[str, Any]) -> dict[str, Any]:
    diagnostics = native.get("diagnostics") or {}
    iota_min = finite_float(diagnostics.get("iota_min"))
    iota_max = finite_float(diagnostics.get("iota_max"))
    return {
        "score": finite_float(native.get("score")),
        "status": str(native.get("status", "missing")),
        "iota_min": iota_min,
        "iota_max": iota_max,
        "iota_sign": classify_iota_interval(iota_min, iota_max),
    }


def scalar_iota_endpoint(score: Any, iota: Any) -> dict[str, Any]:
    value = finite_float(iota)
    if value is None:
        sign = "missing"
    elif value < -1.0e-10:
        sign = "negative"
    elif value > 1.0e-10:
        sign = "positive"
    else:
        sign = "crosses_zero"
    return {"score": finite_float(score), "iota": value, "iota_sign": sign}


def row_sha256(row: dict[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def optimizer_command(
    *,
    checkpoint: Path,
    start_path: Path,
    score_lib: Path,
    output_dir: Path,
    nfp: int,
    n_base_coils: int,
    iterations: int,
    max_wall_s: float,
    optimizer_seed: int,
    device: int,
) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "optimize_flow_latent.py"),
        "--checkpoint",
        str(checkpoint),
        "--initial-case",
        str(start_path),
        "--lib",
        str(score_lib),
        "--out-dir",
        str(output_dir),
        "--nfp",
        str(nfp),
        "--n-base-coils",
        str(n_base_coils),
        "--target-helicity-sign",
        str(TARGET_HELICITY_SIGN),
        "--iterations",
        str(iterations),
        "--max-wall-s",
        f"{max_wall_s:.6f}",
        "--parameter-space",
        "data",
        "--data-start-mode",
        "exact-unclipped",
        "--recorded-initial-score-tolerance",
        "0.1",
        "--perturbation",
        "0.0025",
        "--gradient-mode",
        "random-orthogonal",
        "--random-directions",
        "64",
        "--seed",
        str(optimizer_seed),
        "--optimizer",
        "adam",
        "--learning-rate",
        "0.01",
        "--beta1",
        "0.7",
        "--beta2",
        "0.999",
        "--flow-device",
        str(device),
        "--score-device",
        str(device),
        "--plot-every",
        "0",
        "--progress-every",
        "10",
        "--trajectory-every",
        "0",
        "--state-every",
        str(iterations),
    ]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Stream axis-flipped compact-prior samples into ABI-11 Adam200."
    )
    value.add_argument("--run-root", type=Path, required=True)
    value.add_argument("--protocol-path", type=Path, required=True)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--score-lib", type=Path, required=True)
    value.add_argument("--expected-commit", required=True)
    value.add_argument("--expected-lib-sha", required=True)
    value.add_argument("--expected-checkpoint-sha", required=True)
    value.add_argument("--worker-index", type=int, required=True)
    value.add_argument("--worker-count", type=int, default=6)
    value.add_argument("--seed", type=int, default=20260905)
    value.add_argument("--device", type=int, default=0)
    value.add_argument("--discovery-wall-s", type=float, default=14400.0)
    value.add_argument("--hard-wall-s", type=float, default=17700.0)
    value.add_argument("--iterations", type=int, default=200)
    value.add_argument("--max-valid-cases", type=int, default=0)
    value.add_argument("--max-screened-cases", type=int, default=0)
    value.add_argument("--start-sequence-index", type=int, default=0)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.discovery_wall_s <= 0.0:
        raise ValueError("discovery-wall-s must be positive")
    if args.hard_wall_s <= args.discovery_wall_s:
        raise ValueError("hard-wall-s must exceed discovery-wall-s")
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    if args.start_sequence_index < 0:
        raise ValueError("start-sequence-index must be nonnegative")
    case_id_for_worker(args.worker_index, args.worker_count, 0)

    protocol = json.loads(args.protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("protocol ID mismatch")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    if commit != args.expected_commit:
        raise RuntimeError(f"repository commit {commit} != {args.expected_commit}")
    if file_sha256(args.score_lib) != args.expected_lib_sha:
        raise RuntimeError("score-library hash mismatch")
    if file_sha256(args.checkpoint) != args.expected_checkpoint_sha:
        raise RuntimeError("checkpoint hash mismatch")

    conditions = supported_conditions()
    if not conditions or any(nc > 4 for _, nc in conditions):
        raise RuntimeError("registered condition set must be nonempty and exclude nc>4")
    checkpoint_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    normalizer = CoilNormalizer.from_dict(checkpoint_payload["normalizer"])

    from stellarator_gpu import score_coils_native

    worker_dir = args.run_root / "workers" / f"worker_{args.worker_index:02d}"
    screening_dir = args.run_root / "screening"
    starts_dir = args.run_root / "starts"
    trajectories_dir = args.run_root / "trajectories"
    failures_dir = args.run_root / "failures"
    incomplete_dir = args.run_root / "incomplete"
    for path in (
        worker_dir,
        screening_dir,
        starts_dir,
        trajectories_dir,
        failures_dir,
        incomplete_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    rows_path = screening_dir / f"worker_{args.worker_index:02d}.jsonl"
    done_path = worker_dir / "done.json"
    if rows_path.exists() or done_path.exists():
        raise FileExistsError(f"worker {args.worker_index} output already exists")

    started = time.perf_counter()
    sequence_index = args.start_sequence_index
    screened = 0
    valid = 0
    completed = 0
    failed_valid = 0
    score_errors = 0
    status_counts: Counter[str] = Counter()
    initial_sign_counts: Counter[str] = Counter()
    best_sign_counts: Counter[str] = Counter()
    best_ge_70: list[dict[str, Any]] = []
    max_best_score: float | None = None
    active_case: dict[str, Any] | None = None
    stop_reason = "discovery_wall_s"
    previous_failure: str | None = None
    repeated_failure_count = 0

    def progress(stage: str) -> None:
        elapsed = time.perf_counter() - started
        replace_json(
            worker_dir / "progress.json",
            {
                "format": ARTIFACT_FORMATS["worker"],
                "protocol_id": PROTOCOL_ID,
                "worker_index": args.worker_index,
                "worker_count": args.worker_count,
                "stage": stage,
                "stop_reason": stop_reason if stage != "running" else None,
                "screened_count": screened,
                "screening_status_counts": dict(sorted(status_counts.items())),
                "score_exception_count": score_errors,
                "valid_count": valid,
                "completed_adam_count": completed,
                "failed_valid_count": failed_valid,
                "initial_valid_iota_sign_counts": dict(sorted(initial_sign_counts.items())),
                "best_iota_sign_counts": dict(sorted(best_sign_counts.items())),
                "max_best_score": max_best_score,
                "best_ge_70_count": len(best_ge_70),
                "best_ge_70_cases": best_ge_70,
                "active_case": active_case,
                "next_sequence_index": sequence_index,
                "elapsed_s": elapsed,
                "discovery_wall_s": args.discovery_wall_s,
                "discovery_remaining_s": max(0.0, args.discovery_wall_s - elapsed),
                "soft_deadline_overrun_s": max(0.0, elapsed - args.discovery_wall_s),
                "updated_unix_s": time.time(),
            },
        )

    progress("running")
    while discovery_is_open(time.perf_counter() - started, args.discovery_wall_s):
        if args.max_valid_cases and valid >= args.max_valid_cases:
            stop_reason = "max_valid_cases"
            break
        if args.max_screened_cases and screened >= args.max_screened_cases:
            stop_reason = "max_screened_cases"
            break

        case_id = case_id_for_worker(
            args.worker_index, args.worker_count, sequence_index
        )
        nfp, n_base_coils = conditions[case_id % len(conditions)]
        generated = sample_shaped_prior_prototype(
            seed=args.seed + case_id,
            nfp=nfp,
            n_base_coils=n_base_coils,
            preset=PRESET,
            surface_phi_samples=96,
            surface_theta_samples=48,
            sample_role="registered_scoring",
            axis_chirality=-1,
        )
        metadata = generated.metadata
        if metadata.get("format") != GENERATOR_FORMAT:
            raise RuntimeError("axis-flipped generator format mismatch")
        if metadata.get("construction_axis_chirality") != -1:
            raise RuntimeError("construction-axis chirality was not flipped")
        if float(metadata["axis_radial_coefficients"][0]) <= 0.0:
            raise RuntimeError("dominant construction-axis radial harmonic changed sign")
        if float(metadata["axis_vertical_coefficients"][0]) >= 0.0:
            raise RuntimeError("dominant construction-axis vertical harmonic was not flipped")

        parameters, current_l1_a, optimizer_start_tokens, roundtrip = (
            exact_standardized_start(
                generated.tokens, normalizer, condition=(nfp, n_base_coils)
            )
        )
        if max(
            roundtrip["geometry_relative_rms"],
            roundtrip["current_relative_rms"],
        ) > 2.0e-6:
            raise RuntimeError("exact-data start roundtrip exceeds tolerance")
        case = token_case(
            optimizer_start_tokens,
            nfp=nfp,
            target="QH",
            metadata={"case_id": case_id, **metadata},
        )
        score_started = time.perf_counter()
        native: dict[str, Any] | None = None
        error: str | None = None
        try:
            native = compact_result(
                score_coils_native(
                    args.score_lib,
                    case["raw"]["x"],
                    case["raw"]["y"],
                    case["raw"]["z"],
                    case["raw"]["current"],
                    nfp,
                    device_id=args.device,
                    target_helicity=(1, nfp),
                    config_overrides=formal_screening_score_config(),
                )
            )
            status = str(native["status"])
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            status = "score_exception"
            score_errors += 1
        row = {
            "format": ARTIFACT_FORMATS["screening"],
            "protocol_id": PROTOCOL_ID,
            "case_id": case_id,
            "worker_index": args.worker_index,
            "sequence_index": sequence_index,
            "seed": args.seed + case_id,
            "nfp": nfp,
            "n_base_coils": n_base_coils,
            "family": PRESET,
            "tokens": optimizer_start_tokens.tolist(),
            "reference_axis": generated.reference_axis[::8].tolist(),
            "generator": metadata,
            "screening_representation": "optimizer_exact_unclipped_reconstruction",
            "data_parameterization": {
                "current_l1_a": current_l1_a,
                "roundtrip": roundtrip,
            },
            "native": native,
            "score_wall_s": time.perf_counter() - score_started,
            "error": error,
        }
        append_jsonl(rows_path, row)
        screened += 1
        status_counts[status] += 1
        sequence_index += 1

        if status != "ok" or native is None:
            if screened % 10 == 0:
                progress("running")
            continue

        valid += 1
        initial_endpoint = native_endpoint(native)
        initial_sign_counts[initial_endpoint["iota_sign"]] += 1
        trajectory_id = f"axisflip_case_{case_id:07d}"
        destination = trajectories_dir / trajectory_id
        partial = incomplete_dir / (
            f"{trajectory_id}.worker{args.worker_index}.{os.getpid()}.partial"
        )
        partial.mkdir()
        optimizer_seed = args.seed + 100_000_000 + case_id
        case_record: dict[str, Any] = {
            "trajectory_id": trajectory_id,
            "case_id": case_id,
            "worker_index": args.worker_index,
            "sequence_index": sequence_index - 1,
            "nfp": nfp,
            "n_base_coils": n_base_coils,
            "initial_score": initial_endpoint["score"],
            "initial_iota": initial_endpoint,
            "optimizer_seed": optimizer_seed,
            "source_row_sha256": row_sha256(row),
        }
        write_json(partial / "case.json", case_record)
        active_case = {
            **case_record,
            "partial_path": str(partial.resolve()),
            "started_unix_s": time.time(),
        }
        progress("running")
        case_started = time.perf_counter()
        try:
            start = token_case(optimizer_start_tokens, nfp=nfp, target="QH")
            start["data_prior_screening"] = {
                "format": ARTIFACT_FORMATS["start"],
                "protocol_id": PROTOCOL_ID,
                "normalized_coil_tokens": parameters.tolist(),
                "current_l1_a": current_l1_a,
                "native_score": native,
                "source_case_id": case_id,
                "source_worker_index": args.worker_index,
                "source_row_sha256": case_record["source_row_sha256"],
                "roundtrip": roundtrip,
                "generator": {
                    "format": GENERATOR_FORMAT,
                    "construction_axis_chirality": -1,
                    "construction_axis_transform": "z_reflection_of_same_seed_baseline",
                },
            }
            start_path = starts_dir / f"case_{case_id:07d}.json"
            write_json(start_path, start)
            shutil.copy2(start_path, partial / "start.json")
            case_record["start"] = str(start_path.resolve())
            case_record["start_sha256"] = file_sha256(start_path)
            write_json(partial / "case.json", case_record)

            elapsed = time.perf_counter() - started
            optimizer_wall_s = max(60.0, args.hard_wall_s - elapsed - 180.0)
            optimization_dir = partial / "optimization"
            optimization_process_wall_s = run_logged(
                optimizer_command(
                    checkpoint=args.checkpoint,
                    start_path=partial / "start.json",
                    score_lib=args.score_lib,
                    output_dir=optimization_dir,
                    nfp=nfp,
                    n_base_coils=n_base_coils,
                    iterations=args.iterations,
                    max_wall_s=optimizer_wall_s,
                    optimizer_seed=optimizer_seed,
                    device=args.device,
                ),
                partial / "optimization.log",
            )
            summary = json.loads(
                (optimization_dir / "summary.json").read_text(encoding="utf-8")
            )
            optimizer_manifest = json.loads(
                (optimization_dir / "manifest.json").read_text(encoding="utf-8")
            )
            if (
                summary.get("status") != "ok"
                or summary.get("stop_reason") != "completed_iterations"
                or int(summary.get("completed_iterations", -1)) != args.iterations
            ):
                raise RuntimeError("optimizer did not complete requested iterations")
            if abs(float(summary["initial_score"]) - float(native["score"])) > 0.1:
                raise RuntimeError("optimizer initial score differs from screening by more than 0.1")
            if optimizer_manifest.get("target_helicity") != [1, nfp]:
                raise RuntimeError("optimizer did not use positive-hand target helicity")
            optimizer_roundtrip = optimizer_manifest["data_parameterization"]
            if float(optimizer_roundtrip["initial_roundtrip_relative_rms"]) > 2.0e-6:
                raise RuntimeError("optimizer step-0 roundtrip exceeds tolerance")

            best_payload = json.loads(
                (optimization_dir / "best.json").read_text(encoding="utf-8")
            )
            best_native = best_payload["original_space_local_gradient_adam"]["native_score"]
            best_endpoint = native_endpoint(best_native)
            history = [
                json.loads(line)
                for line in (optimization_dir / "history.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            final_iota = history[-1].get("current_iota") if history else None
            final_endpoint = scalar_iota_endpoint(summary["final_score"], final_iota)
            trajectory_wall_s = time.perf_counter() - case_started
            trajectory_manifest = {
                "format": ARTIFACT_FORMATS["trajectory"],
                "protocol_id": PROTOCOL_ID,
                "trajectory_id": trajectory_id,
                "case": case_record,
                "target_helicity": [1, nfp],
                "endpoints": {
                    "initial": initial_endpoint,
                    "best": best_endpoint,
                    "final": final_endpoint,
                },
                "optimization": summary,
                "optimizer_protocol": optimizer_manifest["protocol"],
                "initial_consistency_gate": optimizer_manifest[
                    "initial_consistency_gate"
                ],
                "data_parameterization": optimizer_roundtrip,
                "timing": {
                    "optimization_process_wall_s": optimization_process_wall_s,
                    "trajectory_wall_s": trajectory_wall_s,
                },
                "provenance": {
                    "protocol": str(args.protocol_path.resolve()),
                    "code_commit": commit,
                    "score_library_sha256": args.expected_lib_sha,
                    "checkpoint_sha256": args.expected_checkpoint_sha,
                },
            }
            atomic_write_json(partial / "trajectory_manifest.json", trajectory_manifest)
            os.replace(partial, destination)
            completed += 1
            best_sign_counts[best_endpoint["iota_sign"]] += 1
            best_score = float(summary["best_score"])
            max_best_score = (
                best_score if max_best_score is None else max(max_best_score, best_score)
            )
            if best_score >= 70.0:
                best_ge_70.append(
                    {
                        "trajectory_id": trajectory_id,
                        "case_id": case_id,
                        "best_score": best_score,
                        "best_iteration": int(summary["best_iteration"]),
                        "best_iota_sign": best_endpoint["iota_sign"],
                    }
                )
            previous_failure = None
            repeated_failure_count = 0
            print(
                json.dumps(
                    {
                        "event": "trajectory_complete",
                        "worker_index": args.worker_index,
                        "case_id": case_id,
                        "initial_score": summary["initial_score"],
                        "initial_iota_sign": initial_endpoint["iota_sign"],
                        "best_score": summary["best_score"],
                        "best_iota_sign": best_endpoint["iota_sign"],
                        "wall_s": trajectory_wall_s,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
        except Exception as exc:
            signature = f"{type(exc).__name__}: {exc}"
            failure = {
                "format": ARTIFACT_FORMATS["failure"],
                "protocol_id": PROTOCOL_ID,
                "trajectory_id": trajectory_id,
                "case": case_record,
                "error": signature,
                "wall_s": time.perf_counter() - case_started,
            }
            write_json(partial / "failure.json", failure)
            failure_destination = failures_dir / trajectory_id
            if failure_destination.exists():
                failure_destination = failures_dir / f"{trajectory_id}.{int(time.time())}"
            os.replace(partial, failure_destination)
            failed_valid += 1
            if signature == previous_failure:
                repeated_failure_count += 1
            else:
                previous_failure = signature
                repeated_failure_count = 1
            print(json.dumps({"event": "trajectory_failed", **failure}), flush=True)
        active_case = None
        progress("running")
        if repeated_failure_count >= 3:
            stop_reason = "three_identical_consecutive_failures"
            break

    if not discovery_is_open(time.perf_counter() - started, args.discovery_wall_s):
        stop_reason = "discovery_wall_s"
    final_stage = (
        "failed"
        if stop_reason == "three_identical_consecutive_failures"
        else "complete"
    )
    progress(final_stage)
    final = json.loads((worker_dir / "progress.json").read_text(encoding="utf-8"))
    final["finished_unix_s"] = time.time()
    write_json(done_path, final)
    print(json.dumps(final, indent=2), flush=True)
    if final_stage == "failed":
        raise RuntimeError("worker stopped after three identical consecutive failures")


if __name__ == "__main__":
    main()
