from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

from flow_matching.collection import replace_json
from flow_matching.data import file_sha256
from flow_matching.trajectory_dataset import atomic_write_json


def worker_cases(manifest: dict[str, Any], worker_index: int) -> list[dict[str, Any]]:
    return sorted(
        (case for case in manifest["cases"] if int(case["worker_index"]) == worker_index),
        key=lambda case: int(case["selection_rank"]),
    )


def run_logged(command: list[str], log_path: Path) -> float:
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
    wall_s = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(f"optimizer exited {completed.returncode}; see {log_path}")
    return wall_s


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Run one balanced-v2 direct-data Adam200 worker.")
    value.add_argument("--run-root", type=Path, required=True)
    value.add_argument("--worker-index", type=int, required=True)
    value.add_argument("--device", type=int, default=0)
    value.add_argument("--max-wall-s", type=float, default=17400.0)
    value.add_argument("--max-new-cases", type=int, default=0)
    value.add_argument("--iterations", type=int, default=200)
    value.add_argument("--allow-partial", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    manifest = json.loads((args.run_root / "selection_manifest.json").read_text(encoding="utf-8"))
    cases = worker_cases(manifest, args.worker_index)
    if not cases:
        raise ValueError(f"worker {args.worker_index} has no assigned cases")
    worker_dir = args.run_root / "workers" / f"worker_{args.worker_index:02d}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir = args.run_root / "trajectories"
    failures_dir = args.run_root / "failures"
    incomplete_dir = args.run_root / "incomplete"
    for path in (trajectories_dir, failures_dir, incomplete_dir):
        path.mkdir(exist_ok=True)

    checkpoint = manifest["artifacts"]["checkpoint"]
    score_lib = manifest["artifacts"]["score_library"]
    started = time.perf_counter()
    completed = 0
    attempted = 0
    skipped = 0
    durations: list[float] = []
    outcomes: Counter[str] = Counter()
    stop_reason = "all_assigned_cases_complete"
    previous_failure: str | None = None
    repeated_failure_count = 0

    for case in cases:
        destination = trajectories_dir / case["trajectory_id"]
        if destination.exists():
            skipped += 1
            continue
        if args.max_new_cases and completed >= args.max_new_cases:
            stop_reason = "max_new_cases"
            break
        elapsed = time.perf_counter() - started
        reserve = max(2400.0, 1.35 * max(durations[-3:], default=0.0))
        if elapsed + reserve >= args.max_wall_s:
            stop_reason = "max_wall_s"
            break

        attempted += 1
        partial = incomplete_dir / f"{case['trajectory_id']}.worker{args.worker_index}.{os.getpid()}.partial"
        partial.mkdir()
        case_started = time.perf_counter()
        try:
            start_path = Path(case["start"])
            if file_sha256(start_path) != case["start_sha256"]:
                raise RuntimeError("start hash mismatch")
            shutil.copy2(start_path, partial / "start.json")
            optimization_dir = partial / "optimization"
            optimization_wall_s = run_logged(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "optimize_flow_latent.py"),
                    "--checkpoint",
                    checkpoint,
                    "--initial-case",
                    str(partial / "start.json"),
                    "--lib",
                    score_lib,
                    "--out-dir",
                    str(optimization_dir),
                    "--nfp",
                    str(case["nfp"]),
                    "--n-base-coils",
                    str(case["n_base_coils"]),
                    "--iterations",
                    str(args.iterations),
                    "--max-wall-s",
                    "2400",
                    "--parameter-space",
                    "data",
                    "--data-start-mode",
                    "exact-unclipped",
                    "--perturbation",
                    "0.0025",
                    "--gradient-mode",
                    "random-orthogonal",
                    "--random-directions",
                    "64",
                    "--seed",
                    str(case["optimizer_seed"]),
                    "--optimizer",
                    "adam",
                    "--learning-rate",
                    "0.01",
                    "--beta1",
                    "0.7",
                    "--beta2",
                    "0.999",
                    "--flow-device",
                    str(args.device),
                    "--score-device",
                    str(args.device),
                    "--plot-every",
                    "0",
                    "--progress-every",
                    "10",
                    "--trajectory-every",
                    "0",
                    "--state-every",
                    str(args.iterations),
                ],
                partial / "optimization.log",
            )
            summary = json.loads((optimization_dir / "summary.json").read_text(encoding="utf-8"))
            if (
                summary.get("status") != "ok"
                or summary.get("stop_reason") != "completed_iterations"
                or int(summary.get("completed_iterations", -1)) != args.iterations
            ):
                raise RuntimeError("optimizer did not complete requested iterations")
            optimizer_manifest = json.loads(
                (optimization_dir / "manifest.json").read_text(encoding="utf-8")
            )
            roundtrip = optimizer_manifest["data_parameterization"]
            if float(roundtrip["initial_roundtrip_relative_rms"]) > 2.0e-6:
                raise RuntimeError("optimizer step-0 roundtrip exceeds tolerance")
            trajectory_wall_s = time.perf_counter() - case_started
            trajectory_manifest = {
                "format": "axis_surface_prior_balanced_v2_adam200_trajectory_v1",
                "protocol_id": manifest["protocol_id"],
                "trajectory_id": case["trajectory_id"],
                "case": case,
                "optimization": summary,
                "optimizer_protocol": optimizer_manifest["protocol"],
                "data_parameterization": roundtrip,
                "timing": {
                    "optimization_process_wall_s": optimization_wall_s,
                    "trajectory_wall_s": trajectory_wall_s,
                },
                "provenance": {
                    "selection_manifest": str((args.run_root / "selection_manifest.json").resolve()),
                    "code_commit": manifest["code_commit"],
                    "score_library_sha256": manifest["artifacts"]["score_library_sha256"],
                    "checkpoint_sha256": manifest["artifacts"]["checkpoint_sha256"],
                },
            }
            atomic_write_json(partial / "trajectory_manifest.json", trajectory_manifest)
            os.replace(partial, destination)
            completed += 1
            durations.append(trajectory_wall_s)
            outcomes["ok"] += 1
            previous_failure = None
            repeated_failure_count = 0
            print(
                json.dumps(
                    {
                        "event": "trajectory_complete",
                        "trajectory_id": case["trajectory_id"],
                        "initial_score": summary["initial_score"],
                        "best_score": summary["best_score"],
                        "wall_s": trajectory_wall_s,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
        except Exception as exc:
            signature = f"{type(exc).__name__}: {exc}"
            failure = {
                "format": "axis_surface_prior_balanced_v2_adam200_failure_v1",
                "trajectory_id": case["trajectory_id"],
                "worker_index": args.worker_index,
                "error": signature,
                "wall_s": time.perf_counter() - case_started,
            }
            atomic_write_json(partial / "failure.json", failure)
            os.replace(partial, failures_dir / f"{case['trajectory_id']}.{int(time.time())}")
            outcomes["runtime_failure"] += 1
            print(json.dumps({"event": "trajectory_failed", **failure}), flush=True)
            if signature == previous_failure:
                repeated_failure_count += 1
            else:
                previous_failure = signature
                repeated_failure_count = 1

        replace_json(
            worker_dir / "progress.json",
            {
                "format": "axis_surface_prior_balanced_v2_adam200_worker_v1",
                "worker_index": args.worker_index,
                "stage": "running",
                "assigned_cases": len(cases),
                "new_completed": completed,
                "attempted": attempted,
                "skipped_existing": skipped,
                "outcomes": dict(sorted(outcomes.items())),
                "elapsed_s": time.perf_counter() - started,
                "mean_trajectory_wall_s": float(np.mean(durations)) if durations else None,
                "updated_unix_s": time.time(),
            },
        )
        if repeated_failure_count >= 3:
            stop_reason = "three_identical_consecutive_failures"
            break

    finished = {path.name for path in trajectories_dir.iterdir() if path.is_dir()}
    missing = [case["trajectory_id"] for case in cases if case["trajectory_id"] not in finished]
    final = {
        "format": "axis_surface_prior_balanced_v2_adam200_worker_v1",
        "worker_index": args.worker_index,
        "stage": "complete" if not missing else "incomplete",
        "stop_reason": stop_reason,
        "assigned_cases": len(cases),
        "new_completed": completed,
        "attempted": attempted,
        "skipped_existing": skipped,
        "missing_cases": missing,
        "outcomes": dict(sorted(outcomes.items())),
        "elapsed_s": time.perf_counter() - started,
        "mean_trajectory_wall_s": float(np.mean(durations)) if durations else None,
        "finished_unix_s": time.time(),
    }
    replace_json(worker_dir / "progress.json", final)
    print(json.dumps(final, indent=2), flush=True)
    if missing and not args.allow_partial:
        raise RuntimeError(f"worker left {len(missing)} assigned trajectories incomplete")


if __name__ == "__main__":
    main()
