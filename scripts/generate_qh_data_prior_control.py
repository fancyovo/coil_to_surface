from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flow_matching.collection import replace_json  # noqa: E402
from flow_matching.trajectory_dataset import atomic_write_json  # noqa: E402


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
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}; see {log_path}"
        )
    return wall_s


def worker_cases(manifest: dict[str, Any], worker_index: int) -> list[dict[str, Any]]:
    return sorted(
        (
            row
            for row in manifest["cases"]
            if int(row["worker_index"]) == worker_index
        ),
        key=lambda row: row["trajectory_id"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one worker of the matched standardized-data-prior control."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--max-wall-s", type=float, default=93600.0)
    parser.add_argument("--max-new-cases", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = json.loads(
        (args.run_root / "control_manifest.json").read_text(encoding="utf-8")
    )
    worker_count = int(manifest["worker_count"])
    if not 0 <= args.worker_index < worker_count:
        raise ValueError("worker index is outside the prepared worker range")
    checkpoint = Path(manifest["checkpoint"])
    library = Path(manifest["score_library"])
    cases = worker_cases(manifest, args.worker_index)
    worker_dir = args.run_root / "workers" / f"worker_{args.worker_index:02d}"
    worker_dir.mkdir(exist_ok=True)

    started = time.perf_counter()
    new_completed = 0
    attempted = 0
    skipped = 0
    outcomes: Counter[str] = Counter()
    durations: list[float] = []
    stop_reason = "all_assigned_cases_complete"
    for case in cases:
        destination = args.run_root / "cases" / case["trajectory_id"]
        if destination.exists():
            skipped += 1
            continue
        if args.max_new_cases and new_completed >= args.max_new_cases:
            stop_reason = "max_new_cases"
            break
        elapsed = time.perf_counter() - started
        reserve = max(2400.0, 1.35 * max(durations[-3:], default=0.0))
        if elapsed + reserve >= args.max_wall_s:
            stop_reason = "max_wall_s"
            break

        attempted += 1
        partial = args.run_root / "incomplete" / (
            f"{case['trajectory_id']}.worker{args.worker_index}.{os.getpid()}.partial"
        )
        partial.mkdir()
        case_started = time.perf_counter()
        try:
            screening_dir = partial / "screening"
            screening_wall_s = run_logged(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "screen_qh_adam_starts.py"),
                    "--checkpoint",
                    str(checkpoint),
                    "--lib",
                    str(library),
                    "--out-dir",
                    str(screening_dir),
                    "--nfp",
                    str(case["nfp"]),
                    "--n-base-coils",
                    str(case["n_base_coils"]),
                    "--candidate-count",
                    "32",
                    "--parameter-space",
                    "data",
                    "--seed",
                    str(case["screening_seed"]),
                    "--device",
                    str(args.device),
                ],
                partial / "screening.log",
            )
            screening = json.loads(
                (screening_dir / "summary.json").read_text(encoding="utf-8")
            )
            optimization = None
            optimization_wall_s = 0.0
            if screening["status"] == "ok":
                optimization_dir = partial / "optimization"
                optimization_wall_s = run_logged(
                    [
                        sys.executable,
                        str(
                            REPO_ROOT
                            / "scripts"
                            / "optimize_flow_prior_local_full_gradient_adam.py"
                        ),
                        "--checkpoint",
                        str(checkpoint),
                        "--initial-case",
                        str(screening_dir / "selected_start.json"),
                        "--lib",
                        str(library),
                        "--out-dir",
                        str(optimization_dir),
                        "--nfp",
                        str(case["nfp"]),
                        "--n-base-coils",
                        str(case["n_base_coils"]),
                        "--iterations",
                        "200",
                        "--max-wall-s",
                        "3600",
                        "--flow-steps",
                        "128",
                        "--parameter-space",
                        "data",
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
                        "--trajectory-every",
                        "0",
                        "--progress-every",
                        "20",
                        "--state-every",
                        "200",
                    ],
                    partial / "optimization.log",
                )
                optimization = json.loads(
                    (optimization_dir / "summary.json").read_text(encoding="utf-8")
                )
                if (
                    optimization["status"] != "ok"
                    or optimization["stop_reason"] != "completed_iterations"
                    or int(optimization["completed_iterations"]) != 200
                ):
                    raise RuntimeError("optimizer did not complete 200 iterations")
                status = "optimization_ok"
            elif screening["status"] == "no_valid_candidate":
                status = "no_valid_start"
            else:
                raise RuntimeError(f"unexpected screening status {screening['status']}")

            wall_s = time.perf_counter() - case_started
            case_manifest = {
                "format": "qh_data_prior_end_to_end_case_v1",
                "trajectory_id": case["trajectory_id"],
                "status": status,
                "condition": {
                    "nfp": int(case["nfp"]),
                    "n_base_coils": int(case["n_base_coils"]),
                    "group": case["condition_group"],
                },
                "seeds": {
                    "screening": int(case["screening_seed"]),
                    "optimizer": int(case["optimizer_seed"]),
                },
                "reference": case,
                "screening": screening,
                "optimization": optimization,
                "timing": {
                    "screening_process_wall_s": screening_wall_s,
                    "optimization_process_wall_s": optimization_wall_s,
                    "case_wall_s": wall_s,
                },
            }
            atomic_write_json(partial / "case_manifest.json", case_manifest)
            os.replace(partial, destination)
            new_completed += 1
            outcomes[status] += 1
            durations.append(wall_s)
            print(
                json.dumps(
                    {
                        "event": "case_complete",
                        "trajectory_id": case["trajectory_id"],
                        "status": status,
                        "selected_score": screening.get("selected_score"),
                        "best_score": None if optimization is None else optimization["best_score"],
                        "wall_s": wall_s,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
        except Exception as exc:
            failure = {
                "format": "qh_data_prior_end_to_end_failure_v1",
                "trajectory_id": case["trajectory_id"],
                "worker_index": args.worker_index,
                "error": f"{type(exc).__name__}: {exc}",
                "wall_s": time.perf_counter() - case_started,
            }
            atomic_write_json(partial / "failure.json", failure)
            failure_destination = args.run_root / "failures" / (
                f"{case['trajectory_id']}.worker{args.worker_index}.{int(time.time())}"
            )
            os.replace(partial, failure_destination)
            outcomes["runtime_failure"] += 1
            print(json.dumps({"event": "case_failed", **failure}), flush=True)

        replace_json(
            worker_dir / "progress.json",
            {
                "format": "qh_data_prior_end_to_end_worker_v1",
                "worker_index": args.worker_index,
                "new_completed": new_completed,
                "attempted": attempted,
                "skipped_existing": skipped,
                "outcomes": dict(sorted(outcomes.items())),
                "elapsed_s": time.perf_counter() - started,
                "mean_case_wall_s": float(np.mean(durations)) if durations else None,
                "updated_unix_s": time.time(),
                "stage": "running",
            },
        )

    final = {
        "format": "qh_data_prior_end_to_end_worker_v1",
        "worker_index": args.worker_index,
        "stage": "complete",
        "stop_reason": stop_reason,
        "assigned_cases": len(cases),
        "new_completed": new_completed,
        "attempted": attempted,
        "skipped_existing": skipped,
        "outcomes": dict(sorted(outcomes.items())),
        "elapsed_s": time.perf_counter() - started,
        "mean_case_wall_s": float(np.mean(durations)) if durations else None,
        "finished_unix_s": time.time(),
    }
    replace_json(worker_dir / "progress.json", final)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
