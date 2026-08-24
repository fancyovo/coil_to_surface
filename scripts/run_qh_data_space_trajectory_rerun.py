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
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flow_matching.collection import replace_json  # noqa: E402
from flow_matching.data import CoilNormalizer, file_sha256  # noqa: E402
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


def coil_tokens(payload: dict[str, Any]) -> np.ndarray:
    raw = payload["raw"]
    return np.column_stack(
        (
            np.asarray(raw["x"], dtype=np.float64),
            np.asarray(raw["y"], dtype=np.float64),
            np.asarray(raw["z"], dtype=np.float64),
            np.asarray(raw["current"], dtype=np.float64),
        )
    )


def make_data_start(
    source: dict[str, Any],
    normalizer: CoilNormalizer,
    *,
    nfp: int,
    n_base_coils: int,
    source_path: Path,
) -> tuple[dict[str, Any], float]:
    tokens = coil_tokens(source)
    if tokens.shape != (n_base_coils, 100):
        raise ValueError(
            f"saved screening winner has shape {tokens.shape}, expected {(n_base_coils, 100)}"
        )
    normalized, clipped_fraction = normalizer.transform(
        tokens[None], (nfp, n_base_coils)
    )
    payload = dict(source)
    original = source.get("flow_prior_screening", {})
    payload["data_prior_screening"] = {
        "format": "qh_reused_flow_winner_data_start_v1",
        "normalized_coil_tokens": normalized[0].astype(np.float32).tolist(),
        "native_score": original.get("native_score"),
        "source_flow_start": str(source_path.resolve()),
        "source_flow_start_sha256": file_sha256(source_path),
        "post_projection_clipped_fraction": float(clipped_fraction),
        "candidate_index": original.get("candidate_index"),
        "candidate_count": original.get("candidate_count"),
    }
    return payload, float(clipped_fraction)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one worker of the 309-case same-start data-space rerun."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-wall-s", type=float, default=81000.0)
    parser.add_argument("--max-new-cases", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--allow-partial", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = json.loads((args.run_root / "run_manifest.json").read_text(encoding="utf-8"))
    if not 0 <= args.worker_index < int(manifest["worker_count"]):
        raise ValueError("worker index is outside the prepared worker range")
    checkpoint_path = Path(manifest["checkpoint"])
    library = Path(manifest["score_library"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    normalizer = CoilNormalizer.from_dict(checkpoint["normalizer"])
    if int(checkpoint["step"]) != 30000:
        raise RuntimeError("unexpected Flow checkpoint step")

    cases = worker_cases(manifest, args.worker_index)
    worker_dir = args.run_root / "workers" / f"worker_{args.worker_index:02d}"
    worker_dir.mkdir(exist_ok=True)
    started = time.perf_counter()
    completed = 0
    skipped = 0
    attempted = 0
    durations: list[float] = []
    outcomes: Counter[str] = Counter()
    stop_reason = "all_assigned_cases_complete"

    for case in cases:
        destination = args.run_root / "trajectories" / case["trajectory_id"]
        if destination.exists():
            skipped += 1
            continue
        if args.max_new_cases and completed >= args.max_new_cases:
            stop_reason = "max_new_cases"
            break
        elapsed = time.perf_counter() - started
        reserve = max(1800.0, 1.35 * max(durations[-3:], default=0.0))
        if elapsed + reserve >= args.max_wall_s:
            stop_reason = "max_wall_s"
            break

        attempted += 1
        trajectory_id = case["trajectory_id"]
        partial = args.run_root / "incomplete" / (
            f"{trajectory_id}.worker{args.worker_index}.{os.getpid()}.partial"
        )
        partial.mkdir()
        case_started = time.perf_counter()
        try:
            source_path = Path(case["reference_selected_start"])
            source = json.loads(source_path.read_text(encoding="utf-8"))
            data_start, clipped_fraction = make_data_start(
                source,
                normalizer,
                nfp=int(case["nfp"]),
                n_base_coils=int(case["n_base_coils"]),
                source_path=source_path,
            )
            screening_dir = partial / "screening"
            screening_dir.mkdir()
            shutil.copy2(source_path, screening_dir / "reference_selected_start.json")
            atomic_write_json(screening_dir / "selected_start.json", data_start)

            optimization_dir = partial / "optimization"
            optimization_wall_s = run_logged(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "optimize_flow_prior_local_full_gradient_adam.py"),
                    "--checkpoint",
                    str(checkpoint_path),
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
                    str(args.iterations),
                    "--max-wall-s",
                    "7200",
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
                    "--progress-every",
                    "20",
                    "--trajectory-every",
                    "0",
                    "--state-every",
                    str(args.iterations),
                    "--save-training-trace",
                ],
                partial / "optimization.log",
            )
            optimization = json.loads(
                (optimization_dir / "summary.json").read_text(encoding="utf-8")
            )
            if (
                optimization["status"] != "ok"
                or optimization["stop_reason"] != "completed_iterations"
                or int(optimization["completed_iterations"]) != args.iterations
            ):
                raise RuntimeError(
                    f"optimizer did not complete {args.iterations} iterations"
                )

            reference_manifest = json.loads(
                Path(case["reference_manifest"]).read_text(encoding="utf-8")
            )
            trajectory_wall_s = time.perf_counter() - case_started
            trajectory_manifest = {
                "format": "qh_data_space_same_start_trajectory_v1",
                "trajectory_id": trajectory_id,
                "stream_name": f"data_worker_{args.worker_index:02d}",
                "stream_index": args.worker_index,
                "condition": reference_manifest["condition"],
                "seeds": reference_manifest["seeds"],
                "screening": {
                    "status": "reused_reference_winner",
                    "screening_process_wall_s": 0.0,
                    "reference": reference_manifest.get("screening", {}),
                },
                "optimization": optimization,
                "timing": {
                    "screening_process_wall_s": 0.0,
                    "optimization_process_wall_s": optimization_wall_s,
                    "trajectory_wall_s": trajectory_wall_s,
                },
                "same_start_control": {
                    "reference_manifest": case["reference_manifest"],
                    "reference_selected_start": str(source_path.resolve()),
                    "reference_start_score": case["reference_start_score"],
                    "reference_best_score": case["reference_best_score"],
                    "data_start_clipped_fraction": clipped_fraction,
                    "initial_score_difference": (
                        float(optimization["initial_score"])
                        - float(case["reference_start_score"])
                    ),
                },
                "provenance": {
                    "run_manifest": str((args.run_root / "run_manifest.json").resolve()),
                    "code_commit": manifest["code_commit"],
                    "checkpoint_sha256": manifest["checkpoint_sha256"],
                    "score_library_sha256": manifest["score_library_sha256"],
                },
            }
            atomic_write_json(partial / "trajectory_manifest.json", trajectory_manifest)
            os.replace(partial, destination)
            completed += 1
            outcomes["ok"] += 1
            durations.append(trajectory_wall_s)
            print(
                json.dumps(
                    {
                        "event": "trajectory_complete",
                        "trajectory_id": trajectory_id,
                        "start_score": optimization["initial_score"],
                        "best_score": optimization["best_score"],
                        "wall_s": trajectory_wall_s,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
        except Exception as exc:
            failure = {
                "format": "qh_data_space_same_start_failure_v1",
                "trajectory_id": trajectory_id,
                "worker_index": args.worker_index,
                "error": f"{type(exc).__name__}: {exc}",
                "wall_s": time.perf_counter() - case_started,
            }
            atomic_write_json(partial / "failure.json", failure)
            failure_destination = args.run_root / "failures" / (
                f"{trajectory_id}.worker{args.worker_index}.{int(time.time())}"
            )
            os.replace(partial, failure_destination)
            outcomes["runtime_failure"] += 1
            print(json.dumps({"event": "trajectory_failed", **failure}), flush=True)

        replace_json(
            worker_dir / "progress.json",
            {
                "format": "qh_data_space_same_start_worker_v1",
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

    finished_ids = {
        path.name for path in (args.run_root / "trajectories").iterdir() if path.is_dir()
    }
    missing = [row["trajectory_id"] for row in cases if row["trajectory_id"] not in finished_ids]
    final = {
        "format": "qh_data_space_same_start_worker_v1",
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
