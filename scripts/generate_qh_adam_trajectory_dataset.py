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
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flow_matching.collection import (  # noqa: E402
    derive_stream_seed,
    group_label,
    load_train_condition_prior,
    replace_json,
)
from flow_matching.data import CoilNormalizer, file_sha256  # noqa: E402
from flow_matching.trajectory_dataset import (  # noqa: E402
    FORMAT,
    atomic_write_json,
)


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def normalizer_keys(normalizer: CoilNormalizer) -> set[tuple[int, int]]:
    keys = set()
    for text in normalizer.current_l1_a:
        nfp, n_coils = text.split(":", maxsplit=1)
        keys.add((int(nfp), int(n_coils)))
    return keys


def ensure_dataset_manifest(
    dataset_root: Path,
    *,
    checkpoint: Path,
    library: Path,
    prior: dict[str, Any],
) -> None:
    dataset_root.mkdir(parents=True, exist_ok=True)
    for name in ("trajectories", "incomplete", "failures", "streams"):
        (dataset_root / name).mkdir(exist_ok=True)
    path = dataset_root / "dataset_manifest.json"
    expected = {
        "format": FORMAT,
        "unit": "one joint-empirical condition, 32 global starts, then 200 Adam steps",
        "trajectory_glob": "trajectories/*/trajectory_manifest.json",
        "condition_prior": prior,
        "flow_checkpoint_sha256": file_sha256(checkpoint),
        "score_library_sha256": file_sha256(library),
        "optimizer": {
            "gradient": "64 fresh orthogonal random directions, centered difference",
            "perturbation": 0.005,
            "adam": {"lr": 0.02, "beta1": 0.7, "beta2": 0.999},
            "flow": "FP32 RK4-128",
            "formal_center_score": True,
            "local_endpoint_score_omits_coordinate_gradient": True,
        },
        "code_commit": git_commit(),
        "created_unix_s": time.time(),
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for key in ("format", "flow_checkpoint_sha256", "score_library_sha256"):
            if existing.get(key) != expected[key]:
                raise ValueError(f"dataset manifest mismatch for {key}")
        return
    temporary = path.with_name(f"{path.name}.{os.getpid()}.partial")
    temporary.write_text(
        json.dumps(expected, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    try:
        os.link(temporary, path)
    except FileExistsError:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("format") != FORMAT:
            raise ValueError("concurrent dataset manifest has the wrong format")
    finally:
        temporary.unlink(missing_ok=True)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continuously generate screened QH Adam trajectories on one GPU."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--training-run-manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--stream-index", type=int, required=True)
    parser.add_argument("--stream-name", required=True)
    parser.add_argument("--seed-base", type=int, default=20260813)
    parser.add_argument("--max-wall-s", type=float, default=75600.0)
    parser.add_argument("--max-trajectories", type=int, default=0)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--device", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0 <= args.stream_index < 16:
        raise ValueError("stream-index must be in [0, 16)")
    if args.iterations < 1 or args.candidate_count < 1:
        raise ValueError("iterations and candidate-count must be positive")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    normalizer = CoilNormalizer.from_dict(checkpoint["normalizer"])
    prior_manifest = args.training_run_manifest
    if prior_manifest is None:
        candidate = args.checkpoint.parent / "run_manifest.json"
        prior_manifest = candidate if candidate.is_file() else None
    prior = load_train_condition_prior(
        args.data_dir,
        training_run_manifest=prior_manifest,
        supported_keys=normalizer_keys(normalizer),
    )
    ensure_dataset_manifest(
        args.dataset_root,
        checkpoint=args.checkpoint,
        library=args.lib,
        prior=prior.to_dict(),
    )

    stream_seed = derive_stream_seed(args.seed_base, args.stream_index)
    rng = np.random.default_rng(stream_seed)
    stream_dir = args.dataset_root / "streams" / args.stream_name
    stream_dir.mkdir(exist_ok=False)
    stream_manifest = {
        "format": FORMAT,
        "stage": "running",
        "stream_name": args.stream_name,
        "stream_index": args.stream_index,
        "stream_seed": stream_seed,
        "pid": os.getpid(),
        "started_unix_s": time.time(),
        "max_wall_s": args.max_wall_s,
        "code_commit": git_commit(),
    }
    atomic_write_json(stream_dir / "manifest.json", stream_manifest)

    job_started = time.perf_counter()
    completed = 0
    attempted = 0
    failures: Counter[str] = Counter()
    consecutive_failure_reason: str | None = None
    consecutive_failure_count = 0
    durations: list[float] = []
    condition_counts: Counter[str] = Counter()
    stop_reason = "max_wall_s"
    while True:
        elapsed = time.perf_counter() - job_started
        reserve = max(1800.0, 1.25 * max(durations[-3:], default=0.0))
        if elapsed + reserve >= args.max_wall_s:
            break
        if args.max_trajectories and completed >= args.max_trajectories:
            stop_reason = "max_trajectories"
            break

        attempted += 1
        condition_index = int(prior.sample_indices(rng, 1)[0])
        nfp, n_coils = prior.keys[condition_index]
        condition = group_label((nfp, n_coils))
        condition_counts[condition] += 1
        screen_seed = int(rng.integers(0, 2**63 - 1))
        optimizer_seed = int(rng.integers(0, 2**31 - 1))
        trajectory_id = f"{args.stream_name}_{attempted:06d}"
        partial = args.dataset_root / "incomplete" / f"{trajectory_id}.partial"
        partial.mkdir()
        trajectory_started = time.perf_counter()
        failure_reason = None
        try:
            screening_dir = partial / "screening"
            screening_wall_s = run_logged(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "screen_qh_adam_starts.py"),
                    "--checkpoint",
                    str(args.checkpoint),
                    "--lib",
                    str(args.lib),
                    "--out-dir",
                    str(screening_dir),
                    "--nfp",
                    str(nfp),
                    "--n-base-coils",
                    str(n_coils),
                    "--candidate-count",
                    str(args.candidate_count),
                    "--flow-steps",
                    "128",
                    "--seed",
                    str(screen_seed),
                    "--device",
                    str(args.device),
                ],
                partial / "screening.log",
            )
            screening_summary = json.loads(
                (screening_dir / "summary.json").read_text(encoding="utf-8")
            )
            if screening_summary["status"] != "ok":
                raise RuntimeError("screening produced no valid candidate")

            optimization_dir = partial / "optimization"
            optimization_wall_s = run_logged(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "optimize_flow_prior_local_full_gradient_adam.py"),
                    "--checkpoint",
                    str(args.checkpoint),
                    "--initial-case",
                    str(screening_dir / "selected_start.json"),
                    "--lib",
                    str(args.lib),
                    "--out-dir",
                    str(optimization_dir),
                    "--nfp",
                    str(nfp),
                    "--n-base-coils",
                    str(n_coils),
                    "--iterations",
                    str(args.iterations),
                    "--max-wall-s",
                    "7200",
                    "--flow-steps",
                    "128",
                    "--perturbation",
                    "0.005",
                    "--gradient-mode",
                    "random-orthogonal",
                    "--random-directions",
                    "64",
                    "--seed",
                    str(optimizer_seed),
                    "--optimizer",
                    "adam",
                    "--learning-rate",
                    "0.02",
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
            optimization_summary = json.loads(
                (optimization_dir / "summary.json").read_text(encoding="utf-8")
            )
            if (
                optimization_summary["status"] != "ok"
                or optimization_summary["stop_reason"] != "completed_iterations"
                or int(optimization_summary["completed_iterations"]) != args.iterations
            ):
                raise RuntimeError("optimizer did not complete the required trajectory")

            trajectory_wall_s = time.perf_counter() - trajectory_started
            trajectory_manifest = {
                "format": FORMAT,
                "trajectory_id": trajectory_id,
                "stream_name": args.stream_name,
                "stream_index": args.stream_index,
                "stream_attempt_index": attempted,
                "condition": {
                    "nfp": nfp,
                    "n_base_coils": n_coils,
                    "group": condition,
                    "joint_prior_probability": float(prior.probabilities[condition_index]),
                },
                "seeds": {"screening": screen_seed, "optimizer": optimizer_seed},
                "screening": screening_summary,
                "optimization": optimization_summary,
                "timing": {
                    "screening_process_wall_s": screening_wall_s,
                    "optimization_process_wall_s": optimization_wall_s,
                    "trajectory_wall_s": trajectory_wall_s,
                },
                "provenance": {
                    "code_commit": git_commit(),
                    "checkpoint_sha256": file_sha256(args.checkpoint),
                    "score_library_sha256": file_sha256(args.lib),
                },
            }
            atomic_write_json(partial / "trajectory_manifest.json", trajectory_manifest)
            destination = args.dataset_root / "trajectories" / trajectory_id
            os.replace(partial, destination)
            completed += 1
            durations.append(trajectory_wall_s)
            consecutive_failure_reason = None
            consecutive_failure_count = 0
            print(
                json.dumps(
                    {
                        "event": "trajectory_complete",
                        "trajectory_id": trajectory_id,
                        "condition": condition,
                        "start_score": optimization_summary["initial_score"],
                        "best_score": optimization_summary["best_score"],
                        "wall_s": trajectory_wall_s,
                        "completed": completed,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
        except Exception as exc:
            failure_reason = f"{type(exc).__name__}: {exc}"
            failures[failure_reason] += 1
            if failure_reason == consecutive_failure_reason:
                consecutive_failure_count += 1
            else:
                consecutive_failure_reason = failure_reason
                consecutive_failure_count = 1
            atomic_write_json(
                partial / "failure.json",
                {
                    "format": FORMAT,
                    "trajectory_id": trajectory_id,
                    "condition": condition,
                    "failure": failure_reason,
                    "wall_s": time.perf_counter() - trajectory_started,
                },
            )
            destination = args.dataset_root / "failures" / trajectory_id
            os.replace(partial, destination)
            print(
                json.dumps(
                    {"event": "trajectory_failed", "trajectory_id": trajectory_id, "failure": failure_reason},
                    separators=(",", ":"),
                ),
                flush=True,
            )

        elapsed = time.perf_counter() - job_started
        mean_duration = float(np.mean(durations)) if durations else None
        progress = {
            "format": FORMAT,
            "stage": "running",
            "stream_name": args.stream_name,
            "completed_trajectories": completed,
            "attempted_trajectories": attempted,
            "failure_counts": dict(sorted(failures.items())),
            "condition_counts": dict(sorted(condition_counts.items())),
            "elapsed_s": elapsed,
            "mean_completed_trajectory_wall_s": mean_duration,
            "projected_trajectories_per_day_single_gpu": (
                86400.0 / mean_duration if mean_duration else None
            ),
            "last_failure": failure_reason,
            "updated_unix_s": time.time(),
        }
        replace_json(stream_dir / "progress.json", progress)
        if consecutive_failure_count >= 3:
            stop_reason = "three_identical_consecutive_failures"
            break

    stream_manifest.update(
        {
            "stage": "complete",
            "stop_reason": stop_reason,
            "completed_trajectories": completed,
            "attempted_trajectories": attempted,
            "failure_counts": dict(sorted(failures.items())),
            "condition_counts": dict(sorted(condition_counts.items())),
            "elapsed_s": time.perf_counter() - job_started,
            "mean_completed_trajectory_wall_s": (
                float(np.mean(durations)) if durations else None
            ),
            "finished_unix_s": time.time(),
        }
    )
    replace_json(stream_dir / "manifest.json", stream_manifest)
    replace_json(
        stream_dir / "progress.json",
        {**stream_manifest, "updated_unix_s": time.time()},
    )


if __name__ == "__main__":
    main()
