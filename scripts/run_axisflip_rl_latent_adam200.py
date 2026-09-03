from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "qh-axisflip-rl-round12-flow-screen32-adam200-64d-abi11-v1"
SPEC_PATH = REPO_ROOT / "evaluation" / "axisflip_rl_round12_latent_adam200_abi11_v1.json"
NFP = 8
N_BASE_COILS = 3
SCREEN_COUNT = 32
FLOW_STEPS = 128
ITERATIONS = 200
DIRECTIONS = 64
PERTURBATION = 0.005
LEARNING_RATE = 0.02
BETA1 = 0.7
BETA2 = 0.999


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def repository_state() -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    tracked = subprocess.check_output(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "status",
            "--short",
            "--untracked-files=no",
        ],
        text=True,
    ).strip()
    return {"commit": commit, "tracked_dirty": bool(tracked)}


def case_indices(worker_index: int, worker_count: int, start_count: int) -> list[int]:
    return [worker_index + worker_count * index for index in range(start_count)]


def screen_command(
    *, checkpoint: Path, score_lib: Path, out_dir: Path, seed: int
) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "screen_flow_starts.py"),
        "--checkpoint",
        str(checkpoint),
        "--lib",
        str(score_lib),
        "--out-dir",
        str(out_dir),
        "--nfp",
        str(NFP),
        "--n-base-coils",
        str(N_BASE_COILS),
        "--candidate-count",
        str(SCREEN_COUNT),
        "--parameter-space",
        "latent",
        "--flow-steps",
        str(FLOW_STEPS),
        "--seed",
        str(seed),
        "--device",
        "0",
        "--iota-degree",
        "3",
        "--surface-theta-count",
        "128",
    ]


def optimization_command(
    *,
    checkpoint: Path,
    score_lib: Path,
    initial_case: Path,
    out_dir: Path,
    seed: int,
    max_wall_s: float,
) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "optimize_flow_latent.py"),
        "--checkpoint",
        str(checkpoint),
        "--initial-case",
        str(initial_case),
        "--lib",
        str(score_lib),
        "--out-dir",
        str(out_dir),
        "--nfp",
        str(NFP),
        "--n-base-coils",
        str(N_BASE_COILS),
        "--target-helicity-sign",
        "1",
        "--iterations",
        str(ITERATIONS),
        "--max-wall-s",
        str(max_wall_s),
        "--flow-steps",
        str(FLOW_STEPS),
        "--parameter-space",
        "latent",
        "--recorded-initial-score-tolerance",
        "0.1",
        "--perturbation",
        str(PERTURBATION),
        "--gradient-mode",
        "random-orthogonal",
        "--random-directions",
        str(DIRECTIONS),
        "--seed",
        str(seed),
        "--optimizer",
        "adam",
        "--learning-rate",
        str(LEARNING_RATE),
        "--beta1",
        str(BETA1),
        "--beta2",
        str(BETA2),
        "--flow-device",
        "0",
        "--score-device",
        "0",
        "--flow-pipeline",
        "--formal-surface-theta-count",
        "128",
        "--local-surface-theta-count",
        "64",
        "--iota-degree",
        "3",
        "--plot-every",
        "20",
        "--progress-every",
        "1",
        "--trajectory-every",
        "20",
        "--state-every",
        "1",
    ]


def run_logged(command: list[str], *, stdout_path: Path, stderr_path: Path) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    return int(completed.returncode)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    state = repository_state()
    if state != {"commit": args.expected_commit, "tracked_dirty": False}:
        raise RuntimeError(f"repository state mismatch: {state}")
    if file_sha256(args.checkpoint) != args.expected_checkpoint_sha:
        raise RuntimeError("frozen Flow checkpoint SHA-256 mismatch")
    if file_sha256(args.score_lib) != args.expected_score_lib_sha:
        raise RuntimeError("native score library SHA-256 mismatch")
    spec = load_json(SPEC_PATH)
    if spec.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("experiment specification protocol mismatch")

    import torch

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    required = {"format", "stage", "model_config", "ema", "normalizer", "step"}
    missing = required - checkpoint.keys()
    if missing:
        raise RuntimeError(f"Flow checkpoint lacks keys: {sorted(missing)}")
    if checkpoint.get("stage") != "online_policy":
        raise RuntimeError("Flow checkpoint is not an online-policy checkpoint")
    if int(checkpoint.get("outer_round", -1)) != args.expected_outer_round:
        raise RuntimeError("Flow checkpoint outer-round mismatch")
    conditions = checkpoint["normalizer"].get("current_l1_a", {})
    if f"{NFP}:{N_BASE_COILS}" not in conditions:
        raise RuntimeError("Flow checkpoint normalizer lacks nfp=8,nc=3")
    return state, {
        "format": checkpoint["format"],
        "stage": checkpoint["stage"],
        "outer_round": int(checkpoint["outer_round"]),
        "step": int(checkpoint["step"]),
        "model_config": checkpoint["model_config"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen and optimize frozen online-RL Flow latents on one GPU."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha", required=True)
    parser.add_argument("--expected-outer-round", type=int, required=True)
    parser.add_argument("--score-lib", type=Path, required=True)
    parser.add_argument("--expected-score-lib-sha", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, default=2)
    parser.add_argument("--start-count", type=int, default=4)
    parser.add_argument("--seed-base", type=int, default=202609031200)
    parser.add_argument("--per-case-max-wall-s", type=float, default=7200.0)
    args = parser.parse_args()
    if not 0 <= args.worker_index < args.worker_count:
        raise ValueError("worker-index must be in [0, worker-count)")
    if args.start_count < 1 or args.per_case_max_wall_s <= 0.0:
        raise ValueError("start-count and per-case-max-wall-s must be positive")

    state, checkpoint_metadata = validate_inputs(args)
    worker_dir = args.run_root / f"worker_{args.worker_index:02d}"
    if worker_dir.exists():
        raise FileExistsError(f"refusing to overwrite {worker_dir}")
    worker_dir.mkdir(parents=True)
    started = time.time()
    indices = case_indices(args.worker_index, args.worker_count, args.start_count)
    manifest = {
        "format": "axisflip_rl_frozen_latent_adam200_worker_v1",
        "protocol": {
            "id": PROTOCOL_ID,
            "status": "registered-experimental",
            "default_impact": "none",
            "specification": str(SPEC_PATH.relative_to(REPO_ROOT)),
            "specification_sha256": file_sha256(SPEC_PATH),
        },
        "repository": state,
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
        "condition": {"nfp": NFP, "n_base_coils": N_BASE_COILS},
        "frozen_flow": {
            "source": str(args.checkpoint.resolve()),
            "sha256": args.expected_checkpoint_sha,
            **checkpoint_metadata,
            "weights": "ema",
            "decode": {"method": "rk4", "steps": FLOW_STEPS, "dtype": "fp32"},
        },
        "evaluator": {
            "library": str(args.score_lib.resolve()),
            "sha256": args.expected_score_lib_sha,
            "abi": 11,
            "target_helicity": [1, NFP],
        },
        "screening": {"candidate_count": SCREEN_COUNT, "parameter_space": "latent"},
        "optimization": {
            "parameter_space": "latent",
            "optimizer": "adam",
            "iterations": ITERATIONS,
            "directions": DIRECTIONS,
            "direction_policy": "fresh random orthogonal centered directions per update",
            "perturbation": PERTURBATION,
            "learning_rate": LEARNING_RATE,
            "beta": [BETA1, BETA2],
            "flow_pipeline": True,
            "recorded_initial_score_tolerance": 0.1,
        },
        "parallelism": {
            "worker_index": args.worker_index,
            "worker_count": args.worker_count,
            "case_indices": indices,
            "serial_reason_within_worker": (
                "Each worker owns one GPU; independent workers run concurrently, "
                "while trajectories sharing that GPU run sequentially."
            ),
        },
        "started_unix_s": started,
    }
    atomic_write_json(worker_dir / "manifest.json", manifest)
    rows: list[dict[str, Any]] = []

    for case_index in indices:
        case_dir = worker_dir / "cases" / f"case_{case_index:03d}"
        screening_dir = case_dir / "screening"
        optimization_dir = case_dir / "optimization"
        log_dir = case_dir / "logs"
        screen_seed = args.seed_base + 2 * case_index
        optimizer_seed = screen_seed + 1
        row: dict[str, Any] = {
            "case_index": case_index,
            "screen_seed": screen_seed,
            "optimizer_seed": optimizer_seed,
            "status": "screening",
        }
        rows.append(row)
        atomic_write_json(
            worker_dir / "progress.json",
            {"status": "running", "cases": rows, "updated_unix_s": time.time()},
        )

        screen = screen_command(
            checkpoint=args.checkpoint,
            score_lib=args.score_lib,
            out_dir=screening_dir,
            seed=screen_seed,
        )
        row["screen_command"] = screen
        return_code = run_logged(
            screen,
            stdout_path=log_dir / "screening.out",
            stderr_path=log_dir / "screening.err",
        )
        row["screen_return_code"] = return_code
        if return_code != 0:
            row["status"] = "screening_failed"
            atomic_write_json(
                worker_dir / "progress.json",
                {"status": "failed", "cases": rows, "updated_unix_s": time.time()},
            )
            raise RuntimeError(f"screening failed for case {case_index}")
        screening = load_json(screening_dir / "summary.json")
        row["screening"] = {
            "status": screening["status"],
            "candidate_count": screening["candidate_count"],
            "valid_candidate_count": screening["valid_candidate_count"],
            "selected_index": screening["selected_index"],
            "selected_score": screening["selected_score"],
            "timing": screening["timing"],
        }
        if screening["status"] != "ok":
            row["status"] = "no_valid_screened_start"
            continue

        row["status"] = "optimizing"
        atomic_write_json(
            worker_dir / "progress.json",
            {"status": "running", "cases": rows, "updated_unix_s": time.time()},
        )
        optimize = optimization_command(
            checkpoint=args.checkpoint,
            score_lib=args.score_lib,
            initial_case=screening_dir / "selected_start.json",
            out_dir=optimization_dir,
            seed=optimizer_seed,
            max_wall_s=args.per_case_max_wall_s,
        )
        row["optimization_command"] = optimize
        return_code = run_logged(
            optimize,
            stdout_path=log_dir / "optimization.out",
            stderr_path=log_dir / "optimization.err",
        )
        row["optimization_return_code"] = return_code
        if return_code != 0:
            row["status"] = "optimization_failed"
            atomic_write_json(
                worker_dir / "progress.json",
                {"status": "failed", "cases": rows, "updated_unix_s": time.time()},
            )
            raise RuntimeError(f"optimization failed for case {case_index}")
        summary = load_json(optimization_dir / "summary.json")
        row["status"] = "complete"
        row["optimization"] = summary
        atomic_write_json(
            worker_dir / "progress.json",
            {"status": "running", "cases": rows, "updated_unix_s": time.time()},
        )

    final_status = "complete"
    summary = {
        "format": "axisflip_rl_frozen_latent_adam200_worker_v1",
        "protocol_id": PROTOCOL_ID,
        "status": final_status,
        "worker_index": args.worker_index,
        "case_count": len(rows),
        "complete_count": sum(row["status"] == "complete" for row in rows),
        "no_valid_screened_start_count": sum(
            row["status"] == "no_valid_screened_start" for row in rows
        ),
        "cases": rows,
        "wall_s": time.time() - started,
        "finished_unix_s": time.time(),
    }
    atomic_write_json(worker_dir / "summary.json", summary)
    atomic_write_json(
        worker_dir / "progress.json",
        {"status": final_status, "cases": rows, "updated_unix_s": time.time()},
    )
    print(json.dumps(summary, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
