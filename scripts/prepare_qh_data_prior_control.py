from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flow_matching.data import file_sha256
from flow_matching.trajectory_dataset import atomic_write_json


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_reference_cases(root: Path) -> list[dict[str, Any]]:
    cases = []
    for path in sorted((root / "trajectories").glob("*/trajectory_manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        condition = payload["condition"]
        seeds = payload["seeds"]
        cases.append(
            {
                "trajectory_id": str(payload["trajectory_id"]),
                "nfp": int(condition["nfp"]),
                "n_base_coils": int(condition["n_base_coils"]),
                "condition_group": str(condition["group"]),
                "screening_seed": int(seeds["screening"]),
                "optimizer_seed": int(seeds["optimizer"]),
                "reference_manifest": str(path.resolve()),
                "reference_wall_s": float(payload["timing"]["trajectory_wall_s"]),
                "reference_start_score": float(payload["optimization"]["initial_score"]),
                "reference_best_score": float(payload["optimization"]["best_score"]),
            }
        )
    if not cases:
        raise FileNotFoundError(f"no reference trajectories found below {root}")
    return cases


def assign_workers(
    cases: list[dict[str, Any]], worker_count: int
) -> tuple[list[dict[str, Any]], list[float]]:
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    loads = [0.0] * worker_count
    assigned: list[dict[str, Any]] = []
    for case in sorted(
        cases,
        key=lambda item: (-item["reference_wall_s"], item["trajectory_id"]),
    ):
        worker = min(range(worker_count), key=lambda index: (loads[index], index))
        row = dict(case)
        row["worker_index"] = worker
        row["case_index"] = len(assigned)
        assigned.append(row)
        loads[worker] += float(case["reference_wall_s"])
    return sorted(assigned, key=lambda item: item["trajectory_id"]), loads


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a matched standardized-data-prior control for QH trajectories."
    )
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--worker-count", type=int, default=4)
    args = parser.parse_args()

    if args.run_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.run_root}")
    args.run_root.mkdir(parents=True)
    for name in ("cases", "incomplete", "failures", "workers", "logs"):
        (args.run_root / name).mkdir()

    cases, loads = assign_workers(
        load_reference_cases(args.reference_root), args.worker_count
    )
    manifest = {
        "format": "qh_data_prior_end_to_end_control_v1",
        "question": (
            "Does Flow help mainly by supplying the best-of-32 initial coil, rather "
            "than by improving local Adam updates?"
        ),
        "reference_root": str(args.reference_root.resolve()),
        "run_root": str(args.run_root.resolve()),
        "case_count": len(cases),
        "worker_count": args.worker_count,
        "worker_reference_load_s": loads,
        "screening": {
            "candidate_count": 32,
            "prior": "independent N(0,1) in per-coordinate standardized coil space",
            "condition_and_seed_pairing": "exactly matched to each reference Flow trajectory",
        },
        "optimizer": {
            "parameter_space": "standardized coil data",
            "iterations": 200,
            "directions": 64,
            "difference": "centered",
            "perturbation": 0.0025,
            "learning_rate": 0.01,
            "beta1": 0.7,
            "beta2": 0.999,
        },
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "score_library": str(args.lib.resolve()),
        "score_library_sha256": file_sha256(args.lib),
        "code_commit": git_commit(),
        "created_unix_s": time.time(),
        "cases": cases,
    }
    atomic_write_json(args.run_root / "control_manifest.json", manifest)
    print(
        json.dumps(
            {
                "case_count": len(cases),
                "worker_count": args.worker_count,
                "worker_reference_load_s": loads,
            }
        )
    )


if __name__ == "__main__":
    main()
