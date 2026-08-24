from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flow_matching.data import file_sha256
from flow_matching.trajectory_dataset import atomic_write_json
from scripts.prepare_qh_data_prior_control import assign_workers, load_reference_cases


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a six-worker direct-data rerun from every saved Flow-screening winner."
        )
    )
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--worker-count", type=int, default=6)
    args = parser.parse_args()

    if args.run_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.run_root}")
    args.run_root.mkdir(parents=True)
    for name in ("trajectories", "incomplete", "failures", "workers", "logs"):
        (args.run_root / name).mkdir()

    reference_cases = load_reference_cases(args.reference_root)
    cases, loads = assign_workers(reference_cases, args.worker_count)
    for case in cases:
        source = (
            args.reference_root
            / "trajectories"
            / case["trajectory_id"]
            / "screening"
            / "selected_start.json"
        )
        if not source.is_file():
            raise FileNotFoundError(source)
        case["reference_selected_start"] = str(source.resolve())

    manifest = {
        "format": "qh_data_space_same_start_trajectory_rerun_v1",
        "question": (
            "How does direct standardized-coil-parameter Adam behave on all 309 "
            "saved best-of-32 Flow starts?"
        ),
        "reference_root": str(args.reference_root.resolve()),
        "run_root": str(args.run_root.resolve()),
        "case_count": len(cases),
        "worker_count": args.worker_count,
        "worker_reference_load_s": loads,
        "start_policy": (
            "reuse each reference trajectory's saved screening winner; no candidate "
            "screening is rerun"
        ),
        "optimizer": {
            "parameter_space": "per-coordinate standardized coil data",
            "iterations": 200,
            "random_orthogonal_directions": 64,
            "difference": "centered",
            "perturbation": 0.0025,
            "learning_rate": 0.01,
            "beta1": 0.7,
            "beta2": 0.999,
            "optimizer_seed": "exactly reused from each reference trajectory",
        },
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "score_library": str(args.lib.resolve()),
        "score_library_sha256": file_sha256(args.lib),
        "code_commit": git_commit(),
        "created_unix_s": time.time(),
        "cases": cases,
    }
    atomic_write_json(args.run_root / "run_manifest.json", manifest)
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
