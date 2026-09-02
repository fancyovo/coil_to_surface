from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.native_score_runtime import write_json
from scripts.prepare_axis_surface_prior_continuation import continuation_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare one frozen axis-flip v4 best case for long Adam continuation.")
    parser.add_argument("--source-best", type=Path, required=True)
    parser.add_argument("--source-start", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--nfp", type=int, required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    if commit != args.expected_commit:
        raise RuntimeError(f"repository commit {commit} != expected {args.expected_commit}")
    dirty = subprocess.check_output(
        ["git", "status", "--short", "--untracked-files=no"], cwd=REPO_ROOT, text=True
    )
    if dirty.strip():
        raise RuntimeError("tracked experiment worktree is dirty")
    if args.output.exists():
        raise FileExistsError(args.output)
    best = json.loads(args.source_best.read_text(encoding="utf-8"))
    source_start = json.loads(args.source_start.read_text(encoding="utf-8"))
    if int(best["nfp"]) != args.nfp:
        raise ValueError("source best nfp differs from requested continuation")
    payload = continuation_payload(
        best,
        source_start,
        protocol_id=args.protocol_id,
        target_helicity=(1, args.nfp),
    )
    payload["long_continuation"] = {
        "protocol_id": args.protocol_id,
        "source_best": str(args.source_best.resolve()),
        "source_start": str(args.source_start.resolve()),
        "source_best_iteration": int(
            best["original_space_local_gradient_adam"]["best_iteration"]
        ),
        "requested_iterations": 2000,
        "target_helicity": [1, args.nfp],
        "code_commit": commit,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_json(args.output, payload)
    print(
        json.dumps(
            {
                "event": "long_start_prepared",
                "output": str(args.output),
                "source_score": payload["data_prior_screening"]["native_score"]["score"],
            }
        )
    )


if __name__ == "__main__":
    main()
