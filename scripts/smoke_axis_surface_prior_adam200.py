from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]

from flow_matching.data import file_sha256
from scripts.native_score_runtime import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-step exact-start smoke test for analytic-prior Adam200.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads((args.run_root / "selection_manifest.json").read_text(encoding="utf-8"))
    case = min(manifest["cases"], key=lambda value: int(value["selection_rank"]))
    smoke_dir = args.run_root / "smoke"
    smoke_dir.mkdir(exist_ok=False)
    start_path = Path(case["start"])
    if file_sha256(start_path) != case["start_sha256"]:
        raise RuntimeError("smoke start hash mismatch")
    shutil.copy2(start_path, smoke_dir / "start.json")
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "optimize_flow_latent.py"),
        "--checkpoint",
        manifest["artifacts"]["checkpoint"],
        "--initial-case",
        str(smoke_dir / "start.json"),
        "--lib",
        manifest["artifacts"]["score_library"],
        "--out-dir",
        str(smoke_dir / "optimization"),
        "--nfp",
        str(case["nfp"]),
        "--n-base-coils",
        str(case["n_base_coils"]),
        "--iterations",
        "3",
        "--max-wall-s",
        "600",
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
        "1",
        "--trajectory-every",
        "0",
        "--state-every",
        "3",
    ]
    with (smoke_dir / "optimization.log").open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError("smoke optimizer failed")
    summary = json.loads(
        (smoke_dir / "optimization" / "summary.json").read_text(encoding="utf-8")
    )
    optimizer_manifest = json.loads(
        (smoke_dir / "optimization" / "manifest.json").read_text(encoding="utf-8")
    )
    score_difference = float(summary["initial_score"]) - float(case["initial_score"])
    roundtrip = float(
        optimizer_manifest["data_parameterization"]["initial_roundtrip_relative_rms"]
    )
    if summary.get("stop_reason") != "completed_iterations":
        raise RuntimeError("smoke did not complete three iterations")
    if abs(score_difference) > 0.1:
        raise RuntimeError(f"smoke step-0 score difference {score_difference} exceeds 0.1")
    if roundtrip > 2.0e-6:
        raise RuntimeError("smoke exact-start roundtrip exceeds tolerance")
    result = {
        "status": "passed",
        "case_id": case["case_id"],
        "nfp": case["nfp"],
        "n_base_coils": case["n_base_coils"],
        "source_initial_score": case["initial_score"],
        "optimizer_initial_score": summary["initial_score"],
        "initial_score_difference": score_difference,
        "initial_roundtrip_relative_rms": roundtrip,
        "completed_iterations": summary["completed_iterations"],
        "best_score": summary["best_score"],
    }
    write_json(smoke_dir / "smoke_result.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
