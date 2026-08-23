from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark_summary1_evaluator_modes import discover_cases
from scripts.optimize_native_score_cem import token_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-count", type=int, default=8)
    parser.add_argument("--probe-iterations", default="1,50,150,200")
    parser.add_argument("--selection-seed", type=int, default=2026082402)
    args = parser.parse_args()
    requested = tuple(int(value) for value in args.probe_iterations.split(","))
    selected = discover_cases(
        args.trajectory_root,
        case_count=args.case_count,
        probe_iterations=requested,
        seed=args.selection_seed,
    )
    case_dir = args.output_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, case in enumerate(selected):
        manifest_path = Path(case["source_manifest"])
        trace_path = manifest_path.parent / "optimization" / "training_trace.npz"
        with np.load(trace_path, allow_pickle=False) as trace:
            iterations = np.asarray(trace["iteration"], dtype=int)
            trace_index = int(np.argmin(np.abs(iterations - case["probe_iteration"])))
            noise = np.asarray(trace["probe_noise"][trace_index], dtype=np.float32)
            tokens = np.asarray(trace["probe_tokens"][trace_index], dtype=np.float64)
        payload = token_case(
            tokens,
            nfp=case["nfp"],
            target="QH",
            metadata={
                "summary1_pair_case_id": case["case_id"],
                "source_trajectory": case["trajectory_id"],
                "probe_iteration": case["probe_iteration"],
            },
        )
        payload["noise"] = noise.tolist()
        path = case_dir / f"case_{index:02d}.json"
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        rows.append(
            {
                "index": index,
                "case_id": case["case_id"],
                "trajectory_id": case["trajectory_id"],
                "probe_iteration": case["probe_iteration"],
                "nfp": case["nfp"],
                "n_base_coils": case["n_base_coils"],
                "case_path": str(path.resolve()),
                "source_manifest": case["source_manifest"],
            }
        )
    manifest = {
        "format": "summary1_flow_parameterization_pairs_v1",
        "trajectory_root": str(args.trajectory_root.resolve()),
        "selection_seed": args.selection_seed,
        "probe_iterations": requested,
        "cases": rows,
    }
    (args.output_dir / "pair_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"case_count": len(rows), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
