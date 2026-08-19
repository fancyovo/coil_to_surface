from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from scripts.evaluate_qh_equal_s_surface_qs_gpu import evaluate_case, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one shard of saved trajectory cases on strict equal-s surfaces.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--gpu-lib", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--n-phi", type=int, default=96)
    parser.add_argument("--n-theta", type=int, default=96)
    parser.add_argument("--output-name", default="equal_s_qs_summary.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    case_dirs = sorted(path for path in (args.experiment_root / "cases").iterdir() if path.is_dir())
    selected = [path for index, path in enumerate(case_dirs) if index % args.shard_count == args.shard_index]
    started = time.perf_counter()
    rows = []
    for index, case_dir in enumerate(selected, 1):
        result = evaluate_case(
            case_dir,
            gpu_lib=args.gpu_lib,
            device=args.device,
            n_phi=args.n_phi,
            n_theta=args.n_theta,
            output_name=args.output_name,
            overwrite=args.overwrite,
        )
        rows.append({"case_id": result["case_id"], "status": result["status"], "wall_s": result["total_wall_s"]})
        if index % 10 == 0 or index == len(selected):
            print(json.dumps({"shard": args.shard_index, "completed": index, "total": len(selected), "last": rows[-1]}), flush=True)
    summary = {
        "format": "qh_equal_s_differential_qs_shard_v1",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "case_count": len(selected),
        "status_counts": {status: sum(row["status"] == status for row in rows) for status in sorted({row["status"] for row in rows})},
        "total_wall_s": time.perf_counter() - started,
        "rows": rows,
    }
    write_json(args.experiment_root / f"equal_s_qs_shard_{args.shard_index:02d}.json", summary)


if __name__ == "__main__":
    main()
