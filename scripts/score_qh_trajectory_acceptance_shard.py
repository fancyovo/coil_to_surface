from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "gpu_backend" / "python"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from flow_matching.data import load_raw_groups  # noqa: E402
from scripts.evaluate_corrected_score_calibration import select_quasr_cases  # noqa: E402
from scripts.optimize_flow_prior_local_full_gradient_adam import score_center  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def case_tokens(payload: dict[str, Any]) -> np.ndarray:
    raw = payload["raw"]
    return np.concatenate(
        [
            np.asarray(raw["x"], dtype=np.float32),
            np.asarray(raw["y"], dtype=np.float32),
            np.asarray(raw["z"], dtype=np.float32),
            np.asarray(raw["current"], dtype=np.float32)[:, None],
        ],
        axis=1,
    )


def load_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    groups, _ = load_raw_groups(args.data_dir, "test", verify_hashes=False)
    quasr = select_quasr_cases(
        groups, args.quasr_count, np.random.default_rng(args.seed)
    )
    cases = [
        {
            "kind": "quasr",
            "case_id": f"quasr_{row['sample_index']:06d}",
            "sample_index": int(row["sample_index"]),
            "source_id": int(row["source_id"]),
            "nfp": int(row["key"][0]),
            "n_base_coils": int(row["key"][1]),
            "tokens": row["tokens"],
        }
        for row in quasr
    ]
    for manifest_path in sorted(
        (args.dataset_root / "trajectories").glob("*/trajectory_manifest.json")
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        best_path = manifest_path.parent / "optimization" / "best.json"
        best = json.loads(best_path.read_text(encoding="utf-8"))
        cases.append(
            {
                "kind": "adam_best",
                "case_id": manifest["trajectory_id"],
                "sample_index": None,
                "source_id": None,
                "nfp": int(manifest["condition"]["nfp"]),
                "n_base_coils": int(manifest["condition"]["n_base_coils"]),
                "tokens": case_tokens(best),
                "online_best_score": float(manifest["optimization"]["best_score"]),
                "online_best_iteration": int(
                    manifest["optimization"]["best_iteration"]
                ),
            }
        )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independently score one shard of QUASR references and Adam best cases."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quasr-count", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=6)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"shard_{args.shard_index:02d}.jsonl"
    summary_path = args.output_dir / f"shard_{args.shard_index:02d}_summary.json"
    if output_path.exists() or summary_path.exists():
        raise FileExistsError(f"refusing to overwrite shard {args.shard_index}")

    all_cases = load_cases(args)
    cases = all_cases[args.shard_index :: args.shard_count]
    started = time.perf_counter()
    status_counts: dict[str, int] = {}
    with output_path.open("x", encoding="utf-8") as stream:
        for ordinal, case in enumerate(cases):
            row = {key: value for key, value in case.items() if key != "tokens"}
            try:
                result, elapsed_s = score_center(
                    args.lib,
                    case["tokens"],
                    nfp=case["nfp"],
                    score_device=args.device,
                    iota_degree=3,
                    surface_theta_count=128,
                    previous_result=None,
                )
                row.update(
                    {
                        "native_score": result,
                        "score_wall_s": elapsed_s,
                        "error": None,
                    }
                )
                status = str(result.get("status", "missing"))
            except Exception as exc:
                row.update(
                    {
                        "native_score": None,
                        "score_wall_s": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                status = "python_error"
            status_counts[status] = status_counts.get(status, 0) + 1
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=True))
            stream.write("\n")
            stream.flush()
            if (ordinal + 1) % 25 == 0:
                print(
                    json.dumps(
                        {
                            "completed": ordinal + 1,
                            "assigned": len(cases),
                            "elapsed_s": time.perf_counter() - started,
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
    summary = {
        "format": "qh_trajectory_acceptance_rescore_v1",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "case_count": len(cases),
        "total_case_count": len(all_cases),
        "quasr_count": args.quasr_count,
        "adam_best_count": len(all_cases) - args.quasr_count,
        "status_counts": status_counts,
        "wall_s": time.perf_counter() - started,
        "score_library_sha256": file_sha256(args.lib),
        "scoring": {
            "history": "none",
            "axis_hint": "none",
            "iota_degree": 3,
            "surface_theta_count": 128,
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
