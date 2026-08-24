from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def distribution(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if not array.size:
        return {"count": 0, "p10": None, "p50": None, "p90": None, "mean": None}
    return {
        "count": int(array.size),
        "p10": float(np.percentile(array, 10)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "mean": float(np.mean(array)),
    }


def collect(run_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    expected = {row["trajectory_id"]: row for row in run_manifest["cases"]}
    rows = []
    for trajectory_id, reference in sorted(expected.items()):
        path = run_root / "trajectories" / trajectory_id / "trajectory_manifest.json"
        if not path.is_file():
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        optimization = manifest["optimization"]
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "nfp": int(reference["nfp"]),
                "n_base_coils": int(reference["n_base_coils"]),
                "initial_online_score": float(optimization["initial_score"]),
                "final_online_score": float(optimization["final_score"]),
                "best_online_score": float(optimization["best_score"]),
                "best_global_score": float(optimization["best_score"]),
                "best_iteration": int(optimization["best_iteration"]),
                "selected_global_score": float(optimization["initial_score"]),
                "optimization_wall_s": float(optimization["total_wall_s"]),
                "trajectory_wall_s": float(manifest["timing"]["trajectory_wall_s"]),
                "reference_initial_score": float(reference["reference_start_score"]),
                "reference_best_score": float(reference["reference_best_score"]),
                "initial_score_difference": (
                    float(optimization["initial_score"])
                    - float(reference["reference_start_score"])
                ),
                "best_score_difference_data_minus_latent": (
                    float(optimization["best_score"])
                    - float(reference["reference_best_score"])
                ),
            }
        )
    missing = sorted(set(expected) - {row["trajectory_id"] for row in rows})
    summary = {
        "format": "qh_data_space_same_start_trajectory_summary_v1",
        "expected_case_count": len(expected),
        "completed_case_count": len(rows),
        "missing_case_count": len(missing),
        "missing_cases": missing,
        "initial_score": distribution(row["initial_online_score"] for row in rows),
        "final_score": distribution(row["final_online_score"] for row in rows),
        "best_score": distribution(row["best_online_score"] for row in rows),
        "best_gain": distribution(
            row["best_online_score"] - row["initial_online_score"] for row in rows
        ),
        "initial_roundtrip_difference": distribution(
            row["initial_score_difference"] for row in rows
        ),
        "paired_best_difference_data_minus_latent": distribution(
            row["best_score_difference_data_minus_latent"] for row in rows
        ),
        "data_wins": sum(
            row["best_score_difference_data_minus_latent"] > 0.0 for row in rows
        ),
        "latent_wins": sum(
            row["best_score_difference_data_minus_latent"] < 0.0 for row in rows
        ),
        "ties": sum(
            row["best_score_difference_data_minus_latent"] == 0.0 for row in rows
        ),
        "optimization_wall_s": distribution(row["optimization_wall_s"] for row in rows),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the complete same-start data-space trajectory rerun."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, summary = collect(args.run_root)
    if summary["missing_case_count"]:
        raise RuntimeError(
            f"rerun is incomplete: {summary['missing_case_count']} trajectories are missing"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "trajectory_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
