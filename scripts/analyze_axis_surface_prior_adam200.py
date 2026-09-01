from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p10": None, "median": None, "p90": None, "max": None}
    data = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "p10": float(np.quantile(data, 0.1)),
        "median": float(np.median(data)),
        "p90": float(np.quantile(data, 0.9)),
        "max": float(np.max(data)),
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    initial = [float(row["initial_score"]) for row in rows]
    best = [float(row["best_score"]) for row in rows]
    gains = [right - left for left, right in zip(initial, best, strict=True)]
    return {
        "count": len(rows),
        "initial_score": distribution(initial),
        "best_score": distribution(best),
        "best_gain": distribution(gains),
        "best_ge_20": sum(value >= 20.0 for value in best),
        "best_ge_50": sum(value >= 50.0 for value in best),
        "best_ge_70": sum(value >= 70.0 for value in best),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize analytic-prior random valid Adam200 trajectories.")
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.run_root / "analysis"
    output_dir.mkdir(exist_ok=True)
    selection = json.loads((args.run_root / "selection_manifest.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    curves: list[tuple[int, np.ndarray, np.ndarray]] = []
    for path in sorted((args.run_root / "trajectories").glob("*/trajectory_manifest.json")):
        trajectory = json.loads(path.read_text(encoding="utf-8"))
        case = trajectory["case"]
        optimization = trajectory["optimization"]
        row = {
            "trajectory_id": trajectory["trajectory_id"],
            "case_id": case["case_id"],
            "worker_index": case["worker_index"],
            "nfp": case["nfp"],
            "n_base_coils": case["n_base_coils"],
            "initial_score": optimization["initial_score"],
            "final_score": optimization["final_score"],
            "best_score": optimization["best_score"],
            "best_iteration": optimization["best_iteration"],
            "best_gain": float(optimization["best_score"]) - float(optimization["initial_score"]),
            "trajectory_wall_s": trajectory["timing"]["trajectory_wall_s"],
        }
        rows.append(row)
        progress_path = path.parent / "optimization" / "history.jsonl"
        if progress_path.exists():
            progress = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines() if line]
            if progress:
                curves.append(
                    (
                        int(case["n_base_coils"]),
                        np.asarray([0] + [int(item["iteration"]) for item in progress]),
                        np.asarray([float(optimization["initial_score"])] + [float(item["current_score"]) for item in progress]),
                    )
                )

    rows.sort(key=lambda row: int(row["case_id"]))
    failures = list((args.run_root / "failures").glob("*/failure.json"))
    workers = []
    for path in sorted((args.run_root / "workers").glob("worker_*/progress.json")):
        workers.append(json.loads(path.read_text(encoding="utf-8")))
    by_nc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_nfp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_nc[str(row["n_base_coils"])].append(row)
        by_nfp[str(row["nfp"])].append(row)
    overall = summarize_group(rows)
    summary = {
        "format": selection.get("artifact_formats", {}).get(
            "summary", "axis_surface_prior_balanced_v2_adam200_summary_v1"
        ),
        "protocol_id": selection["protocol_id"],
        "expected_count": len(selection["cases"]),
        "completed_count": len(rows),
        "failure_count": len(failures),
        "worker_states": Counter(worker.get("stage", "missing") for worker in workers),
        "overall": overall,
        "conditional_success_rate_best_ge_50": (
            overall["best_ge_50"] / len(rows) if rows else None
        ),
        "estimated_unconditional_rate_best_ge_50": (
            selection["input"]["valid_rate"] * overall["best_ge_50"] / len(rows)
            if rows
            else None
        ),
        "by_n_base_coils": {key: summarize_group(value) for key, value in sorted(by_nc.items())},
        "by_nfp": {key: summarize_group(value) for key, value in sorted(by_nfp.items())},
        "trajectory_wall_s": distribution([float(row["trajectory_wall_s"]) for row in rows]),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if rows:
        with (output_dir / "trajectories.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), constrained_layout=True)
        for nc, steps, scores in curves:
            axes[0].plot(steps, scores, alpha=0.22, linewidth=0.8, label=f"nc={nc}")
        axes[0].axhline(50.0, color="#b04a3a", linestyle="--", linewidth=1.0)
        axes[0].set(xlabel="Adam update", ylabel="ABI-11 score", title="Random valid starts: Adam200 trajectories")
        axes[0].grid(alpha=0.2)
        initial = np.asarray([row["initial_score"] for row in rows])
        best = np.asarray([row["best_score"] for row in rows])
        axes[1].scatter(initial, best, c=[row["n_base_coils"] for row in rows], cmap="viridis", s=28, alpha=0.8)
        bounds = [min(initial.min(), best.min()), max(initial.max(), best.max())]
        axes[1].plot(bounds, bounds, color="#777777", linewidth=1.0)
        axes[1].axhline(50.0, color="#b04a3a", linestyle="--", linewidth=1.0)
        axes[1].set(xlabel="Initial score", ylabel="Best score", title="Initial versus Adam200 best")
        axes[1].grid(alpha=0.2)
        figure.savefig(output_dir / "adam200_trajectories.png", dpi=190)
        plt.close(figure)
    print(json.dumps(summary, indent=2, default=dict), flush=True)


if __name__ == "__main__":
    main()
