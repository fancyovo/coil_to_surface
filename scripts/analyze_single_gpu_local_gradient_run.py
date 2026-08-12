from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def distribution(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "p50": float(np.quantile(data, 0.50)),
        "p95": float(np.quantile(data, 0.95)),
        "p99": float(np.quantile(data, 0.99)),
        "max": float(np.max(data)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.run_dir / "history.jsonl")
    if not rows:
        raise ValueError("run contains no completed iterations")
    steady = rows[1:-1] if len(rows) > 2 else rows
    stage_keys = {
        "iteration": lambda row: row["iteration_wall_s"],
        "flow_pipeline": lambda row: row.get(
            "flow_pipeline_decode_wall_s", row["proposal_decode_wall_s"]
        ),
        "formal_score": lambda row: row["proposal_score_wall_s"],
        "gradient_native": lambda row: row["gradient_pipeline"]["timing_s"]["total"],
        "axis_refine": lambda row: row["gradient_pipeline"]["timing_s"]["axis_refine"],
        "psi": lambda row: row["gradient_pipeline"]["timing_s"]["psi"],
        "local_score": lambda row: row["gradient_pipeline"]["timing_s"]["local_score"],
    }
    local_keys = ("surface_s", "flux_s", "volume_s", "alpha_s", "qs_s")
    timing = {
        key: distribution([float(getter(row)) for row in steady])
        for key, getter in stage_keys.items()
    }
    timing["local_breakdown"] = {
        key: distribution(
            [float(row["gradient_pipeline"]["local_stats"][key]) for row in steady]
        )
        for key in local_keys
    }
    summary = {
        "format": "single_gpu_local_gradient_timing_v1",
        "run_dir": str(args.run_dir),
        "iteration_count": len(rows),
        "steady_iteration_count": len(steady),
        "initial_score": float(
            json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))[
                "initial_score"
            ]
        ),
        "final_score": float(rows[-1]["current_score"]),
        "best_score": float(max(row["best_score"] for row in rows)),
        "best_iteration": int(max(rows, key=lambda row: row["best_score"])["best_iteration"]),
        "cache_hit_fraction": float(
            np.mean([bool(row.get("endpoint_decode_cache_hit", False)) for row in steady])
        ),
        "accepted_fraction": float(
            np.mean([bool(row["center_update_accepted"]) for row in rows])
        ),
        "wasted_endpoint_count": int(
            sum(int(row.get("flow_pipeline_wasted_endpoint_count", 0)) for row in rows)
        ),
        "timing_s": timing,
    }
    (args.output_dir / "timing_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    iteration = np.asarray([row["iteration"] for row in rows])
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].plot(iteration, [row["current_score"] for row in rows], label="current")
    axes[0, 0].plot(iteration, [row["best_score"] for row in rows], label="best")
    axes[0, 0].set(ylabel="formal score", title="Single-GPU optimization")
    axes[0, 0].legend()
    for key, label in (
        ("current_qh_error", "QH"),
        ("current_qa_error", "QA"),
        ("current_qp_error", "QP"),
    ):
        axes[0, 1].plot(iteration, [row[key] for row in rows], label=label)
    axes[0, 1].set(yscale="log", ylabel="volume QS residual", title="QS components")
    axes[0, 1].legend()
    axes[1, 0].plot(iteration, [row["current_iota"] for row in rows])
    axes[1, 0].set(xlabel="iteration", ylabel="minimum iota", title="Iota")
    axes[1, 1].plot(iteration, [row["iteration_wall_s"] for row in rows], label="total")
    axes[1, 1].plot(
        iteration,
        [row["gradient_pipeline"]["timing_s"]["total"] for row in rows],
        label="gradient native",
    )
    axes[1, 1].plot(
        iteration,
        [row.get("flow_pipeline_decode_wall_s", row["proposal_decode_wall_s"]) for row in rows],
        label="flow batch",
    )
    axes[1, 1].plot(
        iteration, [row["proposal_score_wall_s"] for row in rows], label="formal score"
    )
    axes[1, 1].set(xlabel="iteration", ylabel="seconds", title="Per-step timing")
    axes[1, 1].legend()
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
    figure.savefig(args.output_dir / "optimization_and_timing.png", dpi=190)
    plt.close(figure)


if __name__ == "__main__":
    main()
