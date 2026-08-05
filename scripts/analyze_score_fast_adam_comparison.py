from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def trajectory(directory: Path) -> tuple[dict, list[dict]]:
    summary = read_json(directory / "summary.json")
    history = read_jsonl(directory / "history.jsonl")
    initial = {
        "iteration": 0,
        "current_score": float(summary["initial_score"]),
        "total_wall_s": 0.0,
        "iteration_wall_s": float("nan"),
        "current_qh_error": float("nan"),
        "current_qa_error": float("nan"),
        "current_qp_error": float("nan"),
        "current_iota": float("nan"),
    }
    return summary, [initial, *history]


def values(rows: list[dict], name: str) -> np.ndarray:
    return np.asarray([float(row.get(name, float("nan"))) for row in rows])


def normalized_progress(rows: list[dict]) -> np.ndarray:
    score = values(rows, "current_score")
    denominator = max(float(np.max(score) - score[0]), 1.0e-12)
    return (score - score[0]) / denominator


def run_summary(summary: dict, rows: list[dict]) -> dict:
    score = values(rows, "current_score")
    deltas = np.diff(score)
    iteration_time = values(rows[1:], "iteration_wall_s")
    return {
        "initial_score": float(summary["initial_score"]),
        "final_score": float(summary["final_score"]),
        "best_score": float(summary["best_score"]),
        "best_iteration": int(summary["best_iteration"]),
        "gain_at_best": float(summary["best_score"] - summary["initial_score"]),
        "total_wall_s": float(summary["total_wall_s"]),
        "mean_iteration_wall_s": float(np.mean(iteration_time)),
        "median_iteration_wall_s": float(np.median(iteration_time)),
        "p95_iteration_wall_s": float(np.quantile(iteration_time, 0.95)),
        "negative_step_count": int(np.count_nonzero(deltas < 0.0)),
        "largest_step_drop": float(np.min(deltas)),
        "final_qh_error": float(rows[-1]["current_qh_error"]),
        "final_qa_error": float(rows[-1]["current_qa_error"]),
        "final_qp_error": float(rows[-1]["current_qp_error"]),
        "final_iota": float(rows[-1]["current_iota"]),
    }


def plot_score(
    old_rows: list[dict], new_rows: list[dict], output: Path
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
    styles = (("legacy", old_rows, "#5b5f97"), ("continuous", new_rows, "#d1495b"))
    for label, rows, color in styles:
        iteration = values(rows, "iteration")
        score = values(rows, "current_score")
        wall_min = values(rows, "total_wall_s") / 60.0
        axes[0, 0].plot(iteration, score, label=label, color=color, lw=1.7)
        axes[0, 1].plot(iteration, normalized_progress(rows), label=label, color=color, lw=1.7)
        axes[1, 0].plot(wall_min, normalized_progress(rows), label=label, color=color, lw=1.7)
        axes[1, 1].plot(
            iteration[1:], values(rows[1:], "iteration_wall_s"),
            label=label, color=color, lw=1.0, alpha=0.75,
        )
    axes[0, 0].set(title="Raw score (definitions differ)", xlabel="Adam step", ylabel="score")
    axes[0, 1].set(title="Progress within each score definition", xlabel="Adam step", ylabel="fraction of observed gain")
    axes[1, 0].set(title="Progress versus wall time", xlabel="optimizer wall time (min)", ylabel="fraction of observed gain")
    axes[1, 1].set(title="Per-step wall time", xlabel="Adam step", ylabel="seconds")
    for axis in axes.flat:
        axis.grid(alpha=0.22)
        axis.legend()
    figure.savefig(output, dpi=190)
    plt.close(figure)


def plot_physics(old_rows: list[dict], new_rows: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
    styles = (("legacy", old_rows, "#5b5f97"), ("continuous", new_rows, "#d1495b"))
    metrics = (
        ("current_qh_error", "QH error", True),
        ("current_qa_error", "QA error", True),
        ("current_qp_error", "QP error", True),
        ("current_iota", "minimum |iota|", False),
    )
    for axis, (field, title, logarithmic) in zip(axes.flat, metrics):
        for label, rows, color in styles:
            axis.plot(values(rows, "iteration"), values(rows, field), label=label, color=color, lw=1.5)
        if logarithmic:
            axis.set_yscale("log")
        axis.set(title=title, xlabel="Adam step")
        axis.grid(alpha=0.22)
        axis.legend()
    figure.savefig(output, dpi=190)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-dir", type=Path, required=True)
    parser.add_argument("--continuous-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cross-score", type=Path)
    args = parser.parse_args()
    old_summary, old_rows = trajectory(args.legacy_dir)
    new_summary, new_rows = trajectory(args.continuous_dir)
    result = {
        "legacy": run_summary(old_summary, old_rows),
        "continuous": run_summary(new_summary, new_rows),
    }
    result["speedup_mean_iteration"] = (
        result["legacy"]["mean_iteration_wall_s"] /
        result["continuous"]["mean_iteration_wall_s"]
    )
    result["speedup_total_wall"] = (
        result["legacy"]["total_wall_s"] / result["continuous"]["total_wall_s"]
    )
    if args.cross_score:
        cross_rows = read_json(args.cross_score)
        result["cross_score"] = {}
        for label, row in zip(("legacy_best", "continuous_best"), cross_rows):
            result["cross_score"][label] = {
                mode: {
                    "status": row[mode]["status"],
                    "score": float(row[mode]["score"]),
                    "surface_level": float(row[mode]["diagnostics"]["surface_level"]),
                    "inverse_aspect_ratio": float(
                        row[mode]["diagnostics"]["surface_inverse_aspect_ratio"]
                    ),
                    "qh_error_per_helicity": float(
                        row[mode]["diagnostics"]["qs_target_global_error_per_helicity"]
                    ),
                    "iota_min": float(row[mode]["diagnostics"]["iota_min"]),
                }
                for mode in ("legacy", "continuous")
            }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    plot_score(old_rows, new_rows, args.output_dir / "score_and_timing.png")
    plot_physics(old_rows, new_rows, args.output_dir / "physics_trajectory.png")
    print(json.dumps(result, allow_nan=True))


if __name__ == "__main__":
    main()
