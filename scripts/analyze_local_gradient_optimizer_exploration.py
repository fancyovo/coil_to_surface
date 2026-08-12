from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_run(root: Path, name: str, max_iteration: int | None = None) -> dict[str, Any]:
    directory = root / name
    summary = read_json(directory / "summary.json")
    rows = read_jsonl(directory / "history.jsonl")
    if max_iteration is not None:
        rows = [row for row in rows if int(row["iteration"]) <= max_iteration]
    if not rows:
        raise ValueError(f"no history rows for {directory}")

    endpoint_budget = 0
    running_best = float(summary["initial_score"])
    for row in rows:
        endpoint_budget += int(row["gradient_endpoint_count"])
        running_best = max(running_best, float(row["current_score"]))
        row["analysis_endpoint_budget"] = endpoint_budget
        row["analysis_best_score"] = running_best
    return {
        "name": name,
        "directory": str(directory),
        "summary": summary,
        "rows": rows,
    }


def run_metrics(run: dict[str, Any]) -> dict[str, Any]:
    rows = run["rows"]
    initial = float(run["summary"]["initial_score"])
    accepted = np.asarray(
        [bool(row["center_update_accepted"]) for row in rows], dtype=np.bool_
    )
    current = np.asarray([float(row["current_score"]) for row in rows])
    previous = np.concatenate(([initial], current[:-1]))
    iteration_wall = np.asarray([float(row["iteration_wall_s"]) for row in rows])
    best_index = int(np.argmax(current))
    thresholds: dict[str, dict[str, float | int] | None] = {}
    running_best = np.maximum.accumulate(current)
    for threshold in (92.0, 92.5, 93.0, 93.5):
        indices = np.flatnonzero(running_best >= threshold)
        if not len(indices):
            thresholds[str(threshold)] = None
            continue
        index = int(indices[0])
        thresholds[str(threshold)] = {
            "iteration": int(rows[index]["iteration"]),
            "wall_s": float(rows[index]["total_wall_s"]),
            "endpoint_evaluations": int(rows[index]["analysis_endpoint_budget"]),
        }
    return {
        "initial_score": initial,
        "final_score": float(current[-1]),
        "best_score": float(current[best_index]),
        "best_iteration": int(rows[best_index]["iteration"]),
        "score_gain": float(current[best_index] - initial),
        "iterations": len(rows),
        "wall_s": float(rows[-1]["total_wall_s"]),
        "endpoint_evaluations": int(rows[-1]["analysis_endpoint_budget"]),
        "accepted_updates": int(np.count_nonzero(accepted)),
        "accepted_fraction": float(np.mean(accepted)),
        "negative_step_fraction": float(np.mean(current < previous)),
        "iteration_wall_mean_s": float(np.mean(iteration_wall)),
        "iteration_wall_p95_s": float(np.quantile(iteration_wall, 0.95)),
        "thresholds": thresholds,
    }


def series(run: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in run["rows"]])


def best_at_budget(run: dict[str, Any], key: str, limit: float) -> float:
    best = float(run["summary"]["initial_score"])
    for row in run["rows"]:
        if float(row[key]) > limit:
            break
        best = max(best, float(row["current_score"]))
    return best


def with_initial(run: dict[str, Any], x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate(([0.0], x)),
        np.concatenate(([float(run["summary"]["initial_score"])], y)),
    )


def plot_adam_learning_rates(runs: list[tuple[str, dict[str, Any]]], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for label, run in runs:
        x = series(run, "iteration")
        best = series(run, "analysis_best_score")
        current = series(run, "current_score")
        xb, yb = with_initial(run, x, best)
        xc, yc = with_initial(run, x, current)
        axes[0].plot(xb, yb, linewidth=2, label=label)
        axes[1].plot(xc, yc, linewidth=1.6, label=label)
    axes[0].set(title="Best score: full 300-D gradient", xlabel="optimizer step", ylabel="score")
    axes[1].set(title="Current score and stability", xlabel="optimizer step", ylabel="score")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(output, dpi=190)
    plt.close(figure)


def plot_direction_sweep(runs: list[tuple[str, dict[str, Any]]], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for label, run in runs:
        rows = run["rows"]
        step = series(run, "iteration")
        wall_min = series(run, "total_wall_s") / 60.0
        endpoints = series(run, "analysis_endpoint_budget")
        best = series(run, "analysis_best_score")
        current = series(run, "current_score")
        for axis, x in zip(axes.ravel()[:3], (step, wall_min, endpoints), strict=True):
            xp, yp = with_initial(run, x, best)
            axis.plot(xp, yp, linewidth=1.8, label=label)
        xp, yp = with_initial(run, step, current)
        axes[1, 1].plot(xp, yp, linewidth=1.4, label=label, alpha=0.9)
    axes[0, 0].set(title="Best score by step", xlabel="optimizer step", ylabel="score")
    axes[0, 1].set(title="Best score by wall time", xlabel="wall time (min)", ylabel="score")
    axes[1, 0].set(
        title="Best score by local-score endpoint budget",
        xlabel="cumulative centered-difference endpoints",
        ylabel="score",
    )
    axes[1, 1].set(title="Current-score stability", xlabel="optimizer step", ylabel="score")
    random_wall_limit = max(float(run["rows"][-1]["total_wall_s"]) for _, run in runs[1:]) / 60.0
    random_endpoint_limit = max(
        int(run["rows"][-1]["analysis_endpoint_budget"]) for _, run in runs[1:]
    )
    axes[0, 1].set_xlim(0.0, random_wall_limit)
    axes[1, 0].set_xlim(0.0, random_endpoint_limit)
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(output, dpi=190)
    plt.close(figure)


def adam_continuation(
    baseline: dict[str, Any], anchor_iteration: int, wall_limit_s: float
) -> tuple[np.ndarray, np.ndarray]:
    all_rows = baseline["rows"]
    anchor = next(row for row in all_rows if int(row["iteration"]) == anchor_iteration)
    anchor_wall = float(anchor["total_wall_s"])
    anchor_score = float(anchor["current_score"])
    elapsed = [0.0]
    gain = [0.0]
    running_best = anchor_score
    for row in all_rows:
        row_elapsed = float(row["total_wall_s"]) - anchor_wall
        if row_elapsed <= 0.0:
            continue
        if row_elapsed > wall_limit_s:
            break
        running_best = max(running_best, float(row["current_score"]))
        elapsed.append(row_elapsed / 60.0)
        gain.append(running_best - anchor_score)
    return np.asarray(elapsed), np.asarray(gain)


def adam_continuation_metrics(
    baseline: dict[str, Any], anchor_iteration: int, wall_limit_s: float
) -> dict[str, float | int]:
    elapsed, gains = adam_continuation(baseline, anchor_iteration, wall_limit_s)
    return {
        "anchor_iteration": anchor_iteration,
        "wall_limit_s": wall_limit_s,
        "observed_wall_s": float(elapsed[-1] * 60.0),
        "best_score_gain": float(gains[-1]),
        "steps_observed": int(len(gains) - 1),
    }


def plot_bfgs(
    baseline: dict[str, Any],
    bfgs_runs: list[tuple[str, int, dict[str, Any]]],
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for label, anchor_iteration, run in bfgs_runs:
        rows = run["rows"]
        initial = float(run["summary"]["initial_score"])
        elapsed = series(run, "total_wall_s") / 60.0
        gain = series(run, "analysis_best_score") - initial
        xb, yb = (np.concatenate(([0.0], elapsed)), np.concatenate(([0.0], gain)))
        axes[0].plot(xb, yb, linewidth=2, label=f"BFGS {label}")
        ax, ay = adam_continuation(baseline, anchor_iteration, float(rows[-1]["total_wall_s"]))
        axes[0].plot(ax, ay, linestyle="--", linewidth=1.8, label=f"Adam continuation {label}")

        step = series(run, "iteration")
        trust = series(run, "bfgs_trust_rms")
        accepted = np.asarray([bool(row["center_update_accepted"]) for row in rows])
        axes[1].plot(step, trust, linewidth=1.5, label=label)
        axes[1].scatter(step[accepted], trust[accepted], s=24, marker="o")
    axes[0].set(title="BFGS versus recorded Adam continuation", xlabel="elapsed wall time (min)", ylabel="best score gain")
    axes[1].set(title="BFGS trust radius (dots are accepted steps)", xlabel="BFGS iteration", ylabel="trust RMS", yscale="log")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(output, dpi=190)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_full = load_run(args.baseline_dir.parent, args.baseline_dir.name)
    lr_runs = [
        ("lr=0.01", load_run(args.baseline_dir.parent, args.baseline_dir.name, 100)),
        ("lr=0.02", load_run(args.asset_root, "full_adam_lr020_100")),
        ("lr=0.03", load_run(args.asset_root, "full_adam_lr030_100")),
        ("lr=0.05", load_run(args.asset_root, "full_adam_lr050_100")),
    ]
    direction_specs = [
        ("full, lr=0.01", args.baseline_dir.parent, args.baseline_dir.name),
        ("K=32, lr=0.01", args.asset_root, "random32_start_300"),
        ("K=64, lr=0.01", args.asset_root, "random64_start_300"),
        ("K=32, lr=0.02", args.asset_root, "random32_lr020_300"),
        ("K=64, lr=0.02", args.asset_root, "random64_lr020_300"),
        ("K=64, lr=0.03", args.asset_root, "random64_lr030_300"),
    ]
    direction_runs = [
        (label, load_run(root, name, 300)) for label, root, name in direction_specs
    ]
    bfgs_runs = [
        ("anchor step 100", 100, load_run(args.asset_root, "bfgs_step100_40")),
        ("anchor step 1250", 1250, load_run(args.asset_root, "bfgs_step1250_40")),
    ]

    plot_adam_learning_rates(lr_runs, args.output_dir / "adam_learning_rate_comparison.png")
    plot_direction_sweep(direction_runs, args.output_dir / "random_direction_comparison.png")
    plot_bfgs(baseline_full, bfgs_runs, args.output_dir / "bfgs_comparison.png")

    payload = {
        "format": "local_gradient_optimizer_exploration_v1",
        "adam_learning_rate": {label: run_metrics(run) for label, run in lr_runs},
        "random_direction": {label: run_metrics(run) for label, run in direction_runs},
        "random_direction_common_budgets": {
            "wall_s": 1500.0,
            "endpoint_evaluations": 19200,
            "score_at_wall_s": {
                label: best_at_budget(run, "total_wall_s", 1500.0)
                for label, run in direction_runs
            },
            "score_at_endpoint_evaluations": {
                label: best_at_budget(run, "analysis_endpoint_budget", 19200)
                for label, run in direction_runs
            },
        },
        "bfgs": {
            label: {
                **run_metrics(run),
                "recorded_adam_continuation": adam_continuation_metrics(
                    baseline_full,
                    anchor_iteration,
                    float(run["rows"][-1]["total_wall_s"]),
                ),
                "formal_proposal_evaluations": int(
                    sum(len(row["proposal_attempts"]) for row in run["rows"])
                ),
                "curvature_updates": int(
                    sum(bool(row["bfgs_update"].get("updated")) for row in run["rows"])
                ),
                "maximum_inverse_hessian_condition": max(
                    (
                        float(row["bfgs_update"]["condition_number"])
                        for row in run["rows"]
                        if row["bfgs_update"].get("updated")
                    ),
                    default=None,
                ),
            }
            for label, anchor_iteration, run in bfgs_runs
        },
    }
    (args.output_dir / "optimizer_exploration_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
