from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import statistics
from typing import Any

import numpy as np


LEARNING_RATES = (0.003, 0.01, 0.03)
BETA1_VALUES = (0.5, 0.7, 0.9)
RANDOM_DIRECTIONS = (0, 1, 2)
COMPONENTS = ("axis", "psi", "surface", "coordinate", "volume_qs", "iota", "coil")
BASELINE_SCORE = 93.1655597


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_at_budget(
    initial_score: float,
    history: list[dict[str, Any]],
    budget: int,
    counter: str,
) -> float:
    value = initial_score
    for row in history:
        if int(row[counter]) > budget:
            break
        value = float(row["best_score"])
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the 27-run QH direction sweep.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    expected = set(itertools.product(LEARNING_RATES, BETA1_VALUES, RANDOM_DIRECTIONS))
    runs: dict[tuple[float, float, int], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    center_repeat_errors: list[float] = []
    accepted_score_gains: list[float] = []
    for directory in sorted(path for path in args.raw_root.iterdir() if path.is_dir()):
        summary = read_json(directory / "summary.json")
        manifest = summary["manifest"]
        history = read_jsonl(directory / "history.jsonl")
        key = (
            float(manifest["learning_rate"]),
            float(manifest["betas"][0]),
            int(manifest["random_directions"]),
        )
        if key in runs:
            raise ValueError(f"duplicate sweep point: {key}")
        if len(history) != 100 or [int(row["iteration"]) for row in history] != list(
            range(1, 101)
        ):
            raise ValueError(f"incomplete history: {directory}")
        initial_score = float(summary["initial_score"])
        previous_score = initial_score
        for row in history:
            current_score = float(row["current_score"])
            gain = current_score - previous_score
            if gain < -1.0e-10:
                raise ValueError(f"non-monotone accepted score in {directory}: {gain}")
            if row["accepted_mode"] != "rejected":
                accepted_score_gains.append(gain)
            if row.get("center_rescored", False):
                center_repeat_errors.append(abs(float(row["secant_center_score_delta"])))
            previous_score = current_score
        if abs(previous_score - float(summary["final_score"])) > 1.0e-10:
            raise ValueError(f"summary/history final score mismatch: {directory}")

        g3_valid_steps = sum(math.isfinite(float(row["g3_slope"])) for row in history)
        valid_direction_count = sum(int(row["valid_directions"]) for row in history)
        total_direction_slots = len(history) * int(manifest["total_secant_directions"])
        row = {
            "run_name": directory.name,
            "learning_rate": key[0],
            "beta1": key[1],
            "random_directions_k": key[2],
            "total_directions_per_step": key[2] + 1,
            "initial_score": initial_score,
            "final_score": float(summary["final_score"]),
            "best_score": float(summary["best_score"]),
            "score_gain": float(summary["best_score"]) - initial_score,
            "best_iteration": int(summary["best_iteration"]),
            "accepted_steps": int(summary["accepted_steps"]),
            "adam_accepted_steps": int(summary["adam_accepted_steps"]),
            "probe_accepted_steps": int(summary["probe_accepted_steps"]),
            "branch_accepted_steps": int(summary["branch_accepted_steps"]),
            "rejected_steps": int(summary["rejected_steps"]),
            "g3_valid_steps": g3_valid_steps,
            "valid_direction_count": valid_direction_count,
            "valid_direction_fraction": valid_direction_count / total_direction_slots,
            "evaluated_directions": int(summary["cumulative_direction_evaluations"]),
            "blackbox_score_calls": int(summary["cumulative_blackbox_score_evaluations"]),
            "score_calls_per_direction": int(
                summary["cumulative_blackbox_score_evaluations"]
            )
            / int(summary["cumulative_direction_evaluations"]),
            "total_wall_s": float(summary["total_wall_s"]),
            "mean_iteration_wall_s": float(summary["mean_iteration_wall_s"]),
            "best_qh_error": float(summary["best_diagnostics"]["qs_global_error"]),
            "best_qa_error": float(summary["best_diagnostics"]["qs_qa_global_error"]),
            "best_qp_error": float(summary["best_diagnostics"]["qs_qp_global_error"]),
            "best_iota": float(summary["best_diagnostics"]["iota_min"]),
            "best_surface_level": float(summary["best_diagnostics"]["surface_level"]),
            **{
                f"best_component_{name}": float(summary["best_components"][name])
                for name in COMPONENTS
            },
        }
        rows.append(row)
        runs[key] = {
            "directory": directory,
            "summary": summary,
            "manifest": manifest,
            "history": history,
            "row": row,
        }
    if set(runs) != expected:
        raise ValueError(f"sweep grid mismatch: missing={expected - set(runs)}, extra={set(runs) - expected}")

    shard_durations = []
    for shard_index in range(6):
        shard = read_json(args.raw_root / f"shard_{shard_index:02d}_manifest.json")
        if shard.get("status") != "completed" or len(shard["completed_runs"]) != len(
            shard["configs"]
        ):
            raise ValueError(f"shard {shard_index} is incomplete")
        shard_durations.append(
            (float(shard["completed_unix_s"]) - float(shard["started_unix_s"])) / 3600.0
        )
        postflight = (
            args.raw_root / f"shard_{shard_index}_gpu_postflight.csv"
        ).read_text(encoding="utf-8").strip()
        fields = [field.strip() for field in postflight.split(",")]
        if len(fields) < 6 or int(fields[3]) != 0 or int(fields[4]) > 16:
            raise ValueError(f"shard {shard_index} postflight is not idle: {postflight}")

    rows.sort(key=lambda row: (-float(row["best_score"]), str(row["run_name"])))
    write_csv(args.output_dir / "run_summary.csv", rows)

    grouped: dict[str, Any] = {}
    for field, values in (
        ("random_directions_k", RANDOM_DIRECTIONS),
        ("learning_rate", LEARNING_RATES),
        ("beta1", BETA1_VALUES),
    ):
        grouped[field] = {}
        for value in values:
            selected = [row for row in rows if row[field] == value]
            grouped[field][str(value)] = {
                "count": len(selected),
                "mean_best_score": statistics.mean(float(row["best_score"]) for row in selected),
                "median_best_score": statistics.median(
                    float(row["best_score"]) for row in selected
                ),
                "max_best_score": max(float(row["best_score"]) for row in selected),
                "mean_score_calls": statistics.mean(
                    int(row["blackbox_score_calls"]) for row in selected
                ),
            }

    paired_k_effects = []
    for learning_rate in LEARNING_RATES:
        for beta1 in BETA1_VALUES:
            scores = [float(runs[learning_rate, beta1, k]["row"]["best_score"]) for k in RANDOM_DIRECTIONS]
            paired_k_effects.append(
                {
                    "learning_rate": learning_rate,
                    "beta1": beta1,
                    "score_k0": scores[0],
                    "score_k1": scores[1],
                    "score_k2": scores[2],
                    "k1_minus_k0": scores[1] - scores[0],
                    "k2_minus_k1": scores[2] - scores[1],
                }
            )
    write_csv(args.output_dir / "paired_k_effects.csv", paired_k_effects)

    total_directions = sum(int(row["evaluated_directions"]) for row in rows)
    total_calls = sum(int(row["blackbox_score_calls"]) for row in rows)
    best = rows[0]
    aggregate = {
        "format": "qh_reference_direction_sweep_analysis_v1",
        "validated_run_count": len(rows),
        "validated_step_count": len(rows) * 100,
        "accepted_score_drop_count": 0,
        "center_repeat_count": len(center_repeat_errors),
        "max_abs_center_repeat_error": max(center_repeat_errors),
        "minimum_accepted_score_gain": min(accepted_score_gains),
        "total_evaluated_directions": total_directions,
        "required_plus_minus_endpoint_calls": 2 * total_directions,
        "actual_blackbox_score_calls": total_calls,
        "conservative_blackbox_call_upper_bound": 27162,
        "actual_to_upper_bound_fraction": total_calls / 27162.0,
        "score_calls_per_evaluated_direction": total_calls / total_directions,
        "summed_gpu_hours": sum(float(row["total_wall_s"]) for row in rows) / 3600.0,
        "parallel_shard_wall_hours": shard_durations,
        "parallel_wall_hours": max(shard_durations),
        "best_run": best,
        "gap_to_historical_k4_score": BASELINE_SCORE - float(best["best_score"]),
        "grouped": grouped,
    }
    write_json(args.output_dir / "analysis.json", aggregate)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {0: "#6b7280", 1: "#16857b", 2: "#d97706"}
    figure, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True, constrained_layout=True)
    for row_index, learning_rate in enumerate(LEARNING_RATES):
        for column_index, beta1 in enumerate(BETA1_VALUES):
            axis = axes[row_index, column_index]
            for k in RANDOM_DIRECTIONS:
                run = runs[learning_rate, beta1, k]
                history = run["history"]
                x = [0, *[int(item["cumulative_direction_evaluations"]) for item in history]]
                y = [float(run["summary"]["initial_score"]), *[float(item["best_score"]) for item in history]]
                axis.plot(x, y, color=colors[k], linewidth=1.8, label=f"K={k}")
            axis.axhline(BASELINE_SCORE, color="black", linestyle=":", linewidth=0.8)
            axis.set_title(f"lr={learning_rate:g}, beta1={beta1:g}")
            axis.grid(alpha=0.22)
            if row_index == 2:
                axis.set_xlabel("cumulative evaluated directions")
            if column_index == 0:
                axis.set_ylabel("best exact score")
    axes[0, 0].legend(loc="lower right")
    figure.suptitle("All 27 trajectories by evaluated-direction budget", fontsize=15)
    figure.savefig(args.output_dir / "score_by_direction_grid.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True, constrained_layout=True)
    for row_index, learning_rate in enumerate(LEARNING_RATES):
        for column_index, beta1 in enumerate(BETA1_VALUES):
            axis = axes[row_index, column_index]
            for k in RANDOM_DIRECTIONS:
                run = runs[learning_rate, beta1, k]
                history = run["history"]
                x = [1, *[int(item["cumulative_blackbox_score_evaluations"]) for item in history]]
                y = [float(run["summary"]["initial_score"]), *[float(item["best_score"]) for item in history]]
                axis.plot(x, y, color=colors[k], linewidth=1.8, label=f"K={k}")
            axis.axhline(BASELINE_SCORE, color="black", linestyle=":", linewidth=0.8)
            axis.set_title(f"lr={learning_rate:g}, beta1={beta1:g}")
            axis.grid(alpha=0.22)
            if row_index == 2:
                axis.set_xlabel("cumulative black-box score calls")
            if column_index == 0:
                axis.set_ylabel("best exact score")
    axes[0, 0].legend(loc="lower right")
    figure.suptitle("All 27 trajectories by actual score-call budget", fontsize=15)
    figure.savefig(args.output_dir / "score_by_call_grid.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    heatmap_values = np.asarray([float(row["best_score"]) for row in rows])
    for axis, k in zip(axes, RANDOM_DIRECTIONS, strict=True):
        matrix = np.asarray(
            [
                [float(runs[learning_rate, beta1, k]["row"]["best_score"]) for learning_rate in LEARNING_RATES]
                for beta1 in BETA1_VALUES
            ]
        )
        image = axis.imshow(
            matrix,
            cmap="viridis",
            vmin=float(np.min(heatmap_values)),
            vmax=BASELINE_SCORE,
            aspect="auto",
        )
        for row_index in range(3):
            for column_index in range(3):
                axis.text(column_index, row_index, f"{matrix[row_index, column_index]:.2f}", ha="center", va="center", color="white" if matrix[row_index, column_index] < 90.5 else "black", fontsize=10)
        axis.set_title(f"K={k} random directions")
        axis.set_xticks(range(3), [f"{value:g}" for value in LEARNING_RATES])
        axis.set_yticks(range(3), [f"{value:g}" for value in BETA1_VALUES])
        axis.set_xlabel("learning rate")
        axis.set_ylabel("beta1")
    figure.colorbar(image, ax=axes, label="best exact score", shrink=0.85)
    figure.savefig(args.output_dir / "best_score_heatmaps.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for k in RANDOM_DIRECTIONS:
        selected = [runs[learning_rate, beta1, k] for learning_rate in LEARNING_RATES for beta1 in BETA1_VALUES]
        max_direction_budget = (k + 1) * 100
        direction_budgets = np.arange(0, max_direction_budget + 1)
        matrix = np.asarray(
            [
                [score_at_budget(float(run["summary"]["initial_score"]), run["history"], int(budget), "cumulative_direction_evaluations") for budget in direction_budgets]
                for run in selected
            ]
        )
        axes[0].plot(direction_budgets, np.max(matrix, axis=0), color=colors[k], linewidth=2.0)
        axes[0].plot(direction_budgets, np.median(matrix, axis=0), color=colors[k], linewidth=1.2, linestyle="--", alpha=0.85)
        max_common_calls = min(int(run["row"]["blackbox_score_calls"]) for run in selected)
        call_budgets = np.arange(1, max_common_calls + 1)
        call_matrix = np.asarray(
            [
                [score_at_budget(float(run["summary"]["initial_score"]), run["history"], int(budget), "cumulative_blackbox_score_evaluations") for budget in call_budgets]
                for run in selected
            ]
        )
        axes[1].plot(call_budgets, np.max(call_matrix, axis=0), color=colors[k], linewidth=2.0)
        axes[1].plot(call_budgets, np.median(call_matrix, axis=0), color=colors[k], linewidth=1.2, linestyle="--", alpha=0.85)
    legend_handles = [
        *(Line2D([0], [0], color=colors[k], linewidth=2.0, label=f"K={k}") for k in RANDOM_DIRECTIONS),
        Line2D([0], [0], color="black", linewidth=2.0, label="post-hoc max over 9 settings"),
        Line2D([0], [0], color="black", linewidth=1.2, linestyle="--", label="median over 9 settings"),
        Line2D([0], [0], color="black", linewidth=0.9, linestyle=":", label="historical K=4 marker"),
    ]
    for axis in axes:
        axis.axhline(BASELINE_SCORE, color="black", linestyle=":", linewidth=0.9)
        axis.grid(alpha=0.22)
        axis.legend(handles=legend_handles, fontsize=8)
        axis.set_ylabel("best exact score")
    axes[0].set(title="Post-hoc tuned envelope by direction budget", xlabel="cumulative evaluated directions")
    axes[1].set(title="Post-hoc tuned envelope by score-call budget", xlabel="cumulative black-box score calls")
    figure.savefig(args.output_dir / "budget_efficiency_envelopes.png", dpi=180)
    plt.close(figure)

    best_key = (
        float(best["learning_rate"]),
        float(best["beta1"]),
        int(best["random_directions_k"]),
    )
    best_run = runs[best_key]
    history = best_run["history"]
    iterations = [int(row["iteration"]) for row in history]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].plot([0, *iterations], [float(best_run["summary"]["initial_score"]), *[float(row["best_score"]) for row in history]], color="#1f2937")
    axes[0, 0].axhline(BASELINE_SCORE, color="black", linestyle=":", linewidth=0.9)
    axes[0, 0].set(title="Best run score", xlabel="iteration", ylabel="best exact score")
    for component in COMPONENTS:
        axes[0, 1].plot(iterations, [float(row["components"][component]) for row in history], label=component)
    axes[0, 1].set(title="Score components", xlabel="iteration", ylabel="component score")
    axes[0, 1].legend(ncol=2, fontsize=8)
    for field, label in (("qh_error", "QH"), ("qa_error", "QA"), ("qp_error", "QP")):
        axes[1, 0].plot(iterations, [float(row[field]) for row in history], label=label)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set(title="Differential symmetry residuals", xlabel="iteration", ylabel="residual")
    axes[1, 0].legend()
    axes[1, 1].plot(iterations, [int(row["valid_directions"]) for row in history], label="valid directions")
    calls_axis = axes[1, 1].twinx()
    calls_axis.plot(iterations, [int(row["cumulative_blackbox_score_evaluations"]) for row in history], color="#d97706", label="score calls")
    axes[1, 1].set(title="Direction validity and cost", xlabel="iteration", ylabel="valid directions")
    calls_axis.set_ylabel("cumulative score calls")
    axes[1, 1].legend(loc="upper left")
    calls_axis.legend(loc="lower right")
    for axis in axes.ravel():
        axis.grid(alpha=0.22)
    figure.savefig(args.output_dir / "best_run_diagnostics.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
