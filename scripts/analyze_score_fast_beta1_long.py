from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


RUNS = (("beta1_0p5", 0.5), ("beta1_0p7", 0.7), ("beta1_0p9", 0.9))
CHECKPOINT_STEPS = (100, 200, 300, 400, 500, 600)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize_run(
    path: Path, expected_beta1: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(path / "manifest.json")
    summary = read_json(path / "summary.json")
    best = read_json(path / "best.json")["flow_prior_standard_adam"]
    history = read_jsonl(path / "history.jsonl")
    if len(history) != 600 or int(summary["completed_iterations"]) != 600:
        raise ValueError(f"incomplete 600-step run: {path}")
    beta1 = float(manifest["betas"][0])
    if not np.isclose(beta1, expected_beta1):
        raise ValueError(f"unexpected beta1 for {path}: {beta1}")
    if manifest["gpu_ids"] != [0, 1]:
        raise ValueError(f"run is not a two-score-GPU result: {path}")
    if (
        int(manifest["flow_steps"]) != 128
        or int(manifest["directions"]) != 2
        or manifest["gradient_estimator"] != "central"
    ):
        raise ValueError(f"run does not use the accepted default configuration: {path}")

    scores = np.asarray([float(row["current_score"]) for row in history])
    running_best = np.asarray([float(row["best_score"]) for row in history])
    iteration_wall = np.asarray([float(row["iteration_wall_s"]) for row in history])
    tail_iterations = np.arange(501, 601, dtype=np.float64)
    tail_slope = float(np.polyfit(tail_iterations, scores[-100:], 1)[0] * 100.0)
    native = best["native_score"]
    diagnostics = native["diagnostics"]
    components = native["components"]
    row: dict[str, Any] = {
        "name": path.name,
        "beta1": beta1,
        "initial_score": float(summary["initial_score"]),
        "final_score": float(summary["final_score"]),
        "best_score": float(summary["best_score"]),
        "best_iteration": int(summary["best_iteration"]),
        "best_gain": float(summary["best_score"] - summary["initial_score"]),
        "completed_adam_steps": int(summary["completed_adam_steps"]),
        "total_wall_s": float(summary["total_wall_s"]),
        "mean_iteration_wall_s": float(summary["mean_iteration_wall_s"]),
        "p95_iteration_wall_s": float(np.quantile(iteration_wall, 0.95)),
        "flow_wall_s": sum(
            float(item["pair_decode_wall_s"]) + float(item["center_decode_wall_s"])
            for item in history
        ),
        "score_wall_s": sum(
            float(item["pair_score_wall_s"]) + float(item["center_score_wall_s"])
            for item in history
        ),
        "pipeline_cache_hits": sum(
            bool(item["endpoint_decode_cache_hit"]) for item in history
        ),
        "pipeline_wasted_endpoints": sum(
            int(item["pipeline_prefetch_wasted_endpoints"]) for item in history
        ),
        "gradient_steps_applied": sum(
            bool(item["gradient_step_applied"]) for item in history
        ),
        "accepted_center_updates": sum(
            bool(item["center_update_accepted"]) for item in history
        ),
        "non_ok_endpoints": sum(
            status != "ok" for item in history for status in item["pair_statuses"]
        ),
        "temporal_rejections": sum(
            bool(item["temporal_step_rejected"]) for item in history
        ),
        "negative_score_steps": int(np.sum(np.diff(scores) < 0.0)),
        "last_100_running_best_gain": float(running_best[-1] - running_best[499]),
        "last_100_score_slope_per_100_steps": tail_slope,
        "best_qh_error_per_helicity": float(
            diagnostics["qs_target_global_error_per_helicity"]
        ),
        "best_iota_min": float(diagnostics["iota_min"]),
        "best_surface_inverse_aspect_ratio": float(
            diagnostics["surface_inverse_aspect_ratio"]
        ),
    }
    for step in CHECKPOINT_STEPS:
        row[f"score_step_{step}"] = float(scores[step - 1])
        row[f"best_step_{step}"] = float(running_best[step - 1])
    for name, value in components.items():
        row[f"component_{name}"] = float(value)
    return row, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    histories: dict[float, list[dict[str, Any]]] = {}
    for name, beta1 in RUNS:
        summary, history = summarize_run(args.root / name, beta1)
        summaries.append(summary)
        histories[beta1] = history

    fields = list(summaries[0])
    with (args.output_dir / "beta1_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    (args.output_dir / "beta1_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )

    colors = {0.5: "#2369a1", 0.7: "#bc3c29", 0.9: "#16865b"}
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    for beta1, history in histories.items():
        iterations = np.asarray([int(row["iteration"]) for row in history])
        scores = np.asarray([float(row["current_score"]) for row in history])
        running_best = np.asarray([float(row["best_score"]) for row in history])
        qh_error = np.asarray([float(row["current_qh_error"]) for row in history]) / 4.0
        update_rms = np.asarray([float(row["update_rms"]) for row in history])
        kernel = np.ones(21, dtype=np.float64) / 21.0
        smooth_update = np.convolve(update_rms, kernel, mode="valid")
        label = rf"$\beta_1={beta1:g}$"
        axes[0, 0].plot(iterations, scores, color=colors[beta1], label=label)
        axes[0, 1].plot(iterations, running_best, color=colors[beta1], label=label)
        axes[1, 0].semilogy(iterations, qh_error, color=colors[beta1], label=label)
        axes[1, 1].plot(
            iterations[20:], smooth_update, color=colors[beta1], label=label
        )
    axes[0, 0].set(title="Current score", ylabel="continuous QH score")
    axes[0, 1].set(title="Running best", ylabel="continuous QH score")
    axes[1, 0].set(
        title="Current differential QH error",
        xlabel="Adam step",
        ylabel="QH error / helicity",
    )
    axes[1, 1].set(
        title="Update RMS (21-step mean)",
        xlabel="Adam step",
        ylabel="latent update RMS",
    )
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0, 0].set_xlabel("Adam step")
    axes[0, 1].set_xlabel("Adam step")
    figure.tight_layout()
    figure.savefig(args.output_dir / "beta1_long_trajectories.png", dpi=200)
    plt.close(figure)

    component_names = (
        "axis",
        "psi",
        "surface",
        "coordinate",
        "volume_qs",
        "iota",
        "coil",
    )
    x = np.arange(len(component_names), dtype=np.float64)
    width = 0.24
    figure, axis = plt.subplots(figsize=(11.5, 5.4))
    for index, row in enumerate(summaries):
        axis.bar(
            x + (index - 1) * width,
            [row[f"component_{name}"] for name in component_names],
            width=width,
            color=colors[float(row["beta1"])],
            label=rf"$\beta_1={float(row['beta1']):g}$",
        )
    axis.set_xticks(x, component_names)
    axis.set_ylim(55, 101)
    axis.set(ylabel="component score", title="Components at each run's best point")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "beta1_best_components.png", dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
