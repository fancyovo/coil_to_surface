from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("run must be LABEL=PATH")
    label, path = value.split("=", maxsplit=1)
    return label, Path(path)


def run_summary(label: str, directory: Path) -> dict[str, Any]:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    history = read_jsonl(directory / "history.jsonl")
    initial = float(summary["initial_score"])
    best = float(summary["best_score"])
    return {
        "label": label,
        "directory": str(directory.resolve()),
        "summary": summary,
        "history": history,
        "score_gain": best - initial,
        "mean_iteration_wall_s": float(
            np.mean([row["iteration_wall_s"] for row in history])
        ),
        "valid_center_fraction": float(
            np.mean([row["current_status"] == "ok" for row in history])
        ),
        "applied_fraction": float(
            np.mean([row["center_update_accepted"] for row in history])
        ),
    }


def plot_optimization(runs: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(runs)))
    for color, run in zip(colors, runs, strict=True):
        rows = run["history"]
        step = np.asarray([row["iteration"] for row in rows])
        wall = np.asarray([row["total_wall_s"] for row in rows]) / 60.0
        current = np.asarray([row["current_score"] for row in rows])
        best = np.asarray([row["best_score"] for row in rows])
        qh = np.asarray([row["current_qh_error"] for row in rows])
        axes[0, 0].plot(step, current, color=color, alpha=0.45, linewidth=1.0)
        axes[0, 0].plot(step, best, color=color, linewidth=2.0, label=run["label"])
        axes[0, 1].plot(wall, best, color=color, linewidth=2.0, label=run["label"])
        axes[1, 0].plot(step, qh, color=color, linewidth=1.6, label=run["label"])
        axes[1, 1].plot(
            step,
            [row["iteration_wall_s"] for row in rows],
            color=color,
            alpha=0.8,
            linewidth=1.0,
            label=run["label"],
        )
    axes[0, 0].set(xlabel="Adam step", ylabel="formal score", title="Score vs step")
    axes[0, 1].set(xlabel="wall time (min)", ylabel="best formal score", title="Score vs time")
    axes[1, 0].set(
        xlabel="Adam step",
        ylabel="volume QH residual",
        title="QH residual",
        yscale="log",
    )
    axes[1, 1].set(
        xlabel="Adam step",
        ylabel="seconds",
        title="Single-GPU iteration time",
    )
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_benchmark(benchmark: dict[str, Any], output: Path) -> None:
    rows = benchmark["rows"]
    counts = np.asarray([row["endpoint_count"] for row in rows])
    total = np.asarray([row["timing"]["total"]["median_s"] for row in rows])
    amortized = np.asarray([row["amortized_ms_per_endpoint"] for row in rows])
    endpoint_rate = np.asarray([row["endpoints_per_s"] for row in rows])
    flux_tflops = np.asarray([row["flux_nominal_tflops"] for row in rows])
    psi_tflops = np.asarray([row["psi_matvec_equivalent_tflops"] for row in rows])

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].plot(counts, total, marker="o", label="gradient batch")
    axes[0, 0].set(xscale="log", xlabel="parallel endpoints", ylabel="seconds", title="Batch wall time")
    axes[0, 1].plot(counts, amortized, marker="o", color="#c44e52")
    axes[0, 1].set(
        xscale="log",
        yscale="log",
        xlabel="parallel endpoints",
        ylabel="ms / endpoint",
        title="Amortized endpoint cost",
    )
    stage_names = ["axis_refine", "axis_samples", "psi", "local_score"]
    bottom = np.zeros_like(counts, dtype=np.float64)
    for stage in stage_names:
        values = np.asarray([row["timing"][stage]["median_s"] for row in rows])
        axes[1, 0].bar(counts, values, bottom=bottom, width=np.maximum(1, counts * 0.12), label=stage)
        bottom += values
    axes[1, 0].set(xscale="log", xlabel="parallel endpoints", ylabel="seconds", title="Major stages")
    axes[1, 1].plot(counts, endpoint_rate, marker="o", label="endpoints/s")
    tflops_axis = axes[1, 1].twinx()
    tflops_axis.plot(counts, flux_tflops, marker="s", label="flux nominal TFLOP/s", color="#55a868")
    tflops_axis.plot(counts, psi_tflops, marker="^", label="psi matvec-equiv TFLOP/s", color="#8172b3")
    axes[1, 1].set(xscale="log", xlabel="parallel endpoints", ylabel="endpoints/s", title="Throughput and arithmetic model")
    tflops_axis.set(ylabel="modeled TFLOP/s")
    lines, labels = axes[1, 1].get_legend_handles_labels()
    lines2, labels2 = tflops_axis.get_legend_handles_labels()
    axes[1, 1].legend(lines + lines2, labels + labels2, fontsize=8)
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
    axes[1, 0].legend(fontsize=8)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.run:
        raise ValueError("at least one --run LABEL=PATH is required")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = [run_summary(*parse_run(value)) for value in args.run]
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    plot_optimization(runs, args.output_dir / "optimization_comparison.png")
    plot_benchmark(benchmark, args.output_dir / "batch_scaling.png")
    compact_runs = [
        {key: value for key, value in run.items() if key != "history"}
        for run in runs
    ]
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {"format": "original_space_optimization_analysis_v1", "runs": compact_runs, "benchmark": benchmark},
            indent=2,
            allow_nan=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
