from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


CONFIGS = tuple(
    (steps, estimator, directions)
    for steps in (256, 128, 64)
    for estimator, directions in (
        ("central", 4),
        ("one-sided", 4),
        ("central", 2),
    )
)


def config_name(steps: int, estimator: str, directions: int) -> str:
    label = "one_sided" if estimator == "one-sided" else "central"
    return f"rk4_{steps:03d}_{label}{directions}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize_run(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(path / "manifest.json")
    summary = read_json(path / "summary.json")
    history = read_jsonl(path / "history.jsonl")
    if int(summary["completed_iterations"]) != 200 or len(history) != 200:
        raise ValueError(f"incomplete 200-step run: {path}")
    flow_wall = sum(
        float(row["pair_decode_wall_s"]) + float(row["center_decode_wall_s"])
        for row in history
    )
    score_wall = sum(
        float(row["pair_score_wall_s"]) + float(row["center_score_wall_s"])
        for row in history
    )
    endpoint_calls = sum(int(row["endpoint_count"]) for row in history)
    center_calls = sum(len(row["center_score_elapsed_s"]) for row in history)
    cache_hits = sum(bool(row["endpoint_decode_cache_hit"]) for row in history)
    row = {
        "name": path.name,
        "rk4_steps": int(manifest["flow_steps"]),
        "estimator": str(manifest["gradient_estimator"]),
        "directions": int(manifest["directions"]),
        "initial_score": float(summary["initial_score"]),
        "final_score": float(summary["final_score"]),
        "best_score": float(summary["best_score"]),
        "best_iteration": int(summary["best_iteration"]),
        "total_wall_s": float(summary["total_wall_s"]),
        "mean_iteration_wall_s": float(summary["mean_iteration_wall_s"]),
        "flow_wall_s": flow_wall,
        "score_wall_s": score_wall,
        "endpoint_score_calls": endpoint_calls,
        "center_score_calls": center_calls,
        "pipeline_cache_hits": cache_hits,
        "pipeline_wasted_endpoints": sum(
            int(row["pipeline_prefetch_wasted_endpoints"]) for row in history
        ),
    }
    return row, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    histories: dict[tuple[int, str, int], list[dict[str, Any]]] = {}
    for config in CONFIGS:
        path = args.root / config_name(*config)
        summary, history = summarize_run(path)
        summaries.append(summary)
        histories[config] = history

    historical = read_jsonl(args.historical / "history.jsonl")
    historical_iterations = [int(row["iteration"]) for row in historical]
    historical_scores = [float(row["current_score"]) for row in historical]

    fields = list(summaries[0])
    with (args.output_dir / "matrix_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    (args.output_dir / "matrix_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True, sharey=True)
    panels = (("central", 4), ("one-sided", 4), ("central", 2))
    colors = {256: "#bc3c29", 128: "#2369a1", 64: "#16865b"}
    for axis, (estimator, directions) in zip(axes, panels):
        axis.plot(
            historical_iterations,
            historical_scores,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="historical RK4-256 central-4",
        )
        for steps in (256, 128, 64):
            history = histories[(steps, estimator, directions)]
            axis.plot(
                [int(row["iteration"]) for row in history],
                [float(row["current_score"]) for row in history],
                color=colors[steps],
                linewidth=1.5,
                label=f"RK4-{steps}",
            )
        label = "one-sided" if estimator == "one-sided" else "central"
        axis.set_title(f"{label}, {directions} directions")
        axis.set_xlabel("Adam step")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("continuous QH score")
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(args.output_dir / "score_curves_9_vs_historical.png", dpi=200)
    plt.close(figure)

    ordered = sorted(
        summaries,
        key=lambda row: (row["estimator"], row["directions"], -row["rk4_steps"]),
    )
    labels = [
        f"{row['rk4_steps']}-{row['estimator'][:3]}{row['directions']}"
        for row in ordered
    ]
    flow = np.asarray([row["flow_wall_s"] for row in ordered]) / 60.0
    score = np.asarray([row["score_wall_s"] for row in ordered]) / 60.0
    total = np.asarray([row["total_wall_s"] for row in ordered]) / 60.0
    other = np.maximum(total - flow - score, 0.0)
    x = np.arange(len(ordered))
    figure, axis = plt.subplots(figsize=(12, 5.2))
    axis.bar(x, flow, label="flow")
    axis.bar(x, score, bottom=flow, label="native score")
    axis.bar(x, other, bottom=flow + score, label="other")
    axis.set_xticks(x, labels, rotation=30, ha="right")
    axis.set(ylabel="wall time [min]", title="Single-GPU 200-step wall time")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "optimizer_wall_time.png", dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
