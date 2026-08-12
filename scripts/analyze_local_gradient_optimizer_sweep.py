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


def first_at_or_above(rows: list[dict[str, Any]], score: float) -> dict[str, Any] | None:
    for row in rows:
        if float(row["best_score"]) >= score:
            return {
                "iteration": int(row["iteration"]),
                "wall_s": float(row["total_wall_s"]),
                "endpoint_evaluations": int(row["cumulative_endpoint_evaluations"]),
            }
    return None


def quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "p50": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def load_run(label: str, directory: Path) -> dict[str, Any]:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    rows = read_jsonl(directory / "history.jsonl")
    cumulative = 0
    for row in rows:
        cumulative += int(row["gradient_endpoint_count"])
        row["cumulative_endpoint_evaluations"] = cumulative
    accepted = [bool(row["center_update_accepted"]) for row in rows]
    improvements = [
        float(row["current_score"]) - float(rows[index - 1]["current_score"])
        if index
        else float(row["current_score"]) - float(summary["initial_score"])
        for index, row in enumerate(rows)
    ]
    return {
        "label": label,
        "directory": str(directory),
        "summary": summary,
        "rows": rows,
        "metrics": {
            "initial_score": float(summary["initial_score"]),
            "final_score": float(summary["final_score"]),
            "best_score": float(summary["best_score"]),
            "best_iteration": int(summary["best_iteration"]),
            "score_gain": float(summary["best_score"] - summary["initial_score"]),
            "iterations": len(rows),
            "wall_s": float(summary["total_wall_s"]),
            "endpoint_evaluations": cumulative,
            "accepted_fraction": float(np.mean(accepted)) if accepted else None,
            "negative_step_fraction": float(np.mean(np.asarray(improvements) < 0.0))
            if improvements
            else None,
            "iteration_timing_s": quantiles(
                [float(row["iteration_wall_s"]) for row in rows]
            ),
            "gradient_timing_s": quantiles(
                [float(row["gradient_wall_s"]) for row in rows]
            ),
            "thresholds": {
                str(score): first_at_or_above(rows, score)
                for score in (90.0, 91.0, 92.0, 92.5, 93.0, 93.5)
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="LABEL=DIR")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    runs = []
    for item in args.run:
        label, raw_directory = item.split("=", 1)
        runs.append(load_run(label, Path(raw_directory)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    serializable = {
        "format": "local_gradient_optimizer_sweep_v1",
        "runs": [
            {
                "label": run["label"],
                "directory": run["directory"],
                "manifest": run["summary"]["manifest"],
                "metrics": run["metrics"],
            }
            for run in runs
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(serializable, indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for run in runs:
        rows = run["rows"]
        axes[0, 0].plot(
            [row["iteration"] for row in rows],
            [row["best_score"] for row in rows],
            label=run["label"],
        )
        axes[0, 1].plot(
            np.asarray([row["total_wall_s"] for row in rows]) / 60.0,
            [row["best_score"] for row in rows],
            label=run["label"],
        )
        axes[1, 0].plot(
            [row["cumulative_endpoint_evaluations"] for row in rows],
            [row["best_score"] for row in rows],
            label=run["label"],
        )
        axes[1, 1].plot(
            [row["iteration"] for row in rows],
            [row["current_score"] for row in rows],
            label=run["label"],
            alpha=0.85,
        )
    axes[0, 0].set(title="Best score by step", xlabel="optimizer step", ylabel="score")
    axes[0, 1].set(title="Best score by wall time", xlabel="wall time (min)", ylabel="score")
    axes[1, 0].set(
        title="Best score by local endpoint budget",
        xlabel="cumulative local endpoint evaluations",
        ylabel="score",
    )
    axes[1, 1].set(title="Current-score stability", xlabel="optimizer step", ylabel="score")
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(args.output_dir / "optimizer_comparison.png", dpi=190)
    plt.close(figure)


if __name__ == "__main__":
    main()
