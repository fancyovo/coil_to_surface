from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from flow_matching.monitoring import read_metrics


VALIDATION_KEYS = (
    "validation_loss",
    "validation_geometry_physical_loss",
    "validation_geometry_relative_loss",
    "validation_current_loss",
)


def validation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    for row in rows:
        if "step" in row and "validation_loss" in row:
            by_step[int(row["step"])] = row
    return [by_step[step] for step in sorted(by_step)]


def summarize_rows(
    rows: list[dict[str, Any]], *, tail_points: int
) -> dict[str, Any]:
    validation = validation_rows(rows)
    if not validation:
        raise ValueError("run contains no validation rows")
    tail = validation[-min(tail_points, len(validation)) :]
    steps = np.asarray([row["step"] for row in tail], dtype=np.float64)
    summary: dict[str, Any] = {
        "validation_points": len(validation),
        "tail_points": len(tail),
        "tail_step_start": int(steps[0]),
        "tail_step_end": int(steps[-1]),
    }
    for key in VALIDATION_KEYS:
        values = np.asarray([row[key] for row in validation], dtype=np.float64)
        tail_values = values[-len(tail) :]
        slope = (
            float(np.polyfit(steps, tail_values, 1)[0] * 1000.0)
            if len(tail_values) > 1
            else None
        )
        summary[key] = {
            "minimum": float(values.min()),
            "minimum_step": int(validation[int(values.argmin())]["step"]),
            "final": float(values[-1]),
            "tail_mean": float(tail_values.mean()),
            "tail_slope_per_1000_steps": slope,
        }
    samples = [row for row in rows if row.get("event") == "sample"]
    if samples:
        summary["final_sample"] = samples[-1]
    completed = [row for row in rows if row.get("event") == "complete"]
    if completed:
        summary["complete"] = completed[-1]
    return summary


def parse_run(specification: str) -> tuple[str, Path]:
    label, separator, raw_path = specification.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("run must be LABEL=METRICS_JSONL")
    path = Path(raw_path)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"metrics file does not exist: {path}")
    return label, path


def plot_runs(
    runs: list[tuple[str, list[dict[str, Any]]]], output_path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = (
        "Weighted validation loss",
        "Physical geometry loss",
        "Normalized-coordinate auxiliary loss",
        "Current loss",
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, key, title in zip(axes.flat, VALIDATION_KEYS, titles, strict=True):
        for label, rows in runs:
            validation = validation_rows(rows)
            axis.plot(
                [row["step"] for row in validation],
                [row[key] for row in validation],
                label=label,
                linewidth=1.6,
            )
        axis.set(title=title, xlabel="step", ylabel="loss")
        axis.grid(alpha=0.25)
        axis.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)
    figure.savefig(temporary, dpi=160)
    plt.close(figure)
    temporary.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, type=parse_run)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tail-points", type=int, default=25)
    args = parser.parse_args()
    if args.tail_points <= 0:
        raise ValueError("tail-points must be positive")

    loaded = [(label, read_metrics(path)) for label, path in args.run]
    summary = {
        "tail_points_requested": args.tail_points,
        "runs": {
            label: summarize_rows(rows, tail_points=args.tail_points)
            for label, rows in loaded
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    plot_runs(loaded, args.output_dir / "validation_comparison.png")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
