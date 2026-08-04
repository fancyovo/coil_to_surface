from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


COMPONENTS = ("axis", "psi", "surface", "coordinate", "volume_qs", "iota", "coil")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def parse_run(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("runs must use LABEL=PATH")
    return label, Path(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_percentile(values: list[float], percentile: float) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return float(np.percentile(finite, percentile)) if finite.size else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze G3-informed subspace optimizer runs.")
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-score", type=float, default=93.1655597)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    for label, directory in args.run:
        summary = read_json(directory / "summary.json")
        history = read_jsonl(directory / "history.jsonl")
        runs.append({"label": label, "directory": directory, "summary": summary, "history": history})
        walls = [float(row["iteration_wall_s"]) for row in history]
        summary_rows.append(
            {
                "label": label,
                "directory": str(directory.resolve()),
                "iterations": len(history),
                "initial_score": float(summary["initial_score"]),
                "final_score": float(summary["final_score"]),
                "best_score": float(summary["best_score"]),
                "best_iteration": int(summary["best_iteration"]),
                "accepted_steps": int(summary.get("accepted_steps", 0)),
                "adam_accepted_steps": int(summary.get("adam_accepted_steps", 0)),
                "branch_accepted_steps": int(summary.get("branch_accepted_steps", 0)),
                "rejected_steps": int(summary.get("rejected_steps", 0)),
                "mean_iteration_wall_s": float(np.mean(walls)),
                "p95_iteration_wall_s": finite_percentile(walls, 95.0),
                "max_iteration_wall_s": float(np.max(walls)),
                "total_wall_s": float(summary["total_wall_s"]),
                "random_directions": int(summary["manifest"]["random_directions"]),
                "perturbation": float(summary["manifest"]["perturbation"]),
                "seed": int(summary["manifest"]["seed"]),
            }
        )
        for row in history:
            history_rows.append(
                {
                    "label": label,
                    "iteration": int(row["iteration"]),
                    "current_score": float(row["current_score"]),
                    "best_score": float(row["best_score"]),
                    "qh_error": float(row["qh_error"]),
                    "qa_error": float(row["qa_error"]),
                    "qp_error": float(row["qp_error"]),
                    "iota": float(row["iota"]),
                    "surface_level": float(row["surface_level"]),
                    "valid_directions": int(row["valid_directions"]),
                    "accepted_mode": str(row.get("accepted_mode", "adam")),
                    "iteration_wall_s": float(row["iteration_wall_s"]),
                }
            )
    write_csv(args.output_dir / "run_summary.csv", summary_rows)
    write_csv(args.output_dir / "history.csv", history_rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for run in runs:
        history = run["history"]
        steps = [0, *[int(row["iteration"]) for row in history]]
        initial = float(run["summary"]["initial_score"])
        current = [initial, *[float(row["current_score"]) for row in history]]
        best = [initial, *[float(row["best_score"]) for row in history]]
        axes[0, 0].plot(steps, current, label=run["label"])
        axes[0, 1].plot(steps, best, label=run["label"])
        axes[1, 0].plot(
            [int(row["iteration"]) for row in history],
            [float(row["qh_error"]) for row in history],
            label=run["label"],
        )
        axes[1, 1].plot(
            [int(row["iteration"]) for row in history],
            [int(row["valid_directions"]) for row in history],
            label=run["label"],
        )
    axes[0, 0].axhline(args.baseline_score, color="black", linestyle="--", linewidth=1.0, label="K=4 SPSA baseline")
    axes[0, 1].axhline(args.baseline_score, color="black", linestyle="--", linewidth=1.0, label="K=4 SPSA baseline")
    axes[0, 0].set(title="Exact ABI-9 score", xlabel="iteration", ylabel="current score")
    axes[0, 1].set(title="Monotone best score", xlabel="iteration", ylabel="best score")
    axes[1, 0].set(title="QH differential residual", xlabel="iteration", ylabel="QH error")
    axes[1, 1].set(title="Same-branch secant directions", xlabel="iteration", ylabel="valid directions")
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(args.output_dir / "optimizer_comparison.png", dpi=180)
    plt.close(figure)

    selected = max(runs, key=lambda run: float(run["summary"]["best_score"]))
    history = selected["history"]
    steps = [int(row["iteration"]) for row in history]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for component in COMPONENTS:
        axes[0, 0].plot(steps, [float(row["components"][component]) for row in history], label=component)
    for key, label in (("qh_error", "QH"), ("qa_error", "QA"), ("qp_error", "QP")):
        axes[0, 1].plot(steps, [float(row[key]) for row in history], label=label)
    axes[1, 0].plot(steps, [float(row["iota"]) for row in history], label="iota")
    level_axis = axes[1, 0].twinx()
    level_axis.plot(
        steps,
        [float(row["surface_level"]) for row in history],
        color="tab:red",
        alpha=0.6,
        label="surface level",
    )
    axes[1, 1].plot(steps, [float(row["iteration_wall_s"]) for row in history], label="iteration")
    axes[1, 1].plot(steps, [float(row["pair_score_wall_s"]) for row in history], label="secant score batch")
    axes[0, 0].set(title=f"Score components: {selected['label']}", xlabel="iteration", ylabel="component score")
    axes[0, 1].set(title="Helicity residuals", xlabel="iteration", ylabel="error")
    axes[1, 0].set(title="Iota and selected surface", xlabel="iteration", ylabel="iota")
    level_axis.set_ylabel("surface level")
    axes[1, 1].set(title="Per-step cost", xlabel="iteration", ylabel="seconds")
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    level_axis.legend(loc="lower right", fontsize=8)
    figure.savefig(args.output_dir / "best_run_diagnostics.png", dpi=180)
    plt.close(figure)

    (args.output_dir / "analysis.json").write_text(
        json.dumps(
            {
                "format": "qh_g3_subspace_optimizer_analysis_v1",
                "baseline_score": float(args.baseline_score),
                "selected_best_run": selected["label"],
                "runs": summary_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
