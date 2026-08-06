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
    label, separator, path = value.rpartition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("runs must use LABEL=PATH")
    return label, Path(path)


def parse_offset(value: str) -> tuple[str, int]:
    label, separator, offset = value.rpartition("=")
    if not separator or not label or not offset:
        raise argparse.ArgumentTypeError("offsets must use LABEL=ITERATION")
    try:
        parsed = int(offset)
    except ValueError as error:
        raise argparse.ArgumentTypeError("offset iteration must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("offset iteration must be nonnegative")
    return label, parsed


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


def normalize_history_row(row: dict[str, Any], *, direction_count: int) -> dict[str, Any]:
    accepted_mode = row.get("accepted_mode")
    if accepted_mode is None:
        accepted_mode = "adam" if row.get("center_update_accepted", False) else "rejected"
    valid_directions = row.get("valid_directions")
    if valid_directions is None:
        rejected = sum(bool(value) for value in row.get("filtered_invalid_directions", [])) + sum(
            bool(value) for value in row.get("filtered_outlier_directions", [])
        )
        valid_directions = max(direction_count - rejected, 0)
    return {
        "iteration": int(row["iteration"]),
        "current_score": float(row["current_score"]),
        "best_score": float(row["best_score"]),
        "qh_error": float(row.get("qh_error", row.get("current_qh_error", float("nan")))),
        "qa_error": float(row.get("qa_error", row.get("current_qa_error", float("nan")))),
        "qp_error": float(row.get("qp_error", row.get("current_qp_error", float("nan")))),
        "iota": float(row.get("iota", row.get("current_iota", float("nan")))),
        "surface_level": float(row.get("surface_level", float("nan"))),
        "valid_directions": int(valid_directions),
        "accepted_mode": str(accepted_mode),
        "center_rescored": bool(row.get("center_rescored", True)),
        "secant_center_score": float(row.get("secant_center_score", float("nan"))),
        "direction_evaluations": int(row.get("direction_evaluations", direction_count)),
        "cumulative_direction_evaluations": row.get(
            "cumulative_direction_evaluations"
        ),
        "blackbox_score_evaluations": row.get("blackbox_score_evaluations"),
        "cumulative_blackbox_score_evaluations": row.get(
            "cumulative_blackbox_score_evaluations"
        ),
        "iteration_wall_s": float(row["iteration_wall_s"]),
        "pair_score_wall_s": float(row["pair_score_wall_s"]),
        "components": row.get("components"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze G3-informed subspace optimizer runs.")
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument(
        "--offset",
        action="append",
        type=parse_offset,
        default=[],
        help="Plot a run after an iteration offset, using LABEL=ITERATION.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-score", type=float, default=93.1655597)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    offsets = dict(args.offset)
    labels = [label for label, _ in args.run]
    unknown_offsets = set(offsets) - set(labels)
    if unknown_offsets:
        raise ValueError(f"offset labels have no matching run: {sorted(unknown_offsets)}")

    runs: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    for label, directory in args.run:
        summary_path = directory / "summary.json"
        summary = read_json(summary_path) if summary_path.is_file() else {}
        manifest = summary.get("manifest")
        if manifest is None:
            manifest = read_json(directory / "manifest.json")
        random_direction_count = int(
            manifest.get("random_directions", manifest.get("directions", 0))
        )
        direction_count = int(
            manifest.get("total_secant_directions", random_direction_count + 1)
        )
        history = [
            normalize_history_row(row, direction_count=direction_count)
            for row in read_jsonl(directory / "history.jsonl")
        ]
        if not history:
            raise ValueError(f"run has no history rows: {directory}")
        if not summary:
            initial = read_json(directory / "trajectory" / "step_0000.json")
            best_row = max(history, key=lambda row: row["best_score"])
            summary = {
                "initial_score": float(initial["score"]["score"]),
                "final_score": float(history[-1]["current_score"]),
                "best_score": float(best_row["best_score"]),
                "best_iteration": int(best_row["iteration"]),
            }
            # total_wall_s is recorded cumulatively in the raw history.
            raw_last = read_jsonl(directory / "history.jsonl")[-1]
            summary["total_wall_s"] = float(raw_last["total_wall_s"])
        previous_score = float(summary["initial_score"])
        running_direction_evaluations = 0
        running_blackbox_score_evaluations = 0
        center_repeat_deltas: list[float] = []
        accepted_gains: list[float] = []
        for row in history:
            center_repeat_delta = (
                float(row["secant_center_score"]) - previous_score
                if row["center_rescored"]
                else float("nan")
            )
            accepted_gain = float(row["current_score"]) - previous_score
            row["center_repeat_delta"] = center_repeat_delta
            row["accepted_score_gain"] = accepted_gain
            saved_direction_total = row["cumulative_direction_evaluations"]
            if saved_direction_total is None:
                running_direction_evaluations += int(row["direction_evaluations"])
            else:
                running_direction_evaluations = int(saved_direction_total)
            row["cumulative_direction_evaluations"] = running_direction_evaluations
            saved_score_total = row["cumulative_blackbox_score_evaluations"]
            if saved_score_total is None:
                row["cumulative_blackbox_score_evaluations"] = None
            else:
                running_blackbox_score_evaluations = int(saved_score_total)
                row["cumulative_blackbox_score_evaluations"] = (
                    running_blackbox_score_evaluations
                )
            if np.isfinite(center_repeat_delta):
                center_repeat_deltas.append(center_repeat_delta)
            if row["accepted_mode"] != "rejected":
                accepted_gains.append(accepted_gain)
            previous_score = float(row["current_score"])
        runs.append(
            {
                "label": label,
                "iteration_offset": int(offsets.get(label, 0)),
                "directory": directory,
                "summary": summary,
                "manifest": manifest,
                "history": history,
            }
        )
        walls = [float(row["iteration_wall_s"]) for row in history]
        accepted_steps = sum(row["accepted_mode"] != "rejected" for row in history)
        adam_steps = sum(row["accepted_mode"] == "adam" for row in history)
        projected_steps = sum(row["accepted_mode"] == "projected" for row in history)
        quadratic_steps = sum(row["accepted_mode"] == "quadratic" for row in history)
        quadratic_axis_steps = sum(
            row["accepted_mode"] == "quadratic_axis" for row in history
        )
        branch_steps = sum(row["accepted_mode"] == "branch_endpoint" for row in history)
        probe_steps = sum(row["accepted_mode"] == "probe_endpoint" for row in history)
        summary_rows.append(
            {
                "label": label,
                "iteration_offset": int(offsets.get(label, 0)),
                "directory": str(directory.resolve()),
                "completed": summary_path.is_file(),
                "iterations": len(history),
                "initial_score": float(summary["initial_score"]),
                "final_score": float(summary["final_score"]),
                "best_score": float(summary["best_score"]),
                "best_iteration": int(summary["best_iteration"]),
                "plotted_best_iteration": int(offsets.get(label, 0))
                + int(summary["best_iteration"]),
                "accepted_steps": int(summary.get("accepted_steps", accepted_steps)),
                "adam_accepted_steps": int(summary.get("adam_accepted_steps", adam_steps)),
                "projected_accepted_steps": int(
                    summary.get("projected_accepted_steps", projected_steps)
                ),
                "quadratic_accepted_steps": int(
                    summary.get("quadratic_accepted_steps", quadratic_steps)
                ),
                "quadratic_axis_accepted_steps": int(
                    summary.get("quadratic_axis_accepted_steps", quadratic_axis_steps)
                ),
                "branch_accepted_steps": int(summary.get("branch_accepted_steps", branch_steps)),
                "probe_accepted_steps": int(summary.get("probe_accepted_steps", probe_steps)),
                "rejected_steps": int(summary.get("rejected_steps", len(history) - accepted_steps)),
                "max_abs_center_repeat_delta": (
                    float(np.max(np.abs(center_repeat_deltas)))
                    if center_repeat_deltas
                    else None
                ),
                "minimum_accepted_score_gain": (
                    float(np.min(accepted_gains)) if accepted_gains else None
                ),
                "mean_iteration_wall_s": float(np.mean(walls)),
                "p95_iteration_wall_s": finite_percentile(walls, 95.0),
                "max_iteration_wall_s": float(np.max(walls)),
                "total_wall_s": float(summary["total_wall_s"]),
                "random_directions": random_direction_count,
                "total_secant_directions": direction_count,
                "cumulative_direction_evaluations": int(
                    history[-1]["cumulative_direction_evaluations"]
                ),
                "cumulative_blackbox_score_evaluations": history[-1][
                    "cumulative_blackbox_score_evaluations"
                ],
                "perturbation": float(manifest["perturbation"]),
                "seed": int(manifest["seed"]),
            }
        )
        for row in history:
            history_rows.append(
                {
                    "label": label,
                    "iteration": int(row["iteration"]),
                    "plotted_iteration": int(offsets.get(label, 0))
                    + int(row["iteration"]),
                    "current_score": float(row["current_score"]),
                    "best_score": float(row["best_score"]),
                    "qh_error": float(row["qh_error"]),
                    "qa_error": float(row["qa_error"]),
                    "qp_error": float(row["qp_error"]),
                    "iota": float(row["iota"]),
                    "surface_level": float(row["surface_level"]),
                    "valid_directions": int(row["valid_directions"]),
                    "accepted_mode": str(row.get("accepted_mode", "adam")),
                    "center_repeat_delta": float(row["center_repeat_delta"]),
                    "accepted_score_gain": float(row["accepted_score_gain"]),
                    "direction_evaluations": int(row["direction_evaluations"]),
                    "cumulative_direction_evaluations": int(
                        row["cumulative_direction_evaluations"]
                    ),
                    "blackbox_score_evaluations": row[
                        "blackbox_score_evaluations"
                    ],
                    "cumulative_blackbox_score_evaluations": row[
                        "cumulative_blackbox_score_evaluations"
                    ],
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
        offset = int(run["iteration_offset"])
        steps = [offset, *[offset + int(row["iteration"]) for row in history]]
        initial = float(run["summary"]["initial_score"])
        current = [initial, *[float(row["current_score"]) for row in history]]
        best = [initial, *[float(row["best_score"]) for row in history]]
        axes[0, 0].plot(steps, current, label=run["label"])
        axes[0, 1].plot(steps, best, label=run["label"])
        axes[1, 0].plot(
            [offset + int(row["iteration"]) for row in history],
            [float(row["qh_error"]) for row in history],
            label=run["label"],
        )
        axes[1, 1].plot(
            [offset + int(row["iteration"]) for row in history],
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

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    score_call_runs = 0
    for run in runs:
        history = run["history"]
        initial = float(run["summary"]["initial_score"])
        direction_budget = [
            0,
            *[int(row["cumulative_direction_evaluations"]) for row in history],
        ]
        scores = [initial, *[float(row["current_score"]) for row in history]]
        axes[0].plot(direction_budget, scores, label=run["label"])
        score_call_budget = [
            row["cumulative_blackbox_score_evaluations"] for row in history
        ]
        if all(value is not None for value in score_call_budget):
            score_call_runs += 1
            axes[1].plot(
                [1, *[int(value) for value in score_call_budget]],
                scores,
                label=run["label"],
            )
    axes[0].set(
        title="Score by evaluated direction budget",
        xlabel="cumulative directions (G3 reference + random)",
        ylabel="exact score",
    )
    axes[1].set(
        title="Score by exact black-box calls",
        xlabel="cumulative score calls",
        ylabel="exact score",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    if score_call_runs == 0:
        axes[1].text(0.5, 0.5, "Exact call counts unavailable", ha="center", va="center")
    figure.savefig(args.output_dir / "optimizer_cost_comparison.png", dpi=180)
    plt.close(figure)

    detailed_runs = [run for run in runs if run["history"] and run["history"][0]["components"]]
    if not detailed_runs:
        raise ValueError("at least one run must provide score components")
    selected = max(detailed_runs, key=lambda run: float(run["summary"]["best_score"]))
    history = selected["history"]
    offset = int(selected["iteration_offset"])
    steps = [offset + int(row["iteration"]) for row in history]
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
