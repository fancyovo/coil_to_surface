from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


COLORS = {
    "initial": "#4C78A8",
    "best": "#E45756",
    "threshold": "#3A7D44",
    "accent": "#B279A2",
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in (
            "case_id",
            "worker_index",
            "nfp",
            "n_base_coils",
            "best_iteration",
        ):
            row[key] = int(row[key])
        for key in (
            "initial_score",
            "final_score",
            "best_score",
            "best_gain",
            "trajectory_wall_s",
        ):
            row[key] = float(row[key])
    return rows


def wilson_interval(successes: int, count: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if count <= 0:
        return 0.0, 0.0
    rate = successes / count
    denominator = 1.0 + z * z / count
    center = (rate + z * z / (2.0 * count)) / denominator
    radius = z * math.sqrt(rate * (1.0 - rate) / count + z * z / (4.0 * count * count)) / denominator
    return center - radius, center + radius


def grouped_success(rows: list[dict[str, Any]], key: str) -> list[tuple[int, int, int]]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(int(row[key]), []).append(row)
    return [
        (group, len(values), sum(float(row["best_score"]) >= 50.0 for row in values))
        for group, values in sorted(groups.items())
    ]


def plot_score_distribution(
    rows: list[dict[str, Any]], output: Path, cohort_label: str = "Analytic prior"
) -> None:
    initial = np.asarray([row["initial_score"] for row in rows], dtype=float)
    best = np.asarray([row["best_score"] for row in rows], dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)

    bins = np.arange(0.0, 75.0, 5.0)
    axes[0].hist(initial, bins=bins, alpha=0.72, color=COLORS["initial"], label="Initial")
    axes[0].hist(best, bins=bins, alpha=0.58, color=COLORS["best"], label="Adam200 best")
    axes[0].axvline(20.0, color="#777777", linestyle=":", linewidth=1.2)
    axes[0].axvline(50.0, color=COLORS["threshold"], linestyle="--", linewidth=1.3)
    axes[0].set(
        xlabel="ABI-11 score",
        ylabel="Completed trajectories",
        title=f"{cohort_label}: score distribution before and after Adam200",
    )
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)

    ordered = np.sort(best)
    survival = (len(ordered) - np.arange(len(ordered))) / len(ordered)
    axes[1].step(ordered, survival, where="post", color=COLORS["best"], linewidth=2.0)
    axes[1].axvspan(0.0, 20.0, color="#D9D9D9", alpha=0.35)
    axes[1].axvspan(20.0, 50.0, color="#F2CF5B", alpha=0.18)
    axes[1].axvspan(50.0, 72.0, color="#59A14F", alpha=0.14)
    axes[1].axvline(50.0, color=COLORS["threshold"], linestyle="--", linewidth=1.3)
    success_count = int(np.sum(best >= 50.0))
    axes[1].annotate(
        f"{success_count} / {len(rows)} at score >= 50",
        xy=(50.0, success_count / len(rows)),
        xytext=(35.0, 0.58),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        fontsize=9,
    )
    axes[1].set(
        xlim=(0.0, 72.0),
        ylim=(0.0, 1.02),
        xlabel="Adam200 best ABI-11 score",
        ylabel="Fraction with score at least x",
        title="Best-score survival curve",
    )
    axes[1].grid(alpha=0.2)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def plot_success_rates(
    rows: list[dict[str, Any]], output: Path, cohort_label: str = "Analytic prior"
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    for axis, key, title in (
        (axes[0], "n_base_coils", "Conditional success by base-coil count"),
        (axes[1], "nfp", "Conditional success by field period"),
    ):
        groups = grouped_success(rows, key)
        x = np.arange(len(groups))
        rates = np.asarray([successes / count for _, count, successes in groups])
        intervals = [wilson_interval(successes, count) for _, count, successes in groups]
        lower = np.maximum(
            0.0, rates - np.asarray([interval[0] for interval in intervals])
        )
        upper = np.maximum(
            0.0, np.asarray([interval[1] for interval in intervals]) - rates
        )
        axis.errorbar(
            x,
            rates,
            yerr=np.vstack((lower, upper)),
            fmt="o",
            markersize=7,
            capsize=4,
            color=COLORS["accent"],
            ecolor="#666666",
            linewidth=1.5,
        )
        for index, (_, count, successes) in enumerate(groups):
            axis.annotate(
                f"{successes}/{count}",
                (index, rates[index]),
                xytext=(0, 11),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        axis.set_xticks(x, [str(group) for group, _, _ in groups])
        axis.set_ylim(0.0, 1.05)
        axis.set(
            xlabel="nc" if key == "n_base_coils" else "nfp",
            ylabel="Fraction with Adam200 best >= 50",
            title=title,
        )
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(cohort_label)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def plot_runtime_and_convergence(
    rows: list[dict[str, Any]], output: Path, cohort_label: str = "Analytic prior"
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    nc_values = sorted({int(row["n_base_coils"]) for row in rows})
    runtime_groups = [
        np.asarray(
            [row["trajectory_wall_s"] / 60.0 for row in rows if row["n_base_coils"] == nc],
            dtype=float,
        )
        for nc in nc_values
    ]
    axes[0].boxplot(runtime_groups, tick_labels=[str(value) for value in nc_values], showfliers=True)
    axes[0].set(
        xlabel="nc",
        ylabel="Trajectory wall time (minutes)",
        title="Adam200 cost by base-coil count",
    )
    axes[0].grid(axis="y", alpha=0.2)

    scatter = axes[1].scatter(
        [row["best_iteration"] for row in rows],
        [row["best_score"] for row in rows],
        c=[row["n_base_coils"] for row in rows],
        cmap="viridis",
        s=34,
        alpha=0.82,
    )
    axes[1].axhline(50.0, color=COLORS["threshold"], linestyle="--", linewidth=1.3)
    axes[1].set(
        xlim=(0, 205),
        xlabel="Update at best score",
        ylabel="Adam200 best ABI-11 score",
        title="When each trajectory reached its best score",
    )
    axes[1].grid(alpha=0.2)
    colorbar = figure.colorbar(scatter, ax=axes[1], pad=0.02)
    colorbar.set_label("nc")
    figure.suptitle(cohort_label)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def load_curves(run_root: Path) -> list[dict[str, Any]]:
    curves: list[dict[str, Any]] = []
    for manifest_path in sorted(
        (run_root / "trajectories").glob("*/trajectory_manifest.json")
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        history_path = manifest_path.parent / "optimization" / "history.jsonl"
        history = [
            json.loads(line)
            for line in history_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        curves.append(
            {
                "trajectory_id": manifest["trajectory_id"],
                "nc": int(manifest["case"]["n_base_coils"]),
                "steps": np.asarray(
                    [0] + [int(item["iteration"]) for item in history], dtype=int
                ),
                "scores": np.asarray(
                    [float(manifest["optimization"]["initial_score"])]
                    + [float(item["current_score"]) for item in history],
                    dtype=float,
                ),
            }
        )
    return curves


def plot_trajectory_curves(
    curves: list[dict[str, Any]], output: Path, cohort_label: str = "Analytic prior"
) -> None:
    palette = {1: "#4C78A8", 2: "#F28E2B", 3: "#59A14F", 4: "#B279A2"}
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    for curve in curves:
        axes[0].plot(
            curve["steps"],
            curve["scores"],
            color=palette[curve["nc"]],
            alpha=0.28,
            linewidth=0.8,
        )
    axes[0].axhline(50.0, color=COLORS["threshold"], linestyle="--", linewidth=1.3)
    axes[0].set(
        xlim=(0, 200),
        xlabel="Adam update",
        ylabel="ABI-11 score",
        title=f"{cohort_label}: all {len(curves)} completed Adam200 trajectories",
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend(
        handles=[
            Line2D([0], [0], color=palette[nc], linewidth=2.0, label=f"nc={nc}")
            for nc in sorted(palette)
        ],
        frameon=False,
        loc="upper left",
        ncols=2,
    )

    for nc in sorted(palette):
        group = [curve for curve in curves if curve["nc"] == nc]
        first_crossings: list[int] = []
        for curve in group:
            indices = np.flatnonzero(curve["scores"] >= 50.0)
            if len(indices):
                first_crossings.append(int(curve["steps"][indices[0]]))
        steps = np.arange(0, 201)
        reached = np.asarray(
            [sum(crossing <= step for crossing in first_crossings) / len(group) for step in steps]
        )
        axes[1].step(
            steps,
            reached,
            where="post",
            color=palette[nc],
            linewidth=2.0,
            label=f"nc={nc}: {len(first_crossings)}/{len(group)}",
        )
    axes[1].set(
        xlim=(0, 200),
        ylim=(0.0, 1.02),
        xlabel="Adam update",
        ylabel="Fraction first reaching score >= 50",
        title="First-passage timing for the 50-point threshold",
    )
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, loc="upper left")
    figure.savefig(output, dpi=200)
    plt.close(figure)


def first_passage_summary(curves: list[dict[str, Any]]) -> dict[str, Any]:
    crossings: list[dict[str, int]] = []
    for curve in curves:
        indices = np.flatnonzero(curve["scores"] >= 50.0)
        if len(indices):
            crossings.append(
                {
                    "trajectory_id": str(curve["trajectory_id"]),
                    "nc": int(curve["nc"]),
                    "step": int(curve["steps"][indices[0]]),
                }
            )
    checkpoints = (25, 50, 100, 150, 200)
    return {
        "threshold": 50.0,
        "completed_trajectories": len(curves),
        "crossing_trajectories": len(crossings),
        "crossings": sorted(crossings, key=lambda item: (item["step"], item["trajectory_id"])),
        "crossed_by_step": {
            str(step): sum(item["step"] <= step for item in crossings)
            for step in checkpoints
        },
        "current_score_ge_threshold_at_step": {
            str(step): sum(
                float(curve["scores"][np.flatnonzero(curve["steps"] == step)[0]])
                >= 50.0
                for curve in curves
            )
            for step in checkpoints
        },
        "by_nc": {
            str(nc): {
                "count": sum(curve["nc"] == nc for curve in curves),
                "crossing_count": sum(item["nc"] == nc for item in crossings),
                "crossed_by_step": {
                    str(step): sum(
                        item["nc"] == nc and item["step"] <= step
                        for item in crossings
                    )
                    for step in checkpoints
                },
            }
            for nc in sorted({int(curve["nc"]) for curve in curves})
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render analytic-prior Adam200 report figures.")
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--cohort-label", default="Analytic prior")
    args = parser.parse_args()
    rows = load_rows(args.asset_dir / "trajectories.csv")
    summary = json.loads((args.asset_dir / "summary.json").read_text(encoding="utf-8"))
    if len(rows) != int(summary["completed_count"]):
        raise RuntimeError("trajectory CSV count does not match frozen summary")
    plot_score_distribution(
        rows, args.asset_dir / "score_distribution.png", args.cohort_label
    )
    plot_success_rates(rows, args.asset_dir / "success_rates.png", args.cohort_label)
    plot_runtime_and_convergence(
        rows, args.asset_dir / "runtime_and_convergence.png", args.cohort_label
    )
    if args.run_root is not None:
        curves = load_curves(args.run_root)
        if len(curves) != len(rows):
            raise RuntimeError("trajectory history count does not match frozen summary")
        plot_trajectory_curves(
            curves, args.asset_dir / "trajectory_curves.png", args.cohort_label
        )
        (args.asset_dir / "first_passage_summary.json").write_text(
            json.dumps(first_passage_summary(curves), indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
