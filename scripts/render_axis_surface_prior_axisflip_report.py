from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


COLORS = {
    1: "#4C78A8",
    2: "#F28E2B",
    3: "#59A14F",
    4: "#B279A2",
}
THRESHOLD_COLOR = "#8C2D2D"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    integer_fields = ("case_id", "worker_index", "nfp", "n_base_coils", "best_iteration")
    float_fields = ("initial_score", "best_score", "final_score", "trajectory_wall_s")
    for row in rows:
        for field in integer_fields:
            row[field] = int(row[field])
        for field in float_fields:
            row[field] = float(row[field])
        row["gain"] = float(row["best_score"]) - float(row["initial_score"])
    return rows


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "p90": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def wilson(successes: int, count: int) -> tuple[float, float]:
    if count == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    rate = successes / count
    denominator = 1.0 + z * z / count
    center = (rate + z * z / (2.0 * count)) / denominator
    radius = z * math.sqrt(
        rate * (1.0 - rate) / count + z * z / (4.0 * count * count)
    ) / denominator
    return center - radius, center + radius


def load_curves(run_root: Path) -> list[dict[str, Any]]:
    curves: list[dict[str, Any]] = []
    for manifest_path in sorted((run_root / "trajectories").glob("*/trajectory_manifest.json")):
        manifest = read_json(manifest_path)
        history_path = manifest_path.parent / "optimization" / "history.jsonl"
        history = [
            json.loads(line)
            for line in history_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        curves.append(
            {
                "trajectory_id": manifest["trajectory_id"],
                "nc": int(manifest["case"]["n_base_coils"]),
                "steps": np.asarray([0] + [int(row["iteration"]) for row in history]),
                "scores": np.asarray(
                    [float(manifest["optimization"]["initial_score"])]
                    + [float(row["current_score"]) for row in history]
                ),
            }
        )
    return curves


def load_component_endpoints(run_root: Path) -> list[dict[str, Any]]:
    endpoints: list[dict[str, Any]] = []
    for manifest_path in sorted((run_root / "trajectories").glob("*/trajectory_manifest.json")):
        manifest = read_json(manifest_path)
        start = read_json(manifest_path.parent / "start.json")
        best = read_json(manifest_path.parent / "optimization" / "best.json")
        initial_native = start["data_prior_screening"]["native_score"]
        best_native = best["original_space_local_gradient_adam"]["native_score"]
        initial = initial_native["components"]
        optimized = best_native["components"]
        endpoints.append(
            {
                "trajectory_id": manifest["trajectory_id"],
                "case_id": int(manifest["case"]["case_id"]),
                "nfp": int(manifest["case"]["nfp"]),
                "n_base_coils": int(manifest["case"]["n_base_coils"]),
                "initial_total": float(initial_native["score"]),
                "initial_volume_qs": float(initial["volume_qs"]),
                "initial_coil": float(initial["coil"]),
                "best_total": float(best_native["score"]),
                "best_volume_qs": float(optimized["volume_qs"]),
                "best_coil": float(optimized["coil"]),
            }
        )
    for row in endpoints:
        row["delta_volume_qs"] = row["best_volume_qs"] - row["initial_volume_qs"]
        row["delta_coil"] = row["best_coil"] - row["initial_coil"]
    return endpoints


def first_passage(curves: list[dict[str, Any]], threshold: float = 50.0) -> dict[str, Any]:
    crossings: dict[str, int | None] = {}
    for curve in curves:
        indices = np.flatnonzero(curve["scores"] >= threshold)
        crossings[str(curve["trajectory_id"])] = (
            int(curve["steps"][indices[0]]) if len(indices) else None
        )
    checkpoints = (25, 50, 100, 150, 200)
    return {
        "threshold": threshold,
        "by_trajectory": crossings,
        "crossed_by_step": {
            str(step): sum(value is not None and value <= step for value in crossings.values())
            for step in checkpoints
        },
    }


def plot_trajectories(curves: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
    for curve in curves:
        color = COLORS[int(curve["nc"])]
        axes[0].plot(curve["steps"], curve["scores"], color=color, alpha=0.28, linewidth=0.9)
        axes[1].plot(
            curve["steps"],
            np.maximum.accumulate(curve["scores"]),
            color=color,
            alpha=0.28,
            linewidth=0.9,
        )
    for axis in axes:
        axis.axhline(50.0, color=THRESHOLD_COLOR, linestyle="--", linewidth=1.2)
        axis.set(xlim=(0, 200), xlabel="Adam update", ylabel="ABI-11 score")
        axis.grid(alpha=0.2)
    axes[0].set_title(f"Current score across {len(curves)} completed trajectories")
    axes[1].set_title("Best score reached by each update")
    axes[0].legend(
        handles=[Line2D([0], [0], color=COLORS[nc], label=f"nc={nc}") for nc in COLORS],
        frameon=False,
        ncols=2,
        loc="lower right",
    )
    figure.savefig(output, dpi=200)
    plt.close(figure)


def plot_score_outcomes(rows: list[dict[str, Any]], output: Path) -> None:
    initial = np.asarray([float(row["initial_score"]) for row in rows])
    best = np.asarray([float(row["best_score"]) for row in rows])
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
    upper = 5.0 * math.ceil(max(85.0, float(np.max(best)) + 2.0) / 5.0)
    bins = np.arange(0.0, upper + 5.0, 5.0)
    axes[0].hist(initial, bins=bins, alpha=0.70, color="#4C78A8", label="Initial")
    axes[0].hist(best, bins=bins, alpha=0.55, color="#E45756", label="Adam200 best")
    axes[0].axvline(50.0, color=THRESHOLD_COLOR, linestyle="--", linewidth=1.2)
    axes[0].set(xlabel="ABI-11 score", ylabel="Completed trajectories", title="Initial and best-score distributions")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)
    for nc in COLORS:
        group = [row for row in rows if int(row["n_base_coils"]) == nc]
        axes[1].scatter(
            [float(row["initial_score"]) for row in group],
            [float(row["best_score"]) for row in group],
            color=COLORS[nc],
            label=f"nc={nc}",
            alpha=0.76,
            s=30,
        )
    axes[1].axhline(50.0, color=THRESHOLD_COLOR, linestyle="--", linewidth=1.2)
    axes[1].set(xlabel="Initial ABI-11 score", ylabel="Adam200 best ABI-11 score", title="Optimization outcome from each valid start")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, ncols=2)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def plot_group_rates(summary: dict[str, Any], rows: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
    screen_groups = summary["by_n_base_coils_screening"]
    nc_values = sorted(int(value) for value in screen_groups)
    valid_rates = [float(screen_groups[str(nc)]["valid_rate"]) for nc in nc_values]
    axes[0].bar([str(value) for value in nc_values], valid_rates, color="#4C78A8", alpha=0.82)
    for index, nc in enumerate(nc_values):
        group = screen_groups[str(nc)]
        axes[0].text(index, valid_rates[index] + 0.012, f"{group['valid_count']}/{group['count']}", ha="center", fontsize=9)
    axes[0].set(ylim=(0.0, max(0.35, max(valid_rates) + 0.08)), xlabel="nc", ylabel="Initial valid fraction", title="Formal-config screening validity")
    axes[0].grid(axis="y", alpha=0.2)

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["n_base_coils"])].append(row)
    rates = []
    errors = []
    labels = []
    for nc in sorted(grouped):
        values = grouped[nc]
        successes = sum(float(row["best_score"]) >= 50.0 for row in values)
        rate = successes / len(values)
        low, high = wilson(successes, len(values))
        rates.append(rate)
        errors.append((max(0.0, rate - low), max(0.0, high - rate)))
        labels.append(f"{successes}/{len(values)}")
    positions = np.arange(len(rates))
    axes[1].errorbar(
        positions,
        rates,
        yerr=np.asarray(errors).T,
        fmt="o",
        color="#B279A2",
        ecolor="#666666",
        capsize=4,
    )
    for index, label in enumerate(labels):
        axes[1].text(index, rates[index] + 0.07, label, ha="center", fontsize=9)
    axes[1].set_xticks(positions, [str(value) for value in sorted(grouped)])
    axes[1].set(ylim=(0.0, 1.1), xlabel="nc", ylabel="Fraction with Adam200 best >= 50", title="Conditional optimization success")
    axes[1].grid(axis="y", alpha=0.2)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def plot_component_tradeoff(endpoints: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), constrained_layout=True)
    box_values = [
        [row["initial_volume_qs"] for row in endpoints],
        [row["best_volume_qs"] for row in endpoints],
        [row["initial_coil"] for row in endpoints],
        [row["best_coil"] for row in endpoints],
    ]
    boxes = axes[0].boxplot(
        box_values,
        tick_labels=["QS initial", "QS best", "Coil initial", "Coil best"],
        patch_artist=True,
        showfliers=True,
    )
    for patch, color in zip(boxes["boxes"], ("#4C78A8", "#E45756", "#4C78A8", "#E45756")):
        patch.set_facecolor(color)
        patch.set_alpha(0.58)
    axes[0].set(ylabel="ABI-11 component score", title="Endpoint component distributions")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.2)

    for row in endpoints:
        color = COLORS[int(row["n_base_coils"])]
        axes[1].plot(
            [row["initial_volume_qs"], row["best_volume_qs"]],
            [row["initial_coil"], row["best_coil"]],
            color=color,
            alpha=0.22,
            linewidth=0.8,
        )
        axes[1].scatter(row["initial_volume_qs"], row["initial_coil"], marker="x", color=color, alpha=0.62, s=22)
        axes[1].scatter(row["best_volume_qs"], row["best_coil"], marker="o", color=color, alpha=0.72, s=24)
    axes[1].set(
        xlabel="Volume-QS component score",
        ylabel="Coil-engineering component score",
        title="Paired movement from initial (x) to best (o)",
    )
    axes[1].grid(alpha=0.2)
    axes[1].legend(
        handles=[Line2D([0], [0], marker="o", color=COLORS[nc], linestyle="", label=f"nc={nc}") for nc in COLORS],
        frameon=False,
        ncols=2,
        loc="lower left",
    )

    for nc in COLORS:
        group = [row for row in endpoints if int(row["n_base_coils"]) == nc]
        axes[2].scatter(
            [row["delta_volume_qs"] for row in group],
            [row["delta_coil"] for row in group],
            color=COLORS[nc],
            label=f"nc={nc}",
            alpha=0.76,
            s=30,
        )
    axes[2].axhline(0.0, color="#777777", linewidth=1.0)
    axes[2].axvline(0.0, color="#777777", linewidth=1.0)
    axes[2].set(
        xlabel="Change in volume-QS component",
        ylabel="Change in coil-engineering component",
        title="Joint component changes under Adam200",
    )
    axes[2].grid(alpha=0.2)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(np.asarray(left, dtype=float), np.asarray(right, dtype=float))[0, 1])


def component_metrics(endpoints: list[dict[str, Any]]) -> dict[str, Any]:
    initial_qs = [float(row["initial_volume_qs"]) for row in endpoints]
    initial_coil = [float(row["initial_coil"]) for row in endpoints]
    best_qs = [float(row["best_volume_qs"]) for row in endpoints]
    best_coil = [float(row["best_coil"]) for row in endpoints]
    delta_qs = [float(row["delta_volume_qs"]) for row in endpoints]
    delta_coil = [float(row["delta_coil"]) for row in endpoints]
    quadrants = {
        "both_improved": sum(dq >= 0.0 and dc >= 0.0 for dq, dc in zip(delta_qs, delta_coil)),
        "qs_improved_coil_declined": sum(dq >= 0.0 and dc < 0.0 for dq, dc in zip(delta_qs, delta_coil)),
        "qs_declined_coil_improved": sum(dq < 0.0 and dc >= 0.0 for dq, dc in zip(delta_qs, delta_coil)),
        "both_declined": sum(dq < 0.0 and dc < 0.0 for dq, dc in zip(delta_qs, delta_coil)),
    }
    by_outcome: dict[str, Any] = {}
    for label, predicate in (
        ("best_below_50", lambda row: row["best_total"] < 50.0),
        ("best_50_to_70", lambda row: 50.0 <= row["best_total"] < 70.0),
        ("best_at_least_70", lambda row: row["best_total"] >= 70.0),
    ):
        group = [row for row in endpoints if predicate(row)]
        by_outcome[label] = {
            "count": len(group),
            "best_volume_qs": distribution([float(row["best_volume_qs"]) for row in group]),
            "best_coil": distribution([float(row["best_coil"]) for row in group]),
        }
    return {
        "initial_volume_qs": distribution(initial_qs),
        "initial_coil": distribution(initial_coil),
        "best_volume_qs": distribution(best_qs),
        "best_coil": distribution(best_coil),
        "delta_volume_qs": distribution(delta_qs),
        "delta_coil": distribution(delta_coil),
        "pearson_volume_qs_vs_coil": {
            "initial": pearson(initial_qs, initial_coil),
            "best": pearson(best_qs, best_coil),
            "change": pearson(delta_qs, delta_coil),
        },
        "pearson_total_vs_component": {
            "initial_total_vs_volume_qs": pearson(
                [float(row["initial_total"]) for row in endpoints], initial_qs
            ),
            "initial_total_vs_coil": pearson(
                [float(row["initial_total"]) for row in endpoints], initial_coil
            ),
            "best_total_vs_volume_qs": pearson(
                [float(row["best_total"]) for row in endpoints], best_qs
            ),
            "best_total_vs_coil": pearson(
                [float(row["best_total"]) for row in endpoints], best_coil
            ),
        },
        "joint_change_counts": quadrants,
        "by_best_total_outcome": by_outcome,
    }


def build_metrics(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    curves: list[dict[str, Any]],
    endpoints: list[dict[str, Any]],
) -> dict[str, Any]:
    by_nc: dict[str, Any] = {}
    for nc in sorted({int(row["n_base_coils"]) for row in rows}):
        group = [row for row in rows if int(row["n_base_coils"]) == nc]
        component_group = [
            row for row in endpoints if int(row["n_base_coils"]) == nc
        ]
        screening = summary["by_n_base_coils_screening"][str(nc)]
        by_nc[str(nc)] = {
            "screened": screening["count"],
            "valid": screening["valid_count"],
            "valid_rate": screening["valid_rate"],
            "completed_adam200": len(group),
            "best_ge_50": sum(float(row["best_score"]) >= 50.0 for row in group),
            "best_ge_70": sum(float(row["best_score"]) >= 70.0 for row in group),
            "best_score": distribution([float(row["best_score"]) for row in group]),
            "best_volume_qs": distribution(
                [float(row["best_volume_qs"]) for row in component_group]
            ),
            "best_coil": distribution(
                [float(row["best_coil"]) for row in component_group]
            ),
        }
    threshold_rates = {}
    for threshold in (50.0, 60.0, 70.0, 80.0):
        successes = sum(float(row["best_score"]) >= threshold for row in rows)
        low, high = wilson(successes, len(rows))
        threshold_rates[str(threshold)] = {
            "count": successes,
            "rate": successes / len(rows),
            "wilson_95": [low, high],
        }
    valid_low, valid_high = wilson(
        int(summary["screening"]["valid_count"]), int(summary["screening"]["count"])
    )
    return {
        "screening": summary["screening"],
        "screening_valid_wilson_95": [valid_low, valid_high],
        "accounting": summary["accounting"],
        "initial_score": distribution([float(row["initial_score"]) for row in rows]),
        "best_score": distribution([float(row["best_score"]) for row in rows]),
        "gain": distribution([float(row["gain"]) for row in rows]),
        "threshold_counts": {
            str(threshold): sum(float(row["best_score"]) >= threshold for row in rows)
            for threshold in (50.0, 60.0, 70.0, 80.0)
        },
        "threshold_rates": threshold_rates,
        "endpoint_iota_sign_counts": {
            stage: dict(sorted({sign: sum(row[f"{stage}_iota_sign"] == sign for row in rows) for sign in {row[f"{stage}_iota_sign"] for row in rows}}.items()))
            for stage in ("initial", "best", "final")
        },
        "score_components": component_metrics(endpoints),
        "first_passage_50": first_passage(curves),
        "by_nc": by_nc,
        "top10": sorted(rows, key=lambda row: float(row["best_score"]), reverse=True)[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the axis-flip v4 report evidence.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    args = parser.parse_args()
    args.asset_dir.mkdir(parents=True, exist_ok=True)
    summary = read_json(args.run_root / "analysis" / "summary.json")
    rows = read_rows(args.run_root / "analysis" / "trajectories.csv")
    curves = load_curves(args.run_root)
    endpoints = load_component_endpoints(args.run_root)
    if len(rows) != len(curves) or len(rows) != len(endpoints) or len(rows) != int(summary["adam200"]["count"]):
        raise RuntimeError("trajectory table, histories, components, and summary have different counts")
    if any(len(curve["steps"]) != 201 for curve in curves):
        raise RuntimeError("a completed trajectory does not contain 200 Adam updates")
    plot_trajectories(curves, args.asset_dir / "adam200_trajectories.png")
    plot_score_outcomes(rows, args.asset_dir / "score_outcomes.png")
    plot_group_rates(summary, rows, args.asset_dir / "group_rates.png")
    plot_component_tradeoff(endpoints, args.asset_dir / "component_tradeoff.png")
    with (args.asset_dir / "component_endpoints.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(endpoints[0]))
        writer.writeheader()
        writer.writerows(endpoints)
    (args.asset_dir / "report_metrics.json").write_text(
        json.dumps(build_metrics(summary, rows, curves, endpoints), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
