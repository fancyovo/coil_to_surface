from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def q_down(value: float, scale: float, power: float, fallback: float = 0.0) -> float:
    if not math.isfinite(value) or scale <= 0.0:
        return fallback
    return 1.0 / (1.0 + (max(value, 0.0) / scale) ** power)


def q_up(value: float, scale: float, power: float, fallback: float = 0.0) -> float:
    if not math.isfinite(value) or value <= 0.0 or scale <= 0.0:
        return fallback
    return 1.0 / (1.0 + (scale / value) ** power)


def coil_decomposition(native: dict[str, Any]) -> dict[str, float]:
    diagnostics = native["diagnostics"]
    quality = {
        "length": q_down(float(diagnostics["coil_length_mean"]), 7.0, 1.4, 0.6),
        "curvature_p95": q_down(
            float(diagnostics["coil_curvature_p95"]), 25.0, 1.3, 0.5
        ),
        "curvature_max": q_down(
            float(diagnostics["coil_curvature_max"]), 35.0, 1.2, 0.5
        ),
        "spacing": q_up(
            float(diagnostics["coil_min_intercoil_distance"]), 0.08, 1.1, 0.45
        ),
        "axis_distance": q_up(
            float(diagnostics["coil_min_axis_distance"]), 0.20, 1.2, 0.45
        ),
        "high_mode": q_down(
            float(diagnostics["coil_high_mode_energy_fraction"]), 0.05, 1.0, 0.7
        ),
        "current": q_down(
            float(diagnostics["coil_current_abs_max_a"]), 2.0e6, 1.0, 0.7
        ),
    }
    weights = {
        "length": 0.16,
        "curvature_p95": 0.20,
        "curvature_max": 0.12,
        "spacing": 0.20,
        "axis_distance": 0.12,
        "high_mode": 0.13,
        "current": 0.07,
    }
    points = {name: 100.0 * weights[name] * value for name, value in quality.items()}
    total = sum(points.values())
    recorded = float(native["components"]["coil"])
    if abs(total - recorded) > 3.0e-8:
        raise ValueError(f"coil decomposition mismatch: {total} versus {recorded}")
    return {
        "curvature_points": points["curvature_p95"] + points["curvature_max"],
        "distance_points": points["spacing"] + points["axis_distance"],
        "other_points": points["length"] + points["high_mode"] + points["current"],
        "curvature_p95_points": points["curvature_p95"],
        "curvature_max_points": points["curvature_max"],
        "spacing_points": points["spacing"],
        "axis_distance_points": points["axis_distance"],
    }


def endpoint(native: dict[str, Any]) -> dict[str, float]:
    diagnostics = native["diagnostics"]
    return {
        "score": float(native["score"]),
        "volume_qs": float(native["components"]["volume_qs"]),
        "coil": float(native["components"]["coil"]),
        "length_mean_m": float(diagnostics["coil_length_mean"]),
        "effective_radius_m": float(diagnostics["coil_length_mean"]) / (2.0 * math.pi),
        "curvature_p95_per_m": float(diagnostics["coil_curvature_p95"]),
        "curvature_max_per_m": float(diagnostics["coil_curvature_max"]),
        "min_intercoil_distance_m": float(diagnostics["coil_min_intercoil_distance"]),
        "min_z_axis_distance_m": float(diagnostics["coil_min_axis_distance"]),
        "high_mode_fraction": float(diagnostics["coil_high_mode_energy_fraction"]),
        **coil_decomposition(native),
    }


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half_width = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [center - half_width, center + half_width]


def load_screening(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "screening").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.asset_root

    summary = json.loads((root / "analysis" / "summary.json").read_text(encoding="utf-8"))
    screening = load_screening(root)
    if len(screening) != 384:
        raise RuntimeError(f"expected 384 screening rows, found {len(screening)}")
    screening_by_case = {int(row["case_id"]): row for row in screening}

    rows: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, Any]]] = {}
    for case_dir in sorted((root / "trajectories").glob("axisflip_r012_case_*")):
        manifest = json.loads((case_dir / "trajectory_manifest.json").read_text(encoding="utf-8"))
        start_payload = json.loads((case_dir / "start.json").read_text(encoding="utf-8"))
        best_payload = json.loads((case_dir / "optimization" / "best.json").read_text(encoding="utf-8"))
        start_native = start_payload["data_prior_screening"]["native_score"]
        best_native = best_payload["original_space_local_gradient_adam"]["native_score"]
        start = endpoint(start_native)
        best = endpoint(best_native)
        case_id = int(manifest["case"]["case_id"])
        source = screening_by_case[case_id]
        generator = source["generator"]
        row: dict[str, Any] = {
            "trajectory_id": case_dir.name,
            "case_id": case_id,
            "worker_index": int(manifest["case"]["worker_index"]),
            "best_iteration": int(manifest["optimization"]["best_iteration"]),
            "prior_minor_radius_m": float(generator["parameters"]["minor_radius"]),
            "winding_section_mean_radius_m": float(generator["winding_section_mean_radius_m"]),
        }
        for prefix, values in (("initial", start), ("best", best)):
            row.update({f"{prefix}_{name}": value for name, value in values.items()})
        for name in start:
            row[f"delta_{name}"] = best[name] - start[name]
        rows.append(row)
        histories[case_dir.name] = [
            json.loads(line)
            for line in (case_dir / "optimization" / "history.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    if len(rows) != 12:
        raise RuntimeError(f"expected 12 trajectories, found {len(rows)}")

    metrics: dict[str, Any] = {
        "format": "axisflip_r012_curvature_r04_report_metrics_v1",
        "protocol_id": summary["protocol_id"],
        "screening": summary["screening"],
        "valid_rate_wilson_95": wilson(summary["screening"]["valid_count"], summary["screening"]["count"]),
        "adam200": summary["adam200"],
        "adam_ge_70_rate_wilson_95": wilson(summary["adam200"]["best_ge_70"], summary["adam200"]["count"]),
        "definitions": {
            "effective_radius_m": "mean base-coil arc length divided by 2*pi",
            "curvature_points": "20*q_down(p95;25,1.3)+12*q_down(max;35,1.2)",
            "distance_points": "20*q_up(intercoil;0.08,1.1)+12*q_up(z-axis;0.20,1.2)",
            "min_z_axis_distance_m": "distance to the cylindrical symmetry axis used by the engineering score",
        },
        "trajectory_rows": rows,
        "paired": {},
    }
    names = [
        "score",
        "volume_qs",
        "coil",
        "effective_radius_m",
        "length_mean_m",
        "curvature_p95_per_m",
        "curvature_max_per_m",
        "min_intercoil_distance_m",
        "curvature_points",
        "distance_points",
        "other_points",
    ]
    for name in names:
        initial = [float(row[f"initial_{name}"]) for row in rows]
        best = [float(row[f"best_{name}"]) for row in rows]
        delta = [float(row[f"delta_{name}"]) for row in rows]
        metrics["paired"][name] = {
            "initial": distribution(initial),
            "best": distribution(best),
            "delta": distribution(delta),
            "increase_count": sum(value > 0.0 for value in delta),
            "decrease_count": sum(value < 0.0 for value in delta),
        }
    metrics["prior"] = {
        "minor_radius_m": distribution([float(row["prior_minor_radius_m"]) for row in rows]),
        "winding_section_mean_radius_m": distribution(
            [float(row["winding_section_mean_radius_m"]) for row in rows]
        ),
    }
    (root / "report_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    with (root / "trajectory_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, len(rows)))
    figure, axis = plt.subplots(figsize=(11.8, 6.8), constrained_layout=True)
    for color, row in zip(colors, rows, strict=True):
        history = histories[row["trajectory_id"]]
        steps = [0] + [int(item["iteration"]) for item in history]
        scores = [float(row["initial_score"])] + [float(item["current_score"]) for item in history]
        label = f"case {row['case_id']} (best {row['best_score']:.2f})"
        linewidth = 2.4 if row["best_score"] < 50.0 or row["best_score"] == max(r["best_score"] for r in rows) else 1.25
        axis.plot(steps, scores, color=color, linewidth=linewidth, alpha=0.9, label=label)
    axis.axhline(50.0, color="#666666", linestyle="--", linewidth=1.1, label="50-point basin threshold")
    axis.axhline(70.0, color="#222222", linestyle=":", linewidth=1.1, label="70 points")
    axis.set(xlabel="Adam update", ylabel="Experimental native score", xlim=(0, 200))
    axis.set_title("R012 / curvature-R04: all 12 Adam200 trajectories")
    axis.grid(alpha=0.22)
    axis.legend(ncol=3, fontsize=8.2, loc="lower right")
    figure.savefig(root / "adam200_trajectories.png", dpi=190)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.5), constrained_layout=True)
    for row in rows:
        color = "#ba3b2e" if row["best_score"] < 50.0 else "#167d8d"
        axes[0].plot(
            [row["initial_volume_qs"], row["best_volume_qs"]],
            [row["initial_coil"], row["best_coil"]],
            color=color,
            alpha=0.58,
            linewidth=1.2,
        )
        axes[0].scatter(
            row["initial_volume_qs"], row["initial_coil"], facecolors="none", edgecolors=color, s=28
        )
        axes[0].scatter(row["best_volume_qs"], row["best_coil"], color=color, s=30)
    axes[0].set(xlabel="Volume-QS component", ylabel="Coil engineering component")
    axes[0].set_title("Start to best endpoint")
    axes[0].grid(alpha=0.22)

    status_counts = summary["screening"]["status_counts"]
    labels = ["ok", "no_surface", "no_axis", "flux_rejected"]
    values = [int(status_counts.get(label, 0)) for label in labels]
    axes[1].bar(labels, values, color=["#167d8d", "#d4a72c", "#ba3b2e", "#7868a6"])
    for index, value in enumerate(values):
        axes[1].text(index, value + 3, str(value), ha="center", va="bottom")
    axes[1].set(ylabel="Samples", ylim=(0, max(values) * 1.15))
    axes[1].set_title("Fixed 384-sample legality audit")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", alpha=0.22)
    figure.savefig(root / "legality_and_component_joint.png", dpi=190)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(12.4, 9.2), constrained_layout=True)
    plot_specs = [
        ("effective_radius_m", "Effective radius: mean length / 2pi [m]"),
        ("min_intercoil_distance_m", "Minimum intercoil distance [m]"),
        ("curvature_points", "Curvature contribution [points]"),
        ("distance_points", "Distance contribution [points]"),
    ]
    for axis, (name, label) in zip(axes.flat, plot_specs, strict=True):
        initial = np.asarray([float(row[f"initial_{name}"]) for row in rows])
        best = np.asarray([float(row[f"best_{name}"]) for row in rows])
        low = min(float(initial.min()), float(best.min()))
        high = max(float(initial.max()), float(best.max()))
        margin = max((high - low) * 0.08, 1.0e-3)
        point_colors = ["#ba3b2e" if row["best_score"] < 50.0 else "#167d8d" for row in rows]
        axis.scatter(initial, best, c=point_colors, s=42, alpha=0.88)
        axis.plot([low - margin, high + margin], [low - margin, high + margin], "--", color="#666666")
        axis.set(xlabel=f"Initial {label}", ylabel=f"Best {label}")
        axis.grid(alpha=0.22)
    figure.suptitle("Coil geometry and engineering-score changes over Adam200")
    figure.savefig(root / "coil_geometry_and_score_changes.png", dpi=190)
    plt.close(figure)

    print(json.dumps({"event": "complete", "screening": len(screening), "trajectories": len(rows)}))


if __name__ == "__main__":
    main()
