from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.axisflip_coil_scale import coil_score_decomposition
from scripts.native_score_runtime import write_json


def native_score(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload[key]["native_score"]
    if value.get("status") != "ok":
        raise ValueError(f"{key} native score is not ok")
    return value


def endpoint(native: dict[str, Any]) -> dict[str, float]:
    diagnostics = native["diagnostics"]
    pieces = coil_score_decomposition(native)
    return {
        "total": float(native["score"]),
        "volume_qs": float(native["components"]["volume_qs"]),
        "coil": float(native["components"]["coil"]),
        "length_mean_m": float(diagnostics["coil_length_mean"]),
        "effective_radius_m": float(diagnostics["coil_length_mean"]) / (2.0 * np.pi),
        "curvature_p95_per_m": float(diagnostics["coil_curvature_p95"]),
        "curvature_max_per_m": float(diagnostics["coil_curvature_max"]),
        "min_intercoil_distance_m": float(diagnostics["coil_min_intercoil_distance"]),
        "min_z_axis_distance_m": float(diagnostics["coil_min_axis_distance"]),
        **pieces,
    }


def paired_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze coil-scale drift in axis-flip v4 Adam200 trajectories.")
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.trajectory_root.glob("axisflip_case_*/optimization/best.json"))
    if len(paths) != 50:
        raise RuntimeError(f"expected exactly 50 complete trajectories, found {len(paths)}")
    rows: list[dict[str, Any]] = []
    for best_path in paths:
        case_dir = best_path.parents[1]
        start_path = case_dir / "start.json"
        best_payload = json.loads(best_path.read_text(encoding="utf-8"))
        start_payload = json.loads(start_path.read_text(encoding="utf-8"))
        start = endpoint(native_score(start_payload, "data_prior_screening"))
        best = endpoint(native_score(best_payload, "original_space_local_gradient_adam"))
        row: dict[str, Any] = {
            "trajectory_id": case_dir.name,
            "nfp": int(best_payload["nfp"]),
            "nc": len(best_payload["raw"]["current"]),
        }
        for prefix, values in (("initial", start), ("best", best)):
            row.update({f"{prefix}_{key}": value for key, value in values.items()})
        for key in start:
            row[f"delta_{key}"] = best[key] - start[key]
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "coil_growth_pairs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    tracked = [
        "effective_radius_m",
        "length_mean_m",
        "curvature_p95_per_m",
        "curvature_max_per_m",
        "min_intercoil_distance_m",
        "min_z_axis_distance_m",
        "points_curvature",
        "points_curvature_p95",
        "points_curvature_max",
        "points_distance",
        "points_spacing",
        "points_axis_distance",
        "points_other",
        "coil",
    ]
    summary: dict[str, Any] = {
        "format": "axisflip_v4_coil_growth_analysis_v1",
        "cohort": "50 complete original-space Adam200 trajectories",
        "trajectory_count": len(rows),
        "definitions": {
            "effective_radius_m": "mean base-coil arc length divided by 2*pi",
            "curvature_points": "20*q_down(curvature_p95;10,1.3)+12*q_down(curvature_max;35,1.2)",
            "distance_points": "20*q_up(min_intercoil;0.08,1.1)+12*q_up(min_z_axis;0.20,1.2)",
            "min_z_axis": "distance to the cylindrical symmetry axis used by the ABI-11 engineering score",
        },
        "metrics": {},
    }
    for key in tracked:
        initial = np.asarray([row[f"initial_{key}"] for row in rows])
        best = np.asarray([row[f"best_{key}"] for row in rows])
        delta = best - initial
        summary["metrics"][key] = {
            "initial": paired_summary(initial),
            "best": paired_summary(best),
            "delta": paired_summary(delta),
            "increase_count": int(np.count_nonzero(delta > 0.0)),
            "decrease_count": int(np.count_nonzero(delta < 0.0)),
        }
    radius_delta = np.asarray([row["delta_effective_radius_m"] for row in rows])
    spacing_delta = np.asarray([row["delta_min_intercoil_distance_m"] for row in rows])
    distance_delta = np.asarray([row["delta_points_distance"] for row in rows])
    curvature_delta = np.asarray([row["delta_points_curvature"] for row in rows])
    summary["correlations"] = {
        "delta_radius_vs_delta_min_intercoil": float(np.corrcoef(radius_delta, spacing_delta)[0, 1]),
        "delta_radius_vs_delta_distance_points": float(np.corrcoef(radius_delta, distance_delta)[0, 1]),
        "delta_radius_vs_delta_curvature_points": float(np.corrcoef(radius_delta, curvature_delta)[0, 1]),
    }
    summary["by_nc"] = {}
    for nc in sorted({int(row["nc"]) for row in rows}):
        group = [row for row in rows if int(row["nc"]) == nc]
        summary["by_nc"][str(nc)] = {
            "count": len(group),
            "delta_effective_radius_m": paired_summary(
                np.asarray([row["delta_effective_radius_m"] for row in group])
            ),
            "delta_min_intercoil_distance_m": paired_summary(
                np.asarray([row["delta_min_intercoil_distance_m"] for row in group])
            ),
            "delta_curvature_points": paired_summary(
                np.asarray([row["delta_points_curvature"] for row in group])
            ),
            "delta_distance_points": paired_summary(
                np.asarray([row["delta_points_distance"] for row in group])
            ),
            "spacing_decrease_count": int(
                sum(row["delta_min_intercoil_distance_m"] < 0.0 for row in group)
            ),
        }
    write_json(args.output_dir / "coil_growth_summary.json", summary)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    initial_radius = np.asarray([row["initial_effective_radius_m"] for row in rows])
    best_radius = np.asarray([row["best_effective_radius_m"] for row in rows])
    initial_spacing = np.asarray([row["initial_min_intercoil_distance_m"] for row in rows])
    best_spacing = np.asarray([row["best_min_intercoil_distance_m"] for row in rows])
    initial_curvature = np.asarray([row["initial_points_curvature"] for row in rows])
    best_curvature = np.asarray([row["best_points_curvature"] for row in rows])
    initial_distance = np.asarray([row["initial_points_distance"] for row in rows])
    best_distance = np.asarray([row["best_points_distance"] for row in rows])
    delta_coil = np.asarray([row["delta_coil"] for row in rows])

    figure, axes = plt.subplots(2, 2, figsize=(12.4, 9.6), constrained_layout=True)
    for axis, initial, best, label in (
        (axes[0, 0], initial_radius, best_radius, "Effective coil radius [m]"),
        (axes[0, 1], initial_spacing, best_spacing, "Minimum intercoil distance [m]"),
    ):
        limits = (min(initial.min(), best.min()), max(initial.max(), best.max()))
        margin = max((limits[1] - limits[0]) * 0.06, 1.0e-3)
        axis.scatter(initial, best, c="#176b87", alpha=0.78, s=34)
        axis.plot([limits[0] - margin, limits[1] + margin], [limits[0] - margin, limits[1] + margin], "--", color="#555555")
        axis.set(xlabel=f"Initial {label}", ylabel=f"Adam200 best {label}")
        axis.grid(alpha=0.22)

    axes[1, 0].scatter(initial_curvature, best_curvature, c="#b44b2a", alpha=0.78, s=34, label="curvature")
    axes[1, 0].scatter(initial_distance, best_distance, c="#27824a", alpha=0.78, s=34, label="distance")
    axes[1, 0].plot([0, 32], [0, 32], "--", color="#555555")
    axes[1, 0].set(xlabel="Initial weighted contribution [score points]", ylabel="Adam200 best weighted contribution [score points]")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.22)

    scatter = axes[1, 1].scatter(radius_delta, spacing_delta * 1000.0, c=delta_coil, cmap="coolwarm", s=42, alpha=0.85)
    axes[1, 1].axhline(0.0, color="#555555", linewidth=1)
    axes[1, 1].axvline(0.0, color="#555555", linewidth=1)
    axes[1, 1].set(xlabel="Change in effective radius [m]", ylabel="Change in minimum intercoil distance [mm]")
    axes[1, 1].grid(alpha=0.22)
    figure.colorbar(scatter, ax=axes[1, 1], label="Change in coil component [points]")
    figure.suptitle("Axis-flip v4: coil geometry drift over original-space Adam200")
    figure.savefig(args.output_dir / "coil_growth_population.png", dpi=185)
    plt.close(figure)
    print(json.dumps({"event": "complete", "rows": len(rows), "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
