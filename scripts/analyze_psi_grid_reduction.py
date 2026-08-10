from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GRID_SIZE = {
    "baseline": 80,
    "psi_grid72": 72,
    "psi_grid64": 64,
    "psi_grid56": 56,
    "psi_grid48": 48,
}
VARIANT_ORDER = tuple(GRID_SIZE)
ANGLE_KEYS = ("psi_angle_mean", "psi_angle_p95", "psi_angle_l2")
DOWNSTREAM_KEYS = (
    "surface_level",
    "surface_drift_relative_p95",
    "alpha_relative_l2",
    "alpha_normal_B_relative_l2",
    "iota_min",
    "qs_target_global_error_per_helicity",
    "qs_target_edge_error_per_helicity",
)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def rankdata(values: list[float]) -> np.ndarray:
    values_array = np.asarray(values, dtype=float)
    order = np.argsort(values_array, kind="mergesort")
    ranks = np.empty(len(values_array), dtype=float)
    ranks[order] = np.arange(len(values_array), dtype=float)
    return ranks


def spearman(left: list[float], right: list[float]) -> float:
    if len(left) < 3:
        return float("nan")
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def physical_point_count(grid: int) -> int:
    accepted = 0
    for ir in range(grid):
        dr = -1.0 + 2.0 * ir / max(grid - 1, 1)
        for iz in range(grid):
            dz = -1.0 + 2.0 * iz / max(grid - 1, 1)
            if dr * dr + dz * dz <= 1.0:
                accepted += 1
    return accepted * grid


def median_by_case(rows: list[dict]) -> dict[int, dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["case_id"])].append(row)
    result = {}
    for case_id, case_rows in grouped.items():
        ok_rows = [row for row in case_rows if row["result"]["status"] == "ok"]
        if not ok_rows:
            continue
        first = ok_rows[0]
        result[case_id] = {
            "score": float(np.median([row["result"]["score"] for row in ok_rows])),
            "components": {
                key: float(np.median([row["result"]["components"][key] for row in ok_rows]))
                for key in first["result"]["components"]
            },
            "diagnostics": {
                key: float(np.median([row["result"]["diagnostics"][key] for row in ok_rows]))
                for key in (*ANGLE_KEYS, "psi_train_rms", *DOWNSTREAM_KEYS)
            },
        }
    return result


def finite_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0.0:
        return float("nan")
    return numerator / denominator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    missing = [variant for variant in VARIANT_ORDER if variant not in grouped]
    if missing:
        raise ValueError(f"missing variants: {missing}")

    per_case = {variant: median_by_case(grouped[variant]) for variant in VARIANT_ORDER}
    baseline = per_case["baseline"]
    baseline_wall = percentile([float(row["caller_wall_s"]) for row in grouped["baseline"]], 50)
    baseline_psi = percentile(
        [float(row["result"]["timing"]["psi_fit_s"]) for row in grouped["baseline"]], 50
    )
    summary = {}
    csv_rows = []
    for variant in VARIANT_ORDER:
        variant_rows = grouped[variant]
        cases = per_case[variant]
        common = sorted(set(baseline) & set(cases))
        baseline_scores = [baseline[case_id]["score"] for case_id in common]
        candidate_scores = [cases[case_id]["score"] for case_id in common]
        score_deltas = [candidate - base for candidate, base in zip(candidate_scores, baseline_scores)]
        grid = GRID_SIZE[variant]
        point_count = physical_point_count(grid)
        caller_walls = [float(row["caller_wall_s"]) for row in variant_rows]
        psi_fit = [float(row["result"]["timing"]["psi_fit_s"]) for row in variant_rows]
        psi_points = [float(row["result"]["timing"]["psi_points_s"]) for row in variant_rows]
        angle_ratios = {
            key: [
                finite_ratio(cases[case_id]["diagnostics"][key], baseline[case_id]["diagnostics"][key])
                for case_id in common
            ]
            for key in ANGLE_KEYS
        }
        component_max_abs_delta = {
            key: max(
                abs(cases[case_id]["components"][key] - baseline[case_id]["components"][key])
                for case_id in common
            )
            for key in baseline[common[0]]["components"]
        }
        downstream_max_abs_delta = {
            key: max(
                abs(cases[case_id]["diagnostics"][key] - baseline[case_id]["diagnostics"][key])
                for case_id in common
            )
            for key in DOWNSTREAM_KEYS
        }
        top_count = max(1, math.ceil(0.1 * len(common)))
        baseline_top = set(sorted(common, key=lambda case_id: baseline[case_id]["score"], reverse=True)[:top_count])
        candidate_top = set(sorted(common, key=lambda case_id: cases[case_id]["score"], reverse=True)[:top_count])
        ok_calls = sum(row["result"]["status"] == "ok" for row in variant_rows)
        record = {
            "grid": grid,
            "physical_point_count": point_count,
            "augmented_row_count": point_count + 1574,
            "calls": len(variant_rows),
            "ok_call_fraction": ok_calls / len(variant_rows),
            "paired_ok_cases": len(common),
            "caller_wall_p50_s": percentile(caller_walls, 50),
            "caller_wall_p95_s": percentile(caller_walls, 95),
            "caller_wall_max_s": max(caller_walls),
            "psi_fit_p50_s": percentile(psi_fit, 50),
            "psi_fit_p95_s": percentile(psi_fit, 95),
            "psi_points_p50_s": percentile(psi_points, 50),
            "total_speedup_vs_grid80": baseline_wall / percentile(caller_walls, 50),
            "psi_fit_speedup_vs_grid80": baseline_psi / percentile(psi_fit, 50),
            "score_delta_median": float(np.median(score_deltas)),
            "score_delta_p95_abs": percentile(list(map(abs, score_deltas)), 95),
            "score_delta_max_abs": max(map(abs, score_deltas)),
            "score_spearman": spearman(baseline_scores, candidate_scores),
            "top_decile_overlap": len(baseline_top & candidate_top) / top_count,
            "psi_train_rms_ratio_median": float(np.median([
                finite_ratio(cases[case_id]["diagnostics"]["psi_train_rms"], baseline[case_id]["diagnostics"]["psi_train_rms"])
                for case_id in common
            ])),
            "angle_ratio_median": {
                key: float(np.median(values)) for key, values in angle_ratios.items()
            },
            "angle_ratio_p95": {
                key: percentile(values, 95) for key, values in angle_ratios.items()
            },
            "component_max_abs_delta": component_max_abs_delta,
            "downstream_max_abs_delta": downstream_max_abs_delta,
        }
        summary[variant] = record
        csv_rows.append({
            "variant": variant,
            **{key: value for key, value in record.items() if not isinstance(value, dict)},
            "angle_p95_ratio_median": record["angle_ratio_median"]["psi_angle_p95"],
            "angle_p95_ratio_p95": record["angle_ratio_p95"]["psi_angle_p95"],
            "angle_l2_ratio_median": record["angle_ratio_median"]["psi_angle_l2"],
            "angle_l2_ratio_p95": record["angle_ratio_p95"]["psi_angle_l2"],
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    grids = [GRID_SIZE[variant] for variant in VARIANT_ORDER]
    points = [summary[variant]["physical_point_count"] for variant in VARIANT_ORDER]
    total_p50 = [summary[variant]["caller_wall_p50_s"] for variant in VARIANT_ORDER]
    psi_p50 = [summary[variant]["psi_fit_p50_s"] for variant in VARIANT_ORDER]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(points, total_p50, "o-", label="full score call")
    ax.plot(points, psi_p50, "s-", label="psi fit")
    for point, wall, grid in zip(points, total_p50, grids):
        ax.annotate(f"{grid}^3", (point, wall), xytext=(4, 5), textcoords="offset points")
    ax.set_xlabel("Physical psi training rows")
    ax.set_ylabel("P50 wall time (s)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "runtime_vs_training_points.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    angle_median = [summary[variant]["angle_ratio_median"]["psi_angle_p95"] for variant in VARIANT_ORDER]
    angle_p95 = [summary[variant]["angle_ratio_p95"]["psi_angle_p95"] for variant in VARIANT_ORDER]
    ax.plot(total_p50, angle_median, "o-", label="median case ratio")
    ax.plot(total_p50, angle_p95, "s--", label="P95 case ratio")
    for wall, ratio, grid in zip(total_p50, angle_median, grids):
        ax.annotate(f"{grid}^3", (wall, ratio), xytext=(4, 5), textcoords="offset points")
    ax.axhline(1.0, color="black", linewidth=1, linestyle=":")
    ax.set_xlabel("Full score-call P50 (s)")
    ax.set_ylabel("Independent psi-angle P95 / grid80")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "physical_accuracy_vs_runtime.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    for variant in VARIANT_ORDER[1:]:
        common = sorted(set(baseline) & set(per_case[variant]))
        ax.scatter(
            [baseline[case_id]["score"] for case_id in common],
            [per_case[variant][case_id]["score"] for case_id in common],
            s=22,
            alpha=0.7,
            label=f"{GRID_SIZE[variant]}^3",
        )
    all_scores = [entry["score"] for entry in baseline.values()]
    low, high = min(all_scores) - 1.0, max(all_scores) + 1.0
    ax.plot([low, high], [low, high], color="black", linewidth=1, linestyle="--")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Grid80 score")
    ax.set_ylabel("Reduced-grid score")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "score_preservation.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
