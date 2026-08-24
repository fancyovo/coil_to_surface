from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


PREDICTORS = {
    "native_global_qh": "volume differential QH",
    "native_edge_shell_qh": "outer-shell differential QH",
    "equal_s_area_qh": "strict equal-s differential QH",
}
ROOT_RESIDUAL_TOLERANCE = 1.0e-8


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def nested(payload: Any, *keys: str) -> float:
    value = payload
    try:
        for key in keys:
            value = value[key]
        return float(value)
    except (KeyError, TypeError, ValueError):
        return float("nan")


def flatten_case(case_dir: Path, output_name: str) -> list[dict[str, Any]]:
    metadata = read_json(case_dir / "metadata.json")
    diagnostics = metadata.get("native_score", {}).get("diagnostics", {})
    validation_path = case_dir / "face_qs" / "validation_summary.json"
    equal_s_path = case_dir / "face_qs" / output_name
    validation = read_json(validation_path) if validation_path.is_file() else {}
    equal_s = read_json(equal_s_path) if equal_s_path.is_file() else {}
    validation_by_name = {row["name"]: row for row in validation.get("surfaces", [])}
    equal_s_by_name = {row["name"]: row for row in equal_s.get("surfaces", [])}
    rows = []
    for target in metadata["surface_targets"]:
        name = target["name"]
        surface = validation_by_name.get(name, {})
        strict = equal_s_by_name.get(name, {})
        metrics = strict.get("metrics", {})
        checks = surface.get("acceptance_checks", {})
        accepted = bool(surface.get("accepted", False))
        solved_regular = bool(
            checks.get("solver_converged", False)
            and checks.get("toroidal_winding", False)
            and checks.get("normal_nonzero", False)
            and nested(surface, "surface_qs_error", "QH_1_1") > 0.0
        )
        rows.append(
            {
                "case_id": metadata["case_id"],
                "trajectory_id": metadata["trajectory_id"],
                "iteration": int(metadata["iteration"]),
                "scheduled_snapshot": bool(metadata.get("scheduled_snapshot", True)),
                "score_best_snapshot": bool(metadata.get("score_best_snapshot", False)),
                "nfp": int(metadata["nfp"]),
                "n_base_coils": int(metadata["n_base_coils"]),
                "surface_name": name,
                "target_s": float(target["s_level"]),
                "accepted": accepted,
                "solved_regular": solved_regular,
                "native_global_qh": nested(diagnostics, "qs_target_global_error_per_helicity"),
                "native_edge_shell_qh": (
                    nested(diagnostics, "qs_target_edge_error_per_helicity")
                    if name == "adaptive_edge"
                    else float("nan")
                ),
                "equal_s_status": equal_s.get("status", "missing"),
                "equal_s_area_qh": nested(metrics, "per_helicity_area_rms"),
                "equal_s_raw_qh": nested(metrics, "raw_area_rms"),
                "equal_s_normal_rms": nested(metrics, "normal_B_sine_area_rms"),
                "equal_s_normal_p95": nested(metrics, "normal_B_sine_p95"),
                "equal_s_s_residual_rms": nested(metrics, "s_residual_rms"),
                "equal_s_root_residual_max": nested(metrics, "root_residual_max"),
                "equal_s_root_valid": bool(
                    math.isfinite(nested(metrics, "root_residual_max"))
                    and nested(metrics, "root_residual_max") <= ROOT_RESIDUAL_TOLERANCE
                ),
                "equal_s_area_ess": nested(metrics, "area_weight_effective_fraction"),
                "equal_s_wall_s": nested(strict, "wall_s"),
                "face_qh": nested(surface, "surface_qs_error", "QH_1_1"),
                "face_iota": nested(surface, "final_iota"),
                "equal_s_iota": nested(strict, "iota"),
            }
        )
    return rows


def valid_pairs(
    rows: list[dict[str, Any]],
    predictor: str,
    acceptance_key: str,
    *,
    require_equal_s_root: bool = True,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row[acceptance_key]
        and (
            predictor != "equal_s_area_qh"
            or not require_equal_s_root
            or row["equal_s_root_valid"]
        )
        and row[predictor] > 0.0
        and row["face_qh"] > 0.0
        and math.isfinite(row[predictor])
        and math.isfinite(row["face_qh"])
    ]


def correlation(
    rows: list[dict[str, Any]],
    predictor: str,
    *,
    acceptance_key: str,
    seed: int,
    require_equal_s_root: bool = True,
) -> dict[str, Any]:
    usable = valid_pairs(
        rows,
        predictor,
        acceptance_key,
        require_equal_s_root=require_equal_s_root,
    )
    if len(usable) < 3:
        return {"count": len(usable), "spearman": None, "log_pearson": None, "cluster_bootstrap_95": None}
    x = np.log10([row[predictor] for row in usable])
    y = np.log10([row["face_qh"] for row in usable])
    by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        by_trajectory[row["trajectory_id"]].append(row)
    identifiers = sorted(by_trajectory)
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(1000):
        selected = rng.choice(identifiers, size=len(identifiers), replace=True)
        sample = [row for identifier in selected for row in by_trajectory[identifier]]
        bx = np.log10([row[predictor] for row in sample])
        by = np.log10([row["face_qh"] for row in sample])
        if np.ptp(bx) > 0.0 and np.ptp(by) > 0.0:
            bootstrap.append(float(spearmanr(bx, by).statistic))
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    return {
        "count": len(usable),
        "trajectory_count": len(identifiers),
        "spearman": float(spearmanr(x, y).statistic),
        "log_pearson": float(pearsonr(x, y).statistic),
        "cluster_bootstrap_95": [float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))],
        "log_fit": {
            "slope": float(slope),
            "intercept": float(intercept),
            "residual_std_decades": float(np.std(residual)),
        },
    }


def distribution(values) -> dict[str, Any]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if not len(array):
        return {"count": 0, "p50": None, "p95": None, "max": None}
    return {
        "count": int(len(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "format": "qh_equal_s_face_qs_analysis_v1",
        "row_count": len(rows),
        "case_count": len({row["case_id"] for row in rows}),
        "equal_s_status_counts": {
            status: len({row["case_id"] for row in rows if row["equal_s_status"] == status})
            for status in sorted({row["equal_s_status"] for row in rows})
        },
        "surface_types": {},
    }
    for surface_index, surface_name in enumerate(("fixed_probe", "adaptive_edge")):
        subset = [row for row in rows if row["surface_name"] == surface_name]
        predictors = ["native_global_qh", "equal_s_area_qh"]
        if surface_name == "adaptive_edge":
            predictors.insert(1, "native_edge_shell_qh")
        summary["surface_types"][surface_name] = {
            "row_count": len(subset),
            "strict_accepted": sum(row["accepted"] for row in subset),
            "solved_regular": sum(row["solved_regular"] for row in subset),
            "strict_correlations": {
                predictor: correlation(subset, predictor, acceptance_key="accepted", seed=20260819 + 10 * surface_index + index)
                for index, predictor in enumerate(predictors)
            },
            "solved_regular_correlations": {
                predictor: correlation(subset, predictor, acceptance_key="solved_regular", seed=20260919 + 10 * surface_index + index)
                for index, predictor in enumerate(predictors)
            },
            "equal_s_root_valid": {
                "all": sum(row["equal_s_root_valid"] for row in subset),
                "strict_accepted": sum(row["accepted"] and row["equal_s_root_valid"] for row in subset),
                "solved_regular": sum(row["solved_regular"] and row["equal_s_root_valid"] for row in subset),
                "tolerance": ROOT_RESIDUAL_TOLERANCE,
            },
            "equal_s_unfiltered_strict_correlation": correlation(
                subset,
                "equal_s_area_qh",
                acceptance_key="accepted",
                seed=20261019 + surface_index,
                require_equal_s_root=False,
            ),
            "equal_s_quality_strata": {
                f"normal_rms_le_{threshold:.0e}": correlation(
                    [row for row in subset if row["equal_s_normal_rms"] <= threshold],
                    "equal_s_area_qh",
                    acceptance_key="accepted",
                    seed=20261119 + 10 * surface_index + index,
                )
                for index, threshold in enumerate((1.0e-4, 3.0e-5, 1.0e-5))
            },
            "equal_s_diagnostics": {
                "normal_B_sine_area_rms": distribution(row["equal_s_normal_rms"] for row in subset),
                "normal_B_sine_p95": distribution(row["equal_s_normal_p95"] for row in subset),
                "s_residual_rms": distribution(row["equal_s_s_residual_rms"] for row in subset),
                "root_residual_max": distribution(row["equal_s_root_residual_max"] for row in subset),
                "area_weight_effective_fraction": distribution(row["equal_s_area_ess"] for row in subset),
                "wall_s": distribution(row["equal_s_wall_s"] for row in subset),
            },
        }
    return summary


def plot_scatter(rows: list[dict[str, Any]], summary: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 10.2), constrained_layout=True)
    configurations = [
        ("fixed_probe", "native_global_qh"),
        ("fixed_probe", "equal_s_area_qh"),
        ("adaptive_edge", "native_global_qh"),
        ("adaptive_edge", "native_edge_shell_qh"),
        ("adaptive_edge", "equal_s_area_qh"),
    ]
    axis_positions = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]
    axes[0, 2].axis("off")
    scatter = None
    for (surface_name, predictor), position in zip(configurations, axis_positions, strict=True):
        axis = axes[position]
        subset = valid_pairs([row for row in rows if row["surface_name"] == surface_name], predictor, "accepted")
        if not subset:
            axis.text(0.5, 0.5, "No accepted pairs", ha="center", va="center", transform=axis.transAxes)
            continue
        x = np.asarray([row[predictor] for row in subset])
        y = np.asarray([row["face_qh"] for row in subset])
        iterations = np.asarray([row["iteration"] for row in subset])
        scatter = axis.scatter(x, y, c=iterations, cmap="viridis", s=13, alpha=0.62, linewidths=0)
        relation = summary["surface_types"][surface_name]["strict_correlations"][predictor]
        xline = np.geomspace(np.min(x), np.max(x), 200)
        fit = relation["log_fit"]
        axis.plot(xline, 10 ** (fit["intercept"] + fit["slope"] * np.log10(xline)), color="#b94e35", linewidth=1.8)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.18)
        axis.set(
            xlabel=PREDICTORS[predictor],
            ylabel="Simsopt face QH",
            title=f"{surface_name}: rho={relation['spearman']:.3f}, n={relation['count']}",
        )
    if scatter is not None:
        figure.colorbar(scatter, ax=axes, label="Adam iteration", shrink=0.8)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def plot_correlations(summary: dict[str, Any], path: Path) -> None:
    labels = []
    values = []
    lower = []
    upper = []
    colors = []
    palette = {"native_global_qh": "#557c8d", "native_edge_shell_qh": "#d18b2c", "equal_s_area_qh": "#7a4f8f"}
    for surface_name in ("fixed_probe", "adaptive_edge"):
        for predictor, result in summary["surface_types"][surface_name]["strict_correlations"].items():
            labels.append(f"{surface_name}\n{PREDICTORS[predictor]}")
            values.append(result["spearman"])
            interval = result["cluster_bootstrap_95"]
            lower.append(result["spearman"] - interval[0])
            upper.append(interval[1] - result["spearman"])
            colors.append(palette[predictor])
    figure, axis = plt.subplots(figsize=(12.5, 5.8), constrained_layout=True)
    positions = np.arange(len(labels))
    axis.bar(positions, values, color=colors, alpha=0.9)
    axis.errorbar(positions, values, yerr=np.asarray([lower, upper]), fmt="none", ecolor="#222222", capsize=5)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Spearman correlation with Simsopt face QH")
    axis.set_ylim(0.0, 1.02)
    axis.grid(True, axis="y", alpha=0.2)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare volume, shell, strict equal-s, and Simsopt face QH.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--equal-s-output-name", default="equal_s_qs_summary.json")
    args = parser.parse_args()
    rows = [
        row
        for case_dir in sorted(path for path in (args.experiment_root / "cases").iterdir() if path.is_dir())
        for row in flatten_case(case_dir, args.equal_s_output_name)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with (args.output_dir / "equal_s_surface_records.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    scheduled_rows = [row for row in rows if row["scheduled_snapshot"]]
    summary = build_summary(scheduled_rows)
    summary["all_row_count_including_score_best_extras"] = len(rows)
    summary["score_best_extra_row_count"] = sum(
        row["score_best_snapshot"] and not row["scheduled_snapshot"] for row in rows
    )
    write_json(args.output_dir / "equal_s_summary.json", summary)
    plot_scatter(scheduled_rows, summary, args.output_dir / "equal_s_vs_face_qh.png")
    plot_correlations(summary, args.output_dir / "equal_s_correlation_summary.png")
    print(json.dumps({"case_count": summary["case_count"], "row_count": summary["row_count"], "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
