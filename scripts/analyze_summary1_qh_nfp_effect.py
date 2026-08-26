from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def load_rows(path: Path, surface_name: str) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    output = []
    for row in rows:
        native = float(row["native_qh"])
        face = float(row["face_qh"])
        if (
            row["surface_name"] == surface_name
            and parse_bool(row["accepted"])
            and math.isfinite(native)
            and math.isfinite(face)
            and native > 0.0
            and face > 0.0
        ):
            output.append(row)
    return output


def design_matrix(
    x: np.ndarray,
    nfp: np.ndarray,
    categories: np.ndarray,
    model: str,
) -> np.ndarray:
    helicity = np.log10(np.sqrt(1.0 + nfp.astype(float) ** 2))
    if model == "pooled":
        return np.column_stack([np.ones(len(x)), x])
    if model == "undo_helicity_normalization":
        return np.column_stack([np.ones(len(x)), x + helicity])
    if model == "continuous_nfp":
        return np.column_stack([np.ones(len(x)), x, helicity])
    if model == "nfp_intercepts":
        columns = [np.ones(len(x)), x]
        columns.extend((nfp == category).astype(float) for category in categories[1:])
        return np.column_stack(columns)
    if model == "nfp_separate_lines":
        columns = []
        for category in categories:
            selected = (nfp == category).astype(float)
            columns.extend([selected, selected * x])
        return np.column_stack(columns)
    raise ValueError(f"unknown model: {model}")


def trajectory_folds(rows: list[dict], fold_count: int = 3) -> np.ndarray:
    assignments: dict[str, int] = {}
    rng = np.random.default_rng(20260826)
    for nfp in sorted({int(row["nfp"]) for row in rows}):
        trajectories = sorted(
            {row["trajectory_id"] for row in rows if int(row["nfp"]) == nfp}
        )
        rng.shuffle(trajectories)
        for index, trajectory in enumerate(trajectories):
            assignments[trajectory] = index % fold_count
    return np.asarray([assignments[row["trajectory_id"]] for row in rows], dtype=int)


def regression_metrics(
    x: np.ndarray,
    y: np.ndarray,
    nfp: np.ndarray,
    folds: np.ndarray,
    categories: np.ndarray,
    model: str,
) -> dict:
    design = design_matrix(x, nfp, categories, model)
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    predictions = np.empty_like(y)
    for fold in sorted(set(folds.tolist())):
        train = folds != fold
        test = folds == fold
        train_design = design_matrix(x[train], nfp[train], categories, model)
        test_design = design_matrix(x[test], nfp[test], categories, model)
        fold_coefficients, *_ = np.linalg.lstsq(train_design, y[train], rcond=None)
        predictions[test] = test_design @ fold_coefficients
    cv_residual = y - predictions
    return {
        "coefficients": coefficients.tolist(),
        "in_sample_residual_std_decades": float(np.std(residual)),
        "grouped_cv_rmse_decades": float(np.sqrt(np.mean(cv_residual**2))),
        "grouped_cv_mae_decades": float(np.mean(np.abs(cv_residual))),
    }


def analyze_surface(path: Path, surface_name: str) -> dict:
    rows = load_rows(path, surface_name)
    x = np.log10(np.asarray([float(row["native_qh"]) for row in rows]))
    y = np.log10(np.asarray([float(row["face_qh"]) for row in rows]))
    nfp = np.asarray([int(row["nfp"]) for row in rows], dtype=int)
    categories = np.unique(nfp)
    folds = trajectory_folds(rows)
    models = {
        model: regression_metrics(x, y, nfp, folds, categories, model)
        for model in (
            "pooled",
            "undo_helicity_normalization",
            "continuous_nfp",
            "nfp_intercepts",
            "nfp_separate_lines",
        )
    }
    continuous = models["continuous_nfp"]["coefficients"]
    models["continuous_nfp"]["helicity_coefficient_over_qh_slope"] = (
        float(continuous[2] / continuous[1]) if continuous[1] != 0.0 else math.nan
    )

    by_nfp = {}
    for category in categories:
        selected = nfp == category
        slope, intercept = np.polyfit(x[selected], y[selected], 1)
        residual = y[selected] - (slope * x[selected] + intercept)
        trajectory_count = len(
            {row["trajectory_id"] for row in rows if int(row["nfp"]) == category}
        )
        by_nfp[str(int(category))] = {
            "count": int(np.sum(selected)),
            "trajectory_count": trajectory_count,
            "slope": float(slope),
            "intercept": float(intercept),
            "residual_std_decades": float(np.std(residual)),
            "spearman": float(spearmanr(x[selected], y[selected]).statistic),
        }
    return {
        "surface_name": surface_name,
        "count": len(rows),
        "trajectory_count": len({row["trajectory_id"] for row in rows}),
        "nfp_values": categories.tolist(),
        "models": models,
        "by_nfp": by_nfp,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.surface_records.resolve()
    try:
        source_label = source.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        source_label = str(source)
    summary = {
        "format": "summary1_qh_nfp_calibration_v1",
        "source": source_label,
        "cross_validation": "three folds grouped by trajectory and stratified by nfp",
        "fixed_probe": analyze_surface(args.surface_records, "fixed_probe"),
        "adaptive_edge": analyze_surface(args.surface_records, "adaptive_edge"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
