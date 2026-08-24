from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def nested(payload: Any, *keys: str, default: float = float("nan")) -> float:
    value = payload
    try:
        for key in keys:
            value = value[key]
        return float(value)
    except (KeyError, TypeError, ValueError):
        return default


def finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    array = finite(values)
    if array.size == 0:
        return {"count": 0, "p10": None, "p50": None, "p90": None, "p95": None, "mean": None, "max": None}
    return {
        "count": int(array.size),
        "p10": float(np.percentile(array, 10)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def correlation_statistic(x: np.ndarray, y: np.ndarray, *, kind: str) -> float:
    if x.size < 3 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return float("nan")
    if kind == "spearman":
        return float(spearmanr(x, y).statistic)
    if kind == "pearson":
        return float(pearsonr(x, y).statistic)
    raise ValueError(kind)


def correlations(
    rows: list[dict[str, Any]],
    *,
    acceptance_key: str = "accepted",
    seed: int = 20260819,
    bootstrap: int = 1000,
) -> dict[str, Any]:
    usable = [
        row
        for row in rows
        if row[acceptance_key]
        and row["native_qh"] > 0.0
        and row["face_qh"] > 0.0
        and math.isfinite(row["native_qh"])
        and math.isfinite(row["face_qh"])
    ]
    if len(usable) < 3:
        return {"count": len(usable), "spearman": None, "log_pearson": None, "cluster_bootstrap_95": None}
    x = np.log10([row["native_qh"] for row in usable])
    y = np.log10([row["face_qh"] for row in usable])
    by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        by_trajectory[row["trajectory_id"]].append(row)
    identifiers = sorted(by_trajectory)
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(bootstrap):
        draw = rng.choice(identifiers, size=len(identifiers), replace=True)
        bootstrap_rows = [row for identifier in draw for row in by_trajectory[identifier]]
        bx = np.log10([row["native_qh"] for row in bootstrap_rows])
        by = np.log10([row["face_qh"] for row in bootstrap_rows])
        statistic = correlation_statistic(bx, by, kind="spearman")
        if math.isfinite(statistic):
            samples.append(statistic)
    spearman = correlation_statistic(x, y, kind="spearman")
    log_pearson = correlation_statistic(x, y, kind="pearson")
    return {
        "count": len(usable),
        "trajectory_count": len(identifiers),
        "spearman": spearman if math.isfinite(spearman) else None,
        "log_pearson": log_pearson if math.isfinite(log_pearson) else None,
        "cluster_bootstrap_95": (
            [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]
            if samples
            else None
        ),
    }


def paired_correlations(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    acceptance_key: str = "accepted",
) -> dict[str, float | int | None]:
    usable = [
        row
        for row in rows
        if row[acceptance_key]
        and math.isfinite(row[x_key])
        and math.isfinite(row[y_key])
    ]
    if len(usable) < 3:
        return {"count": len(usable), "spearman": None, "pearson": None}
    x = np.asarray([row[x_key] for row in usable], dtype=float)
    y = np.asarray([row[y_key] for row in usable], dtype=float)
    spearman = correlation_statistic(x, y, kind="spearman")
    pearson = correlation_statistic(x, y, kind="pearson")
    return {
        "count": len(usable),
        "spearman": spearman if math.isfinite(spearman) else None,
        "pearson": pearson if math.isfinite(pearson) else None,
    }


def log_linear_relation(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    usable = [
        row
        for row in rows
        if row["accepted"] and row["native_qh"] > 0.0 and row["face_qh"] > 0.0
    ]
    if len(usable) < 3:
        return {"count": len(usable), "slope": None, "intercept": None, "residual_std_decades": None}
    x = np.log10([row["native_qh"] for row in usable])
    y = np.log10([row["face_qh"] for row in usable])
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    return {
        "count": len(usable),
        "slope": float(slope),
        "intercept": float(intercept),
        "residual_std_decades": float(np.std(residual)),
    }


def endpoint_pair_summary(rows: list[dict[str, Any]], *, acceptance_key: str) -> dict[str, Any]:
    fixed = [row for row in rows if row["surface_name"] == "fixed_probe"]
    by_key = {(row["trajectory_id"], row["iteration"]): row for row in fixed}
    pairs = []
    for trajectory_id in sorted({row["trajectory_id"] for row in fixed}):
        start = by_key.get((trajectory_id, 0))
        end = by_key.get((trajectory_id, 200))
        if (
            start is not None
            and end is not None
            and start[acceptance_key]
            and end[acceptance_key]
            and start["face_qh"] > 0.0
            and end["face_qh"] > 0.0
        ):
            pairs.append((start["face_qh"], end["face_qh"]))
    return {
        "count": len(pairs),
        "start_face_qh": distribution(start for start, _ in pairs),
        "end_face_qh": distribution(end for _, end in pairs),
        "end_over_start": distribution(end / start for start, end in pairs),
        "improved_count": sum(end < start for start, end in pairs),
        "improved_fraction": sum(end < start for start, end in pairs) / max(len(pairs), 1),
    }


def score_best_pairs(
    rows: list[dict[str, Any]], *, acceptance_key: str
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    fixed = [row for row in rows if row["surface_name"] == "fixed_probe"]
    starts = {
        row["trajectory_id"]: row
        for row in fixed
        if row["iteration"] == 0 and row["scheduled_snapshot"]
    }
    best = {
        row["trajectory_id"]: row for row in fixed if row["score_best_snapshot"]
    }
    pairs = []
    for trajectory_id in sorted(starts.keys() & best.keys()):
        start = starts[trajectory_id]
        endpoint = best[trajectory_id]
        if (
            start[acceptance_key]
            and endpoint[acceptance_key]
            and start["face_qh"] > 0.0
            and endpoint["face_qh"] > 0.0
        ):
            pairs.append((start, endpoint))
    return pairs


def score_best_pair_summary(
    rows: list[dict[str, Any]], *, acceptance_key: str
) -> dict[str, Any]:
    pairs = score_best_pairs(rows, acceptance_key=acceptance_key)
    return {
        "count": len(pairs),
        "start_face_qh": distribution(start["face_qh"] for start, _ in pairs),
        "best_face_qh": distribution(best["face_qh"] for _, best in pairs),
        "best_over_start": distribution(
            best["face_qh"] / start["face_qh"] for start, best in pairs
        ),
        "improved_count": sum(best["face_qh"] < start["face_qh"] for start, best in pairs),
        "improved_fraction": (
            sum(best["face_qh"] < start["face_qh"] for start, best in pairs)
            / max(len(pairs), 1)
        ),
    }


def prepare_failure_stage(prepare: dict[str, Any]) -> str:
    if prepare.get("status") == "ok":
        return "none"
    error = str(prepare.get("error", ""))
    for stage in ("source_psi", "alpha_nu", "alpha"):
        if f"/{stage}.log" in error:
            return stage
    return "unknown"


def flatten_case(case_dir: Path) -> list[dict[str, Any]]:
    metadata = read_json(case_dir / "metadata.json")
    native = metadata["native_score"]
    diagnostics = native.get("diagnostics", {})
    face_root = case_dir / "face_qs"
    prepare_path = face_root / "prepare_summary.json"
    validation_path = face_root / "validation_summary.json"
    prepare = read_json(prepare_path) if prepare_path.is_file() else {}
    validation = read_json(validation_path) if validation_path.is_file() else {}
    source_summary = {}
    alpha_summary = {}
    if prepare.get("status") == "ok":
        source_path = Path(prepare["source_psi_dir"]) / "summary.json"
        alpha_path = Path(prepare["alpha_dir"]) / "summary.json"
        if source_path.is_file():
            source_summary = read_json(source_path)
        if alpha_path.is_file():
            alpha_summary = read_json(alpha_path)
    alpha_order = alpha_summary.get("orders", [{}])[-1] if alpha_summary.get("orders") else {}
    surface_by_name = {row["name"]: row for row in validation.get("surfaces", [])}
    rows = []
    for target in metadata["surface_targets"]:
        surface = surface_by_name.get(target["name"], {})
        accepted = bool(surface.get("accepted", False))
        initial_iota = nested(surface, "initial_iota")
        face_iota = nested(surface, "final_iota")
        iota_abs_error = abs(nested(surface, "iota_error"))
        checks = surface.get("acceptance_checks", {})
        rows.append(
            {
                "case_id": metadata["case_id"],
                "trajectory_id": metadata["trajectory_id"],
                "iteration": int(metadata["iteration"]),
                "scheduled_snapshot": bool(metadata.get("scheduled_snapshot", True)),
                "score_best_snapshot": bool(metadata.get("score_best_snapshot", False)),
                "optimizer_wall_s": float(metadata["optimizer_wall_s"]),
                "nfp": int(metadata["nfp"]),
                "n_base_coils": int(metadata["n_base_coils"]),
                "surface_name": target["name"],
                "target_s": float(target["s_level"]),
                "prepare_status": prepare.get("status", "missing"),
                "validation_status": surface.get("status", validation.get("status", "missing")),
                "accepted": accepted,
                "native_score": float(native.get("score", float("nan"))),
                "native_qh": nested(diagnostics, "qs_target_global_error_per_helicity"),
                "native_qa": nested(diagnostics, "qs_qa_global_error_per_helicity"),
                "native_qp": nested(diagnostics, "qs_qp_global_error_per_helicity"),
                "native_iota_mid": 0.5 * (nested(diagnostics, "iota_min") + nested(diagnostics, "iota_max")),
                "native_psi_train_rms": nested(diagnostics, "psi_train_rms"),
                "native_psi_angle_l2": nested(diagnostics, "psi_angle_l2"),
                "native_psi_angle_p95": nested(diagnostics, "psi_angle_p95"),
                "prepare_failure_stage": prepare_failure_stage(prepare),
                "source_psi_train_rms": nested(source_summary, "psi", "fit_info", "train_rms"),
                "source_psi_validation_rms": nested(source_summary, "psi", "fit_info", "validation_rms"),
                "source_psi_angle_l2": nested(source_summary, "psi", "fit_info", "validation_angle_l2"),
                "source_psi_angle_p95": nested(source_summary, "psi", "fit_info", "validation_angle_p95"),
                "alpha_validation_l2": nested(alpha_order, "validation", "relative_l2"),
                "alpha_normal_floor_l2": nested(alpha_order, "validation", "normal_floor_relative_l2"),
                "initial_boozer_l2": nested(surface, "initial", "grids", default=float("nan")),
                "final_boozer_l2": nested(surface, "final", "grids", default=float("nan")),
                "initial_iota": initial_iota,
                "face_iota": face_iota,
                "iota_abs_error": iota_abs_error,
                "iota_relative_error": iota_abs_error / max(abs(face_iota), 1.0e-12),
                "face_qa": nested(surface, "surface_qs_error", "QA_1_0"),
                "face_qh": nested(surface, "surface_qs_error", "QH_1_1"),
                "face_qp": nested(surface, "surface_qs_error", "QP_0_1"),
                "inverse_aspect_ratio": nested(surface, "geometry", "inverse_aspect_ratio"),
                "volume_m3": nested(surface, "geometry", "volume_m3"),
                "prepare_wall_s": nested(prepare, "total_wall_s"),
                "source_psi_wall_s": nested(prepare, "timing_s", "source_psi_s"),
                "alpha_wall_s": nested(prepare, "timing_s", "alpha_s"),
                "alpha_nu_wall_s": nested(prepare, "timing_s", "alpha_nu_s"),
                "cpu_solve_wall_s": nested(surface, "cpu_solve", "total_wall_s"),
                "validation_wall_s": nested(surface, "validation_wall_s"),
                "cpu_solve_status": surface.get("cpu_solve", {}).get("status", "missing"),
                "final_dense_l2": nested(surface, "final", "grids", default=float("nan")),
                "final_normal_p95": nested(surface, "final", "grids", default=float("nan")),
                "has_acceptance_checks": bool(checks),
                **{f"check_{key}": bool(checks.get(key, False)) for key in (
                    "solver_converged",
                    "dense_relative_l2",
                    "dense_normal_field_p95",
                    "toroidal_winding",
                    "normal_nonzero",
                )},
            }
        )
        if surface.get("initial", {}).get("grids"):
            rows[-1]["initial_boozer_l2"] = float(surface["initial"]["grids"][-1]["relative_l2"])
        if surface.get("final", {}).get("grids"):
            rows[-1]["final_boozer_l2"] = float(surface["final"]["grids"][-1]["relative_l2"])
            rows[-1]["final_dense_l2"] = float(surface["final"]["grids"][-1]["relative_l2"])
            rows[-1]["final_normal_p95"] = float(surface["final"]["grids"][-1]["normal_B_sine_p95"])
        rows[-1]["solved_regular"] = bool(
            checks.get("solver_converged", False)
            and checks.get("toroidal_winding", False)
            and checks.get("normal_nonzero", False)
            and math.isfinite(rows[-1]["face_qh"])
            and rows[-1]["face_qh"] > 0.0
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_face_vs_volume(rows: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    colors = {"fixed_probe": "#176b87", "adaptive_edge": "#b64b3b"}
    plotted = 0
    for name in colors:
        subset = [row for row in rows if row["surface_name"] == name and row["accepted"] and row["native_qh"] > 0 and row["face_qh"] > 0]
        plotted += len(subset)
        axes[0].scatter([row["native_qh"] for row in subset], [row["face_qh"] for row in subset], s=14, alpha=0.42, color=colors[name], label=f"{name} (n={len(subset)})")
    if plotted:
        axes[0].set(xscale="log", yscale="log")
    else:
        axes[0].text(0.5, 0.5, "No accepted surfaces", ha="center", va="center", transform=axes[0].transAxes)
    axes[0].set(xlabel="volume differential QH error / helicity", ylabel="surface QH relative variance", title="Volume vs surface QH")
    axes[0].legend(frameon=False)
    accepted = [row for row in rows if row["accepted"] and row["face_qh"] > 0]
    scatter = axes[1].scatter([row["inverse_aspect_ratio"] for row in accepted], [row["face_qh"] for row in accepted], c=[row["iteration"] for row in accepted], s=15, alpha=0.5, cmap="viridis")
    if accepted:
        axes[1].set_yscale("log")
        figure.colorbar(scatter, ax=axes[1], label="Adam iteration")
    else:
        axes[1].text(0.5, 0.5, "No accepted surfaces", ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set(xlabel="surface inverse aspect ratio", ylabel="surface QH relative variance", title="Surface size as a covariate")
    figure.savefig(path, dpi=190)
    plt.close(figure)


def plot_time_evolution(rows: list[dict[str, Any]], path: Path) -> None:
    subset = [row for row in rows if row["surface_name"] == "fixed_probe" and row["accepted"] and row["face_qh"] > 0]
    iterations = sorted({row["iteration"] for row in subset})
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), constrained_layout=True)
    for trajectory_id in sorted({row["trajectory_id"] for row in subset}):
        trajectory = sorted([row for row in subset if row["trajectory_id"] == trajectory_id], key=lambda row: row["optimizer_wall_s"])
        axes[0].plot([row["optimizer_wall_s"] / 60.0 for row in trajectory], [row["face_qh"] for row in trajectory], color="#8aa2ad", alpha=0.12, linewidth=0.7)
    x = []
    median = []
    low = []
    high = []
    for iteration in iterations:
        stage = [row for row in subset if row["iteration"] == iteration]
        x.append(float(np.median([row["optimizer_wall_s"] for row in stage])) / 60.0)
        values = np.asarray([row["face_qh"] for row in stage])
        median.append(float(np.median(values)))
        low.append(float(np.percentile(values, 10)))
        high.append(float(np.percentile(values, 90)))
    axes[0].fill_between(x, low, high, color="#176b87", alpha=0.2, label="P10-P90")
    axes[0].plot(x, median, "o-", color="#176b87", linewidth=2, label="median")
    if subset:
        axes[0].set_yscale("log")
    else:
        axes[0].text(0.5, 0.5, "No accepted fixed-probe surfaces", ha="center", va="center", transform=axes[0].transAxes)
    axes[0].set(xlabel="optimizer wall time [min]", ylabel="fixed-probe surface QH", title="Face QH during optimization")
    axes[0].legend(frameon=False)
    for threshold, color in ((1e-3, "#d18b2c"), (1e-4, "#b64b3b"), (1e-5, "#6d3f8c")):
        fractions = []
        for iteration in iterations:
            stage = [row for row in rows if row["surface_name"] == "fixed_probe" and row["iteration"] == iteration]
            fractions.append(sum(row["accepted"] and row["face_qh"] <= threshold for row in stage) / max(len(stage), 1))
        axes[1].plot(x, fractions, "o-", color=color, label=f"QH <= {threshold:.0e}")
    axes[1].set(xlabel="optimizer wall time [min]", ylabel="fraction of all trajectories", title="Strict threshold attainment", ylim=(-0.02, 1.02))
    axes[1].legend(frameon=False)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def plot_accuracy(rows: list[dict[str, Any]], path: Path) -> None:
    case_rows = [row for row in rows if row["surface_name"] == "fixed_probe"]
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), constrained_layout=True)
    metrics = [
        ("native_psi_angle_p95", "native psi angle P95"),
        ("source_psi_angle_p95", "refit psi angle P95"),
        ("alpha_validation_l2", "alpha validation relative L2"),
        ("final_dense_l2", "final dense Boozer relative L2"),
    ]
    for axis, (key, label) in zip(axes.ravel(), metrics):
        values = finite(row[key] for row in case_rows)
        if values.size:
            positive = np.maximum(values, np.finfo(float).tiny)
            axis.hist(np.log10(positive), bins=35, color="#176b87", alpha=0.82)
        axis.set(xlabel=f"log10({label})", ylabel="count", title=label)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def plot_iota(rows: list[dict[str, Any]], path: Path) -> None:
    subset = [row for row in rows if row["accepted"] and math.isfinite(row["initial_iota"]) and math.isfinite(row["face_iota"])]
    figure, axes = plt.subplots(1, 3, figsize=(16.0, 5.0), constrained_layout=True)
    axes[0].scatter([row["initial_iota"] for row in subset], [row["face_iota"] for row in subset], c=[row["iteration"] for row in subset], s=15, alpha=0.48, cmap="viridis")
    if subset:
        limits = [min(min(row["initial_iota"], row["face_iota"]) for row in subset), max(max(row["initial_iota"], row["face_iota"]) for row in subset)]
        axes[0].plot(limits, limits, "--", color="black", linewidth=1)
    axes[0].set(xlabel="GPU alpha-fit iota", ylabel="Simsopt surface iota", title="Rotational-transform agreement")
    errors = finite(row["iota_abs_error"] for row in subset)
    axes[1].hist(errors, bins=35, color="#b64b3b", alpha=0.82)
    axes[1].set(xlabel="absolute iota error", ylabel="count", title="Absolute iota mismatch")
    relative_errors = finite(row["iota_relative_error"] for row in subset)
    if relative_errors.size:
        axes[2].hist(np.log10(np.maximum(relative_errors, np.finfo(float).tiny)), bins=35, color="#6d3f8c", alpha=0.82)
    axes[2].set(xlabel="log10(relative iota error)", ylabel="count", title="Relative iota mismatch")
    figure.savefig(path, dpi=190)
    plt.close(figure)


def plot_condition_summary(rows: list[dict[str, Any]], path: Path) -> None:
    fixed = [row for row in rows if row["surface_name"] == "fixed_probe"]
    nfps = sorted({row["nfp"] for row in fixed})
    coil_counts = sorted({row["n_base_coils"] for row in fixed})
    count = np.full((len(nfps), len(coil_counts)), np.nan)
    acceptance = np.full_like(count, np.nan)
    rank_correlation = np.full_like(count, np.nan)
    for i, nfp in enumerate(nfps):
        for j, n_coils in enumerate(coil_counts):
            subset = [row for row in fixed if row["nfp"] == nfp and row["n_base_coils"] == n_coils]
            if not subset:
                continue
            count[i, j] = len(subset)
            acceptance[i, j] = sum(row["accepted"] for row in subset) / len(subset)
            statistic = correlations(subset, bootstrap=0)["spearman"]
            if statistic is not None:
                rank_correlation[i, j] = statistic
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    panels = ((count, "sampled snapshots", None, None), (acceptance, "strict acceptance", 0.0, 1.0), (rank_correlation, "volume/face Spearman", 0.0, 1.0))
    for axis, (values, title, vmin, vmax) in zip(axes, panels):
        image = axis.imshow(values, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if math.isfinite(values[i, j]):
                    label = f"{int(values[i, j])}" if title == "sampled snapshots" else f"{values[i, j]:.2f}"
                    axis.text(j, i, label, ha="center", va="center", color="white" if values[i, j] < np.nanmax(values) * 0.65 else "black", fontsize=8)
        axis.set(xticks=range(len(coil_counts)), xticklabels=coil_counts, yticks=range(len(nfps)), yticklabels=nfps, xlabel="base coils", ylabel="NFP", title=title)
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def plot_score_best_comparison(rows: list[dict[str, Any]], path: Path) -> None:
    pairs = score_best_pairs(rows, acceptance_key="accepted")
    if not pairs:
        return
    x = np.asarray([start["face_qh"] for start, _ in pairs], dtype=float)
    y = np.asarray([best["face_qh"] for _, best in pairs], dtype=float)
    nfp = np.asarray([start["nfp"] for start, _ in pairs], dtype=float)
    lower = min(np.min(x), np.min(y)) / 1.5
    upper = max(np.max(x), np.max(y)) * 1.5
    figure, axis = plt.subplots(figsize=(6.5, 5.8), constrained_layout=True)
    scatter = axis.scatter(x, y, c=nfp, cmap="viridis", s=30, alpha=0.72, linewidths=0)
    axis.plot([lower, upper], [lower, upper], "--", color="#222222", linewidth=1.2)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.grid(True, which="both", alpha=0.18)
    axis.set(
        xlabel="Initial Simsopt face QH",
        ylabel="Score-best Simsopt face QH",
        title=f"Exact optimizer score-best snapshots (n={len(pairs)})",
    )
    figure.colorbar(scatter, ax=axis, label="NFP")
    figure.savefig(path, dpi=190)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate and plot trajectory face-QS calibration results.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in read_json(args.experiment_root / "cases.json"):
        rows.extend(flatten_case(args.experiment_root / "cases" / case["case_id"]))
    write_csv(args.output_dir / "surface_records.csv", rows)
    scheduled_rows = [row for row in rows if row["scheduled_snapshot"]]

    by_surface = {}
    for name in ("fixed_probe", "adaptive_edge"):
        subset = [row for row in scheduled_rows if row["surface_name"] == name]
        by_surface[name] = {
            "count": len(subset),
            "accepted": sum(row["accepted"] for row in subset),
            "acceptance_fraction": sum(row["accepted"] for row in subset) / max(len(subset), 1),
            "solved_regular": sum(row["solved_regular"] for row in subset),
            "solved_regular_fraction": sum(row["solved_regular"] for row in subset) / max(len(subset), 1),
            "strict_correlation": correlations(subset),
            "solved_regular_correlation": correlations(subset, acceptance_key="solved_regular"),
            "face_qh": distribution(row["face_qh"] for row in subset if row["accepted"]),
            "inverse_aspect_ratio": distribution(row["inverse_aspect_ratio"] for row in subset if row["accepted"]),
            "iota": {
                "absolute_error": distribution(row["iota_abs_error"] for row in subset if row["accepted"]),
                "relative_error": distribution(row["iota_relative_error"] for row in subset if row["accepted"]),
                "correlation": paired_correlations(subset, x_key="initial_iota", y_key="face_iota"),
            },
            "log10_face_from_log10_volume": log_linear_relation(subset),
        }
    by_iteration = {}
    for iteration in sorted({row["iteration"] for row in scheduled_rows}):
        subset = [row for row in scheduled_rows if row["iteration"] == iteration and row["surface_name"] == "fixed_probe"]
        accepted = [row for row in subset if row["accepted"]]
        by_iteration[str(iteration)] = {
            "count": len(subset),
            "accepted": len(accepted),
            "acceptance_fraction": len(accepted) / max(len(subset), 1),
            "optimizer_wall_minutes": distribution(row["optimizer_wall_s"] / 60.0 for row in subset),
            "face_qh": distribution(row["face_qh"] for row in accepted),
            "native_qh": distribution(row["native_qh"] for row in accepted),
            "below": {
                f"{threshold:.0e}": {
                    "count": sum(row["face_qh"] <= threshold for row in accepted),
                    "fraction_of_strict_accepted": sum(row["face_qh"] <= threshold for row in accepted) / max(len(accepted), 1),
                    "fraction_of_all_trajectories": sum(row["face_qh"] <= threshold for row in accepted) / max(len(subset), 1),
                }
                for threshold in (1e-3, 1e-4, 1e-5)
            },
            "correlation": correlations(subset, seed=20260819 + iteration, bootstrap=500),
        }
    by_condition = {}
    for nfp, n_base_coils in sorted({(row["nfp"], row["n_base_coils"]) for row in scheduled_rows}):
        subset = [row for row in scheduled_rows if row["nfp"] == nfp and row["n_base_coils"] == n_base_coils]
        by_condition[f"nfp{nfp}_nc{n_base_coils}"] = {
            "count": len(subset),
            "accepted": sum(row["accepted"] for row in subset),
            "correlation": correlations(subset, seed=20260819 + 10 * nfp + n_base_coils, bootstrap=500),
        }
    by_condition_fixed_probe = {}
    for nfp, n_base_coils in sorted({(row["nfp"], row["n_base_coils"]) for row in scheduled_rows}):
        subset = [row for row in scheduled_rows if row["surface_name"] == "fixed_probe" and row["nfp"] == nfp and row["n_base_coils"] == n_base_coils]
        by_condition_fixed_probe[f"nfp{nfp}_nc{n_base_coils}"] = {
            "count": len(subset),
            "accepted": sum(row["accepted"] for row in subset),
            "acceptance_fraction": sum(row["accepted"] for row in subset) / max(len(subset), 1),
            "correlation": correlations(subset, seed=20261819 + 10 * nfp + n_base_coils, bootstrap=500),
        }
    threshold_times = {}
    fixed_rows = [row for row in scheduled_rows if row["surface_name"] == "fixed_probe" and row["accepted"]]
    for threshold in (1e-3, 1e-4, 1e-5):
        reached = []
        for trajectory_id in sorted({row["trajectory_id"] for row in fixed_rows}):
            candidates = [row for row in fixed_rows if row["trajectory_id"] == trajectory_id and row["face_qh"] <= threshold]
            if candidates:
                reached.append(min(row["optimizer_wall_s"] / 60.0 for row in candidates))
        threshold_times[f"{threshold:.0e}"] = {
            "trajectory_count_reached": len(reached),
            "fraction_of_sampled_trajectories": len(reached) / max(len({row["trajectory_id"] for row in scheduled_rows}), 1),
            "first_reached_wall_minutes": distribution(reached),
        }
    summary = {
        "row_count": len(scheduled_rows),
        "all_row_count_including_score_best_extras": len(rows),
        "case_count": len(scheduled_rows) // 2,
        "all_case_count_including_score_best_extras": len(rows) // 2,
        "trajectory_count": len({row["trajectory_id"] for row in scheduled_rows}),
        "prepare_status_counts": dict(Counter(row["prepare_status"] for row in scheduled_rows[::2])),
        "prepare_failure_stage_counts": dict(Counter(row["prepare_failure_stage"] for row in scheduled_rows[::2])),
        "cpu_solve_status_counts": dict(Counter(row["cpu_solve_status"] for row in scheduled_rows)),
        "validation_status_counts": dict(Counter(row["validation_status"] for row in scheduled_rows)),
        "by_surface": by_surface,
        "by_iteration_fixed_probe": by_iteration,
        "by_condition": by_condition,
        "by_condition_fixed_probe": by_condition_fixed_probe,
        "threshold_times": threshold_times,
        "paired_endpoints_fixed_probe": {
            "strict_accepted": endpoint_pair_summary(scheduled_rows, acceptance_key="accepted"),
            "solved_regular": endpoint_pair_summary(scheduled_rows, acceptance_key="solved_regular"),
        },
        "paired_score_best_fixed_probe": {
            "strict_accepted": score_best_pair_summary(rows, acceptance_key="accepted"),
            "solved_regular": score_best_pair_summary(rows, acceptance_key="solved_regular"),
        },
        "psi_accuracy": {
            key: distribution(row[key] for row in scheduled_rows if row["surface_name"] == "fixed_probe")
            for key in ("native_psi_train_rms", "native_psi_angle_l2", "native_psi_angle_p95", "source_psi_train_rms", "source_psi_validation_rms", "source_psi_angle_l2", "source_psi_angle_p95", "alpha_validation_l2", "alpha_normal_floor_l2")
        },
        "iota": {
            "absolute_error": distribution(row["iota_abs_error"] for row in scheduled_rows if row["accepted"]),
            "relative_error": distribution(row["iota_relative_error"] for row in scheduled_rows if row["accepted"]),
            "gpu_initial": distribution(row["initial_iota"] for row in scheduled_rows if row["accepted"]),
            "surface_final": distribution(row["face_iota"] for row in scheduled_rows if row["accepted"]),
            "correlation": paired_correlations(scheduled_rows, x_key="initial_iota", y_key="face_iota"),
        },
        "acceptance_check_failures": {
            key: sum(row["has_acceptance_checks"] and not row[f"check_{key}"] for row in scheduled_rows)
            for key in (
                "solver_converged",
                "dense_relative_l2",
                "dense_normal_field_p95",
                "toroidal_winding",
                "normal_nonzero",
            )
        },
        "timing_s": {
            "gpu_prepare_per_case": distribution(row["prepare_wall_s"] for row in scheduled_rows[::2]),
            "source_psi_per_case": distribution(row["source_psi_wall_s"] for row in scheduled_rows[::2]),
            "alpha_per_case": distribution(row["alpha_wall_s"] for row in scheduled_rows[::2]),
            "alpha_nu_per_case": distribution(row["alpha_nu_wall_s"] for row in scheduled_rows[::2]),
            "cpu_solve_per_surface": distribution(row["cpu_solve_wall_s"] for row in scheduled_rows),
            "gpu_validation_per_surface": distribution(row["validation_wall_s"] for row in scheduled_rows),
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    plot_face_vs_volume(scheduled_rows, args.output_dir / "face_vs_volume_qs.png")
    plot_time_evolution(scheduled_rows, args.output_dir / "face_qs_over_time.png")
    plot_accuracy(scheduled_rows, args.output_dir / "psi_and_coordinate_accuracy.png")
    plot_iota(scheduled_rows, args.output_dir / "iota_comparison.png")
    plot_condition_summary(scheduled_rows, args.output_dir / "condition_summary.png")
    plot_score_best_comparison(rows, args.output_dir / "face_qs_start_vs_score_best.png")
    print(json.dumps({"rows": len(rows), "accepted": sum(row["accepted"] for row in rows), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
