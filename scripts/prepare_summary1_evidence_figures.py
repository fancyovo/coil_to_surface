#!/usr/bin/env python3
"""Build the focused evidence figures used by the summary technical report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "reports" / "summary1" / "assets" / "evidence_20260825"
LATENT_COLOR = "#176B87"
DATA_COLOR = "#C65D28"
TANGENT_COLOR = "#6B7280"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(figure: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_DIR / name, dpi=210, bbox_inches="tight")
    plt.close(figure)


def identity_line(axis: plt.Axes, values: list[float], *, log: bool = False) -> None:
    finite = np.asarray([value for value in values if np.isfinite(value) and value > 0.0])
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if log:
        lower *= 0.75
        upper *= 1.35
    else:
        span = upper - lower
        lower -= 0.04 * span
        upper += 0.04 * span
    axis.plot([lower, upper], [lower, upper], color="#111827", linestyle="--", linewidth=1.2)
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)


def plot_landscape_evidence() -> None:
    path = (
        REPO_ROOT
        / "reports"
        / "summary1"
        / "assets"
        / "landscape_current_20260824"
        / "summary.json"
    )
    summary = read_json(path)
    indexed = {
        (int(row["source_id"]), int(row["direction"]), row["path"]): row
        for row in summary["curve_metrics"]
    }
    physical: dict[str, tuple[list[float], list[float]]] = {
        "random_direct": ([], []),
        "tangent_direct": ([], []),
    }
    roughness: dict[str, tuple[list[float], list[float]]] = {
        "random_direct": ([], []),
        "tangent_direct": ([], []),
    }
    for source_id, direction, path_name in sorted(indexed):
        if path_name != "latent":
            continue
        latent = indexed[(source_id, direction, "latent")]
        for comparator in ("random_direct", "tangent_direct"):
            control = indexed[(source_id, direction, comparator)]
            physical[comparator][0].append(control["drop_5_physical_radius_m"]["mean"])
            physical[comparator][1].append(latent["drop_5_physical_radius_m"]["mean"])
            roughness[comparator][0].append(control["roughness"]["second_derivative_rms"])
            roughness[comparator][1].append(latent["roughness"]["second_derivative_rms"])

    figure, axes = plt.subplots(1, 2, figsize=(11.8, 5.0), constrained_layout=True)
    styles = {
        "random_direct": (DATA_COLOR, "Independent data-space directions"),
        "tangent_direct": (TANGENT_COLOR, "Transported local tangents"),
    }
    for comparator, (color, label) in styles.items():
        x, y = physical[comparator]
        axes[0].scatter(x, y, s=46, alpha=0.82, color=color, edgecolor="white", linewidth=0.5, label=label)
    all_physical = [value for pair in physical.values() for values in pair for value in values]
    identity_line(axes[0], all_physical, log=True)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set(
        xlabel="Comparator radius for a 5-point score drop [m]",
        ylabel="Latent-space radius for the same score drop [m]",
        title="High-score basin width at matched coil displacement",
    )
    axes[0].legend(frameon=False, fontsize=9)

    for comparator, (color, label) in styles.items():
        x, y = roughness[comparator]
        axes[1].scatter(x, y, s=46, alpha=0.82, color=color, edgecolor="white", linewidth=0.5, label=label)
    all_roughness = [value for pair in roughness.values() for values in pair for value in values]
    identity_line(axes[1], all_roughness, log=True)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set(
        xlabel="Comparator second-derivative RMS",
        ylabel="Latent-space second-derivative RMS",
        title="Discrete landscape roughness",
    )
    random_radius_control = np.asarray(physical["random_direct"][0])
    random_radius_latent = np.asarray(physical["random_direct"][1])
    random_roughness_control = np.asarray(roughness["random_direct"][0])
    random_roughness_latent = np.asarray(roughness["random_direct"][1])
    radius_ratio = float(np.median(random_radius_latent / random_radius_control))
    roughness_ratio = float(np.median(random_roughness_latent / random_roughness_control))
    wider_count = int(np.sum(random_radius_latent > random_radius_control))
    smoother_count = int(np.sum(random_roughness_latent < random_roughness_control))
    axes[1].text(
        0.04,
        0.96,
        "vs independent directions:\n"
        f"{radius_ratio:.2f}x wider ({wider_count}/12)\n"
        f"{roughness_ratio:.4f}x roughness ({smoother_count}/12)",
        transform=axes[1].transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.92},
    )
    for axis in axes:
        axis.grid(alpha=0.2, which="both")
    save_figure(figure, "flow_landscape_evidence.png")


def plot_paired_optimization() -> None:
    path = (
        REPO_ROOT
        / "reports"
        / "assets"
        / "qh_data_space_large_validation_20260825"
        / "comparison_analysis"
        / "paired_optimizer_trajectories.csv"
    )
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    data_best = np.asarray([float(row["best_online_score_data"]) for row in rows])
    latent_best = np.asarray([float(row["best_online_score_latent"]) for row in rows])
    initial = np.asarray([float(row["reference_initial_score"]) for row in rows])
    difference = latent_best - data_best
    summary_path = (
        REPO_ROOT
        / "reports"
        / "assets"
        / "qh_data_space_large_validation_20260825"
        / "comparison_analysis"
        / "method_comparison_summary.json"
    )
    score_summary = read_json(summary_path)["score_space"]["best_score"]
    data_minus_latent_ci = score_summary["median_difference_bootstrap_95"]
    latent_minus_data_ci = (-float(data_minus_latent_ci[1]), -float(data_minus_latent_ci[0]))
    paired_test = score_summary["paired_test"]
    median_difference = float(np.median(difference))

    figure, axes = plt.subplots(1, 2, figsize=(11.8, 5.0), constrained_layout=True)
    scatter = axes[0].scatter(
        data_best,
        latent_best,
        c=initial,
        cmap="viridis",
        s=34,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.35,
    )
    identity_line(axes[0], [*data_best, *latent_best])
    axes[0].set(
        xlabel="Best formal-center score in standardized data space",
        ylabel="Best formal-center score in Flow latent space",
        title="Same 309 starts, 200 Adam steps",
    )
    colorbar = figure.colorbar(scatter, ax=axes[0], pad=0.02)
    colorbar.set_label("Shared starting score")
    axes[0].text(
        0.04,
        0.96,
        f"Latent wins {paired_test['latent_wins']} / {paired_test['count']} pairs",
        transform=axes[0].transAxes,
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.92},
    )

    bins = np.linspace(float(np.min(difference)), float(np.max(difference)), 31)
    axes[1].hist(difference, bins=bins, color=LATENT_COLOR, alpha=0.82, edgecolor="white")
    axes[1].axvline(0.0, color="#111827", linewidth=1.2)
    axes[1].axvspan(*latent_minus_data_ci, color="#F5C451", alpha=0.35, label="95% CI of median")
    axes[1].axvline(median_difference, color=DATA_COLOR, linewidth=2.0, label=f"median = {median_difference:+.3f}")
    axes[1].set(
        xlabel="Paired best-score difference (latent - data)",
        ylabel="Number of trajectories",
        title="Paired advantage distribution",
    )
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].text(
        0.96,
        0.96,
        f"sign test p = {paired_test['two_sided_sign_test_p']:.2e}\n"
        f"Wilcoxon p = {paired_test['wilcoxon_two_sided_p']:.4f}",
        transform=axes[1].transAxes,
        va="top",
        ha="right",
        fontsize=9,
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    save_figure(figure, "flow_vs_data_paired_optimization.png")


def plot_initialization_evidence() -> None:
    path = REPO_ROOT / "reports" / "assets" / "qh_data_prior_end_to_end_48_20260824" / "summary.json"
    summary = read_json(path)
    rows = summary["rows"]
    data_initial = np.asarray([float(row["data_initial"]) for row in rows])
    flow_initial = np.asarray([float(row["flow_initial"]) for row in rows])
    condition_count = int(summary["analyzed_case_count"])

    figure, axes = plt.subplots(1, 2, figsize=(11.8, 5.0), constrained_layout=True)
    axes[0].scatter(
        data_initial,
        flow_initial,
        color=LATENT_COLOR,
        s=48,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.5,
    )
    identity_line(axes[0], [*data_initial, *flow_initial])
    axes[0].set(
        xlabel="Best-of-32 score from independent data-coordinate Gaussian",
        ylabel="Best-of-32 score from Flow prior",
        title="Paired initialization quality across 48 conditions",
    )
    axes[0].text(
        0.04,
        0.96,
        f"Flow wins {int(np.sum(flow_initial > data_initial))} / {condition_count}\n"
        f"median: {np.median(flow_initial):.3f} vs {np.median(data_initial):.3f}",
        transform=axes[0].transAxes,
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.92},
    )

    labels = ["Valid candidates", "Conditions with\na valid start", "Starts >= 50"]
    flow_rates = np.asarray(
        [
            float(summary["candidate_valid_fraction"]["flow"]),
            float(np.mean(flow_initial > 0.0)),
            float(np.mean(flow_initial >= 50.0)),
        ]
    ) * 100.0
    data_rates = np.asarray(
        [
            float(summary["candidate_valid_fraction"]["data"]),
            float(np.mean(data_initial > 0.0)),
            float(np.mean(data_initial >= 50.0)),
        ]
    ) * 100.0
    locations = np.arange(len(labels))
    width = 0.34
    axes[1].bar(locations - width / 2, flow_rates, width, color=LATENT_COLOR, label="Flow prior")
    axes[1].bar(locations + width / 2, data_rates, width, color=DATA_COLOR, label="Independent data Gaussian")
    axes[1].set_xticks(locations, labels)
    axes[1].set_ylim(0.0, 108.0)
    axes[1].set(ylabel="Fraction [%]", title="Reliability of a 32-candidate screen")
    axes[1].legend(frameon=False, fontsize=9)
    for bars in axes[1].containers:
        axes[1].bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2, axis="y")
    save_figure(figure, "flow_initialization_evidence.png")


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def plot_volume_face_relation() -> None:
    path = REPO_ROOT / "reports" / "assets" / "qh_trajectory_face_qs_calibration_20260819" / "surface_records.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        all_rows = list(csv.DictReader(stream))

    figure, axes = plt.subplots(1, 2, figsize=(11.8, 5.0), constrained_layout=True)
    settings = [
        ("fixed_probe", LATENT_COLOR, "Fixed probe surface"),
        ("adaptive_edge", DATA_COLOR, "Adaptive edge surface"),
    ]
    for axis, (surface_name, color, title) in zip(axes, settings):
        rows = [
            row
            for row in all_rows
            if row["surface_name"] == surface_name
            and parse_bool(row["accepted"])
            and float(row["native_qh"]) > 0.0
            and float(row["face_qh"]) > 0.0
        ]
        volume = np.asarray([float(row["native_qh"]) for row in rows])
        face = np.asarray([float(row["face_qh"]) for row in rows])
        log_volume = np.log10(volume)
        log_face = np.log10(face)
        slope, intercept = np.polyfit(log_volume, log_face, 1)
        residual_std = float(np.std(log_face - (slope * log_volume + intercept)))
        coefficient = 10.0**intercept
        x_grid = np.logspace(float(np.min(log_volume)), float(np.max(log_volume)), 200)
        y_grid = coefficient * x_grid**slope

        axis.scatter(volume, face, s=19, color=color, alpha=0.36, edgecolor="none")
        axis.plot(x_grid, y_grid, color="#111827", linewidth=2.0)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set(
            xlabel=r"Volume differential QH error $E_{\mathrm{vol}}$",
            ylabel=r"Simsopt surface QH error $E_{\mathrm{face}}$",
            title=f"{title} (n={len(rows)})",
        )
        axis.text(
            0.04,
            0.96,
            rf"$E_{{\mathrm{{face}}}}={coefficient:.3f}E_{{\mathrm{{vol}}}}^{{{slope:.3f}}}$"
            "\n"
            rf"$\sigma_{{\log_{{10}}E_{{\mathrm{{face}}}}}}={residual_std:.3f}$ decades",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.92},
        )
        axis.grid(alpha=0.2, which="both")
    save_figure(figure, "volume_to_surface_qh_calibration.png")


def main() -> None:
    plot_landscape_evidence()
    plot_paired_optimization()
    plot_initialization_evidence()
    plot_volume_face_relation()


if __name__ == "__main__":
    main()
