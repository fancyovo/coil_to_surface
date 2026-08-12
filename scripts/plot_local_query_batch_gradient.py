from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads((args.asset_dir / "full600_summary.json").read_text(encoding="utf-8"))
    comparison = summary["local_score"]["gradient_comparison"]
    local = np.asarray(comparison["local_directional"], dtype=np.float64)
    exact = np.asarray(comparison["exact_directional"], dtype=np.float64)

    figure, axis = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    axis.scatter(exact, local, s=18, alpha=0.7, edgecolors="none")
    limit = max(float(np.max(np.abs(exact))), float(np.max(np.abs(local))))
    axis.plot([-limit, limit], [-limit, limit], color="black", linewidth=1, linestyle="--")
    cosine = comparison["coordinate_and_surface_selection_omitted_exact_cosine"]
    axis.set(
        xlabel="Exact fixed-surface directional derivative",
        ylabel="Batched local directional derivative",
        title=f"300-D gradient agreement (cosine = {cosine:.5f})",
    )
    axis.grid(alpha=0.25)
    figure.savefig(args.asset_dir / "full300_gradient_agreement.png", dpi=190)
    plt.close(figure)

    rows = [
        json.loads(line)
        for line in (args.asset_dir / "smoke_history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    first = rows[0]["gradient_pipeline"]["timing_s"]
    stage_names = [
        "endpoint flow",
        "field create",
        "center capture",
        "axis refine",
        "axis samples",
        "psi",
        "surface/flux/alpha/QS",
        "formal proposal",
    ]
    stage_values = [
        rows[0]["endpoint_decode_wall_s"],
        first["field_create"],
        first["center_capture"],
        first["axis_refine"],
        first["axis_samples"],
        first["psi"],
        first["local_score"],
        rows[0]["proposal_decode_wall_s"] + rows[0]["proposal_score_wall_s"],
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].barh(stage_names[::-1], stage_values[::-1], color="#2878B5")
    axes[0].set(xlabel="seconds", title="One full-gradient Adam step")
    axes[0].grid(axis="x", alpha=0.25)
    iterations = [0] + [row["iteration"] for row in rows]
    scores = [json.loads((args.asset_dir / "smoke_summary.json").read_text())["initial_score"]]
    scores.extend(row["current_score"] for row in rows)
    axes[1].plot(iterations, scores, marker="o", color="#C82423")
    axes[1].set(xlabel="Adam step", ylabel="formal score", title="Three-step smoke test")
    axes[1].grid(alpha=0.25)
    figure.savefig(args.asset_dir / "full_gradient_timing_and_smoke.png", dpi=190)
    plt.close(figure)

    long_dir = args.asset_dir / "long_start10_36541"
    baseline_path = (
        REPO_ROOT
        / "reports"
        / "assets"
        / "qh_iota_cubic_direction_compare1000_35902_35903"
        / "random"
        / "history.jsonl"
    )
    if long_dir.is_dir() and baseline_path.is_file():
        long_rows = [
            json.loads(line)
            for line in (long_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        baseline_rows = [
            json.loads(line)
            for line in baseline_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        long_iteration = np.asarray([row["iteration"] for row in long_rows])
        baseline_iteration = np.asarray([row["iteration"] for row in baseline_rows])
        long_wall_min = np.asarray([row["total_wall_s"] for row in long_rows]) / 60.0
        baseline_wall_min = np.asarray([row["total_wall_s"] for row in baseline_rows]) / 60.0

        figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        axes[0, 0].plot(
            long_iteration,
            [row["best_score"] for row in long_rows],
            label="full 300-D gradient",
        )
        axes[0, 0].plot(
            baseline_iteration,
            [row["best_score"] for row in baseline_rows],
            label="2-direction SPSA",
        )
        axes[0, 0].set(xlabel="Adam step", ylabel="best formal score", title="Per-step progress")
        axes[0, 0].legend()

        axes[0, 1].plot(
            long_wall_min,
            [row["best_score"] for row in long_rows],
            label="full 300-D gradient",
        )
        axes[0, 1].plot(
            baseline_wall_min,
            [row["best_score"] for row in baseline_rows],
            label="2-direction SPSA",
        )
        axes[0, 1].set(xlabel="wall time (min)", ylabel="best formal score", title="Wall-time progress")
        axes[0, 1].legend()

        axes[1, 0].plot(
            long_iteration,
            [row["current_qh_error"] for row in long_rows],
            label="QH",
        )
        axes[1, 0].plot(
            long_iteration,
            [row["current_qa_error"] for row in long_rows],
            label="QA",
        )
        axes[1, 0].plot(
            long_iteration,
            [row["current_qp_error"] for row in long_rows],
            label="QP",
        )
        axes[1, 0].set(
            yscale="log",
            xlabel="Adam step",
            ylabel="volume QS residual",
            title="Symmetry residuals",
        )
        axes[1, 0].legend()

        best_payload = json.loads((long_dir / "best.json").read_text(encoding="utf-8"))
        best_score = best_payload["flow_prior_local_full_gradient_adam"]["native_score"]
        start_payload = json.loads(
            (long_dir / "trajectory" / "step_0000.json").read_text(encoding="utf-8")
        )
        start_score = start_payload["flow_prior_local_full_gradient_adam"]["native_score"]
        component_names = ["axis", "psi", "surface", "coordinate", "volume_qs", "iota", "coil"]
        positions = np.arange(len(component_names))
        width = 0.38
        axes[1, 1].bar(
            positions - width / 2,
            [start_score["components"][name] for name in component_names],
            width,
            label="start",
        )
        axes[1, 1].bar(
            positions + width / 2,
            [best_score["components"][name] for name in component_names],
            width,
            label="best",
        )
        axes[1, 1].set(
            xticks=positions,
            xticklabels=component_names,
            ylim=(60, 102),
            ylabel="component score",
            title="Formal-score components",
        )
        axes[1, 1].tick_params(axis="x", rotation=25)
        axes[1, 1].legend()
        for axis in axes.ravel():
            axis.grid(alpha=0.25)
        figure.savefig(long_dir / "acceptance_comparison.png", dpi=190)
        plt.close(figure)


if __name__ == "__main__":
    main()
