#!/usr/bin/env python3
"""Compare two complete physical-evaluation deliveries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--candidate-label", default="full-gradient best")
    parser.add_argument("--baseline-label", default="K64 200-step")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_delivery(root: Path) -> dict:
    native = load_json(root / "native_score_recheck.json")["native_score"]
    selected = load_json(
        root / "candidates" / "s_0p49" / "standard_rho_1" / "summary.json"
    )
    source = load_json(root / "source_psi_candidates" / "a_0p08" / "summary.json")
    desc = load_json(root / "full" / "desc" / "summary.json")
    dense = selected["newton"]["state"]["grids"][-1]
    return {
        "score": native["score"],
        "components": native["components"],
        "volume_m3": abs(selected["newton"]["state"]["geometry"]["signed_volume_m3"]),
        "iota": selected["newton"]["iota"],
        "psi_l2": source["psi"]["fit_info"]["validation_angle_l2"],
        "surface_l2": dense["relative_l2"],
        "normal_p95": dense["normal_B_sine_p95"],
        "surface_qs": selected["surface_qs_error"],
        "desc_initial": {
            "mean": desc["initial_force_mean_abs_normalized"],
            "p95": desc["initial_force_p95_abs_normalized"],
            "max": desc["initial_force_max_abs_normalized"],
        },
        "desc_final": {
            "mean": desc["final_force_mean_abs_normalized"],
            "p95": desc["final_force_p95_abs_normalized"],
            "max": desc["final_force_max_abs_normalized"],
        },
        "desc_cost": desc["optimizer_cost"],
        "desc_success": desc["optimizer_success"],
        "nested_initial": desc["nested_initial"],
        "nested_final": desc["nested_final"],
    }


def grouped_bars(ax, labels, first, second, legend, *, log=False) -> None:
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, first, width, color="#087f8c", label=legend[0])
    ax.bar(x + width / 2, second, width, color="#d1495b", label=legend[1])
    ax.set_xticks(x, labels)
    if log:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)


def main() -> None:
    args = parse_args()
    candidate = load_delivery(args.candidate)
    baseline = load_delivery(args.baseline)
    labels = (args.candidate_label, args.baseline_label)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": 9, "axes.titleweight": "bold"})
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), constrained_layout=True)

    component_names = ["total", "axis", "psi", "surface", "coordinate", "volume QS", "iota", "coil"]
    component_keys = [None, "axis", "psi", "surface", "coordinate", "volume_qs", "iota", "coil"]
    candidate_scores = [candidate["score"]] + [candidate["components"][key] for key in component_keys[1:]]
    baseline_scores = [baseline["score"]] + [baseline["components"][key] for key in component_keys[1:]]
    grouped_bars(axes[0, 0], component_names, candidate_scores, baseline_scores, labels)
    axes[0, 0].set_ylim(60, 102)
    axes[0, 0].tick_params(axis="x", rotation=25)
    axes[0, 0].set_ylabel("score")
    axes[0, 0].set_title("Current native score")
    axes[0, 0].legend(loc="lower left")

    qs_keys = ["QA_1_0", "QH_1_1", "QP_0_1"]
    grouped_bars(
        axes[0, 1],
        ["QA", "QH", "QP"],
        [candidate["surface_qs"][key] for key in qs_keys],
        [baseline["surface_qs"][key] for key in qs_keys],
        labels,
        log=True,
    )
    axes[0, 1].set_ylabel("surface QS error")
    axes[0, 1].set_title("Selected-surface symmetry errors")

    grouped_bars(
        axes[1, 0],
        ["source psi L2", "surface L2", "normal-B P95"],
        [candidate["psi_l2"], candidate["surface_l2"], candidate["normal_p95"]],
        [baseline["psi_l2"], baseline["surface_l2"], baseline["normal_p95"]],
        labels,
        log=True,
    )
    axes[1, 0].set_ylabel("relative residual")
    axes[1, 0].set_title("Independent fit and surface residuals")

    force_labels = ["initial mean", "initial P95", "initial max", "final mean", "final P95", "final max"]
    candidate_force = list(candidate["desc_initial"].values()) + list(candidate["desc_final"].values())
    baseline_force = list(baseline["desc_initial"].values()) + list(baseline["desc_final"].values())
    grouped_bars(axes[1, 1], force_labels, candidate_force, baseline_force, labels, log=True)
    axes[1, 1].tick_params(axis="x", rotation=25)
    axes[1, 1].set_ylabel("normalized force")
    axes[1, 1].set_title("DESC force before and after 50 iterations")

    figure.suptitle("Complete physical evaluation: historical maximum vs 200-step sample", fontsize=14)
    figure.savefig(args.output_dir / "physical_comparison.png", dpi=200)

    summary = {
        "candidate_label": labels[0],
        "baseline_label": labels[1],
        "candidate": candidate,
        "baseline": baseline,
        "candidate_minus_baseline": {
            "score": candidate["score"] - baseline["score"],
            "volume_m3": candidate["volume_m3"] - baseline["volume_m3"],
            "iota": candidate["iota"] - baseline["iota"],
        },
        "candidate_over_baseline": {
            "psi_l2": candidate["psi_l2"] / baseline["psi_l2"],
            "surface_l2": candidate["surface_l2"] / baseline["surface_l2"],
            "normal_p95": candidate["normal_p95"] / baseline["normal_p95"],
            "surface_qh": candidate["surface_qs"]["QH_1_1"] / baseline["surface_qs"]["QH_1_1"],
            "desc_final_mean": candidate["desc_final"]["mean"] / baseline["desc_final"]["mean"],
            "desc_final_p95": candidate["desc_final"]["p95"] / baseline["desc_final"]["p95"],
            "desc_final_max": candidate["desc_final"]["max"] / baseline["desc_final"]["max"],
            "desc_cost": candidate["desc_cost"] / baseline["desc_cost"],
        },
    }
    (args.output_dir / "physical_comparison.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
