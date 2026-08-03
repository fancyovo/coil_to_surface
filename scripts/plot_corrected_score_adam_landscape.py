from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def percentile(values: np.ndarray, value: float) -> float:
    return 100.0 * float(np.mean(values <= value))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay corrected-score calibration points and an Adam trajectory."
    )
    parser.add_argument("--calibration-results", type=Path, required=True)
    parser.add_argument("--adam-history", type=Path, required=True)
    parser.add_argument("--adam-best", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    calibration = read_jsonl(args.calibration_results)
    history = read_jsonl(args.adam_history)
    best_case = read_json(args.adam_best)
    if len(history) != 200:
        raise ValueError(f"expected 200 Adam rows, found {len(history)}")

    background: dict[str, dict[str, np.ndarray]] = {}
    for kind in ("quasr", "random_flow"):
        rows = [
            row["native_score"]
            for row in calibration
            if row["kind"] == kind
            and row["native_score"] is not None
            and row["native_score"]["status"] == "ok"
        ]
        background[kind] = {
            "qh": np.asarray(
                [row["diagnostics"]["qs_target_global_error_per_helicity"] for row in rows]
            ),
            "score": np.asarray([row["score"] for row in rows]),
            "iota": np.asarray(
                [
                    0.5
                    * (
                        abs(row["diagnostics"]["iota_min"])
                        + abs(row["diagnostics"]["iota_max"])
                    )
                    for row in rows
                ]
            ),
            "qa": np.asarray(
                [row["diagnostics"]["qs_qa_global_error_per_helicity"] for row in rows]
            ),
            "qp": np.asarray(
                [row["diagnostics"]["qs_qp_global_error_per_helicity"] for row in rows]
            ),
            "coil": np.asarray([row["components"]["coil"] for row in rows]),
            "surface": np.asarray([row["components"]["surface"] for row in rows]),
        }
        background[kind]["qa_over_qh"] = (
            background[kind]["qa"] / background[kind]["qh"]
        )
        background[kind]["qp_over_qh"] = (
            background[kind]["qp"] / background[kind]["qh"]
        )

    nfp = int(best_case["nfp"])
    helicity_norm = math.hypot(1.0, nfp)
    trajectory = {
        "iteration": np.asarray([row["iteration"] for row in history]),
        "qh": np.asarray([row["current_qh_error"] / helicity_norm for row in history]),
        "score": np.asarray([row["current_score"] for row in history]),
        "iota": np.asarray([abs(row["current_iota"]) for row in history]),
        "qa": np.asarray([row["current_qa_error"] for row in history]),
        "qp": np.asarray([row["current_qp_error"] for row in history]),
    }
    trajectory["qa_over_qh"] = trajectory["qa"] / trajectory["qh"]
    trajectory["qp_over_qh"] = trajectory["qp"] / trajectory["qh"]
    best = best_case["flow_prior_standard_adam"]["native_score"]
    best_diag = best["diagnostics"]
    best_point = {
        "qh": float(best_diag["qs_target_global_error_per_helicity"]),
        "score": float(best["score"]),
        "iota": 0.5
        * (abs(float(best_diag["iota_min"])) + abs(float(best_diag["iota_max"]))),
        "qa": float(best_diag["qs_qa_global_error_per_helicity"]),
        "qp": float(best_diag["qs_qp_global_error_per_helicity"]),
        "coil": float(best["components"]["coil"]),
        "surface": float(best["components"]["surface"]),
    }
    best_point["qa_over_qh"] = best_point["qa"] / best_point["qh"]
    best_point["qp_over_qh"] = best_point["qp"] / best_point["qh"]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(15, 9.2), constrained_layout=True)
    colors = {"quasr": "#176b87", "random_flow": "#bc5a45"}
    labels = {"quasr": "QUASR QH", "random_flow": "random flow"}
    panels = (
        ("score", "native score", False, "Score and QH"),
        ("iota", r"$|\iota|$", False, "Rotational transform and QH"),
        ("qa", "QA error per helicity", True, "QA competitor and QH"),
        ("qp", "QP error per helicity", True, "QP competitor and QH"),
        ("qa_over_qh", "QA error / QH error", True, "QH advantage over QA"),
        ("qp_over_qh", "QP error / QH error", True, "QH advantage over QP"),
    )
    trajectory_scatter = None
    for axis, (metric, ylabel, ylog, title) in zip(axes.flat, panels, strict=True):
        for kind in ("quasr", "random_flow"):
            axis.scatter(
                background[kind]["qh"],
                background[kind][metric],
                s=10,
                alpha=0.26,
                color=colors[kind],
                edgecolors="none",
                label=labels[kind],
            )
        if metric in trajectory:
            axis.plot(
                trajectory["qh"],
                trajectory[metric],
                color="#252525",
                linewidth=1.1,
                alpha=0.7,
                zorder=4,
            )
            trajectory_scatter = axis.scatter(
                trajectory["qh"],
                trajectory[metric],
                c=trajectory["iteration"],
                cmap="viridis",
                s=18,
                zorder=5,
                label="Adam trajectory",
            )
            axis.scatter(
                trajectory["qh"][0],
                trajectory[metric][0],
                marker="s",
                s=60,
                facecolor="white",
                edgecolor="#252525",
                linewidth=1.2,
                zorder=6,
                label="Adam step 1" if metric == "score" else None,
            )
        axis.scatter(
            best_point["qh"],
            best_point[metric],
            marker="*",
            s=145,
            facecolor="#e6b422",
            edgecolor="#252525",
            linewidth=0.8,
            zorder=7,
            label="Adam best" if metric == "score" else None,
        )
        axis.set_xscale("log")
        if ylog:
            axis.set_yscale("log")
        axis.set(
            xlabel="QH differential error per helicity",
            ylabel=ylabel,
            title=title,
        )
        axis.grid(alpha=0.18)
    axes[0, 0].legend(fontsize=8, loc="best")
    if trajectory_scatter is not None:
        colorbar = figure.colorbar(trajectory_scatter, ax=axes[:, :2], shrink=0.85)
        colorbar.set_label("Adam iteration")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)

    quasr = background["quasr"]
    summary = {
        "calibration_points_plotted": {
            "quasr_status_ok": int(len(quasr["qh"])),
            "random_flow_status_ok": int(len(background["random_flow"]["qh"])),
            "excluded_without_QH_coordinate": len(calibration)
            - len(quasr["qh"])
            - len(background["random_flow"]["qh"]),
        },
        "adam_points_plotted": len(history),
        "adam_step1": {key: float(value[0]) for key, value in trajectory.items() if key != "iteration"},
        "adam_best": best_point,
        "adam_best_quasr_percentiles": {
            "score_lower_or_equal_percent": percentile(quasr["score"], best_point["score"]),
            "qh_lower_or_equal_percent": percentile(quasr["qh"], best_point["qh"]),
            "coil_lower_or_equal_percent": percentile(quasr["coil"], best_point["coil"]),
            "surface_lower_or_equal_percent": percentile(quasr["surface"], best_point["surface"]),
        },
        "note": "All six panels contain the full optimizer trajectory. Per-step coil and surface component vectors were not recorded, so those quantities are retained only as final-best background percentiles rather than shown as incomplete trajectories.",
    }
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
