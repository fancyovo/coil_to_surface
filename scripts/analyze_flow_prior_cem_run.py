from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MILESTONES = (1, 2, 4, 8, 16, 32, 64, 96, 119, 128, 144)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def summarize(summary: dict, *, quasr_p10: float, quasr_median: float) -> dict:
    generations = summary["generations"]
    if not generations:
        raise ValueError("summary has no completed generations")
    statuses: Counter[str] = Counter()
    for row in generations:
        statuses.update(row["statuses"])
    total = sum(statuses.values())
    first = generations[0]
    final = generations[-1]
    best = summary["best_diagnostics"]
    nfp = int(summary["manifest"]["nfp"])
    qh_per_helicity = float(best["qs_global_error"]) / math.hypot(1.0, nfp)
    competitor = min(
        float(best["qs_qa_global_error"]), float(best["qs_qp_global_error"])
    )
    generation_by_number = {int(row["generation"]): row for row in generations}
    milestones = []
    for generation in MILESTONES:
        if generation not in generation_by_number:
            continue
        row = generation_by_number[generation]
        milestones.append(
            {
                "generation": generation,
                "score_mean": float(row["score_mean"]),
                "score_median": float(row["score_median"]),
                "score_max": float(row["score_max"]),
                "running_best": float(row["best_score"]),
                "ok_rate": row["statuses"].get("ok", 0) / sum(row["statuses"].values()),
                "sigma_mean": float(row["sigma_mean"]),
            }
        )
    total_wall_s = float(summary["total_wall_s"])
    decode_wall_s = sum(float(row["decode_wall_s"]) for row in generations)
    score_wall_s = sum(float(row["score_wall_s"]) for row in generations)
    return {
        "job": {
            "stop_reason": summary["stop_reason"],
            "completed_generations": len(generations),
            "evaluated_candidates": total,
            "total_wall_s": total_wall_s,
            "wall_s_per_candidate": total_wall_s / total,
            "decode_wall_s": decode_wall_s,
            "decode_fraction": decode_wall_s / total_wall_s,
            "native_score_wall_s": score_wall_s,
        },
        "status_counts": dict(statuses),
        "status_rates": {key: value / total for key, value in statuses.items()},
        "optimization": {
            "first_score_max": float(first["score_max"]),
            "final_score_max": float(final["score_max"]),
            "best_score": float(summary["best_score"]),
            "best_generation": int(summary["best_generation"]),
            "best_candidate": int(summary["best_candidate"]),
            "best_improvement": float(summary["best_score"] - first["score_max"]),
            "first_sigma_mean": float(first["sigma_mean"]),
            "final_sigma_mean": float(final["sigma_mean"]),
            "first_ok_rate": first["statuses"].get("ok", 0) / sum(first["statuses"].values()),
            "final_ok_rate": final["statuses"].get("ok", 0) / sum(final["statuses"].values()),
            "quasr_p10": quasr_p10,
            "quasr_median": quasr_median,
            "above_quasr_p10": float(summary["best_score"]) >= quasr_p10,
            "above_quasr_median": float(summary["best_score"]) >= quasr_median,
        },
        "best_physics": {
            "components": summary["best_components"],
            "surface_level": float(best["surface_level"]),
            "surface_inverse_aspect_ratio": float(best["surface_inverse_aspect_ratio"]),
            "surface_one_period_drift_relative_p95": float(
                best["surface_one_period_drift_relative_p95"]
            ),
            "surface_long_drift_relative_p95": float(best["surface_drift_relative_p95"]),
            "iota": 0.5 * (float(best["iota_min"]) + float(best["iota_max"])),
            "qh_global_error": float(best["qs_global_error"]),
            "qh_error_per_helicity": qh_per_helicity,
            "qa_error": float(best["qs_qa_global_error"]),
            "qp_error": float(best["qs_qp_global_error"]),
            "competitor_error": competitor,
            "qh_to_competitor_ratio": qh_per_helicity / competitor,
            "qh_helicity_advantage": float(best["score_qh_helicity_advantage"]),
            "qh_helicity_quality": float(best["score_qh_helicity_quality"]),
        },
        "milestones": milestones,
    }


def plot(summary: dict, output: Path, *, quasr_p10: float, quasr_median: float) -> None:
    rows = summary["generations"]
    generation = np.asarray([row["generation"] for row in rows])
    status_names = ("ok", "no_axis", "no_surface", "drift_rejected", "flux_rejected")
    totals = np.asarray([sum(row["statuses"].values()) for row in rows], dtype=float)

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    score_axis = axes[0, 0]
    score_axis.plot(generation, [row["score_mean"] for row in rows], label="mean", alpha=0.75)
    score_axis.plot(generation, [row["score_median"] for row in rows], label="median")
    score_axis.plot(generation, [row["score_max"] for row in rows], label="generation max")
    score_axis.plot(generation, [row["best_score"] for row in rows], label="running best", linewidth=2.2)
    score_axis.axhline(quasr_p10, color="#8c564b", linestyle="--", label="QUASR QH P10")
    score_axis.axhline(quasr_median, color="#7f7f7f", linestyle=":", label="QUASR QH median")
    score_axis.set_ylabel("score")
    score_axis.legend(fontsize=8, ncol=2)

    status_axis = axes[0, 1]
    for name in status_names:
        values = np.asarray([row["statuses"].get(name, 0) for row in rows]) / totals
        status_axis.plot(generation, values, label=name)
    status_axis.set_ylabel("generation fraction")
    status_axis.set_ylim(0.0, 1.0)
    status_axis.legend(fontsize=8, ncol=2)

    sigma_axis = axes[1, 0]
    sigma_axis.plot(generation, [row["sigma_mean"] for row in rows], label="sigma mean")
    sigma_axis.plot(generation, [row["sigma_min"] for row in rows], label="sigma min")
    sigma_axis.plot(generation, [row["sigma_max"] for row in rows], label="sigma max")
    sigma_axis.plot(generation, [row["mean_rms"] for row in rows], label="noise mean RMS")
    sigma_axis.set_ylabel("flow-prior noise scale")
    sigma_axis.legend(fontsize=8)

    timing_axis = axes[1, 1]
    timing_axis.plot(generation, [row["score_wall_s"] for row in rows], label="native score")
    timing_axis.plot(generation, [row["decode_wall_s"] for row in rows], label="flow decode")
    timing_axis.plot(generation, [row["wall_s"] for row in rows], label="total", alpha=0.75)
    timing_axis.set_ylabel("wall time per generation [s]")
    timing_axis.legend(fontsize=8)

    for axis in axes.flat:
        axis.set_xlabel("generation")
        axis.grid(alpha=0.22)
    figure.suptitle("CEM over QH flow-prior noise: 4-fold, 3-coil condition")
    figure.tight_layout()
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit CEM optimization over flow-prior noise.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quasr-p10", type=float, default=41.31)
    parser.add_argument("--quasr-median", type=float, default=51.44)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    audit = summarize(
        summary, quasr_p10=args.quasr_p10, quasr_median=args.quasr_median
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "audit.json", audit)
    plot(
        summary,
        args.output_dir / "convergence.png",
        quasr_p10=args.quasr_p10,
        quasr_median=args.quasr_median,
    )


if __name__ == "__main__":
    main()
