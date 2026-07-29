from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MILESTONES = (1, 2, 4, 8, 16, 32, 64, 96, 128, 160)
QH_HELICITY_NORM = math.sqrt(2.0)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def minimum_absolute_interval(lower: float, upper: float) -> float:
    if lower <= 0.0 <= upper:
        return 0.0
    return min(abs(lower), abs(upper))


def compact_candidate(row: dict) -> dict:
    native = row["native_score"]
    diagnostics = native["diagnostics"]
    qh_error = float(diagnostics["qs_global_error"]) / QH_HELICITY_NORM
    qa_error = float(diagnostics["qs_qa_global_error"])
    qp_error = float(diagnostics["qs_qp_global_error"])
    competitor_error = min(qa_error, qp_error)
    iota_min = float(diagnostics["iota_min"])
    iota_max = float(diagnostics["iota_max"])
    return {
        "generation": int(row["generation"]),
        "candidate": int(row["candidate"]),
        "score": float(row["score"]),
        "components": native["components"],
        "iota_min": iota_min,
        "iota_max": iota_max,
        "iota_star": minimum_absolute_interval(iota_min, iota_max),
        "surface_inverse_aspect_ratio": float(
            diagnostics["surface_inverse_aspect_ratio"]
        ),
        "surface_volume": float(diagnostics["surface_volume"]),
        "surface_one_period_drift_relative_p95": float(
            diagnostics["surface_one_period_drift_relative_p95"]
        ),
        "surface_long_drift_relative_p95": float(
            diagnostics["surface_drift_relative_p95"]
        ),
        "qh_error_per_helicity": qh_error,
        "qa_error": qa_error,
        "qp_error": qp_error,
        "competitor_error": competitor_error,
        "qh_to_competitor_ratio": qh_error / max(competitor_error, 1.0e-300),
        "helicity_advantage": float(
            diagnostics["score_qh_helicity_advantage"]
        ),
        "helicity_quality": float(diagnostics["score_qh_helicity_quality"]),
        "helicity_factor": float(
            diagnostics.get("score_qh_total_helicity_factor", math.nan)
        ),
        "iota_factor": float(
            diagnostics.get("score_qh_total_iota_factor", math.nan)
        ),
        "score_before_gates": float(
            diagnostics.get("score_before_qh_iota_gate", math.nan)
        ),
    }


def stream_candidates(path: Path) -> tuple[list[dict], Counter[str]]:
    successful: list[dict] = []
    statuses: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            status = row["status"]
            statuses[status] += 1
            if status == "ok":
                successful.append(compact_candidate(row))
    return successful, statuses


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def selected_cases(rows: list[dict]) -> dict[str, dict | None]:
    iota_rows = [row for row in rows if row["iota_star"] >= 1.0]
    target_rows = [
        row
        for row in rows
        if row["iota_star"] >= 1.0 and row["helicity_advantage"] >= 0.10
    ]
    useful_target_rows = [
        row
        for row in target_rows
        if row["surface_inverse_aspect_ratio"] >= 0.02
    ]
    return {
        "best_total_score": max(rows, key=lambda row: row["score"]),
        "best_helicity_advantage": max(
            rows, key=lambda row: row["helicity_advantage"]
        ),
        "lowest_qh_error": min(
            rows, key=lambda row: row["qh_error_per_helicity"]
        ),
        "lowest_qh_error_iota_ge_1": (
            min(iota_rows, key=lambda row: row["qh_error_per_helicity"])
            if iota_rows
            else None
        ),
        "best_target_like_score": (
            max(target_rows, key=lambda row: row["score"])
            if target_rows
            else None
        ),
        "best_useful_target_like_score": (
            max(useful_target_rows, key=lambda row: row["score"])
            if useful_target_rows
            else None
        ),
    }


def generation_audit(rows: list[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["generation"]].append(row)
    audit = []
    for generation in sorted(grouped):
        current = grouped[generation]
        target_count = sum(
            row["iota_star"] >= 1.0 and row["helicity_advantage"] >= 0.10
            for row in current
        )
        audit.append(
            {
                "generation": generation,
                "successful": len(current),
                "score_max": max(row["score"] for row in current),
                "score_median": float(np.median([row["score"] for row in current])),
                "helicity_advantage_max": max(
                    row["helicity_advantage"] for row in current
                ),
                "helicity_advantage_median": float(
                    np.median([row["helicity_advantage"] for row in current])
                ),
                "target_like_count": target_count,
                "iota_ge_1_count": sum(row["iota_star"] >= 1.0 for row in current),
                "qh_error_min": min(
                    row["qh_error_per_helicity"] for row in current
                ),
            }
        )
    return audit


def build_audit(
    summary: dict,
    rows: list[dict],
    statuses: Counter[str],
    quasr_p10: float,
    quasr_median: float,
) -> dict:
    counts = {
        "iota_ge_1": sum(row["iota_star"] >= 1.0 for row in rows),
        "helicity_advantage_ge_0_10": sum(
            row["helicity_advantage"] >= 0.10 for row in rows
        ),
        "helicity_advantage_ge_0_20": sum(
            row["helicity_advantage"] >= 0.20 for row in rows
        ),
        "helicity_advantage_ge_0_30": sum(
            row["helicity_advantage"] >= 0.30 for row in rows
        ),
        "iota_ge_1_and_advantage_ge_0_10": sum(
            row["iota_star"] >= 1.0 and row["helicity_advantage"] >= 0.10
            for row in rows
        ),
        "iota_ge_1_advantage_ge_0_10_size_ge_0_02": sum(
            row["iota_star"] >= 1.0
            and row["helicity_advantage"] >= 0.10
            and row["surface_inverse_aspect_ratio"] >= 0.02
            for row in rows
        ),
        "score_ge_quasr_p10": sum(row["score"] >= quasr_p10 for row in rows),
        "score_ge_quasr_median": sum(
            row["score"] >= quasr_median for row in rows
        ),
    }
    return {
        "configuration": {
            key: summary["manifest"].get(key)
            for key in (
                "target",
                "nfp",
                "n_base_coils",
                "seed",
                "iterations",
                "popsize",
                "elite",
                "sigma",
                "smoothing",
                "gpu_ids",
            )
        },
        "total_candidates": sum(statuses.values()),
        "successful_candidates": len(rows),
        "status_counts": dict(statuses),
        "total_wall_s": float(summary["total_wall_s"]),
        "best_score": float(summary["best_score"]),
        "quasr_score_reference": {"p10": quasr_p10, "median": quasr_median},
        "successful_quantiles": {
            "score": quantiles([row["score"] for row in rows]),
            "iota_star": quantiles([row["iota_star"] for row in rows]),
            "surface_inverse_aspect_ratio": quantiles(
                [row["surface_inverse_aspect_ratio"] for row in rows]
            ),
            "qh_error_per_helicity": quantiles(
                [row["qh_error_per_helicity"] for row in rows]
            ),
            "helicity_advantage": quantiles(
                [row["helicity_advantage"] for row in rows]
            ),
        },
        "threshold_counts": counts,
        "selected_cases": selected_cases(rows),
        "generation_audit": generation_audit(rows),
        "milestones": [
            row for row in summary["generations"] if row["generation"] in MILESTONES
        ],
    }


def plot_audit(summary: dict, audit: dict, rows: list[dict], output: Path) -> None:
    generations = summary["generations"]
    generation_detail = audit["generation_audit"]
    x = np.asarray([row["generation"] for row in generations])
    best = np.asarray([row["best_score"] for row in generations])
    generation_max = np.asarray([row["score_max"] for row in generations])
    ok_fraction = np.asarray(
        [
            row["statuses"].get("ok", 0) / summary["manifest"]["popsize"]
            for row in generations
        ]
    )
    drift_fraction = np.asarray(
        [
            row["statuses"].get("drift_rejected", 0)
            / summary["manifest"]["popsize"]
            for row in generations
        ]
    )
    sigma = np.asarray([row["sigma_mean"] for row in generations])
    advantage_max = np.asarray(
        [row["helicity_advantage_max"] for row in generation_detail]
    )
    advantage_median = np.asarray(
        [row["helicity_advantage_median"] for row in generation_detail]
    )

    figure, axes = plt.subplots(2, 2, figsize=(11.8, 8.0))
    axes[0, 0].plot(x, best, color="#b43b2f", linewidth=2.1, label="best so far")
    axes[0, 0].plot(
        x, generation_max, color="#277c83", linewidth=1.2, alpha=0.8,
        label="generation max",
    )
    axes[0, 0].axhline(
        audit["quasr_score_reference"]["p10"], color="#555555", linestyle=":",
        label="QUASR QH P10",
    )
    axes[0, 0].set_ylabel("score")
    axes[0, 0].legend(fontsize=8)

    sigma_axis = axes[0, 1].twinx()
    axes[0, 1].plot(x, ok_fraction, color="#277c83", label="ok fraction")
    axes[0, 1].plot(x, drift_fraction, color="#e09f3e", label="drift rejected")
    sigma_axis.plot(x, sigma, color="#b43b2f", label="mean sigma")
    axes[0, 1].set_ylabel("candidate fraction")
    sigma_axis.set_ylabel("CEM mean sigma", color="#b43b2f")
    lines = axes[0, 1].lines + sigma_axis.lines
    axes[0, 1].legend(
        lines, [line.get_label() for line in lines], fontsize=8, loc="center right"
    )

    axes[1, 0].plot(x, advantage_max, color="#b43b2f", label="generation max")
    axes[1, 0].plot(x, advantage_median, color="#277c83", label="generation median")
    axes[1, 0].axhline(0.10, color="#555555", linestyle=":", label="calibrated bad")
    axes[1, 0].axhline(0.30, color="#555555", linestyle="--", label="calibrated good")
    axes[1, 0].set_xlabel("generation")
    axes[1, 0].set_ylabel("QH relative helicity advantage")
    axes[1, 0].legend(fontsize=8)

    qh = np.asarray([row["qh_error_per_helicity"] for row in rows])
    competitor = np.asarray([row["competitor_error"] for row in rows])
    iota = np.asarray([row["iota_star"] for row in rows])
    scatter = axes[1, 1].scatter(
        qh, competitor, c=np.clip(iota, 0.0, 1.5), s=7, alpha=0.35,
        cmap="viridis", rasterized=True,
    )
    positive = np.concatenate([qh[qh > 0.0], competitor[competitor > 0.0]])
    lower = float(np.min(positive))
    upper = float(np.max(positive))
    axes[1, 1].plot([lower, upper], [lower, upper], color="#555555", linestyle=":")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("QH error per helicity")
    axes[1, 1].set_ylabel("best QA/QP competitor error")
    figure.colorbar(scatter, ax=axes[1, 1], label=r"minimum $|\iota|$")

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.suptitle("Score v3: 160-generation QH CEM audit")
    figure.tight_layout()
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a score-v3 QH CEM run.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--quasr-p10", type=float, default=41.31)
    parser.add_argument("--quasr-median", type=float, default=51.44)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = load_json(args.summary)
    rows, statuses = stream_candidates(args.candidates)
    if not rows:
        raise RuntimeError("CEM run contains no successful candidates")
    audit = build_audit(
        summary, rows, statuses, args.quasr_p10, args.quasr_median
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "score_v3_cem_audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    plot_audit(summary, audit, rows, args.output_dir / "score_v3_cem_audit.png")


if __name__ == "__main__":
    main()
