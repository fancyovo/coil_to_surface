from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


VARIANT_ORDER = (
    "baseline",
    "axis_hint_no_fp64",
    "axis_hint_fp64_offset1e3",
    "axis_hint_no_fp64_offset1e3",
)
PAIRS = (
    ("Exact hint", "baseline", "axis_hint_no_fp64"),
    (
        "Hint offset by 1e-3",
        "axis_hint_fp64_offset1e3",
        "axis_hint_no_fp64_offset1e3",
    ),
)
TIMING_KEYS = (
    "total_s",
    "axis_search_s",
    "axis_candidate_refine_s",
    "axis_fp64_verify_s",
    "axis_topology_s",
    "axis_trace_s",
    "psi_fit_s",
    "surface_screen_s",
    "alpha_solve_s",
)
DIAGNOSTIC_KEYS = (
    "axis_R",
    "axis_Z",
    "axis_residual",
    "axis_topology_trace",
    "axis_topology_det",
    "axis_ellipse_aspect",
    "psi_angle_p95",
    "psi_angle_l2",
    "surface_effective_level",
    "surface_drift_relative_p95",
    "alpha_normal_B_relative_l2",
    "iota_min",
    "qs_target_global_error_per_helicity",
)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def rankdata(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    ranks[order] = np.arange(len(array), dtype=float)
    return ranks


def spearman(left: list[float], right: list[float]) -> float:
    if len(left) < 3:
        return float("nan")
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def finite(value: object) -> float:
    result = float(value)
    return result if math.isfinite(result) else float("nan")


def median_result(rows: list[dict]) -> dict | None:
    successful = [row for row in rows if row["result"]["status"] == "ok"]
    if not successful:
        return None
    first = successful[0]["result"]
    return {
        "score": float(np.median([row["result"]["score"] for row in successful])),
        "components": {
            key: float(np.median([row["result"]["components"][key] for row in successful]))
            for key in first["components"]
        },
        "diagnostics": {
            key: float(np.median([finite(row["result"]["diagnostics"][key]) for row in successful]))
            for key in DIAGNOSTIC_KEYS
        },
    }


def timing_summary(rows: list[dict], key: str) -> dict[str, float]:
    if key == "caller_wall_s":
        values = [float(row[key]) for row in rows]
    else:
        values = [float(row["result"]["timing"][key]) for row in rows]
    return {
        "p50_ms": 1000.0 * percentile(values, 50),
        "p95_ms": 1000.0 * percentile(values, 95),
        "max_ms": 1000.0 * max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line
    ]
    grouped: dict[str, list[dict]] = defaultdict(list)
    by_variant_case: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        variant = row["variant"]
        grouped[variant].append(row)
        by_variant_case[variant][int(row["case_id"])].append(row)
    missing = [variant for variant in VARIANT_ORDER if variant not in grouped]
    if missing:
        raise ValueError(f"missing variants: {missing}")

    per_case = {
        variant: {
            case_id: result
            for case_id, case_rows in by_variant_case[variant].items()
            if (result := median_result(case_rows)) is not None
        }
        for variant in VARIANT_ORDER
    }
    summary = {"variants": {}, "pairs": {}}
    for variant in VARIANT_ORDER:
        variant_rows = grouped[variant]
        successful_rows = [row for row in variant_rows if row["result"]["status"] == "ok"]
        summary["variants"][variant] = {
            "calls": len(variant_rows),
            "status_counts": dict(Counter(row["result"]["status"] for row in variant_rows)),
            "timing": {
                "caller_wall_s": timing_summary(variant_rows, "caller_wall_s"),
                **{key: timing_summary(variant_rows, key) for key in TIMING_KEYS},
            },
            "diagnostics": {
                key: {
                    "p50": percentile(
                        [finite(row["result"]["diagnostics"][key]) for row in successful_rows], 50
                    ),
                    "p95": percentile(
                        [finite(row["result"]["diagnostics"][key]) for row in successful_rows], 95
                    ),
                    "max": max(
                        finite(row["result"]["diagnostics"][key]) for row in successful_rows
                    ),
                }
                for key in DIAGNOSTIC_KEYS
            },
        }

    paired_csv = []
    for label, reference_variant, candidate_variant in PAIRS:
        reference = per_case[reference_variant]
        candidate = per_case[candidate_variant]
        common = sorted(set(reference) & set(candidate))
        reference_scores = [reference[case_id]["score"] for case_id in common]
        candidate_scores = [candidate[case_id]["score"] for case_id in common]
        score_deltas = [candidate - base for base, candidate in zip(reference_scores, candidate_scores)]
        reference_wall = summary["variants"][reference_variant]["timing"]["caller_wall_s"]["p50_ms"]
        candidate_wall = summary["variants"][candidate_variant]["timing"]["caller_wall_s"]["p50_ms"]
        top_count = max(1, math.ceil(0.1 * len(common)))
        reference_top = set(np.asarray(common)[np.argsort(reference_scores)[-top_count:]])
        candidate_top = set(np.asarray(common)[np.argsort(candidate_scores)[-top_count:]])
        diagnostic_deltas = {
            key: [
                abs(candidate[case_id]["diagnostics"][key] - reference[case_id]["diagnostics"][key])
                for case_id in common
            ]
            for key in DIAGNOSTIC_KEYS
        }
        component_deltas = {
            key: [
                abs(candidate[case_id]["components"][key] - reference[case_id]["components"][key])
                for case_id in common
            ]
            for key in reference[common[0]]["components"]
        }
        pair_record = {
            "reference_variant": reference_variant,
            "candidate_variant": candidate_variant,
            "paired_ok_cases": len(common),
            "p50_wall_speedup": reference_wall / candidate_wall,
            "score_delta_median": float(np.median(score_deltas)),
            "score_delta_p95_abs": percentile(list(map(abs, score_deltas)), 95),
            "score_delta_max_abs": max(map(abs, score_deltas)),
            "score_spearman": spearman(reference_scores, candidate_scores),
            "top_decile_overlap": len(reference_top & candidate_top) / top_count,
            "diagnostic_delta_p95": {
                key: percentile(values, 95) for key, values in diagnostic_deltas.items()
            },
            "diagnostic_delta_max": {
                key: max(values) for key, values in diagnostic_deltas.items()
            },
            "component_delta_p95": {
                key: percentile(values, 95) for key, values in component_deltas.items()
            },
            "component_delta_max": {
                key: max(values) for key, values in component_deltas.items()
            },
        }
        summary["pairs"][label] = pair_record
        for case_id in common:
            paired_csv.append(
                {
                    "pair": label,
                    "case_id": case_id,
                    "reference_score": reference[case_id]["score"],
                    "candidate_score": candidate[case_id]["score"],
                    "score_delta": candidate[case_id]["score"] - reference[case_id]["score"],
                    **{
                        f"reference_{key}": reference[case_id]["diagnostics"][key]
                        for key in DIAGNOSTIC_KEYS
                    },
                    **{
                        f"candidate_{key}": candidate[case_id]["diagnostics"][key]
                        for key in DIAGNOSTIC_KEYS
                    },
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    with (args.output_dir / "paired_cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired_csv[0]))
        writer.writeheader()
        writer.writerows(paired_csv)

    labels = ["FP64\nexact", "No FP64\nexact", "FP64\noffset", "No FP64\noffset"]
    p50 = [summary["variants"][variant]["timing"]["caller_wall_s"]["p50_ms"] for variant in VARIANT_ORDER]
    p95 = [summary["variants"][variant]["timing"]["caller_wall_s"]["p95_ms"] for variant in VARIANT_ORDER]
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    positions = np.arange(len(labels))
    bars = ax.bar(positions, p50, color=["#6b7280", "#217a8a", "#6b7280", "#217a8a"])
    ax.errorbar(positions, p50, yerr=np.asarray(p95) - np.asarray(p50), fmt="none", color="black", capsize=5)
    for bar, value in zip(bars, p50):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 10, f"{value:.1f} ms", ha="center")
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Score-call wall time (ms)")
    ax.set_title("P50 score latency; error bars reach P95")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "runtime_comparison.png", dpi=180)
    plt.close(fig)

    stack_keys = (
        ("axis_candidate_refine_s", "Mixed Newton"),
        ("axis_fp64_verify_s", "FP64 verification"),
        ("axis_topology_s", "Topology"),
        ("axis_trace_s", "Axis sampling"),
    )
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    bottom = np.zeros(len(VARIANT_ORDER))
    colors = ("#4c78a8", "#e45756", "#72b7b2", "#f2cf5b")
    for (key, label), color in zip(stack_keys, colors):
        values = np.asarray([
            summary["variants"][variant]["timing"][key]["p50_ms"]
            for variant in VARIANT_ORDER
        ])
        ax.bar(positions, values, bottom=bottom, label=label, color=color)
        bottom += values
    ax.set_xticks(positions, labels)
    ax.set_ylabel("P50 stage time (ms)")
    ax.set_title("Magnetic-axis path composition")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "axis_path_breakdown.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.3))
    for ax, (label, reference_variant, candidate_variant) in zip(axes, PAIRS):
        common = sorted(set(per_case[reference_variant]) & set(per_case[candidate_variant]))
        left = [per_case[reference_variant][case_id]["score"] for case_id in common]
        right = [per_case[candidate_variant][case_id]["score"] for case_id in common]
        low = min(left + right) - 1.0
        high = max(left + right) + 1.0
        ax.scatter(left, right, s=24, alpha=0.7, color="#217a8a")
        ax.plot([low, high], [low, high], "k--", linewidth=1)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_xlabel("FP64-verification score")
        ax.set_ylabel("No-FP64 score")
        ax.set_title(label)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "score_preservation.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
