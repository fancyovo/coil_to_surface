from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def rankdata(values: list[float]) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    if "baseline" not in grouped:
        raise ValueError("benchmark has no baseline rows")

    baseline_by_case: dict[int, float] = {}
    for row in grouped["baseline"]:
        baseline_by_case.setdefault(int(row["case_id"]), float(row["result"]["score"]))
    baseline_median = percentile([row["caller_wall_s"] for row in grouped["baseline"]], 50)

    summary = {}
    for variant, variant_rows in grouped.items():
        walls = [float(row["caller_wall_s"]) for row in variant_rows]
        native = [float(row["result"]["timing"]["total_s"]) for row in variant_rows]
        ok = [row for row in variant_rows if row["result"]["status"] == "ok"]
        per_case = {}
        for row in ok:
            per_case.setdefault(int(row["case_id"]), []).append(float(row["result"]["score"]))
        common = sorted(set(per_case) & set(baseline_by_case))
        candidate_scores = [float(np.median(per_case[case_id])) for case_id in common]
        baseline_scores = [baseline_by_case[case_id] for case_id in common]
        deltas = [candidate - baseline for candidate, baseline in zip(candidate_scores, baseline_scores)]
        spearman = float("nan")
        if len(common) >= 3:
            spearman = float(np.corrcoef(rankdata(baseline_scores), rankdata(candidate_scores))[0, 1])
        summary[variant] = {
            "calls": len(variant_rows),
            "ok_fraction": len(ok) / len(variant_rows),
            "caller_wall_p50_s": percentile(walls, 50),
            "caller_wall_p95_s": percentile(walls, 95),
            "caller_wall_max_s": max(walls),
            "native_total_p50_s": percentile(native, 50),
            "speedup_vs_baseline_p50": baseline_median / percentile(walls, 50),
            "score_delta_median": float(np.median(deltas)) if deltas else float("nan"),
            "score_delta_max_abs": max(map(abs, deltas)) if deltas else float("nan"),
            "score_spearman": spearman,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "variant_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )

    variants = list(grouped)
    medians = [summary[name]["caller_wall_p50_s"] for name in variants]
    p95 = [summary[name]["caller_wall_p95_s"] for name in variants]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(variants))
    ax.bar(x, medians, color="#277da1", label="P50")
    ax.scatter(x, p95, color="#d1495b", marker="D", label="P95", zorder=3)
    ax.set_xticks(x, variants, rotation=30, ha="right")
    ax.set_ylabel("Strict-hint score wall time (s)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "variant_runtime.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for variant in variants:
        if variant == "baseline":
            continue
        by_case = defaultdict(list)
        for row in grouped[variant]:
            if row["result"]["status"] == "ok":
                by_case[int(row["case_id"])].append(float(row["result"]["score"]))
        common = sorted(set(by_case) & set(baseline_by_case))
        ax.scatter(
            [baseline_by_case[case_id] for case_id in common],
            [float(np.median(by_case[case_id])) for case_id in common],
            s=24,
            alpha=0.75,
            label=variant,
        )
    limits = [
        min(baseline_by_case.values()) - 1.0,
        max(baseline_by_case.values()) + 1.0,
    ]
    ax.plot(limits, limits, color="black", linewidth=1, linestyle="--")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Baseline score")
    ax.set_ylabel("Candidate score")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "score_preservation.png", dpi=180)
    plt.close(fig)

    baseline_timing = defaultdict(list)
    for row in grouped["baseline"]:
        for name, value in row["result"]["timing"].items():
            baseline_timing[name].append(float(value))
    medians_by_stage = {
        name: float(np.median(values)) for name, values in baseline_timing.items()
        if name != "total_s" and float(np.median(values)) > 1.0e-4
    }
    ordered = sorted(medians_by_stage, key=medians_by_stage.get, reverse=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(ordered[::-1], [medians_by_stage[name] for name in ordered[::-1]], color="#43aa8b")
    ax.set_xlabel("Median native wall time (s)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "baseline_native_stage_timing.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
