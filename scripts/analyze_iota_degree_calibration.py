from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


VARIANTS = ("iota_constant", "iota_cubic")


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def average_result(rows: list[dict]) -> dict:
    first = rows[0]["result"]
    if any(row["result"]["status"] != "ok" for row in rows):
        return {"status": "non_ok"}

    def mean(path: tuple[str, ...]) -> float:
        values = []
        for row in rows:
            value = row["result"]
            for key in path:
                value = value[key]
            values.append(float(value))
        return float(np.mean(values))

    return {
        "status": "ok",
        "score": mean(("score",)),
        "caller_wall_s": float(np.mean([row["caller_wall_s"] for row in rows])),
        "alpha_solve_s": mean(("timing", "alpha_solve_s")),
        "alpha_relative_l2": mean(("diagnostics", "alpha_relative_l2")),
        "alpha_normal_B_relative_l2": mean(
            ("diagnostics", "alpha_normal_B_relative_l2")
        ),
        "iota_min": mean(("diagnostics", "iota_min")),
        "iota_max": mean(("diagnostics", "iota_max")),
        "qh_error": mean(("diagnostics", "qs_target_global_error_per_helicity")),
        "coordinate": mean(("components", "coordinate")),
        "volume_qs": mean(("components", "volume_qs")),
        "iota_component": mean(("components", "iota")),
        "raw_status": first["status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    status_counts: dict[str, Counter] = {variant: Counter() for variant in VARIANTS}
    for row in rows:
        variant = row["variant"]
        if variant not in VARIANTS:
            continue
        grouped[(int(row["case_id"]), variant)].append(row)
        status_counts[variant][row["result"]["status"]] += 1

    cases = sorted({case_id for case_id, _ in grouped})
    paired = []
    for case_id in cases:
        averaged = {
            variant: average_result(grouped[(case_id, variant)]) for variant in VARIANTS
        }
        if all(averaged[variant]["status"] == "ok" for variant in VARIANTS):
            paired.append({"case_id": case_id, **averaged})
    if not paired:
        raise RuntimeError("no paired successful cases")

    def array(variant: str, key: str) -> np.ndarray:
        return np.asarray([item[variant][key] for item in paired], dtype=float)

    constant_score = array("iota_constant", "score")
    cubic_score = array("iota_cubic", "score")
    constant_alpha = array("iota_constant", "alpha_relative_l2")
    cubic_alpha = array("iota_cubic", "alpha_relative_l2")
    cubic_min = array("iota_cubic", "iota_min")
    cubic_max = array("iota_cubic", "iota_max")
    score_delta = cubic_score - constant_score
    alpha_ratio = cubic_alpha / np.maximum(constant_alpha, 1.0e-300)
    span = cubic_max - cubic_min

    count_top = max(1, int(np.ceil(0.1 * len(paired))))
    top_constant = set(np.argsort(constant_score)[-count_top:])
    top_cubic = set(np.argsort(cubic_score)[-count_top:])
    summary = {
        "rows": len(rows),
        "paired_cases": len(paired),
        "status_counts": {key: dict(value) for key, value in status_counts.items()},
        "score": {
            "constant_p50": percentile(constant_score, 50),
            "cubic_p50": percentile(cubic_score, 50),
            "delta_median": percentile(score_delta, 50),
            "delta_p95_abs": percentile(np.abs(score_delta), 95),
            "delta_max_abs": float(np.max(np.abs(score_delta))),
            "spearman": spearman(constant_score, cubic_score),
            "top_decile_overlap": len(top_constant & top_cubic) / count_top,
        },
        "alpha_relative_l2": {
            "constant_p50": percentile(constant_alpha, 50),
            "cubic_p50": percentile(cubic_alpha, 50),
            "ratio_p50": percentile(alpha_ratio, 50),
            "ratio_p95": percentile(alpha_ratio, 95),
            "improved_fraction": float(np.mean(cubic_alpha <= constant_alpha)),
        },
        "iota_profile_span": {
            "p50": percentile(span, 50),
            "p95": percentile(span, 95),
            "max": float(np.max(span)),
        },
        "timing_ms": {},
    }
    for variant in VARIANTS:
        for key in ("caller_wall_s", "alpha_solve_s"):
            values = 1000.0 * array(variant, key)
            summary["timing_ms"].setdefault(variant, {})[key] = {
                "p50": percentile(values, 50),
                "p95": percentile(values, 95),
            }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "paired.jsonl").open("w", encoding="utf-8") as stream:
        for item in paired:
            stream.write(json.dumps(item, allow_nan=True, separators=(",", ":")) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))
    low = min(np.min(constant_score), np.min(cubic_score))
    high = max(np.max(constant_score), np.max(cubic_score))
    axes[0].scatter(constant_score, cubic_score, s=24, alpha=0.75)
    axes[0].plot([low, high], [low, high], "k--", linewidth=1)
    axes[0].set(xlabel="Constant-iota score", ylabel="Cubic-iota score", title="Score")

    low = min(np.min(constant_alpha), np.min(cubic_alpha))
    high = max(np.max(constant_alpha), np.max(cubic_alpha))
    axes[1].scatter(constant_alpha, cubic_alpha, s=24, alpha=0.75)
    axes[1].plot([low, high], [low, high], "k--", linewidth=1)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set(
        xlabel="Constant-iota alpha residual",
        ylabel="Cubic-iota alpha residual",
        title="Joint-fit residual",
    )

    axes[2].scatter(cubic_min, cubic_max, c=cubic_score, cmap="viridis", s=28, alpha=0.8)
    low = min(np.min(cubic_min), np.min(cubic_max))
    high = max(np.max(cubic_min), np.max(cubic_max))
    axes[2].plot([low, high], [low, high], "k--", linewidth=1)
    axes[2].set(xlabel="Cubic iota minimum", ylabel="Cubic iota maximum", title="Fitted profile range")
    fig.tight_layout()
    fig.savefig(args.output_dir / "iota_degree_calibration.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
