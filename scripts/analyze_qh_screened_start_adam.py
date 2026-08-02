from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("no completed screened-start Adam runs")
    initial = np.asarray([row["initial_score"] for row in rows], dtype=np.float64)
    best = np.asarray([row["best_score"] for row in rows], dtype=np.float64)
    elapsed = np.asarray(
        [row["timing_s"]["end_to_end"] for row in rows], dtype=np.float64
    )
    selection_elapsed = np.asarray(
        [row["timing_s"]["candidate_selection"] for row in rows], dtype=np.float64
    )
    return {
        "format": "qh_screened_start_adam_multiseed_v1",
        "run_count": len(rows),
        "nfp_values": sorted({int(row["nfp"]) for row in rows}),
        "n_base_coils_values": sorted({int(row["n_base_coils"]) for row in rows}),
        "candidate_count_values": sorted({int(row["candidate_count"]) for row in rows}),
        "initial_score": {
            "min": float(np.min(initial)),
            "median": float(np.median(initial)),
            "mean": float(np.mean(initial)),
            "max": float(np.max(initial)),
        },
        "best_score": {
            "min": float(np.min(best)),
            "median": float(np.median(best)),
            "mean": float(np.mean(best)),
            "max": float(np.max(best)),
        },
        "gain": {
            "min": float(np.min(best - initial)),
            "median": float(np.median(best - initial)),
            "mean": float(np.mean(best - initial)),
            "max": float(np.max(best - initial)),
        },
        "thresholds": {
            "best_ge_40_count": int(np.sum(best >= 40.0)),
            "best_ge_40_rate": float(np.mean(best >= 40.0)),
            "best_ge_50_count": int(np.sum(best >= 50.0)),
            "best_ge_50_rate": float(np.mean(best >= 50.0)),
        },
        "timing_s": {
            "candidate_selection_mean": float(np.mean(selection_elapsed)),
            "candidate_selection_median": float(np.median(selection_elapsed)),
            "end_to_end_mean": float(np.mean(elapsed)),
            "end_to_end_median": float(np.median(elapsed)),
            "end_to_end_min": float(np.min(elapsed)),
            "end_to_end_max": float(np.max(elapsed)),
        },
        "runs": rows,
    }


def plot(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["candidate_seed"]) for row in rows]
    initial = np.asarray([row["initial_score"] for row in rows], dtype=float)
    best = np.asarray([row["best_score"] for row in rows], dtype=float)
    elapsed = np.asarray([row["timing_s"]["end_to_end"] for row in rows]) / 60.0
    x = np.arange(len(rows))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].scatter(x, initial, color="#486a88", label="best of 128 start", zorder=3)
    axes[0].scatter(x, best, color="#a34137", label="best after 50 steps", zorder=3)
    for index in x:
        axes[0].plot([index, index], [initial[index], best[index]], color="#888888", lw=1)
    axes[0].axhline(40.0, color="#555555", ls="--", label="score 40 / 50")
    axes[0].axhline(50.0, color="#222222", ls=":")
    axes[0].set(xticks=x, xticklabels=labels, xlabel="candidate seed", ylabel="native score")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].bar(x, elapsed, color="#4f7a64")
    axes[1].set(
        xticks=x,
        xticklabels=labels,
        xlabel="candidate seed",
        ylabel="end-to-end minutes",
        title="128-way selection plus 50-step Adam",
    )
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(axis="y", alpha=0.2)
    figure.savefig(output, dpi=190)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate screened-start Adam runs.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.run_root.glob("seed_*/experiment_summary.json"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    summary = analyze(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "multiseed_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    plot(rows, args.output_dir / "screened_start_adam_multiseed.png")
    print(json.dumps({"event": "complete", **summary}, separators=(",", ":")))


if __name__ == "__main__":
    main()
