from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_history(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    iterations = [int(row["iteration"]) for row in rows]
    if iterations != list(range(1, len(rows) + 1)):
        raise ValueError(f"history iterations are not contiguous in {path}")
    return rows


def score_curve(
    initial_score: float, history: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    iterations = np.asarray([0, *[int(row["iteration"]) for row in history]])
    current = np.asarray(
        [initial_score, *[float(row["current_score"]) for row in history]],
        dtype=np.float64,
    )
    best = np.asarray(
        [initial_score, *[float(row["best_score"]) for row in history]],
        dtype=np.float64,
    )
    return iterations, current, best


def tail_progress(
    initial_score: float, history: list[dict[str, Any]], *, tail_steps: int = 10
) -> dict[str, Any]:
    iterations, _, best = score_curve(initial_score, history)
    final_iteration = int(iterations[-1])
    if final_iteration < tail_steps:
        raise ValueError("history is shorter than the requested tail")
    best_by_iteration = dict(zip(iterations.tolist(), best.tolist(), strict=True))
    last_10_start = final_iteration - tail_steps
    last_5_start = final_iteration - 5
    tail_rows = [
        row for row in history if int(row["iteration"]) > last_10_start
    ]
    return {
        "final_iteration": final_iteration,
        "best_gain_last_10_steps": float(
            best_by_iteration[final_iteration] - best_by_iteration[last_10_start]
        ),
        "best_gain_last_5_steps": float(
            best_by_iteration[final_iteration] - best_by_iteration[last_5_start]
        ),
        "applied_adam_steps_last_10": int(
            sum(bool(row["gradient_step_applied"]) for row in tail_rows)
        ),
    }


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


def plot_score_curves(
    rows: list[dict[str, Any]],
    histories: dict[int, list[dict[str, Any]]],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab10").colors
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14.5, 6.2),
        gridspec_kw={"width_ratios": [2.15, 1.0]},
        constrained_layout=True,
    )
    labels = []
    gain_10 = []
    gain_5 = []
    for index, row in enumerate(rows):
        seed = int(row["candidate_seed"])
        color = colors[index % len(colors)]
        iterations, current, best = score_curve(
            float(row["initial_score"]), histories[seed]
        )
        short_label = str(seed)[-2:]
        labels.append(short_label)
        axes[0].plot(iterations, current, color=color, lw=0.9, alpha=0.24)
        axes[0].plot(
            iterations,
            best,
            color=color,
            lw=2.0,
            label=f"seed ...{short_label} (best {best[-1]:.2f})",
        )
        progress = tail_progress(float(row["initial_score"]), histories[seed])
        gain_10.append(progress["best_gain_last_10_steps"])
        gain_5.append(progress["best_gain_last_5_steps"])

    axes[0].axvspan(40, 50, color="#6d6d6d", alpha=0.07, label="last 10 steps")
    axes[0].axhline(40.0, color="#555555", ls="--", lw=1.2)
    axes[0].axhline(50.0, color="#222222", ls=":", lw=1.2)
    axes[0].set(
        xlim=(0, 50),
        xlabel="Adam iteration",
        ylabel="native score",
        title="Eight screened starts: current score (faint) and running best (solid)",
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend(ncol=2, fontsize=8.5, loc="lower right")

    y = np.arange(len(rows))
    bar_colors = [colors[index % len(colors)] for index in range(len(rows))]
    axes[1].barh(y, gain_10, color=bar_colors, alpha=0.82, label="steps 41-50")
    axes[1].scatter(gain_5, y, color="#1f1f1f", marker="|", s=150, label="steps 46-50")
    for index, value in enumerate(gain_10):
        axes[1].text(value + 0.05, index, f"{value:.2f}", va="center", fontsize=8)
    axes[1].set(
        yticks=y,
        yticklabels=[f"seed ...{label}" for label in labels],
        xlabel="running-best score gain",
        title="Remaining progress near step 50",
    )
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", alpha=0.2)
    axes[1].legend(fontsize=8.5, loc="lower right")
    figure.savefig(output, dpi=190)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate screened-start Adam runs.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.run_root.glob("seed_*/experiment_summary.json"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    histories = {
        int(row["candidate_seed"]): load_history(
            args.run_root
            / f"seed_{int(row['candidate_seed'])}"
            / "adam"
            / "history.jsonl"
        )
        for row in rows
    }
    for row in rows:
        seed = int(row["candidate_seed"])
        row["tail_progress"] = tail_progress(float(row["initial_score"]), histories[seed])
    summary = analyze(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "multiseed_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    plot(rows, args.output_dir / "screened_start_adam_multiseed.png")
    plot_score_curves(
        rows,
        histories,
        args.output_dir / "screened_start_adam_score_curves.png",
    )
    print(json.dumps({"event": "complete", **summary}, separators=(",", ":")))


if __name__ == "__main__":
    main()
