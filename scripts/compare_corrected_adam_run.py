from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_history(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or [row["iteration"] for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f"history is empty or non-contiguous: {path}")
    return rows


def run_summary(root: Path) -> dict[str, Any]:
    summary = read_json(root / "summary.json")
    history = read_history(root / "history.jsonl")
    return {
        "initial_score": float(summary["initial_score"]),
        "final_score": float(summary["final_score"]),
        "best_score": float(summary["best_score"]),
        "best_iteration": int(summary["best_iteration"]),
        "gain": float(summary["best_score"] - summary["initial_score"]),
        "iterations_recorded": len(history),
        "applied_steps": int(summary.get("applied_steps", sum(row.get("gradient_step_applied", False) for row in history))),
        "skipped_steps": int(summary.get("skipped_steps", sum(not row.get("gradient_step_applied", False) for row in history))),
        "wall_s": float(history[-1]["total_wall_s"]),
    }


def curve(history: list[dict[str, Any]], name: str, initial: float) -> np.ndarray:
    return np.asarray([initial] + [float(row[name]) for row in history], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the same-start ABI-8 and corrected ABI-9 Adam runs.")
    parser.add_argument("--old-run", type=Path, required=True)
    parser.add_argument("--new-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    old_summary = read_json(args.old_run / "summary.json")
    new_summary = read_json(args.new_run / "summary.json")
    old = read_history(args.old_run / "history.jsonl")
    new = read_history(args.new_run / "history.jsonl")
    norm = math.hypot(1.0, 4.0)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    styles = (("ABI 8, invalid G", old, old_summary, "#8d6b52"), ("ABI 9, corrected G", new, new_summary, "#176b87"))
    for label, history, summary, color in styles:
        iterations = np.arange(len(history) + 1)
        axes[0, 0].plot(iterations, curve(history, "current_score", float(summary["initial_score"])), color=color, alpha=0.6, label=f"{label}: current")
        axes[0, 0].plot(iterations, curve(history, "best_score", float(summary["initial_score"])), color=color, linewidth=2.0, label=f"{label}: best")
        axes[0, 1].plot(iterations[1:], [float(row["current_qh_error"]) / norm for row in history], color=color, label=f"{label}: QH")
        axes[1, 0].plot(iterations[1:], [float(row["current_qa_error"]) for row in history], color=color, linestyle="--", label=f"{label}: QA")
        axes[1, 0].plot(iterations[1:], [float(row["current_qp_error"]) for row in history], color=color, label=f"{label}: QP")
        axes[1, 1].plot(iterations[1:], [float(row["update_rms"]) for row in history], color=color, label=label)
    axes[0, 0].set(title="Same-start Adam score curves", xlabel="iteration", ylabel="native score")
    axes[0, 1].set(title="Target QH error per helicity", xlabel="iteration", ylabel="RMS", yscale="log")
    axes[1, 0].set(title="Competitor errors per helicity", xlabel="iteration", ylabel="RMS", yscale="log")
    axes[1, 1].set(title="Accepted/proposed Adam update scale", xlabel="iteration", ylabel="latent RMS", yscale="log")
    for axis in axes.flat:
        axis.grid(alpha=0.18)
        axis.legend(fontsize=8)
    figure.savefig(args.output_dir / "adam_old_vs_corrected.png", dpi=180)
    plt.close(figure)

    comparison = {
        "old_abi8_invalid_G": run_summary(args.old_run),
        "new_abi9_corrected_G": run_summary(args.new_run),
        "comparison_scope": "same initial latent, seed, Adam settings, and direction schedule; trajectories diverge because the objective changed",
    }
    (args.output_dir / "adam_comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, separators=(",", ":")))


if __name__ == "__main__":
    main()
