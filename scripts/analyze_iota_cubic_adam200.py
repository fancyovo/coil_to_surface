from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    history = [
        json.loads(line)
        for line in (args.run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    if len(history) != 200 or int(history[-1]["iteration"]) != 200:
        raise ValueError("expected one complete 200-step trajectory")

    iterations = np.asarray([int(row["iteration"]) for row in history])
    current = np.asarray([float(row["current_score"]) for row in history])
    best = np.asarray([float(row["best_score"]) for row in history])
    wall = np.asarray([float(row["iteration_wall_s"]) for row in history])
    score_wall = np.asarray(
        [float(row["pair_score_wall_s"]) + float(row["center_score_wall_s"]) for row in history]
    )
    flow_wall = np.asarray(
        [float(row["center_decode_wall_s"]) + float(row["prefetched_endpoint_decode_wall_s"]) for row in history]
    )

    def checkpoint(step: int) -> dict:
        if step == 0:
            return {"iteration": 0, "current_score": float(summary["initial_score"])}
        row = history[step - 1]
        return {
            "iteration": step,
            "current_score": float(row["current_score"]),
            "best_score": float(row["best_score"]),
            "qh_error": float(row["current_qh_error"]),
            "iota_min": float(row["current_iota"]),
        }

    steady = slice(1, None)
    acceptance = {
        "manifest": {
            key: manifest[key]
            for key in (
                "nfp",
                "n_base_coils",
                "seed",
                "directions",
                "gradient_estimator",
                "perturbation",
                "learning_rate",
                "betas",
                "flow_steps",
                "flow_pipeline",
                "iota_degree",
                "axis_hint_verification",
                "native_lib_sha256",
            )
        },
        "summary": summary,
        "checkpoints": [checkpoint(step) for step in (0, 25, 50, 100, 150, 200)],
        "iteration_timing_s": {
            "mean_all": float(np.mean(wall)),
            "p50_all": float(np.percentile(wall, 50)),
            "p95_all": float(np.percentile(wall, 95)),
            "mean_steady_after_first": float(np.mean(wall[steady])),
            "score_mean": float(np.mean(score_wall)),
            "flow_mean": float(np.mean(flow_wall)),
        },
        "step_counts": {
            "gradient_applied": sum(bool(row["gradient_step_applied"]) for row in history),
            "center_accepted": sum(bool(row["center_update_accepted"]) for row in history),
            "temporal_rejected": sum(bool(row["temporal_step_rejected"]) for row in history),
            "non_ok_centers": sum(row["current_status"] != "ok" for row in history),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "acceptance_summary.json").write_text(
        json.dumps(acceptance, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)
    axes[0].plot(iterations, current, label="Current score", linewidth=1.3)
    axes[0].plot(iterations, best, label="Running best", linewidth=2.0)
    axes[0].axhline(float(summary["initial_score"]), color="0.4", linestyle="--", label="Initial")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Cubic-iota Adam optimization from the historical 10000-step start")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(iterations, wall, color="#777777", alpha=0.45, label="Iteration wall time")
    window = 15
    kernel = np.ones(window) / window
    rolling = np.convolve(wall, kernel, mode="valid")
    axes[1].plot(iterations[window - 1 :], rolling, color="#d1495b", linewidth=2, label="15-step mean")
    axes[1].set(xlabel="Iteration", ylabel="Time per step (s)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "score_and_timing.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
