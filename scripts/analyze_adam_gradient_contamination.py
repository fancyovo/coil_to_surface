from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.optimize_flow_prior_zo_adam import (
    cosine_similarity,
    gradient_from_pairs,
    orthogonal_directions,
    rms,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def adam_update(
    first_moment: np.ndarray,
    second_moment: np.ndarray,
    *,
    beta1: float,
    beta2: float,
    step: int,
    learning_rate: float,
    epsilon: float,
) -> np.ndarray:
    first_hat = first_moment / (1.0 - beta1**step)
    second_hat = second_moment / (1.0 - beta2**step)
    return learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)


def recovery_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events = []
    for index, row in enumerate(rows):
        if float(row["valid_endpoint_fraction"]) >= 1.0:
            continue
        prior_best = max(
            [float(item["current_score"]) for item in rows[:index]],
            default=float(row["current_score"]),
        )
        recovered_at = next(
            (
                int(item["iteration"])
                for item in rows[index + 1 :]
                if float(item["current_score"]) >= prior_best
            ),
            None,
        )
        events.append(
            {
                "iteration": int(row["iteration"]),
                "status_fraction": float(row["valid_endpoint_fraction"]),
                "prior_best_score": prior_best,
                "post_step_score": float(row["current_score"]),
                "immediate_drawdown": prior_best - float(row["current_score"]),
                "recovered_at_iteration": recovered_at,
                "recovery_steps": (
                    None
                    if recovered_at is None
                    else recovered_at - int(row["iteration"])
                ),
            }
        )
    return events


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay saved zeroth-order gradients and compare Adam moments with a "
            "counterfactual that removes directions containing invalid endpoints."
        )
    )
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.history)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not rows or [int(row["iteration"]) for row in rows] != list(
        range(1, len(rows) + 1)
    ):
        raise ValueError("history must be a complete zero-moment run starting at step 1")

    shape = tuple(int(value) for value in manifest["noise_shape"])
    direction_count = int(manifest["directions"])
    perturbation = float(manifest["perturbation"])
    learning_rate = float(manifest["learning_rate"])
    beta1, beta2 = (float(value) for value in manifest["betas"])
    epsilon = float(manifest["adam_epsilon"])
    rng = np.random.default_rng(int(manifest["seed"]))

    first = np.zeros(shape, dtype=np.float64)
    second = np.zeros(shape, dtype=np.float64)
    clean_first = np.zeros(shape, dtype=np.float64)
    clean_second = np.zeros(shape, dtype=np.float64)
    diagnostics = []
    update_rms_errors = []

    for row in rows:
        iteration = int(row["iteration"])
        directions = orthogonal_directions(rng, shape, direction_count)
        raw_delta = np.asarray(row["raw_direction_deltas"], dtype=np.float64)
        used_delta = np.asarray(row.get("used_direction_deltas", raw_delta), dtype=np.float64)
        statuses = list(row["pair_statuses"])
        invalid_direction = np.asarray(
            [
                statuses[index] != "ok" or statuses[index + direction_count] != "ok"
                for index in range(direction_count)
            ],
            dtype=bool,
        )
        clean_delta = used_delta.copy()
        clean_delta[invalid_direction] = 0.0

        gradient, _ = gradient_from_pairs(
            0.5 * used_delta,
            -0.5 * used_delta,
            directions,
            perturbation,
            delta_clip=None,
        )
        clean_gradient, _ = gradient_from_pairs(
            0.5 * clean_delta,
            -0.5 * clean_delta,
            directions,
            perturbation,
            delta_clip=None,
        )
        gradient *= float(row.get("gradient_clip_scale", 1.0))

        first = beta1 * first + (1.0 - beta1) * gradient
        second = beta2 * second + (1.0 - beta2) * gradient * gradient
        clean_first = beta1 * clean_first + (1.0 - beta1) * clean_gradient
        clean_second = beta2 * clean_second + (1.0 - beta2) * clean_gradient * clean_gradient
        update = adam_update(
            first,
            second,
            beta1=beta1,
            beta2=beta2,
            step=iteration,
            learning_rate=learning_rate,
            epsilon=epsilon,
        )
        clean_update = adam_update(
            clean_first,
            clean_second,
            beta1=beta1,
            beta2=beta2,
            step=iteration,
            learning_rate=learning_rate,
            epsilon=epsilon,
        )
        update_rms_errors.append(abs(rms(update) - float(row["update_rms"])))
        contamination = first - clean_first
        diagnostics.append(
            {
                "iteration": iteration,
                "current_score": float(row["current_score"]),
                "best_score": float(row["best_score"]),
                "invalid_direction_count": int(np.count_nonzero(invalid_direction)),
                "raw_gradient_rms": rms(gradient),
                "clean_gradient_rms": rms(clean_gradient),
                "first_moment_rms": rms(first),
                "clean_first_moment_rms": rms(clean_first),
                "contamination_first_moment_rms": rms(contamination),
                "contamination_first_moment_fraction": rms(contamination)
                / max(rms(first), 1.0e-30),
                "actual_update_rms": rms(update),
                "clean_counterfactual_update_rms": rms(clean_update),
                "update_difference_rms": rms(update - clean_update),
                "actual_clean_update_cosine": cosine_similarity(update, clean_update),
            }
        )

    maximum_reconstruction_error = max(update_rms_errors)
    if maximum_reconstruction_error > 2.0e-7:
        raise RuntimeError(
            f"Adam replay does not match saved updates: {maximum_reconstruction_error}"
        )

    dirty_steps = [
        int(row["iteration"])
        for row in diagnostics
        if row["invalid_direction_count"] > 0
    ]
    after_first_dirty = [
        row for row in diagnostics if dirty_steps and row["iteration"] >= dirty_steps[0]
    ]
    summary = {
        "history": str(args.history.resolve()),
        "manifest": str(args.manifest.resolve()),
        "iterations": len(rows),
        "dirty_iterations": dirty_steps,
        "maximum_update_rms_reconstruction_error": maximum_reconstruction_error,
        "maximum_contamination_first_moment_fraction_after_first_dirty": (
            max(row["contamination_first_moment_fraction"] for row in after_first_dirty)
            if after_first_dirty
            else 0.0
        ),
        "minimum_actual_clean_update_cosine_after_first_dirty": (
            min(row["actual_clean_update_cosine"] for row in after_first_dirty)
            if after_first_dirty
            else 1.0
        ),
        "recovery_events": recovery_events(rows),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.out_dir / "diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iterations = [row["iteration"] for row in diagnostics]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].plot(iterations, [row["current_score"] for row in diagnostics], label="current")
    axes[0, 0].plot(iterations, [row["best_score"] for row in diagnostics], label="best")
    axes[0, 0].set(ylabel="native score", title="Saved Adam trajectory")
    axes[0, 0].legend()
    axes[0, 1].plot(iterations, [row["raw_gradient_rms"] for row in diagnostics], label="recorded")
    axes[0, 1].plot(iterations, [row["clean_gradient_rms"] for row in diagnostics], label="invalid directions removed")
    axes[0, 1].set(yscale="log", ylabel="gradient RMS", title="Dirty-gradient spikes")
    axes[0, 1].legend()
    axes[1, 0].plot(
        iterations,
        [row["contamination_first_moment_fraction"] for row in diagnostics],
        label="first-moment contamination fraction",
    )
    axes[1, 0].set(ylabel="fraction", xlabel="iteration", title="Persisting first-moment contamination")
    axes[1, 1].plot(
        iterations,
        [row["actual_clean_update_cosine"] for row in diagnostics],
        label="actual vs clean update cosine",
    )
    axes[1, 1].plot(
        iterations,
        [row["update_difference_rms"] for row in diagnostics],
        label="update difference RMS",
    )
    axes[1, 1].set(xlabel="iteration", title="Effect on Adam update")
    axes[1, 1].legend()
    for axis in axes.ravel():
        for dirty_step in dirty_steps:
            axis.axvline(dirty_step, color="tab:red", alpha=0.25, linewidth=1)
        axis.grid(alpha=0.25)
    figure.savefig(args.out_dir / "gradient_contamination.png", dpi=180)
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
