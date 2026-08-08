from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


CHECKPOINTS = (2000, 2500, 3000, 3500, 4000, 4341, 4500, 4750, 5000)
COMPONENTS = ("axis", "psi", "surface", "coordinate", "volume_qs", "iota", "coil")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def best_metadata(path: Path) -> dict[str, Any]:
    return read_json(path)["flow_prior_standard_adam"]


def native_snapshot(metadata: dict[str, Any]) -> dict[str, Any]:
    native = metadata["native_score"]
    diagnostics = native["diagnostics"]
    return {
        "iteration": int(metadata["iteration"]),
        "score": float(metadata["best_score"]),
        "status": native["status"],
        "components": {
            name: float(native["components"][name]) for name in COMPONENTS
        },
        "qh_error": float(diagnostics["qs_global_error"]),
        "qh_error_per_helicity": float(
            diagnostics["qs_target_global_error_per_helicity"]
        ),
        "qa_error": float(diagnostics["qs_qa_global_error"]),
        "qp_error": float(diagnostics["qs_qp_global_error"]),
        "iota": float(diagnostics["iota_min"]),
        "surface_level": float(diagnostics["surface_level"]),
        "surface_inverse_aspect_ratio": float(
            diagnostics["surface_inverse_aspect_ratio"]
        ),
        "surface_volume": float(diagnostics["surface_volume"]),
    }


def tail_slope(values: np.ndarray, count: int) -> float:
    x = np.arange(count, dtype=np.float64)
    return float(np.polyfit(x, values[-count:], 1)[0] * count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline-best", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = read_json(args.run / "manifest.json")
    summary = read_json(args.run / "summary.json")
    resume_events = read_jsonl(args.run / "resume_events.jsonl")
    history = read_jsonl(args.run / "history.jsonl")
    baseline = native_snapshot(best_metadata(args.baseline_best))
    continued = native_snapshot(best_metadata(args.run / "best.json"))

    expected_iterations = list(range(1, 5001))
    actual_iterations = [int(row["iteration"]) for row in history]
    if actual_iterations != expected_iterations:
        raise ValueError("history is not a complete contiguous 1--5000 trajectory")
    if summary["status"] != "ok" or summary["stop_reason"] != "completed_iterations":
        raise ValueError("run did not complete normally")
    if int(summary["resumed_from_iteration"]) != 2000:
        raise ValueError("run did not resume from iteration 2000")
    if int(summary["completed_iterations"]) != 5000:
        raise ValueError("run did not reach iteration 5000")
    if not resume_events or (
        int(resume_events[-1]["saved_iteration"]) != 2000
        or int(resume_events[-1]["requested_iterations"]) != 5000
    ):
        raise ValueError("latest resume event does not describe 2000 -> 5000")
    if int(continued["iteration"]) != int(summary["best_iteration"]):
        raise ValueError("best artifact and summary disagree")
    if manifest["gpu_ids"] != [0, 1]:
        raise ValueError("run is not the requested two-score-GPU configuration")
    if (
        float(manifest["betas"][0]) != 0.7
        or int(manifest["flow_steps"]) != 128
        or int(manifest["directions"]) != 2
        or manifest["gradient_estimator"] != "central"
        or not manifest["flow_pipeline"]
        or manifest["score_surface_mode"] != "continuous"
        or not manifest["axis_continuation"]
    ):
        raise ValueError("run does not match the accepted production configuration")

    iterations = np.asarray(actual_iterations, dtype=np.int64)
    scores = np.asarray([float(row["current_score"]) for row in history])
    running_best = np.asarray([float(row["best_score"]) for row in history])
    qh_per_helicity = np.asarray(
        [float(row["current_qh_error"]) / 4.0 for row in history]
    )
    update_rms = np.asarray([float(row["update_rms"]) for row in history])
    added = history[2000:]
    added_iteration_wall = np.asarray(
        [float(row["iteration_wall_s"]) for row in added], dtype=np.float64
    )
    prior_wall_s = float(history[1999]["total_wall_s"])
    final_history_wall_s = float(history[-1]["total_wall_s"])
    added_wall_s = final_history_wall_s - prior_wall_s

    checkpoint_rows: list[dict[str, Any]] = []
    for step in CHECKPOINTS:
        row = history[step - 1]
        checkpoint_rows.append(
            {
                "iteration": step,
                "current_score": float(row["current_score"]),
                "running_best": float(row["best_score"]),
                "qh_error_per_helicity": float(row["current_qh_error"]) / 4.0,
                "iota": float(row["current_iota"]),
                "update_rms": float(row["update_rms"]),
            }
        )

    windows = (100, 200, 400, 500, 659, 1000, 2000, 3000)
    running_best_tail_gains = {
        str(window): float(running_best[-1] - running_best[-window - 1])
        for window in windows
    }
    result: dict[str, Any] = {
        "status": "accepted_native_score_run",
        "job_id": 33694,
        "completed_iterations": int(summary["completed_iterations"]),
        "completed_adam_steps": int(summary["completed_adam_steps"]),
        "resumed_from_iteration": int(summary["resumed_from_iteration"]),
        "resume_center_score": float(summary["resume_center_score"]),
        "final_score": float(summary["final_score"]),
        "best_score": float(summary["best_score"]),
        "best_iteration": int(summary["best_iteration"]),
        "iterations_after_last_best": 5000 - int(summary["best_iteration"]),
        "gain_from_2000_best": float(summary["best_score"] - running_best[1999]),
        "gain_from_initial": float(summary["best_score"] - summary["initial_score"]),
        "final_gap_below_best": float(summary["best_score"] - summary["final_score"]),
        "best_updates_after_2000": int(
            np.sum(np.diff(running_best[1999:]) > 1.0e-12)
        ),
        "running_best_tail_gains": running_best_tail_gains,
        "current_score_slope_last_200": tail_slope(scores, 200),
        "current_score_slope_last_500": tail_slope(scores, 500),
        "current_score_mean_last_200": float(np.mean(scores[-200:])),
        "current_score_std_last_200": float(np.std(scores[-200:])),
        "summary_total_wall_s": float(summary["total_wall_s"]),
        "prior_history_wall_s": prior_wall_s,
        "additional_history_wall_s": added_wall_s,
        "additional_wall_s_per_iteration": added_wall_s / len(added),
        "additional_iteration_wall_mean_s": float(np.mean(added_iteration_wall)),
        "additional_iteration_wall_p95_s": float(
            np.quantile(added_iteration_wall, 0.95)
        ),
        "additional_iteration_wall_max_s": float(np.max(added_iteration_wall)),
        "additional_flow_wall_s": float(
            sum(
                float(row["pair_decode_wall_s"])
                + float(row["center_decode_wall_s"])
                for row in added
            )
        ),
        "additional_score_wall_s": float(
            sum(
                float(row["pair_score_wall_s"])
                + float(row["center_score_wall_s"])
                for row in added
            )
        ),
        "additional_applied_gradient_steps": sum(
            bool(row["gradient_step_applied"]) for row in added
        ),
        "additional_skipped_gradient_steps": sum(
            not bool(row["gradient_step_applied"]) for row in added
        ),
        "additional_accepted_center_updates": sum(
            bool(row["center_update_accepted"]) for row in added
        ),
        "additional_temporal_rejections": sum(
            bool(row["temporal_step_rejected"]) for row in added
        ),
        "additional_non_ok_endpoints": sum(
            status != "ok" for row in added for status in row["pair_statuses"]
        ),
        "additional_non_ok_centers": sum(
            row["current_status"] != "ok" for row in added
        ),
        "additional_backtracked_centers": sum(
            bool(row["center_backtracking"]) for row in added
        ),
        "additional_pipeline_cache_hits": sum(
            bool(row["endpoint_decode_cache_hit"]) for row in added
        ),
        "resume_event_count": len(resume_events),
        "baseline_best": baseline,
        "continued_best": continued,
        "component_delta": {
            name: continued["components"][name] - baseline["components"][name]
            for name in COMPONENTS
        },
        "checkpoints": checkpoint_rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "acceptance_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "checkpoints.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(checkpoint_rows[0]))
        writer.writeheader()
        writer.writerows(checkpoint_rows)

    figure, axes = plt.subplots(2, 2, figsize=(13.8, 9.0), constrained_layout=True)
    axes[0, 0].plot(iterations, scores, color="#42708f", linewidth=0.65, label="current")
    axes[0, 0].plot(
        iterations, running_best, color="#c23b31", linewidth=1.7, label="running best"
    )
    axes[0, 0].axvline(2000, color="#555555", linestyle="--", linewidth=1.0)
    axes[0, 0].axvline(
        continued["iteration"], color="#c23b31", linestyle=":", linewidth=1.0
    )
    axes[0, 0].set(
        title="Complete score trajectory",
        xlabel="iteration",
        ylabel="continuous QH score",
    )
    axes[0, 0].legend()

    axes[0, 1].plot(
        iterations[1999:], scores[1999:], color="#42708f", linewidth=0.55, alpha=0.55
    )
    axes[0, 1].plot(
        iterations[1999:], running_best[1999:], color="#c23b31", linewidth=1.8
    )
    axes[0, 1].scatter(
        [baseline["iteration"], continued["iteration"]],
        [baseline["score"], continued["score"]],
        color=["#555555", "#c23b31"],
        zorder=3,
    )
    axes[0, 1].set(
        title="Continuation from iteration 2000",
        xlabel="iteration",
        ylabel="continuous QH score",
    )

    axes[1, 0].semilogy(
        iterations[1999:], qh_per_helicity[1999:], color="#16865b", linewidth=0.75
    )
    axes[1, 0].set(
        title="Current differential QH error",
        xlabel="iteration",
        ylabel="QH error / helicity",
    )

    kernel = np.ones(31, dtype=np.float64) / 31.0
    smooth_update = np.convolve(update_rms[1999:], kernel, mode="valid")
    axes[1, 1].plot(
        iterations[2029:], smooth_update, color="#a05a2c", linewidth=1.0
    )
    rejected = np.asarray(
        [not bool(row["gradient_step_applied"]) for row in history], dtype=bool
    )
    continuation_rejected = rejected & (iterations >= 2000)
    axes[1, 1].scatter(
        iterations[continuation_rejected],
        update_rms[continuation_rejected],
        s=12,
        color="#c23b31",
        label="skipped update after 2000",
        zorder=3,
    )
    axes[1, 1].set(
        title="Latent update scale",
        xlabel="iteration",
        ylabel="31-step mean RMS",
    )
    axes[1, 1].set_xlim(2000, 5000)
    axes[1, 1].legend()
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
    figure.savefig(args.output_dir / "score_convergence_5000.png", dpi=200)
    plt.close(figure)

    x = np.arange(len(COMPONENTS), dtype=np.float64)
    width = 0.36
    figure, axis = plt.subplots(figsize=(11.5, 5.2), constrained_layout=True)
    axis.bar(
        x - width / 2,
        [baseline["components"][name] for name in COMPONENTS],
        width,
        color="#6f7c80",
        label=f"iteration {baseline['iteration']}",
    )
    axis.bar(
        x + width / 2,
        [continued["components"][name] for name in COMPONENTS],
        width,
        color="#c23b31",
        label=f"iteration {continued['iteration']}",
    )
    axis.set_xticks(x, COMPONENTS)
    axis.set_ylim(60, 101)
    axis.set(
        title="Native-score components at the two best points",
        ylabel="component score",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(args.output_dir / "best_component_comparison.png", dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
