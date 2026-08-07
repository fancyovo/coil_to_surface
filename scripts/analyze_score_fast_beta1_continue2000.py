from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


CHECKPOINTS = (600, 800, 1000, 1200, 1400, 1600, 1800, 1900, 1945, 2000)
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
    result: dict[str, Any] = {
        "iteration": int(metadata["iteration"]),
        "score": float(metadata["best_score"]),
        "status": native["status"],
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
    result["components"] = {
        name: float(native["components"][name]) for name in COMPONENTS
    }
    return result


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
    current_best = native_snapshot(best_metadata(args.run / "best.json"))

    if len(history) != 2000 or [int(row["iteration"]) for row in history] != list(
        range(1, 2001)
    ):
        raise ValueError("history is not a complete contiguous 1--2000 trajectory")
    if summary["status"] != "ok" or summary["stop_reason"] != "completed_iterations":
        raise ValueError("run did not complete normally")
    if int(summary["resumed_from_iteration"]) != 600:
        raise ValueError("run did not resume from iteration 600")
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
    if int(current_best["iteration"]) != int(summary["best_iteration"]):
        raise ValueError("best artifact and summary disagree")

    iterations = np.asarray([int(row["iteration"]) for row in history])
    scores = np.asarray([float(row["current_score"]) for row in history])
    running_best = np.asarray([float(row["best_score"]) for row in history])
    qh_per_helicity = np.asarray(
        [float(row["current_qh_error"]) / 4.0 for row in history]
    )
    update_rms = np.asarray([float(row["update_rms"]) for row in history])
    added = history[600:]
    added_iteration_wall = np.asarray(
        [float(row["iteration_wall_s"]) for row in added], dtype=np.float64
    )
    prior_wall_s = float(history[599]["total_wall_s"])
    additional_wall_s = float(summary["total_wall_s"]) - prior_wall_s

    checkpoint_rows = []
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

    best_updates_after_600 = int(
        np.sum(np.diff(running_best[599:], prepend=running_best[599]) > 1.0e-12)
    )
    result: dict[str, Any] = {
        "status": "accepted_native_score_run",
        "job_id": 33166,
        "completed_iterations": int(summary["completed_iterations"]),
        "completed_adam_steps": int(summary["completed_adam_steps"]),
        "resumed_from_iteration": int(summary["resumed_from_iteration"]),
        "resume_center_score": float(summary["resume_center_score"]),
        "final_score": float(summary["final_score"]),
        "best_score": float(summary["best_score"]),
        "best_iteration": int(summary["best_iteration"]),
        "gain_from_600_best": float(summary["best_score"] - running_best[599]),
        "gain_from_initial": float(summary["best_score"] - summary["initial_score"]),
        "iterations_after_last_best": 2000 - int(summary["best_iteration"]),
        "best_updates_after_600": best_updates_after_600,
        "running_best_gain_last_100": float(running_best[-1] - running_best[-101]),
        "running_best_gain_last_200": float(running_best[-1] - running_best[-201]),
        "running_best_gain_last_400": float(running_best[-1] - running_best[-401]),
        "current_score_slope_last_100": tail_slope(scores, 100),
        "current_score_slope_last_200": tail_slope(scores, 200),
        "running_best_slope_last_200": tail_slope(running_best, 200),
        "total_wall_s": float(summary["total_wall_s"]),
        "prior_wall_s": prior_wall_s,
        "additional_wall_s": additional_wall_s,
        "additional_wall_s_per_iteration": additional_wall_s / len(added),
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
        "additional_accepted_center_updates": sum(
            bool(row["center_update_accepted"]) for row in added
        ),
        "additional_temporal_rejections": sum(
            bool(row["temporal_step_rejected"]) for row in added
        ),
        "additional_non_ok_endpoints": sum(
            status != "ok" for row in added for status in row["pair_statuses"]
        ),
        "additional_backtracked_centers": sum(bool(row["center_backtracking"]) for row in added),
        "additional_pipeline_cache_hits": sum(
            bool(row["endpoint_decode_cache_hit"]) for row in added
        ),
        "resume_event_count": len(resume_events),
        "baseline_best": baseline,
        "continued_best": current_best,
        "component_delta": {
            name: current_best["components"][name] - baseline["components"][name]
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

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), constrained_layout=True)
    axes[0, 0].plot(iterations, scores, color="#42708f", linewidth=0.8, label="current")
    axes[0, 0].plot(iterations, running_best, color="#c23b31", linewidth=1.6, label="running best")
    axes[0, 0].axvline(600, color="#555555", linestyle="--", linewidth=1.0)
    axes[0, 0].axvline(current_best["iteration"], color="#c23b31", linestyle=":", linewidth=1.0)
    axes[0, 0].set(title="Score trajectory", xlabel="iteration", ylabel="continuous QH score")
    axes[0, 0].legend()

    axes[0, 1].plot(iterations[599:], running_best[599:], color="#c23b31", linewidth=1.8)
    axes[0, 1].scatter(
        [baseline["iteration"], current_best["iteration"]],
        [baseline["score"], current_best["score"]],
        color=["#555555", "#c23b31"],
        zorder=3,
    )
    axes[0, 1].set(title="Running best after resume", xlabel="iteration", ylabel="continuous QH score")

    axes[1, 0].semilogy(iterations[599:], qh_per_helicity[599:], color="#16865b", linewidth=0.9)
    axes[1, 0].set(title="Current differential QH error", xlabel="iteration", ylabel="QH error / helicity")

    kernel = np.ones(31, dtype=np.float64) / 31.0
    smooth_update = np.convolve(update_rms[599:], kernel, mode="valid")
    axes[1, 1].plot(iterations[629:], smooth_update, color="#a05a2c", linewidth=1.1)
    rejected = np.asarray([not bool(row["gradient_step_applied"]) for row in history])
    axes[1, 1].scatter(
        iterations[rejected],
        update_rms[rejected],
        s=13,
        color="#c23b31",
        label="skipped update",
        zorder=3,
    )
    axes[1, 1].set(title="Latent update scale", xlabel="iteration", ylabel="31-step mean RMS")
    axes[1, 1].legend()
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
    figure.savefig(args.output_dir / "score_convergence_2000.png", dpi=200)
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
        [current_best["components"][name] for name in COMPONENTS],
        width,
        color="#c23b31",
        label=f"iteration {current_best['iteration']}",
    )
    axis.set_xticks(x, COMPONENTS)
    axis.set_ylim(60, 101)
    axis.set(title="Native-score components at the two best points", ylabel="component score")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(args.output_dir / "best_component_comparison.png", dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
