#!/usr/bin/env python3
"""Render a frozen stage snapshot of the three active analytic-prior RL jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flow_matching.axis_prior_rl import diversity_summary


STUDENT_RUN = "axisflip_r012_score_gradient_replay10_ema10_rl_20260905_fba88ec"
RADIUS_RUNS = {
    "r015": "axisflip_r015_trajectory_rl_20260906_d8c16a7e",
    "r020": "axisflip_r020_trajectory_rl_20260906_d8c16a7e",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value)!r}")


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    fraction = successes / total
    denominator = 1.0 + z * z / total
    center = (fraction + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * np.sqrt(
            fraction * (1.0 - fraction) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return float(center - radius), float(center + radius)


def mean_window(values: np.ndarray, count: int, *, tail: bool) -> float:
    window = values[-count:] if tail else values[:count]
    return float(np.mean(window))


def load_student(raw_root: Path) -> dict[str, Any]:
    run_root = raw_root / STUDENT_RUN
    progress = load_json(run_root / "progress.json")
    available = sorted(
        int(path.name.split("_")[1])
        for path in (run_root / "rounds").glob("round_*")
        if (path / "collection_summary.json").exists()
        and (path / "training_summary.json").exists()
        and (path / "rank_00.npz").exists()
        and (path / "rank_01.npz").exists()
    )
    if not available or available != list(range(available[-1] + 1)):
        raise RuntimeError("student snapshot does not contain contiguous complete rounds")
    latest = available[-1]
    rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    coordinate_errors: list[float] = []
    coordinate_failure_count = 0
    coordinate_p99_over_tolerance_rounds = 0

    for index in range(latest + 1):
        directory = run_root / "rounds" / f"round_{index:04d}"
        collection = load_json(directory / "collection_summary.json")
        training = load_json(directory / "training_summary.json")["training"]
        current_parts = []
        for rank_path in sorted(directory.glob("rank_*.npz")):
            with np.load(rank_path, allow_pickle=False) as archive:
                current_parts.append(np.asarray(archive["current"], dtype=np.float32))
        if len(current_parts) != 2:
            raise RuntimeError(f"round {index} does not contain two rank snapshots")
        diversity = diversity_summary(
            np.concatenate(current_parts, axis=0), seed=26090600 + index
        )
        for status, count in collection["status_counts"].items():
            status_counts[status] = status_counts.get(status, 0) + int(count)
        for record in collection["records"]:
            error = record.get("coordinate_check_relative_error")
            if error is not None:
                coordinate_errors.append(float(error))
                if not record["gradient_ok"]:
                    coordinate_failure_count += 1
        if collection["coordinate_check"]["relative_error_p99"] > 0.05:
            coordinate_p99_over_tolerance_rounds += 1
        rows.append(
            {
                "round": index,
                "finished_unix_s": float(training["steps"][-1].get("finished_unix_s", collection["finished_unix_s"]))
                if training.get("steps")
                else float(collection["finished_unix_s"]),
                "valid_count": int(collection["valid_count"]),
                "valid_rate": float(collection["valid_rate"]),
                "gradient_ok_count": int(collection["gradient_ok_count"]),
                "gradient_ok_rate_valid": float(collection["gradient_ok_rate_valid"]),
                "initial_score_median_all": float(collection["initial"]["score_median_all"]),
                "initial_score_p90_all": float(collection["initial"]["score_p90_all"]),
                "initial_score_median_valid": float(collection["initial"]["score_median_valid"]),
                "initial_volume_qs_median_valid": float(collection["initial"]["volume_qs_median_valid"]),
                "initial_coil_median_valid": float(collection["initial"]["coil_median_valid"]),
                "initial_qs_coil_correlation_valid": float(collection["initial"]["volume_qs_coil_correlation_valid"]),
                "coordinate_relative_error_median": collection["coordinate_check"]["relative_error_median"],
                "coordinate_relative_error_p99": collection["coordinate_check"]["relative_error_p99"],
                "collection_wall_s": float(collection["timing"]["collection_wall_s"]),
                "initial_score_median_s": float(collection["timing"]["initial_score_median_s"]),
                "gradient_median_s_valid": float(collection["timing"]["gradient_median_s_valid"]),
                "flow_train_wall_s": float(training["wall_s"]),
                "flow_updates": int(training["updates"]),
                "pool_size": int(training["pool_size"]),
                "diversity": diversity,
            }
        )

    sample_count = 64 * len(rows)
    valid_count = sum(row["valid_count"] for row in rows)
    gradient_ok_count = sum(row["gradient_ok_count"] for row in rows)
    valid = np.asarray([row["valid_rate"] for row in rows])
    score = np.asarray([row["initial_score_median_all"] for row in rows])
    p90 = np.asarray([row["initial_score_p90_all"] for row in rows])
    qs = np.asarray([row["initial_volume_qs_median_valid"] for row in rows])
    coil = np.asarray([row["initial_coil_median_valid"] for row in rows])
    collection_wall = np.asarray([row["collection_wall_s"] for row in rows])
    train_wall = np.asarray([row["flow_train_wall_s"] for row in rows])
    rank = np.asarray([row["diversity"]["descriptor_effective_rank"] for row in rows])
    variance = np.asarray([row["diversity"]["descriptor_total_variance"] for row in rows])
    duplicate = np.asarray([row["diversity"]["near_duplicate_rate_1e-4"] for row in rows])
    coordinate_error = np.asarray(coordinate_errors, dtype=np.float64)
    window = min(10, len(rows))
    aggregate = {
        "completed_round_count": len(rows),
        "sample_count": sample_count,
        "valid_count": valid_count,
        "valid_rate": valid_count / sample_count,
        "valid_rate_wilson95": wilson(valid_count, sample_count),
        "gradient_ok_count": gradient_ok_count,
        "gradient_ok_rate_valid": gradient_ok_count / max(valid_count, 1),
        "coordinate_check": {
            "tolerance": 0.05,
            "attempted_count": int(len(coordinate_error)),
            "failure_count": coordinate_failure_count,
            "failure_rate": coordinate_failure_count / max(len(coordinate_error), 1),
            "relative_error_median": float(np.median(coordinate_error)),
            "relative_error_p99": float(np.percentile(coordinate_error, 99.0)),
            "relative_error_maximum": float(np.max(coordinate_error)),
            "rounds_with_reported_p99_over_tolerance": coordinate_p99_over_tolerance_rounds,
        },
        "first_window_count": window,
        "first_window": {
            "valid_rate_mean": mean_window(valid, window, tail=False),
            "initial_score_median_mean": mean_window(score, window, tail=False),
            "initial_score_p90_mean": mean_window(p90, window, tail=False),
            "volume_qs_median_mean": mean_window(qs, window, tail=False),
            "coil_median_mean": mean_window(coil, window, tail=False),
        },
        "last_window": {
            "valid_rate_mean": mean_window(valid, window, tail=True),
            "initial_score_median_mean": mean_window(score, window, tail=True),
            "initial_score_p90_mean": mean_window(p90, window, tail=True),
            "volume_qs_median_mean": mean_window(qs, window, tail=True),
            "coil_median_mean": mean_window(coil, window, tail=True),
        },
        "mean_round_wall_s_accounted": float(np.mean(collection_wall + train_wall)),
        "latest_round_wall_s_accounted": float(collection_wall[-1] + train_wall[-1]),
        "mean_collection_wall_s": float(np.mean(collection_wall)),
        "mean_flow_train_wall_s": float(np.mean(train_wall)),
        "diversity": {
            "first": rows[0]["diversity"],
            "latest": rows[-1]["diversity"],
            "effective_rank_range": [float(np.min(rank)), float(np.max(rank))],
            "total_variance_range": [float(np.min(variance)), float(np.max(variance))],
            "near_duplicate_rate_max": float(np.max(duplicate)),
        },
        "status_counts": status_counts,
    }
    return {
        "job_id": 54423,
        "queue": "stu / Students / qos_stu_medium_2gpu",
        "gpus": 2,
        "manifest": load_json(run_root / "manifest.json"),
        "protocol": load_json(run_root / "protocol.json"),
        "progress": progress,
        "snapshot_completed_round": latest,
        "rows": rows,
        "aggregate": aggregate,
    }


def load_radius(raw_root: Path, key: str, job_id: int) -> dict[str, Any]:
    run_root = raw_root / RADIUS_RUNS[key]
    progress = load_json(run_root / "progress.json")
    available = sorted(
        int(path.name.split("_")[1])
        for path in (run_root / "rounds").glob("round_*")
        if (path / "round_summary.json").exists()
        and (path / "training_summary.json").exists()
    )
    if not available or available != list(range(available[-1] + 1)):
        raise RuntimeError(f"{key} snapshot does not contain contiguous complete rounds")
    latest = available[-1]
    manifest = load_json(run_root / "manifest.json")
    q0 = load_json(run_root / "audit" / "q0" / "summary.json")
    rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}

    for index in range(latest + 1):
        directory = run_root / "rounds" / f"round_{index:03d}"
        summary = load_json(directory / "round_summary.json")
        training = load_json(directory / "training_summary.json")
        for status, count in summary["status_counts"].items():
            status_counts[status] = status_counts.get(status, 0) + int(count)
        adam = summary["adam20"]
        rows.append(
            {
                "round": index,
                "finished_unix_s": float(training["finished_unix_s"]),
                "valid_count": int(summary["valid_count"]),
                "valid_rate": float(summary["valid_rate"]),
                "initial_score_median_all": float(summary["initial"]["score_median_all"]),
                "initial_score_p90_all": float(summary["initial"]["score_p90_all"]),
                "initial_score_median_valid": float(summary["initial"]["score_median_valid"]),
                "initial_volume_qs_median_valid": float(summary["initial"]["volume_qs_median_valid"]),
                "initial_coil_median_valid": float(summary["initial"]["coil_median_valid"]),
                "initial_qs_coil_correlation_valid": float(summary["initial"]["volume_qs_coil_correlation_valid"]),
                "adam20_best_score_median_valid": float(adam["best_score_median_valid"]),
                "adam20_best_score_p90_valid": float(adam["best_score_p90_valid"]),
                "adam20_gain_median_valid": float(adam["gain_median_valid"]),
                "adam20_volume_qs_median_valid": float(adam["volume_qs_median_valid"]),
                "adam20_coil_median_valid": float(adam["coil_median_valid"]),
                "adam20_qs_coil_correlation_valid": float(adam["volume_qs_coil_correlation_valid"]),
                "adam20_threshold_counts": {str(k): int(v) for k, v in adam["threshold_counts"].items()},
                "current_diversity": summary["diversity"]["current"],
                "improved_valid_diversity": summary["diversity"]["improved_valid"],
                "flow_gpu_s": float(summary["runtime"]["flow_gpu_s"]),
                "score_gpu_s": float(summary["runtime"]["score_gpu_s"]),
                "adam20_gpu_s": float(summary["runtime"]["adam20_gpu_s"]),
                "flow_train_wall_s": float(training["wall_s"]),
                "relative_parameter_update_l2": float(training["relative_parameter_update_l2"]),
                "policy_rms_move": float(training["paired_generated_normalized_rms_move"]),
                "reward_point_count": int(summary["training"]["reward_point_count"]),
                "replay_rollout_count": int(summary["training"]["replay_rollout_count_after_round"]),
                "replay_point_count": int(summary["training"]["replay_point_count_after_round"]),
                "reward_effective_sample_size": float(summary["training"]["reward_weight_effective_sample_size"]),
            }
        )

    for index, row in enumerate(rows):
        if index == 0:
            row["observed_round_interval_s"] = None
        else:
            row["observed_round_interval_s"] = row["finished_unix_s"] - rows[index - 1]["finished_unix_s"]
        row["parallel_work_estimate_s"] = (
            (row["flow_gpu_s"] + row["score_gpu_s"] + row["adam20_gpu_s"]) / 2.0
            + row["flow_train_wall_s"]
        )

    sample_count = 64 * len(rows)
    valid_count = sum(row["valid_count"] for row in rows)
    threshold_counts = {
        threshold: sum(row["adam20_threshold_counts"].get(threshold, 0) for row in rows)
        for threshold in ("50", "70", "80")
    }
    valid = np.asarray([row["valid_rate"] for row in rows])
    initial = np.asarray([row["initial_score_median_all"] for row in rows])
    initial_p90 = np.asarray([row["initial_score_p90_all"] for row in rows])
    adam = np.asarray([row["adam20_best_score_median_valid"] for row in rows])
    adam_p90 = np.asarray([row["adam20_best_score_p90_valid"] for row in rows])
    gain = np.asarray([row["adam20_gain_median_valid"] for row in rows])
    qs = np.asarray([row["initial_volume_qs_median_valid"] for row in rows])
    coil = np.asarray([row["initial_coil_median_valid"] for row in rows])
    adam_qs = np.asarray([row["adam20_volume_qs_median_valid"] for row in rows])
    adam_coil = np.asarray([row["adam20_coil_median_valid"] for row in rows])
    rank = np.asarray([row["current_diversity"]["descriptor_effective_rank"] for row in rows])
    variance = np.asarray([row["current_diversity"]["descriptor_total_variance"] for row in rows])
    duplicate = np.asarray([row["current_diversity"]["near_duplicate_rate_1e-4"] for row in rows])
    intervals = np.asarray([
        row["observed_round_interval_s"]
        for row in rows
        if row["observed_round_interval_s"] is not None
    ])
    work = np.asarray([row["parallel_work_estimate_s"] for row in rows])
    window = min(3, len(rows))
    aggregate = {
        "completed_round_count": len(rows),
        "sample_count": sample_count,
        "valid_count": valid_count,
        "valid_rate": valid_count / sample_count,
        "valid_rate_wilson95": wilson(valid_count, sample_count),
        "adam20_threshold_counts": threshold_counts,
        "adam20_threshold_rate_all": {key: value / sample_count for key, value in threshold_counts.items()},
        "adam20_threshold_rate_valid": {key: value / max(valid_count, 1) for key, value in threshold_counts.items()},
        "first_window_count": window,
        "first_window": {
            "valid_rate_mean": mean_window(valid, window, tail=False),
            "initial_score_median_mean": mean_window(initial, window, tail=False),
            "initial_score_p90_mean": mean_window(initial_p90, window, tail=False),
            "adam20_best_median_mean": mean_window(adam, window, tail=False),
            "adam20_best_p90_mean": mean_window(adam_p90, window, tail=False),
            "adam20_gain_median_mean": mean_window(gain, window, tail=False),
            "initial_volume_qs_median_mean": mean_window(qs, window, tail=False),
            "initial_coil_median_mean": mean_window(coil, window, tail=False),
            "adam20_volume_qs_median_mean": mean_window(adam_qs, window, tail=False),
            "adam20_coil_median_mean": mean_window(adam_coil, window, tail=False),
        },
        "last_window": {
            "valid_rate_mean": mean_window(valid, window, tail=True),
            "initial_score_median_mean": mean_window(initial, window, tail=True),
            "initial_score_p90_mean": mean_window(initial_p90, window, tail=True),
            "adam20_best_median_mean": mean_window(adam, window, tail=True),
            "adam20_best_p90_mean": mean_window(adam_p90, window, tail=True),
            "adam20_gain_median_mean": mean_window(gain, window, tail=True),
            "initial_volume_qs_median_mean": mean_window(qs, window, tail=True),
            "initial_coil_median_mean": mean_window(coil, window, tail=True),
            "adam20_volume_qs_median_mean": mean_window(adam_qs, window, tail=True),
            "adam20_coil_median_mean": mean_window(adam_coil, window, tail=True),
        },
        "mean_observed_round_interval_s_excluding_first": float(np.mean(intervals)),
        "mean_parallel_work_estimate_s": float(np.mean(work)),
        "diversity": {
            "q0_flow": q0["by_source"]["flow"]["diversity"],
            "first": rows[0]["current_diversity"],
            "latest": rows[-1]["current_diversity"],
            "effective_rank_range": [float(np.min(rank)), float(np.max(rank))],
            "total_variance_range": [float(np.min(variance)), float(np.max(variance))],
            "near_duplicate_rate_max": float(np.max(duplicate)),
        },
        "status_counts": status_counts,
    }
    return {
        "job_id": job_id,
        "queue": "competition / P107-RTX5090 / qos_p107-rtx5090",
        "gpus": 2,
        "radius_m": float(manifest["teacher"]["generator"]["minor_radius_center_m"]),
        "manifest": manifest,
        "protocol": load_json(run_root / "protocol.json"),
        "progress": progress,
        "snapshot_completed_round": latest,
        "distillation": load_json(run_root / "distillation" / "convergence.json"),
        "q0_audit": q0,
        "rows": rows,
        "aggregate": aggregate,
    }


def render_overview(runs: dict[str, dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13.2, 10.0))
    titles = {
        "student": "R012 score-gradient replay10 / EMA 0.1",
        "r015": "R015 trajectory replay / Adam20",
        "r020": "R020 trajectory replay / Adam20",
    }
    for row_index, key in enumerate(("student", "r015", "r020")):
        run = runs[key]
        rows = run["rows"]
        rounds = np.asarray([row["round"] for row in rows])
        score = np.asarray([row["initial_score_median_all"] for row in rows])
        p90 = np.asarray([row["initial_score_p90_all"] for row in rows])
        valid = np.asarray([row["valid_rate"] for row in rows])
        qs = np.asarray([row["initial_volume_qs_median_valid"] for row in rows])
        coil = np.asarray([row["initial_coil_median_valid"] for row in rows])
        axes[row_index, 0].plot(rounds, score, label="initial median", color="#136f63")
        axes[row_index, 0].plot(rounds, p90, label="initial P90", color="#b23a48")
        axes[row_index, 1].plot(rounds, valid, label="valid rate", color="#3a86ff")
        if key == "student":
            gradient = np.asarray([row["gradient_ok_rate_valid"] for row in rows])
            axes[row_index, 1].plot(rounds, gradient, label="gradient ok | valid", color="#e69f00")
        else:
            hit70 = np.asarray([
                row["adam20_threshold_counts"].get("70", 0) / 64.0 for row in rows
            ])
            axes[row_index, 1].plot(rounds, hit70, label="Adam20 >=70 / all", color="#e69f00")
        axes[row_index, 2].plot(rounds, qs, label="initial volume-QS", color="#3a86ff")
        axes[row_index, 2].plot(rounds, coil, label="initial coil", color="#2a9d8f")
        axes[row_index, 0].set_title(titles[key], loc="left", fontsize=11)
        axes[row_index, 0].set_ylabel("score")
        axes[row_index, 1].set_ylabel("fraction")
        axes[row_index, 1].set_ylim(-0.02, 1.03)
        axes[row_index, 2].set_ylabel("component score")
        for axis in axes[row_index]:
            axis.set_xlabel("completed round")
            axis.grid(alpha=0.22)
            axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Three active RL jobs: Flow initial distribution", fontsize=14)
    fig.tight_layout()
    fig.savefig(output / "initial_overview.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def render_adam20(runs: dict[str, dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.8))
    gain_axis = axes[0, 1].twinx()
    colors = {"r015": "#b23a48", "r020": "#136f63"}
    for key in ("r015", "r020"):
        rows = runs[key]["rows"]
        rounds = np.asarray([row["round"] for row in rows])
        initial = np.asarray([row["initial_score_median_all"] for row in rows])
        best = np.asarray([row["adam20_best_score_median_valid"] for row in rows])
        best_p90 = np.asarray([row["adam20_best_score_p90_valid"] for row in rows])
        gain = np.asarray([row["adam20_gain_median_valid"] for row in rows])
        initial_qs = np.asarray([row["initial_volume_qs_median_valid"] for row in rows])
        adam_qs = np.asarray([row["adam20_volume_qs_median_valid"] for row in rows])
        initial_coil = np.asarray([row["initial_coil_median_valid"] for row in rows])
        adam_coil = np.asarray([row["adam20_coil_median_valid"] for row in rows])
        label = f"{int(runs[key]['radius_m'] * 100)} cm"
        axes[0, 0].plot(rounds, initial, marker="o", color=colors[key], alpha=0.45, label=f"{label} initial")
        axes[0, 0].plot(rounds, best, marker="s", color=colors[key], label=f"{label} Adam20")
        axes[0, 1].plot(rounds, best_p90, marker="o", color=colors[key], label=label)
        axes[1, 0].plot(rounds, initial_qs, linestyle=":", color=colors[key], label=f"{label} initial QS")
        axes[1, 0].plot(rounds, adam_qs, color=colors[key], label=f"{label} Adam20 QS")
        axes[1, 1].plot(rounds, initial_coil, linestyle=":", color=colors[key], label=f"{label} initial coil")
        axes[1, 1].plot(rounds, adam_coil, color=colors[key], label=f"{label} Adam20 coil")
        gain_axis.plot(rounds, gain, linestyle="--", color=colors[key], alpha=0.55)
    gain_axis.set_ylabel("median gain", color="#555555")
    axes[0, 0].set_ylabel("median total score")
    axes[0, 1].set_ylabel("Adam20 best-score P90")
    axes[1, 0].set_ylabel("volume-QS component")
    axes[1, 1].set_ylabel("coil component")
    for axis in axes.flat:
        axis.set_xlabel("completed round")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("P107 trajectory-replay jobs: Adam20 response", fontsize=14)
    fig.tight_layout()
    fig.savefig(output / "adam20_response.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def render_diversity(runs: dict[str, dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13.2, 9.8))
    titles = {
        "student": "R012 score-gradient",
        "r015": "R015 trajectory replay",
        "r020": "R020 trajectory replay",
    }
    for row_index, key in enumerate(("student", "r015", "r020")):
        rows = runs[key]["rows"]
        rounds = np.asarray([row["round"] for row in rows])
        field = "diversity" if key == "student" else "current_diversity"
        rank = np.asarray([row[field]["descriptor_effective_rank"] for row in rows])
        variance = np.asarray([row[field]["descriptor_total_variance"] for row in rows])
        nearest = np.asarray([row[field]["nearest_distance_median"] for row in rows])
        axes[row_index, 0].plot(rounds, rank, color="#6a4c93")
        axes[row_index, 1].plot(rounds, variance, color="#457b9d")
        axes[row_index, 2].plot(rounds, nearest, color="#2a9d8f")
        axes[row_index, 0].set_title(titles[key], loc="left", fontsize=11)
        axes[row_index, 0].set_ylabel("effective rank")
        axes[row_index, 1].set_ylabel("total variance")
        axes[row_index, 2].set_ylabel("nearest-distance median")
        for axis in axes[row_index]:
            axis.set_xlabel("completed round")
            axis.grid(alpha=0.22)
    fig.suptitle("Current-policy sample diversity", fontsize=14)
    fig.tight_layout()
    fig.savefig(output / "diversity_diagnostics.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def render_timing(runs: dict[str, dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    student = runs["student"]["rows"]
    rounds = np.asarray([row["round"] for row in student])
    collection = np.asarray([row["collection_wall_s"] for row in student]) / 60.0
    training = np.asarray([row["flow_train_wall_s"] for row in student]) / 60.0
    axes[0].plot(rounds, collection, label="collection wall", color="#3a86ff")
    axes[0].plot(rounds, training, label="10 Flow updates", color="#e69f00")
    axes[0].set_title("R012 score-gradient", loc="left")
    axes[0].set_ylabel("minutes per round")
    for axis, key in zip(axes[1:], ("r015", "r020")):
        rows = runs[key]["rows"]
        rr = np.asarray([row["round"] for row in rows])
        observed = np.asarray([
            np.nan if row["observed_round_interval_s"] is None else row["observed_round_interval_s"] / 60.0
            for row in rows
        ])
        work = np.asarray([row["parallel_work_estimate_s"] for row in rows]) / 60.0
        axis.plot(rr, observed, marker="o", label="observed interval", color="#136f63")
        axis.plot(rr, work, marker="s", label="2-worker work estimate", color="#b23a48")
        axis.set_title(f"R{int(runs[key]['radius_m'] * 100):03d} trajectory replay", loc="left")
        axis.set_ylabel("minutes per round")
    for axis in axes:
        axis.set_xlabel("completed round")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Round timing and parallel accounting", fontsize=14)
    fig.tight_layout()
    fig.savefig(output / "round_timing.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "assets" / "axisflip_three_rl_stage_acceptance_20260906",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    runs = {
        "student": load_student(args.raw_root.resolve()),
        "r015": load_radius(args.raw_root.resolve(), "r015", 55183),
        "r020": load_radius(args.raw_root.resolve(), "r020", 55185),
    }
    render_overview(runs, output)
    render_adam20(runs, output)
    render_diversity(runs, output)
    render_timing(runs, output)

    payload = {
        "format": "axisflip_three_rl_stage_acceptance_20260906_v1",
        "snapshot": {
            "student_complete_rounds": [0, runs["student"]["snapshot_completed_round"]],
            "r015_complete_rounds": [0, runs["r015"]["snapshot_completed_round"]],
            "r020_complete_rounds": [0, runs["r020"]["snapshot_completed_round"]],
            "first_round_not_in_snapshot": {
                "student": runs["student"]["snapshot_completed_round"] + 1,
                "r015": runs["r015"]["snapshot_completed_round"] + 1,
                "r020": runs["r020"]["snapshot_completed_round"] + 1,
            },
        },
        "runs": runs,
    }
    (output / "report_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: {
                    "rounds": value["aggregate"]["completed_round_count"],
                    "samples": value["aggregate"]["sample_count"],
                    "valid_rate": value["aggregate"]["valid_rate"],
                }
                for key, value in runs.items()
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
