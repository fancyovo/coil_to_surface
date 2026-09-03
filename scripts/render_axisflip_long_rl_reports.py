#!/usr/bin/env python3
"""Render the frozen Adam2000 and analytic-prior online-RL report assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LONG_ROOT = (
    ROOT
    / "reports"
    / "assets"
    / "axis_surface_prior_axisflip_v4_20260902"
    / "adam2000"
)
RL_ROOT = ROOT / "reports" / "assets" / "axisflip_prior_online_rl_20260903"
CASE_LABELS = {
    "axisflip_v4_case18_adam2000_20260903_6350b73": "case 18 (nfp=8, nc=3)",
    "axisflip_v4_case23_adam2000_20260903_6350b73": "case 23 (nfp=6, nc=4)",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def rolling_median(values: np.ndarray, width: int) -> np.ndarray:
    result = np.empty_like(values)
    for index in range(len(values)):
        result[index] = np.median(values[max(0, index - width + 1) : index + 1])
    return result


def first_reach(iterations: np.ndarray, running: np.ndarray, threshold: float) -> int | None:
    selected = np.flatnonzero(running >= threshold)
    return int(iterations[selected[0]]) if selected.size else None


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    fraction = successes / total
    denominator = 1.0 + z * z / total
    center = (fraction + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * np.sqrt(fraction * (1.0 - fraction) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return float(center - radius), float(center + radius)


def render_long_runs() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for directory in sorted(path for path in LONG_ROOT.iterdir() if path.is_dir()):
        summary = load_json(directory / "optimization" / "summary.json")
        start = load_json(directory / "start.json")
        best = load_json(directory / "optimization" / "best.json")
        rows = [
            json.loads(line)
            for line in (directory / "optimization" / "history.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        iterations = np.asarray([0] + [int(row["iteration"]) for row in rows])
        scores = np.asarray(
            [float(summary["initial_score"])] + [float(row["current_score"]) for row in rows]
        )
        running = np.maximum.accumulate(scores)
        start_components = start["data_prior_screening"]["native_score"]["components"]
        best_native = best["original_space_local_gradient_adam"]["native_score"]
        best_components = best_native["components"]
        best_row = rows[int(summary["best_iteration"]) - 1]
        cases.append(
            {
                "directory": directory,
                "label": CASE_LABELS[directory.name],
                "sample_id": Path(start["long_continuation"]["source_best"]).parents[1].name,
                "nfp": int(start["nfp"]),
                "nc": int(summary["manifest"]["n_base_coils"]),
                "iterations": iterations,
                "scores": scores,
                "running": running,
                "summary": summary,
                "start_components": start_components,
                "best_components": best_components,
                "best_qh_error": float(best_row["current_qh_error"]),
                "best_iota": float(best_row["current_iota"]),
                "best_native_status": best_native["status"],
            }
        )

    colors = ["#136f63", "#b23a48"]
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.6), sharex=True)
    for axis, case, color in zip(axes, cases, colors):
        x = case["iterations"]
        y = case["scores"]
        axis.plot(x, y, color=color, alpha=0.24, linewidth=0.8, label="current score")
        axis.plot(x, rolling_median(y, 25), color=color, linewidth=1.4, label="25-step median")
        axis.plot(x, case["running"], color="#202020", linewidth=1.8, label="running best")
        best_iteration = int(case["summary"]["best_iteration"])
        best_score = float(case["summary"]["best_score"])
        axis.scatter([best_iteration], [best_score], s=36, color="#e69f00", zorder=5)
        axis.annotate(
            f"{best_score:.3f} @ {best_iteration}",
            (best_iteration, best_score),
            xytext=(-8, 10),
            textcoords="offset points",
            ha="right",
            fontsize=9,
        )
        axis.set_title(case["label"], loc="left", fontsize=11)
        axis.set_ylabel("ABI-11 total score")
        axis.grid(alpha=0.22)
    axes[0].legend(loc="lower right", ncol=3, frameon=False)
    axes[-1].set_xlabel("Adam update")
    fig.suptitle("Axis-flip v4 continuation: 2,000-step score trajectories", fontsize=13)
    fig.tight_layout()
    fig.savefig(LONG_ROOT / "adam2000_trajectories.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    positions = np.arange(len(cases))
    width = 0.18
    for offset, component, color in (
        (-1.5, "volume_qs", "#3a86ff"),
        (0.5, "coil", "#2a9d8f"),
    ):
        start_values = [float(case["start_components"][component]) for case in cases]
        best_values = [float(case["best_components"][component]) for case in cases]
        left.bar(
            positions + offset * width,
            start_values,
            width,
            color=color,
            alpha=0.45,
            label=f"{component} at Adam200 start",
        )
        left.bar(
            positions + (offset + 1.0) * width,
            best_values,
            width,
            color=color,
            label=f"{component} at Adam2000 best",
        )
    left.set_xticks(positions, ["case 18", "case 23"])
    left.set_ylim(50, 92)
    left.set_ylabel("component score")
    left.grid(axis="y", alpha=0.22)
    left.legend(fontsize=8, frameon=False)

    for case, color in zip(cases, colors):
        start_x = float(case["start_components"]["volume_qs"])
        start_y = float(case["start_components"]["coil"])
        best_x = float(case["best_components"]["volume_qs"])
        best_y = float(case["best_components"]["coil"])
        right.annotate(
            "",
            xy=(best_x, best_y),
            xytext=(start_x, start_y),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 2.0},
        )
        right.scatter([start_x], [start_y], marker="x", s=55, color=color)
        right.scatter([best_x], [best_y], marker="o", s=45, color=color, label=case["label"])
    right.set_xlabel("volume-QS component score")
    right.set_ylabel("coil-engineering component score")
    right.grid(alpha=0.22)
    right.legend(fontsize=8, frameon=False)
    fig.suptitle("Adam200 start to Adam2000 best: score-component tradeoff", fontsize=13)
    fig.tight_layout()
    fig.savefig(LONG_ROOT / "adam2000_components.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    metrics_cases = []
    for case in cases:
        checkpoints = {}
        for iteration in (200, 500, 1000, 1500, 2000):
            checkpoints[str(iteration)] = float(case["running"][iteration])
        metrics_cases.append(
            {
                "sample_id": case["sample_id"],
                "nfp": case["nfp"],
                "nc": case["nc"],
                "status": case["summary"]["status"],
                "completed_iterations": int(case["summary"]["completed_iterations"]),
                "wall_s": float(case["summary"]["total_wall_s"]),
                "initial_score": float(case["summary"]["initial_score"]),
                "final_score": float(case["summary"]["final_score"]),
                "best_score": float(case["summary"]["best_score"]),
                "best_iteration": int(case["summary"]["best_iteration"]),
                "running_best_at": checkpoints,
                "first_reach": {
                    str(threshold): first_reach(case["iterations"], case["running"], threshold)
                    for threshold in (85.0, 87.0, 89.0)
                },
                "start_components": case["start_components"],
                "best_components": case["best_components"],
                "component_delta": {
                    component: float(case["best_components"][component])
                    - float(case["start_components"][component])
                    for component in ("volume_qs", "coil")
                },
                "best_qh_error": case["best_qh_error"],
                "best_iota": case["best_iota"],
                "best_native_status": case["best_native_status"],
            }
        )
    payload = {
        "format": "axisflip_v4_adam2000_report_metrics_v1",
        "source_commit": "6350b7308529e8708d8af6946508070b2aae945b",
        "cases": metrics_cases,
    }
    write_json(LONG_ROOT / "report_metrics.json", payload)
    return payload


def render_rl() -> dict[str, Any]:
    q0 = load_json(RL_ROOT / "q0_audit_summary.json")
    convergence = load_json(RL_ROOT / "distillation_convergence.json")
    rows: list[dict[str, Any]] = []
    joint_total = {"both_up": 0, "qs_up_coil_down": 0, "qs_down_coil_up": 0, "both_down": 0}
    for index in range(9):
        directory = RL_ROOT / f"round_{index:03d}"
        summary = load_json(directory / "round_summary.json")
        training = load_json(directory / "training_summary.json")
        with np.load(directory / "training_data.npz") as data:
            valid = np.asarray(data["current_valid"], dtype=bool)
            initial_scores = np.asarray(data["initial_scores"], dtype=np.float64)
            best_scores = np.asarray(data["best_scores"], dtype=np.float64)
            delta_qs = np.asarray(data["best_volume_qs"] - data["initial_volume_qs"])[valid]
            delta_coil = np.asarray(data["best_coil"] - data["initial_coil"])[valid]
            current_channel = np.asarray(data["current"][..., -1], dtype=np.float64)
        joint = {
            "both_up": int(np.count_nonzero((delta_qs >= 0.0) & (delta_coil >= 0.0))),
            "qs_up_coil_down": int(np.count_nonzero((delta_qs >= 0.0) & (delta_coil < 0.0))),
            "qs_down_coil_up": int(np.count_nonzero((delta_qs < 0.0) & (delta_coil >= 0.0))),
            "both_down": int(np.count_nonzero((delta_qs < 0.0) & (delta_coil < 0.0))),
        }
        for key, value in joint.items():
            joint_total[key] += value
        sample_max_current_deviation = np.max(np.abs(current_channel), axis=1)
        rows.append(
            {
                "round": index,
                "valid_count": int(summary["valid_count"]),
                "valid_rate": float(summary["valid_rate"]),
                "valid_rate_wilson95": wilson_interval(int(summary["valid_count"]), 64),
                "initial_score_median_all": float(summary["initial"]["score_median_all"]),
                "initial_score_p90_all": float(summary["initial"]["score_p90_all"]),
                "initial_score_threshold_counts": {
                    str(threshold): int(np.count_nonzero(initial_scores >= threshold))
                    for threshold in (50.0, 60.0, 70.0)
                },
                "adam20_best_score_median_valid": float(summary["adam20"]["best_score_median_valid"]),
                "adam20_best_score_p90_valid": float(summary["adam20"]["best_score_p90_valid"]),
                "adam20_gain_median_valid": float(summary["adam20"]["gain_median_valid"]),
                "adam20_threshold_counts": summary["adam20"]["threshold_counts"],
                "initial_volume_qs_median_valid": float(summary["initial"]["volume_qs_median_valid"]),
                "initial_coil_median_valid": float(summary["initial"]["coil_median_valid"]),
                "adam20_volume_qs_median_valid": float(summary["adam20"]["volume_qs_median_valid"]),
                "adam20_coil_median_valid": float(summary["adam20"]["coil_median_valid"]),
                "joint_component_changes": joint,
                "current_diversity": summary["diversity"]["current"],
                "improved_diversity": summary["diversity"]["improved_valid"],
                "relative_parameter_update_l2": float(training["relative_parameter_update_l2"]),
                "paired_generated_normalized_rms_move": float(
                    training["paired_generated_normalized_rms_move"]
                ),
                "current_allocation_sample_max_deviation": {
                    "median": float(np.median(sample_max_current_deviation)),
                    "p95": float(np.percentile(sample_max_current_deviation, 95.0)),
                    "maximum": float(np.max(sample_max_current_deviation)),
                    "negative_current_count": int(np.count_nonzero(1.0 + current_channel < 0.0)),
                },
                "sample_score_ranges": {
                    "initial_min": float(np.min(initial_scores)),
                    "initial_max": float(np.max(initial_scores)),
                    "best_valid_min": float(np.min(best_scores[valid])),
                    "best_valid_max": float(np.max(best_scores[valid])),
                },
            }
        )

    rounds = np.asarray([row["round"] for row in rows])
    valid = np.asarray([row["valid_rate"] for row in rows])
    initial_score = np.asarray([row["initial_score_median_all"] for row in rows])
    best_score = np.asarray([row["adam20_best_score_median_valid"] for row in rows])
    gain = np.asarray([row["adam20_gain_median_valid"] for row in rows])
    initial_qs = np.asarray([row["initial_volume_qs_median_valid"] for row in rows])
    initial_coil = np.asarray([row["initial_coil_median_valid"] for row in rows])
    best_qs = np.asarray([row["adam20_volume_qs_median_valid"] for row in rows])
    best_coil = np.asarray([row["adam20_coil_median_valid"] for row in rows])
    rank = np.asarray([row["current_diversity"]["descriptor_effective_rank"] for row in rows])
    variance = np.asarray([row["current_diversity"]["descriptor_total_variance"] for row in rows])
    parameter_update = np.asarray([row["relative_parameter_update_l2"] for row in rows])
    policy_move = np.asarray([row["paired_generated_normalized_rms_move"] for row in rows])

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8))
    axes[0, 0].plot(rounds, initial_score, marker="o", color="#136f63", label="Flow initial")
    axes[0, 0].plot(rounds, best_score, marker="s", color="#b23a48", label="after Adam20")
    axes[0, 0].scatter([-0.55], [q0["by_source"]["flow"]["score_median_all"]], color="#136f63", marker="D", label="q0 audit")
    axes[0, 0].set_ylabel("median total score")
    axes[0, 0].legend(frameon=False, fontsize=8)

    low = np.asarray([row["valid_rate_wilson95"][0] for row in rows])
    high = np.asarray([row["valid_rate_wilson95"][1] for row in rows])
    axes[0, 1].errorbar(rounds, valid, yerr=[valid - low, high - valid], marker="o", capsize=3, color="#3a86ff")
    axes[0, 1].scatter([-0.55], [q0["by_source"]["flow"]["valid_rate"]], color="#3a86ff", marker="D")
    axes[0, 1].set_ylim(0.82, 1.015)
    axes[0, 1].set_ylabel("valid fraction")

    axes[1, 0].plot(rounds, initial_qs, marker="o", color="#3a86ff", label="initial volume-QS")
    axes[1, 0].plot(rounds, initial_coil, marker="o", color="#2a9d8f", label="initial coil")
    axes[1, 0].plot(rounds, best_qs, linestyle="--", color="#3a86ff", label="Adam20 volume-QS")
    axes[1, 0].plot(rounds, best_coil, linestyle="--", color="#2a9d8f", label="Adam20 coil")
    axes[1, 0].set_ylabel("median component score")
    axes[1, 0].set_xlabel("completed online round")
    axes[1, 0].legend(frameon=False, fontsize=8, ncol=2)

    axes[1, 1].plot(rounds, gain, marker="o", color="#e69f00")
    axes[1, 1].set_ylabel("median Adam20 gain")
    axes[1, 1].set_xlabel("completed online round")
    for axis in axes.flat:
        axis.grid(alpha=0.22)
        axis.set_xticks(rounds)
    fig.suptitle("Analytic-prior online Flow: completed rounds 0-8", fontsize=13)
    fig.tight_layout()
    fig.savefig(RL_ROOT / "rl_round_metrics.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9))
    axes[0].plot(rounds, rank, marker="o", color="#6a4c93")
    axes[0].scatter([-0.55], [q0["by_source"]["flow"]["diversity"]["descriptor_effective_rank"]], color="#6a4c93", marker="D")
    axes[0].set_ylabel("descriptor effective rank")
    axes[1].plot(rounds, variance, marker="o", color="#457b9d")
    axes[1].scatter([-0.55], [q0["by_source"]["flow"]["diversity"]["descriptor_total_variance"]], color="#457b9d", marker="D")
    axes[1].set_ylabel("descriptor total variance")
    axes[2].plot(rounds, parameter_update, marker="o", color="#b23a48", label="parameter L2 update")
    axes[2].plot(rounds, policy_move, marker="s", color="#e69f00", label="paired policy RMS move")
    axes[2].set_ylabel("update magnitude")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.set_xlabel("completed online round")
        axis.set_xticks(rounds)
    fig.suptitle("Diversity and policy-update diagnostics", fontsize=13)
    fig.tight_layout()
    fig.savefig(RL_ROOT / "rl_diversity_updates.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    distillation = [
        json.loads(line)
        for line in (RL_ROOT / "distillation_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    epoch = np.asarray([row["epoch"] for row in distillation])
    train_loss = np.asarray([row["train_loss"] for row in distillation])
    validation_loss = np.asarray([row["validation_loss"] for row in distillation])
    moment_error = np.asarray([row["generated"]["moment_error"] for row in distillation])
    fig, left = plt.subplots(figsize=(9.4, 4.6))
    left.plot(epoch, train_loss, color="#136f63", label="train loss")
    left.plot(epoch, validation_loss, color="#b23a48", label="validation loss")
    left.set_xlabel("distillation epoch")
    left.set_ylabel("Flow-matching loss")
    right = left.twinx()
    right.plot(epoch, moment_error, color="#e69f00", alpha=0.8, label="generated moment error")
    right.set_ylabel("generated moment error")
    lines = left.lines + right.lines
    left.legend(lines, [line.get_label() for line in lines], frameon=False)
    left.grid(alpha=0.22)
    left.axvline(convergence["stopped_epoch"], color="#202020", linestyle=":", linewidth=1.2)
    fig.suptitle("q0 distillation convergence", fontsize=13)
    fig.tight_layout()
    fig.savefig(RL_ROOT / "distillation_convergence.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    q0_currents = []
    q0_sources = []
    for path in sorted((RL_ROOT / "q0_audit").glob("worker_*.npz")):
        with np.load(path) as data:
            q0_currents.append(np.asarray(data["normalized"][..., -1], dtype=np.float64))
            q0_sources.append(np.asarray(data["source"]))
    q0_current = np.concatenate(q0_currents)
    q0_source = np.concatenate(q0_sources)
    current_audit = {}
    for source in ("teacher", "flow"):
        selected = q0_current[q0_source == source]
        sample_max = np.max(np.abs(selected), axis=1)
        current_audit[source] = {
            "sample_max_deviation_median": float(np.median(sample_max)),
            "sample_max_deviation_p95": float(np.percentile(sample_max, 95.0)),
            "maximum_deviation": float(np.max(sample_max)),
            "negative_current_count": int(np.count_nonzero(1.0 + selected < 0.0)),
        }

    payload = {
        "format": "axisflip_prior_online_rl_report_metrics_v1",
        "snapshot": {"completed_rounds": list(range(9)), "active_round_excluded": 9},
        "distillation": convergence,
        "q0_audit": q0,
        "q0_current_allocation": current_audit,
        "rounds": rows,
        "aggregate": {
            "completed_sample_count": int(64 * len(rows)),
            "completed_valid_count": int(sum(row["valid_count"] for row in rows)),
            "initial_score_median_change_round0_to_round8": float(initial_score[-1] - initial_score[0]),
            "initial_volume_qs_median_change_round0_to_round8": float(initial_qs[-1] - initial_qs[0]),
            "initial_coil_median_change_round0_to_round8": float(initial_coil[-1] - initial_coil[0]),
            "linear_slope_initial_score_per_round": float(np.polyfit(rounds, initial_score, 1)[0]),
            "linear_slope_valid_rate_per_round": float(np.polyfit(rounds, valid, 1)[0]),
            "current_effective_rank_range": [float(np.min(rank)), float(np.max(rank))],
            "current_effective_rank_round8_over_round0": float(rank[-1] / rank[0]),
            "current_total_variance_round8_over_round0": float(variance[-1] / variance[0]),
            "near_duplicate_rate_max": float(
                max(row["current_diversity"]["near_duplicate_rate_1e-4"] for row in rows)
            ),
            "joint_component_changes_all_valid": joint_total,
        },
    }
    write_json(RL_ROOT / "report_metrics.json", payload)
    return payload


def main() -> None:
    long_metrics = render_long_runs()
    rl_metrics = render_rl()
    print(
        json.dumps(
            {
                "long_cases": len(long_metrics["cases"]),
                "rl_completed_rounds": len(rl_metrics["rounds"]),
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
