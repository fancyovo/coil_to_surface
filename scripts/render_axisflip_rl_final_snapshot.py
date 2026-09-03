from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = REPO_ROOT / "reports" / "assets" / "axisflip_prior_online_rl_20260903"
FINAL_ROUND = 27
FINAL_CHECKPOINT_SHA256 = (
    "0c6db72afce39aa78a4063be1eff611da6fb709b1350991e231a1e05aa168af6"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def load_round(index: int) -> dict[str, Any]:
    directory = ASSET_ROOT / f"round_{index:03d}"
    summary = load_json(directory / "round_summary.json")
    training = load_json(directory / "training_summary.json")
    with np.load(directory / "training_data.npz") as data:
        valid = np.asarray(data["current_valid"], dtype=bool)
        initial_score = np.asarray(data["initial_scores"], dtype=float)
        best_score = np.asarray(data["best_scores"], dtype=float)
        initial_qs = np.asarray(data["initial_volume_qs"], dtype=float)
        initial_coil = np.asarray(data["initial_coil"], dtype=float)
        best_qs = np.asarray(data["best_volume_qs"], dtype=float)
        best_coil = np.asarray(data["best_coil"], dtype=float)
    if summary["round"] != index or training["round"] != index:
        raise ValueError(f"round identity mismatch in {directory}")
    if valid.shape != (64,) or int(valid.sum()) != int(summary["valid_count"]):
        raise ValueError(f"round sample accounting mismatch in {directory}")
    finite_best = np.isfinite(best_score) & valid
    qs_gain = best_qs[finite_best] - initial_qs[finite_best]
    coil_gain = best_coil[finite_best] - initial_coil[finite_best]
    joint = Counter()
    for q_value, c_value in zip(qs_gain, coil_gain, strict=True):
        if q_value >= 0.0 and c_value >= 0.0:
            joint["both_up"] += 1
        elif q_value >= 0.0:
            joint["qs_up_coil_down"] += 1
        elif c_value >= 0.0:
            joint["qs_down_coil_up"] += 1
        else:
            joint["both_down"] += 1
    return {
        "round": index,
        "valid_count": int(valid.sum()),
        "valid_rate": float(valid.mean()),
        "initial_score_median_all": float(np.median(initial_score)),
        "initial_score_p90_all": float(np.percentile(initial_score, 90.0)),
        "initial_score_at_least_70": int(np.count_nonzero(initial_score >= 70.0)),
        "initial_score_at_least_75": int(np.count_nonzero(initial_score >= 75.0)),
        "adam20_best_score_median_valid": float(np.median(best_score[finite_best])),
        "adam20_gain_median_valid": float(
            np.median(best_score[finite_best] - initial_score[finite_best])
        ),
        "adam20_at_least_80": int(np.count_nonzero(best_score[finite_best] >= 80.0)),
        "initial_volume_qs_median_valid": float(np.median(initial_qs[valid])),
        "initial_coil_median_valid": float(np.median(initial_coil[valid])),
        "adam20_volume_qs_median_valid": float(np.median(best_qs[finite_best])),
        "adam20_coil_median_valid": float(np.median(best_coil[finite_best])),
        "descriptor_effective_rank": float(
            summary["diversity"]["current"]["descriptor_effective_rank"]
        ),
        "descriptor_total_variance": float(
            summary["diversity"]["current"]["descriptor_total_variance"]
        ),
        "near_duplicate_rate": float(
            summary["diversity"]["current"]["near_duplicate_rate_1e-4"]
        ),
        "relative_parameter_update_l2": float(training["relative_parameter_update_l2"]),
        "paired_generated_normalized_rms_move": float(
            training["paired_generated_normalized_rms_move"]
        ),
        "joint_component_changes": {
            name: int(joint.get(name, 0))
            for name in (
                "both_up",
                "qs_up_coil_down",
                "qs_down_coil_up",
                "both_down",
            )
        },
    }


def plot_metrics(rows: list[dict[str, Any]]) -> None:
    rounds = np.asarray([row["round"] for row in rows])
    initial = np.asarray([row["initial_score_median_all"] for row in rows])
    optimized = np.asarray([row["adam20_best_score_median_valid"] for row in rows])
    at70 = np.asarray([row["initial_score_at_least_70"] for row in rows])
    at75 = np.asarray([row["initial_score_at_least_75"] for row in rows])
    initial_qs = np.asarray([row["initial_volume_qs_median_valid"] for row in rows])
    initial_coil = np.asarray([row["initial_coil_median_valid"] for row in rows])
    gain = np.asarray([row["adam20_gain_median_valid"] for row in rows])

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4))
    axes[0, 0].plot(rounds, initial, color="#136f63", label="Flow initial median")
    axes[0, 0].plot(rounds, optimized, color="#b23a48", label="Adam20 best median")
    axes[0, 0].set_ylabel("native ABI-11 score")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(rounds, at70, color="#457b9d", label="initial >=70")
    axes[0, 1].plot(rounds, at75, color="#e69f00", label="initial >=75")
    axes[0, 1].set_ylabel("samples per 64")
    axes[0, 1].legend(frameon=False)
    axes[1, 0].plot(rounds, initial_qs, color="#6a4c93", label="volume-QS")
    axes[1, 0].plot(rounds, initial_coil, color="#2a9d8f", label="coil")
    axes[1, 0].set_ylabel("initial component median")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].plot(rounds, gain, color="#d1495b")
    axes[1, 1].set_ylabel("median Adam20 gain")
    for axis in axes.flat:
        axis.set_xlabel("completed online round")
        axis.grid(alpha=0.22)
    fig.suptitle("Analytic-prior online Flow through completed round 27", fontsize=13)
    fig.tight_layout()
    fig.savefig(ASSET_ROOT / "rl_round_metrics_through_027.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    rank = np.asarray([row["descriptor_effective_rank"] for row in rows])
    variance = np.asarray([row["descriptor_total_variance"] for row in rows])
    update = np.asarray([row["relative_parameter_update_l2"] for row in rows])
    move = np.asarray([row["paired_generated_normalized_rms_move"] for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
    axes[0].plot(rounds, rank, color="#6a4c93")
    axes[0].set_ylabel("descriptor effective rank")
    axes[1].plot(rounds, variance, color="#457b9d")
    axes[1].set_ylabel("descriptor total variance")
    axes[2].plot(rounds, update, color="#b23a48", label="parameter L2 update")
    axes[2].plot(rounds, move, color="#e69f00", label="paired policy RMS move")
    axes[2].set_ylabel("update magnitude")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.set_xlabel("completed online round")
        axis.grid(alpha=0.22)
    fig.suptitle("Diversity and policy movement through completed round 27", fontsize=13)
    fig.tight_layout()
    fig.savefig(ASSET_ROOT / "rl_diversity_through_027.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = [load_round(index) for index in range(FINAL_ROUND + 1)]
    plot_metrics(rows)
    joint = Counter()
    for row in rows:
        joint.update(row["joint_component_changes"])
    peak = max(rows, key=lambda row: row["initial_score_median_all"])
    latest = rows[-1]
    first = rows[0]
    round21 = rows[21]
    payload = {
        "format": "axisflip_prior_online_rl_final_acceptance_v1",
        "termination": {
            "requested_by_user": True,
            "scheduler_job_id": 52977,
            "last_atomically_complete_round": FINAL_ROUND,
            "excluded_partial_round": 28,
            "final_checkpoint": "checkpoints/round_028.pt",
            "final_checkpoint_sha256": FINAL_CHECKPOINT_SHA256,
        },
        "rounds": rows,
        "aggregate": {
            "completed_rounds": [0, FINAL_ROUND],
            "completed_sample_count": 64 * len(rows),
            "completed_valid_count": int(sum(row["valid_count"] for row in rows)),
            "round0_to_latest": {
                "initial_score_median_delta": latest["initial_score_median_all"]
                - first["initial_score_median_all"],
                "initial_volume_qs_median_delta": latest[
                    "initial_volume_qs_median_valid"
                ]
                - first["initial_volume_qs_median_valid"],
                "initial_coil_median_delta": latest["initial_coil_median_valid"]
                - first["initial_coil_median_valid"],
                "effective_rank_ratio": latest["descriptor_effective_rank"]
                / first["descriptor_effective_rank"],
                "total_variance_ratio": latest["descriptor_total_variance"]
                / first["descriptor_total_variance"],
            },
            "round21_to_latest": {
                "initial_score_median_delta": latest["initial_score_median_all"]
                - round21["initial_score_median_all"],
                "initial_volume_qs_median_delta": latest[
                    "initial_volume_qs_median_valid"
                ]
                - round21["initial_volume_qs_median_valid"],
                "initial_coil_median_delta": latest["initial_coil_median_valid"]
                - round21["initial_coil_median_valid"],
            },
            "peak_initial_score_median": peak["initial_score_median_all"],
            "peak_initial_score_round": peak["round"],
            "effective_rank_range": [
                min(row["descriptor_effective_rank"] for row in rows),
                max(row["descriptor_effective_rank"] for row in rows),
            ],
            "total_variance_range": [
                min(row["descriptor_total_variance"] for row in rows),
                max(row["descriptor_total_variance"] for row in rows),
            ],
            "near_duplicate_rate_max": max(row["near_duplicate_rate"] for row in rows),
            "joint_component_changes": dict(joint),
        },
    }
    write_json(ASSET_ROOT / "report_metrics_through_round_027.json", payload)
    print(json.dumps(payload["aggregate"], indent=2))


if __name__ == "__main__":
    main()
