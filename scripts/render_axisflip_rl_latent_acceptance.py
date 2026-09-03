#!/usr/bin/env python3
"""Render acceptance assets for frozen-RL latent Adam200 and online RL."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def render_latent(
    source_root: Path,
    source_run_root: str,
    output_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    worker_records: list[dict[str, Any]] = []

    for gpu_record in sorted(source_root.glob("gpu_*flight_worker_*.csv")):
        copy_file(gpu_record, output_root / "scheduler" / gpu_record.name)

    for worker_dir in sorted(source_root.glob("worker_*")):
        worker_manifest = load_json(worker_dir / "manifest.json")
        worker_summary = load_json(worker_dir / "summary.json")
        worker_records.append(
            {
                "worker": worker_dir.name,
                "status": worker_summary["status"],
                "case_count": int(worker_summary["case_count"]),
                "complete_count": int(worker_summary["complete_count"]),
                "no_valid_screened_start_count": int(
                    worker_summary["no_valid_screened_start_count"]
                ),
                "wall_s": float(worker_summary["wall_s"]),
                "repository_commit": worker_manifest["repository"]["commit"],
            }
        )
        copy_file(worker_dir / "manifest.json", output_root / worker_dir.name / "manifest.json")
        copy_file(worker_dir / "summary.json", output_root / worker_dir.name / "summary.json")
        copy_file(worker_dir / "progress.json", output_root / worker_dir.name / "progress.json")

        for case_dir in sorted((worker_dir / "cases").glob("case_*")):
            case_index = int(case_dir.name.split("_")[-1])
            screen_summary = load_json(case_dir / "screening" / "summary.json")
            selected = load_json(case_dir / "screening" / "selected_start.json")
            optimization_dir = case_dir / "optimization"
            summary = load_json(optimization_dir / "summary.json")
            manifest = load_json(optimization_dir / "manifest.json")
            initial = load_json(optimization_dir / "trajectory" / "step_0000.json")
            best = load_json(optimization_dir / "best.json")
            history = load_jsonl(optimization_dir / "history.jsonl")

            initial_native = initial["flow_prior_local_full_gradient_adam"]["native_score"]
            best_native = best["flow_prior_local_full_gradient_adam"]["native_score"]
            initial_components = initial_native["components"]
            best_components = best_native["components"]
            gate = manifest["initial_consistency_gate"]

            iterations = [0] + [int(row["iteration"]) for row in history]
            scores = [float(summary["initial_score"])] + [
                float(row["current_score"]) for row in history
            ]
            running_best = np.maximum.accumulate(scores)
            endpoint_total = sum(int(row["gradient_endpoint_count"]) for row in history)
            endpoint_ok = sum(
                int(row["gradient_endpoint_statuses"].get("ok", 0)) for row in history
            )
            cache_hits = sum(bool(row["endpoint_decode_cache_hit"]) for row in history)
            wasted_endpoints = sum(
                int(row["flow_pipeline_wasted_endpoint_count"]) for row in history
            )
            accepted_centers = sum(bool(row["center_update_accepted"]) for row in history)

            for iteration, score, running in zip(iterations, scores, running_best):
                trajectories.append(
                    {
                        "case": case_index,
                        "iteration": iteration,
                        "current_score": score,
                        "running_best": float(running),
                    }
                )

            worker_case = next(
                item for item in worker_summary["cases"] if int(item["case_index"]) == case_index
            )
            case_record = {
                "case": case_index,
                "worker": worker_dir.name,
                "screen_seed": int(worker_case["screen_seed"]),
                "optimizer_seed": int(worker_case["optimizer_seed"]),
                "screen_valid_count": int(screen_summary["valid_candidate_count"]),
                "screen_selected_score": float(screen_summary["selected_score"]),
                "status": summary["status"],
                "stop_reason": summary["stop_reason"],
                "completed_iterations": int(summary["completed_iterations"]),
                "initial_score": float(summary["initial_score"]),
                "final_score": float(summary["final_score"]),
                "best_score": float(summary["best_score"]),
                "best_iteration": int(summary["best_iteration"]),
                "gain_to_best": float(summary["best_score"] - summary["initial_score"]),
                "wall_s": float(summary["total_wall_s"]),
                "initial_consistency_delta": float(gate["absolute_delta"]),
                "initial_components": {
                    "volume_qs": float(initial_components["volume_qs"]),
                    "coil": float(initial_components["coil"]),
                },
                "best_components": {
                    "volume_qs": float(best_components["volume_qs"]),
                    "coil": float(best_components["coil"]),
                },
                "component_delta": {
                    "volume_qs": float(
                        best_components["volume_qs"] - initial_components["volume_qs"]
                    ),
                    "coil": float(best_components["coil"] - initial_components["coil"]),
                },
                "best_qh_error": float(best_native["diagnostics"]["qs_global_error"]),
                "best_iota_min": float(best_native["diagnostics"]["iota_min"]),
                "best_iota_max": float(best_native["diagnostics"]["iota_max"]),
                "gradient_endpoint_count": endpoint_total,
                "gradient_endpoint_ok_count": endpoint_ok,
                "pipeline_cache_hit_updates": cache_hits,
                "pipeline_wasted_endpoint_count": wasted_endpoints,
                "accepted_center_count": accepted_centers,
                "running_best_at": {
                    str(step): float(running_best[step]) for step in (20, 50, 100, 150, 200)
                },
            }
            cases.append(case_record)

            evidence_dir = output_root / "cases" / f"case_{case_index:03d}"
            copy_file(case_dir / "screening" / "summary.json", evidence_dir / "screening_summary.json")
            copy_file(case_dir / "screening" / "selected_start.json", evidence_dir / "selected_start.json")
            copy_file(optimization_dir / "summary.json", evidence_dir / "optimization_summary.json")
            copy_file(optimization_dir / "manifest.json", evidence_dir / "optimization_manifest.json")
            copy_file(optimization_dir / "best.json", evidence_dir / "best.json")

    cases.sort(key=lambda item: item["case"])
    for expected, case in enumerate(cases):
        if case["case"] != expected:
            raise RuntimeError(f"missing latent case {expected}")

    scores_initial = [case["initial_score"] for case in cases]
    scores_best = [case["best_score"] for case in cases]
    gains = [case["gain_to_best"] for case in cases]
    volume_delta = [case["component_delta"]["volume_qs"] for case in cases]
    coil_delta = [case["component_delta"]["coil"] for case in cases]
    joint = {
        "both_up": int(sum(q >= 0 and c >= 0 for q, c in zip(volume_delta, coil_delta))),
        "qs_up_coil_down": int(sum(q >= 0 and c < 0 for q, c in zip(volume_delta, coil_delta))),
        "qs_down_coil_up": int(sum(q < 0 and c >= 0 for q, c in zip(volume_delta, coil_delta))),
        "both_down": int(sum(q < 0 and c < 0 for q, c in zip(volume_delta, coil_delta))),
    }
    aggregate = {
        "case_count": len(cases),
        "complete_count": sum(case["status"] == "ok" for case in cases),
        "all_completed_200": all(case["completed_iterations"] == 200 for case in cases),
        "screen_candidate_count": 32 * len(cases),
        "screen_valid_count": sum(case["screen_valid_count"] for case in cases),
        "initial_score_median": float(np.median(scores_initial)),
        "initial_score_range": [min(scores_initial), max(scores_initial)],
        "best_score_median": float(np.median(scores_best)),
        "best_score_range": [min(scores_best), max(scores_best)],
        "gain_median": float(np.median(gains)),
        "gain_range": [min(gains), max(gains)],
        "best_at_least_80_count": sum(score >= 80.0 for score in scores_best),
        "best_at_least_78_count": sum(score >= 78.0 for score in scores_best),
        "initial_volume_qs_median": float(
            np.median([case["initial_components"]["volume_qs"] for case in cases])
        ),
        "best_volume_qs_median": float(
            np.median([case["best_components"]["volume_qs"] for case in cases])
        ),
        "initial_coil_median": float(
            np.median([case["initial_components"]["coil"] for case in cases])
        ),
        "best_coil_median": float(
            np.median([case["best_components"]["coil"] for case in cases])
        ),
        "volume_qs_delta_median": float(np.median(volume_delta)),
        "coil_delta_median": float(np.median(coil_delta)),
        "joint_component_changes": joint,
        "maximum_initial_consistency_delta": max(
            case["initial_consistency_delta"] for case in cases
        ),
        "gradient_endpoint_count": sum(case["gradient_endpoint_count"] for case in cases),
        "gradient_endpoint_ok_count": sum(
            case["gradient_endpoint_ok_count"] for case in cases
        ),
        "pipeline_cache_hit_updates": sum(
            case["pipeline_cache_hit_updates"] for case in cases
        ),
        "pipeline_wasted_endpoint_count": sum(
            case["pipeline_wasted_endpoint_count"] for case in cases
        ),
        "accepted_center_count": sum(case["accepted_center_count"] for case in cases),
        "optimizer_wall_s_range": [
            min(case["wall_s"] for case in cases),
            max(case["wall_s"] for case in cases),
        ],
        "worker_wall_s_range": [
            min(worker["wall_s"] for worker in worker_records),
            max(worker["wall_s"] for worker in worker_records),
        ],
    }
    payload = {
        "format": "axisflip_rl_round12_latent_adam200_acceptance_v1",
        "source_run_root": source_run_root,
        "protocol_id": "qh-axisflip-rl-round12-flow-screen32-adam200-64d-abi11-v1",
        "workers": worker_records,
        "cases": cases,
        "aggregate": aggregate,
    }
    write_json(output_root / "report_metrics.json", payload)

    with (output_root / "trajectories.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trajectories[0]))
        writer.writeheader()
        writer.writerows(trajectories)

    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 7.8), sharex=True)
    for case in cases:
        selected = [row for row in trajectories if row["case"] == case["case"]]
        x = np.asarray([row["iteration"] for row in selected])
        current = np.asarray([row["current_score"] for row in selected])
        running = np.asarray([row["running_best"] for row in selected])
        axis = axes[case["case"] // 4]
        color = colors[case["case"]]
        axis.plot(x, current, color=color, alpha=0.18, linewidth=0.8)
        axis.plot(x, running, color=color, linewidth=1.6, label=f"case {case['case']}")
        axis.scatter(
            [case["best_iteration"]],
            [case["best_score"]],
            color=[color],
            edgecolor="white",
            linewidth=0.5,
            s=34,
            zorder=4,
        )
    for axis in axes:
        axis.axhline(80.0, color="#444444", linestyle=":", linewidth=1.0)
        axis.set_ylabel("ABI-11 total score")
        axis.grid(alpha=0.2)
        axis.legend(ncol=4, frameon=False, fontsize=8)
    axes[-1].set_xlabel("latent Adam update")
    fig.suptitle("Frozen round-12 RL Flow: eight latent Adam200 trajectories")
    fig.tight_layout()
    fig.savefig(output_root / "latent_adam200_trajectories.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 6.2))
    for case in cases:
        start_x = case["initial_components"]["volume_qs"]
        start_y = case["initial_components"]["coil"]
        best_x = case["best_components"]["volume_qs"]
        best_y = case["best_components"]["coil"]
        color = colors[case["case"]]
        axis.annotate(
            "",
            xy=(best_x, best_y),
            xytext=(start_x, start_y),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.7},
        )
        axis.scatter([start_x], [start_y], color=[color], marker="x", s=50)
        axis.scatter([best_x], [best_y], color=[color], s=42, label=f"case {case['case']}")
    axis.set_xlabel("volume-QS component score")
    axis.set_ylabel("coil-engineering component score")
    axis.grid(alpha=0.22)
    axis.legend(ncol=2, frameon=False, fontsize=8)
    axis.set_title("Step 0 to each trajectory's Adam200 best")
    fig.tight_layout()
    fig.savefig(output_root / "latent_adam200_components.png", dpi=190, bbox_inches="tight")
    plt.close(fig)
    return payload


def round_directory(index: int, baseline_root: Path, update_root: Path) -> Path:
    name = f"round_{index:03d}"
    candidate = update_root / name
    if candidate.is_dir():
        return candidate
    return baseline_root / name


def render_rl(
    baseline_root: Path,
    update_root: Path,
    output_root: Path,
    max_round: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_joint = {"both_up": 0, "qs_up_coil_down": 0, "qs_down_coil_up": 0, "both_down": 0}
    for index in range(max_round + 1):
        directory = round_directory(index, baseline_root, update_root)
        summary = load_json(directory / "round_summary.json")
        training = load_json(directory / "training_summary.json")
        with np.load(directory / "training_data.npz") as data:
            valid = np.asarray(data["current_valid"], dtype=bool)
            initial_scores = np.asarray(data["initial_scores"], dtype=np.float64)
            best_scores = np.asarray(data["best_scores"], dtype=np.float64)
            delta_qs = np.asarray(data["best_volume_qs"] - data["initial_volume_qs"])[valid]
            delta_coil = np.asarray(data["best_coil"] - data["initial_coil"])[valid]
        joint = {
            "both_up": int(np.count_nonzero((delta_qs >= 0) & (delta_coil >= 0))),
            "qs_up_coil_down": int(np.count_nonzero((delta_qs >= 0) & (delta_coil < 0))),
            "qs_down_coil_up": int(np.count_nonzero((delta_qs < 0) & (delta_coil >= 0))),
            "both_down": int(np.count_nonzero((delta_qs < 0) & (delta_coil < 0))),
        }
        for key in total_joint:
            total_joint[key] += joint[key]
        rows.append(
            {
                "round": index,
                "valid_count": int(summary["valid_count"]),
                "valid_rate": float(summary["valid_rate"]),
                "initial_score_median_all": float(summary["initial"]["score_median_all"]),
                "initial_score_p90_all": float(summary["initial"]["score_p90_all"]),
                "initial_score_at_least_70": int(np.count_nonzero(initial_scores >= 70.0)),
                "initial_score_at_least_75": int(np.count_nonzero(initial_scores >= 75.0)),
                "adam20_best_score_median_valid": float(summary["adam20"]["best_score_median_valid"]),
                "adam20_gain_median_valid": float(summary["adam20"]["gain_median_valid"]),
                "adam20_at_least_80": int(summary["adam20"]["threshold_counts"]["80"]),
                "initial_volume_qs_median_valid": float(summary["initial"]["volume_qs_median_valid"]),
                "initial_coil_median_valid": float(summary["initial"]["coil_median_valid"]),
                "adam20_volume_qs_median_valid": float(summary["adam20"]["volume_qs_median_valid"]),
                "adam20_coil_median_valid": float(summary["adam20"]["coil_median_valid"]),
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
                "joint_component_changes": joint,
            }
        )

    first = rows[0]
    prior_snapshot = rows[8]
    latest = rows[-1]
    scores = np.asarray([row["initial_score_median_all"] for row in rows])
    ranks = np.asarray([row["descriptor_effective_rank"] for row in rows])
    variances = np.asarray([row["descriptor_total_variance"] for row in rows])
    peak_index = int(np.argmax(scores))
    aggregate = {
        "completed_rounds": [0, max_round],
        "completed_sample_count": 64 * len(rows),
        "completed_valid_count": sum(row["valid_count"] for row in rows),
        "round0_to_latest": {
            "initial_score_median_delta": latest["initial_score_median_all"]
            - first["initial_score_median_all"],
            "initial_volume_qs_median_delta": latest["initial_volume_qs_median_valid"]
            - first["initial_volume_qs_median_valid"],
            "initial_coil_median_delta": latest["initial_coil_median_valid"]
            - first["initial_coil_median_valid"],
            "effective_rank_ratio": latest["descriptor_effective_rank"]
            / first["descriptor_effective_rank"],
            "total_variance_ratio": latest["descriptor_total_variance"]
            / first["descriptor_total_variance"],
        },
        "round8_to_latest": {
            "initial_score_median_delta": latest["initial_score_median_all"]
            - prior_snapshot["initial_score_median_all"],
            "initial_volume_qs_median_delta": latest["initial_volume_qs_median_valid"]
            - prior_snapshot["initial_volume_qs_median_valid"],
            "initial_coil_median_delta": latest["initial_coil_median_valid"]
            - prior_snapshot["initial_coil_median_valid"],
        },
        "peak_initial_score_median": float(scores[peak_index]),
        "peak_initial_score_round": peak_index,
        "effective_rank_range": [float(ranks.min()), float(ranks.max())],
        "total_variance_range": [float(variances.min()), float(variances.max())],
        "near_duplicate_rate_max": max(row["near_duplicate_rate"] for row in rows),
        "joint_component_changes": total_joint,
    }
    payload = {
        "format": "axisflip_prior_online_rl_progress_acceptance_v1",
        "snapshot_after_round": max_round,
        "rounds": rows,
        "aggregate": aggregate,
    }
    write_json(output_root / f"report_metrics_through_round_{max_round:03d}.json", payload)
    copy_file(
        update_root / "progress.json",
        output_root / f"progress_snapshot_after_round_{max_round:03d}.json",
    )
    for index in range(9, max_round + 1):
        source = round_directory(index, baseline_root, update_root)
        destination = output_root / f"round_{index:03d}"
        for name in ("round_summary.json", "training_summary.json", "training_data.npz"):
            copy_file(source / name, destination / name)

    rounds = np.asarray([row["round"] for row in rows])
    initial_score = np.asarray([row["initial_score_median_all"] for row in rows])
    best_score = np.asarray([row["adam20_best_score_median_valid"] for row in rows])
    volume_qs = np.asarray([row["initial_volume_qs_median_valid"] for row in rows])
    coil = np.asarray([row["initial_coil_median_valid"] for row in rows])

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), sharex=True)
    axes[0, 0].plot(rounds, initial_score, marker="o", color="#136f63", label="Flow initial")
    axes[0, 0].plot(rounds, best_score, marker="s", color="#b23a48", label="after Adam20")
    axes[0, 0].set_ylabel("median total score")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(
        rounds,
        [row["initial_score_at_least_70"] for row in rows],
        marker="o",
        color="#3a86ff",
        label="initial >= 70",
    )
    axes[0, 1].plot(
        rounds,
        [row["initial_score_at_least_75"] for row in rows],
        marker="s",
        color="#e69f00",
        label="initial >= 75",
    )
    axes[0, 1].set_ylabel("count out of 64")
    axes[0, 1].legend(frameon=False)
    axes[1, 0].plot(rounds, volume_qs, marker="o", color="#3a86ff", label="initial volume-QS")
    axes[1, 0].plot(rounds, coil, marker="s", color="#2a9d8f", label="initial coil")
    axes[1, 0].set_ylabel("median component score")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].plot(
        rounds,
        [row["adam20_gain_median_valid"] for row in rows],
        marker="o",
        color="#6a4c93",
    )
    axes[1, 1].set_ylabel("median Adam20 gain")
    for axis in axes.flat:
        axis.grid(alpha=0.22)
        axis.set_xticks(np.arange(0, max_round + 1, 2))
    axes[1, 0].set_xlabel("completed online round")
    axes[1, 1].set_xlabel("completed online round")
    fig.suptitle(f"Analytic-prior online Flow: complete rounds 0-{max_round}")
    fig.tight_layout()
    fig.savefig(
        output_root / f"rl_round_metrics_through_{max_round:03d}.png",
        dpi=190,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharex=True)
    axes[0].plot(rounds, ranks, marker="o", color="#6a4c93", label="effective rank")
    axes[0].set_ylabel("descriptor effective rank")
    axes[1].plot(rounds, variances, marker="o", color="#457b9d", label="total variance")
    axes[1].set_ylabel("descriptor total variance")
    for axis in axes:
        axis.set_xlabel("completed online round")
        axis.set_xticks(np.arange(0, max_round + 1, 2))
        axis.grid(alpha=0.22)
    fig.suptitle(f"Online Flow diversity through complete round {max_round}")
    fig.tight_layout()
    fig.savefig(
        output_root / f"rl_diversity_through_{max_round:03d}.png",
        dpi=190,
        bbox_inches="tight",
    )
    plt.close(fig)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-root", required=True, type=Path)
    parser.add_argument("--latent-source-run-root", required=True)
    parser.add_argument("--latent-output", required=True, type=Path)
    parser.add_argument("--rl-baseline-root", required=True, type=Path)
    parser.add_argument("--rl-update-root", required=True, type=Path)
    parser.add_argument("--rl-output", required=True, type=Path)
    parser.add_argument("--rl-max-round", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    latent = render_latent(
        args.latent_root,
        args.latent_source_run_root,
        args.latent_output,
    )
    rl = render_rl(
        args.rl_baseline_root,
        args.rl_update_root,
        args.rl_output,
        args.rl_max_round,
    )
    print(
        json.dumps(
            {
                "latent_complete": latent["aggregate"]["complete_count"],
                "latent_best_max": latent["aggregate"]["best_score_range"][1],
                "rl_snapshot_after_round": rl["snapshot_after_round"],
                "rl_latest_initial_median": rl["rounds"][-1]["initial_score_median_all"],
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
