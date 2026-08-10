#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RUNS = {
    "random": ROOT / "random",
    "previous_update": ROOT / "previous_update_after300",
}
COLORS = {"random": "#2457A7", "previous_update": "#D97706"}
LABELS = {
    "random": "Two fresh random directions",
    "previous_update": "Previous update + random after step 300",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_history(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def array(rows, key, dtype=float):
    return np.asarray([row[key] for row in rows], dtype=dtype)


def run_metrics(run_dir: Path):
    rows = load_history(run_dir / "history.jsonl")
    summary = load_json(run_dir / "summary.json")
    manifest = load_json(run_dir / "manifest.json")
    best = load_json(run_dir / "best.json")["flow_prior_standard_adam"]
    native = best["native_score"]
    iterations = array(rows, "iteration", int)
    assert np.array_equal(iterations, np.arange(1, 1001))

    applied = array(rows, "gradient_step_applied", bool)
    rejected = array(rows, "temporal_step_rejected", bool)
    gradient_outlier = array(rows, "temporal_gradient_outlier", bool)
    update_outlier = array(rows, "temporal_update_outlier", bool)
    center_accepted = array(rows, "center_update_accepted", bool)
    acceptance = array(rows, "center_acceptance_fraction")
    invalid_directions = np.asarray(
        [any(row["filtered_invalid_directions"]) for row in rows], dtype=bool
    )
    outlier_directions = np.asarray(
        [any(row["filtered_outlier_directions"]) for row in rows], dtype=bool
    )
    backtracked = center_accepted & (acceptance > 0.0) & (acceptance < 1.0)
    post = iterations > 300
    delta = np.asarray([row["raw_direction_deltas"] for row in rows], dtype=float)

    component_names = ["axis", "psi", "surface", "coordinate", "volume_qs", "iota", "coil"]
    diagnostics = native["diagnostics"]
    metrics = {
        "summary": summary,
        "manifest_direction_policy": manifest["direction_policy"],
        "score_at_300": float(rows[299]["current_score"]),
        "best_gain_after_300": float(summary["best_score"] - rows[299]["current_score"]),
        "post_300_current_score_mean": float(np.mean(array(rows, "current_score")[post])),
        "post_300_iteration_wall_p50_s": float(np.median(array(rows, "iteration_wall_s")[post])),
        "post_300_iteration_wall_p95_s": float(np.quantile(array(rows, "iteration_wall_s")[post], 0.95)),
        "gradient_steps_applied": int(np.count_nonzero(applied)),
        "temporal_steps_rejected": int(np.count_nonzero(rejected)),
        "temporal_gradient_outliers": int(np.count_nonzero(gradient_outlier)),
        "temporal_update_outliers": int(np.count_nonzero(update_outlier)),
        "temporal_gradient_and_update_outliers": int(
            np.count_nonzero(gradient_outlier & update_outlier)
        ),
        "invalid_direction_steps": int(np.count_nonzero(invalid_directions)),
        "outlier_direction_steps": int(np.count_nonzero(outlier_directions)),
        "center_rejections": int(np.count_nonzero(~center_accepted)),
        "center_backtracks": int(np.count_nonzero(backtracked)),
        "non_ok_centers": int(sum(row["current_status"] != "ok" for row in rows)),
        "best_components": {name: float(native["components"][name]) for name in component_names},
        "best_diagnostics": {
            "qh_global_error": float(diagnostics["qs_global_error"]),
            "qh_global_error_per_helicity": float(diagnostics["qs_target_global_error_per_helicity"]),
            "qa_global_error": float(diagnostics["qs_qa_global_error"]),
            "qp_global_error": float(diagnostics["qs_qp_global_error"]),
            "iota_min": float(diagnostics["iota_min"]),
            "iota_max": float(diagnostics["iota_max"]),
            "surface_level": float(diagnostics["surface_level"]),
            "surface_drift_relative_p95": float(diagnostics["surface_drift_relative_p95"]),
            "alpha_relative_l2": float(diagnostics["alpha_relative_l2"]),
        },
        "post_300_direction_signal": {
            "direction_0_abs_median": float(np.median(np.abs(delta[post, 0]))),
            "direction_1_abs_median": float(np.median(np.abs(delta[post, 1]))),
            "direction_0_abs_p95": float(np.quantile(np.abs(delta[post, 0]), 0.95)),
            "direction_1_abs_p95": float(np.quantile(np.abs(delta[post, 1]), 0.95)),
            "direction_0_abs_max": float(np.max(np.abs(delta[post, 0]))),
            "direction_1_abs_max": float(np.max(np.abs(delta[post, 1]))),
            "direction_0_positive_fraction": float(np.mean(delta[post, 0] > 0.0)),
            "direction_1_positive_fraction": float(np.mean(delta[post, 1] > 0.0)),
            "gradient_rms_p50": float(np.median(array(rows, "gradient_rms")[post])),
            "gradient_rms_p95": float(np.quantile(array(rows, "gradient_rms")[post], 0.95)),
            "gradient_rms_max": float(np.max(array(rows, "gradient_rms")[post])),
            "rejected_direction_0_abs_median": (
                float(np.median(np.abs(delta[rejected, 0]))) if np.any(rejected) else None
            ),
            "rejected_direction_1_abs_median": (
                float(np.median(np.abs(delta[rejected, 1]))) if np.any(rejected) else None
            ),
            "rejected_iteration_wall_mean_s": (
                float(np.mean(array(rows, "iteration_wall_s")[rejected]))
                if np.any(rejected)
                else None
            ),
            "accepted_iteration_wall_mean_s": float(
                np.mean(array(rows, "iteration_wall_s")[~rejected])
            ),
        },
    }
    return rows, metrics


rows = {}
metrics = {}
for name, run_dir in RUNS.items():
    rows[name], metrics[name] = run_metrics(run_dir)

pre_fields = [
    "current_score",
    "best_score",
    "gradient_rms",
    "update_rms",
    "current_qh_error",
    "current_iota",
]
pre_switch = {}
for field in pre_fields:
    left = array(rows["random"][:300], field)
    right = array(rows["previous_update"][:300], field)
    pre_switch[f"max_abs_{field}"] = float(np.max(np.abs(left - right)))
left_pairs = np.asarray([row["pair_scores"] for row in rows["random"][:300]])
right_pairs = np.asarray([row["pair_scores"] for row in rows["previous_update"][:300]])
pre_switch["max_abs_pair_score"] = float(np.max(np.abs(left_pairs - right_pairs)))
pre_switch["all_random_sources"] = all(
    row["direction_sources"] == ["random", "random"]
    for data in rows.values()
    for row in data[:300]
)
post_sources = [row["direction_sources"] for row in rows["previous_update"][300:]]
pre_switch["post_switch_source_rows_correct"] = int(
    sum(source == ["previous_update", "random"] for source in post_sources)
)

milestones = {}
for step in [0, 100, 200, 300, 400, 600, 800, 916, 923, 994, 1000]:
    milestones[str(step)] = {}
    for name in RUNS:
        if step == 0:
            summary = metrics[name]["summary"]
            current = best_score = summary["initial_score"]
        else:
            row = rows[name][step - 1]
            current = row["current_score"]
            best_score = row["best_score"]
        milestones[str(step)][name] = {
            "current_score": float(current),
            "best_score": float(best_score),
        }

output = {
    "jobs": {"random": 35902, "previous_update": 35903},
    "pre_switch_identity": pre_switch,
    "milestones": milestones,
    "runs": metrics,
    "best_score_difference_previous_minus_random": float(
        metrics["previous_update"]["summary"]["best_score"]
        - metrics["random"]["summary"]["best_score"]
    ),
}
(ROOT / "analysis_summary.json").write_text(
    json.dumps(output, indent=2, allow_nan=True) + "\n", encoding="utf-8"
)

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.24})
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

for name in RUNS:
    x = array(rows[name], "iteration", int)
    score = array(rows[name], "current_score")
    best = array(rows[name], "best_score")
    axes[0, 0].plot(x, score, color=COLORS[name], alpha=0.32, linewidth=0.8)
    axes[0, 0].plot(x, best, color=COLORS[name], linewidth=1.8, label=LABELS[name])
    axes[0, 1].plot(x, array(rows[name], "current_qh_error"), color=COLORS[name], label=LABELS[name])
    axes[1, 0].plot(x, array(rows[name], "gradient_rms"), color=COLORS[name], alpha=0.75, label=LABELS[name])
    axes[1, 1].plot(x, array(rows[name], "adam_step", int), color=COLORS[name], label=LABELS[name])

for ax in axes.flat:
    ax.axvline(300, color="#555555", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Iteration")
axes[0, 0].set_title("Current score (thin) and running best (thick)")
axes[0, 0].set_ylabel("Score")
axes[0, 0].legend(loc="lower right")
axes[0, 1].set_title("Current raw QH volume error")
axes[0, 1].set_ylabel("QH error")
axes[0, 1].set_yscale("log")
axes[1, 0].set_title("Estimated gradient RMS")
axes[1, 0].set_ylabel("Gradient RMS")
axes[1, 0].set_yscale("log")
rejected_idx = np.flatnonzero(array(rows["previous_update"], "temporal_step_rejected", bool))
if rejected_idx.size:
    x = array(rows["previous_update"], "iteration", int)[rejected_idx]
    y = array(rows["previous_update"], "gradient_rms")[rejected_idx]
    axes[1, 0].scatter(x, y, s=15, color="#B91C1C", label="Rejected temporal outlier", zorder=4)
axes[1, 0].legend(loc="upper right")
axes[1, 1].set_title("Cumulative accepted Adam steps")
axes[1, 1].set_ylabel("Adam step")
axes[1, 1].legend(loc="lower right")
fig.suptitle("1000-step direction-policy comparison (switch after step 300)", fontsize=14)
fig.savefig(ROOT / "trajectory_comparison.png", dpi=180)
plt.close(fig)

component_names = ["axis", "psi", "surface", "coordinate", "volume_qs", "iota", "coil"]
x = np.arange(len(component_names))
width = 0.38
fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
for offset, name in zip([-width / 2, width / 2], RUNS):
    values = [metrics[name]["best_components"][key] for key in component_names]
    ax.bar(x + offset, values, width=width, color=COLORS[name], label=LABELS[name])
ax.set_xticks(x, component_names)
ax.set_ylim(60, 101)
ax.set_ylabel("Component score")
ax.set_title("Components at each run's best score")
ax.legend(loc="lower left")
ax.grid(axis="x", visible=False)
fig.savefig(ROOT / "best_component_comparison.png", dpi=180)
plt.close(fig)
