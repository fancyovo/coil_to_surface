from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, spearmanr, wilcoxon


COLORS = {"latent": "#315A78", "data": "#D17A45"}
LABELS = {"latent": "Flow latent", "data": "Normalized data"}


def load_history(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def quantile(values: list[float] | np.ndarray, probability: float) -> float | None:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, probability)) if array.size else None


def bootstrap_median_interval(
    values: np.ndarray,
    *,
    seed: int = 20260825,
    repeats: int = 10000,
) -> list[float] | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return None
    rng = np.random.default_rng(seed)
    draws = rng.choice(finite, size=(repeats, finite.size), replace=True)
    medians = np.median(draws, axis=1)
    return [float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))]


def running_gain(history: list[dict], initial: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score = np.maximum.accumulate(
        np.asarray([initial] + [float(item["best_score"]) for item in history])
    )
    wall = np.asarray([0.0] + [float(item["total_wall_s"]) for item in history])
    return np.arange(score.size), wall, score - initial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run_root / "pair_manifest.json").read_text(encoding="utf-8"))

    rows: list[dict] = []
    histories: dict[tuple[str, str], list[dict]] = {}
    for case in manifest["cases"]:
        for parameter_space in ("latent", "data"):
            root = args.run_root / case["case_id"] / parameter_space
            run_summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            history = load_history(root / "history.jsonl")
            histories[(case["case_id"], parameter_space)] = history
            rows.append(
                {
                    "case_id": case["case_id"],
                    "trajectory_id": case["trajectory_id"],
                    "source_stage": case.get("source_stage", "unknown"),
                    "nfp": int(case["nfp"]),
                    "n_base_coils": int(case["n_base_coils"]),
                    "parameter_space": parameter_space,
                    "initial_score": float(run_summary["initial_score"]),
                    "best_score": float(run_summary["best_score"]),
                    "best_iteration": int(run_summary["best_iteration"]),
                    "completed_iterations": int(run_summary["completed_iterations"]),
                    "total_wall_s": float(history[-1]["total_wall_s"]),
                    "accepted_updates": int(
                        sum(bool(item["center_update_accepted"]) for item in history)
                    ),
                    "invalid_gradient_steps": int(
                        sum(not bool(item["gradient_step_applied"]) for item in history)
                    ),
                }
            )

    paired: list[dict] = []
    for case in manifest["cases"]:
        by_space = {
            row["parameter_space"]: row
            for row in rows
            if row["case_id"] == case["case_id"]
        }
        latent = by_space["latent"]
        data = by_space["data"]
        paired.append(
            {
                "case_id": case["case_id"],
                "trajectory_id": case["trajectory_id"],
                "nfp": int(case["nfp"]),
                "n_base_coils": int(case["n_base_coils"]),
                "initial_score": 0.5 * (latent["initial_score"] + data["initial_score"]),
                "initial_score_abs_difference": abs(
                    latent["initial_score"] - data["initial_score"]
                ),
                "latent_best": latent["best_score"],
                "data_best": data["best_score"],
                "latent_gain": latent["best_score"] - latent["initial_score"],
                "data_gain": data["best_score"] - data["initial_score"],
                "best_difference": latent["best_score"] - data["best_score"],
                "latent_wall_s": latent["total_wall_s"],
                "data_wall_s": data["total_wall_s"],
            }
        )

    difference = np.asarray([row["best_difference"] for row in paired], dtype=float)
    latent_gain = np.asarray([row["latent_gain"] for row in paired], dtype=float)
    data_gain = np.asarray([row["data_gain"] for row in paired], dtype=float)
    non_ties = difference[np.abs(difference) > 1.0e-10]
    sign_test_p = (
        float(binomtest(int(np.count_nonzero(non_ties > 0.0)), non_ties.size).pvalue)
        if non_ties.size
        else None
    )
    wilcoxon_p = (
        float(wilcoxon(difference, zero_method="wilcox").pvalue)
        if non_ties.size
        else None
    )
    initial_scores = np.asarray([row["initial_score"] for row in paired], dtype=float)
    gain_difference = latent_gain - data_gain
    initial_gain_spearman = (
        float(spearmanr(initial_scores, gain_difference).statistic)
        if len(paired) >= 3
        else None
    )

    common_wall_budget_s = min(
        float(history[-1]["total_wall_s"]) for history in histories.values()
    )
    gain_at_common_wall: dict[str, list[float]] = {"latent": [], "data": []}
    for case in manifest["cases"]:
        for parameter_space in ("latent", "data"):
            history = histories[(case["case_id"], parameter_space)]
            initial = next(
                row["initial_score"]
                for row in rows
                if row["case_id"] == case["case_id"]
                and row["parameter_space"] == parameter_space
            )
            _, wall, gain = running_gain(history, initial)
            gain_at_common_wall[parameter_space].append(
                float(np.interp(common_wall_budget_s, wall, gain))
            )

    condition_counts = Counter(
        f"nfp{row['nfp']}_coils{row['n_base_coils']}" for row in paired
    )
    aggregate = {
        "latent_wins": int(np.count_nonzero(difference > 1.0e-10)),
        "ties": int(np.count_nonzero(np.abs(difference) <= 1.0e-10)),
        "data_wins": int(np.count_nonzero(difference < -1.0e-10)),
        "paired_sign_test_two_sided_p": sign_test_p,
        "paired_wilcoxon_two_sided_p": wilcoxon_p,
        "latent_minus_data_best_p10": quantile(difference, 0.10),
        "latent_minus_data_best_p50": quantile(difference, 0.50),
        "latent_minus_data_best_p90": quantile(difference, 0.90),
        "latent_minus_data_best_median_bootstrap_95": bootstrap_median_interval(difference),
        "latent_gain_p10": quantile(latent_gain, 0.10),
        "latent_gain_p50": quantile(latent_gain, 0.50),
        "latent_gain_p90": quantile(latent_gain, 0.90),
        "data_gain_p10": quantile(data_gain, 0.10),
        "data_gain_p50": quantile(data_gain, 0.50),
        "data_gain_p90": quantile(data_gain, 0.90),
        "latent_zero_gain_fraction": float(np.mean(latent_gain <= 1.0e-10)),
        "data_zero_gain_fraction": float(np.mean(data_gain <= 1.0e-10)),
        "latent_wall_s_p50": quantile(
            [row["total_wall_s"] for row in rows if row["parameter_space"] == "latent"],
            0.50,
        ),
        "data_wall_s_p50": quantile(
            [row["total_wall_s"] for row in rows if row["parameter_space"] == "data"],
            0.50,
        ),
        "latent_to_data_wall_ratio_p50": quantile(
            [row["latent_wall_s"] / row["data_wall_s"] for row in paired], 0.50
        ),
        "common_wall_budget_s": common_wall_budget_s,
        "latent_gain_at_common_wall_p50": quantile(
            gain_at_common_wall["latent"], 0.50
        ),
        "data_gain_at_common_wall_p50": quantile(
            gain_at_common_wall["data"], 0.50
        ),
        "initial_score_p10": quantile(initial_scores, 0.10),
        "initial_score_p50": quantile(initial_scores, 0.50),
        "initial_score_p90": quantile(initial_scores, 0.90),
        "initial_score_abs_difference_max": float(
            max(row["initial_score_abs_difference"] for row in paired)
        ),
        "initial_score_vs_gain_difference_spearman": initial_gain_spearman,
    }
    summary = {
        "format": "summary1_flow_parameterization_pairs_analysis_v2",
        "case_count": len(manifest["cases"]),
        "selection_rule": manifest.get("selection_rule"),
        "condition_counts": dict(sorted(condition_counts.items())),
        "runs": rows,
        "pairs": paired,
        "aggregate": aggregate,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "pair_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.8), constrained_layout=True)
    iteration_curves: dict[str, list[np.ndarray]] = {"latent": [], "data": []}
    for parameter_space in ("latent", "data"):
        for case in manifest["cases"]:
            history = histories[(case["case_id"], parameter_space)]
            initial = next(
                row["initial_score"]
                for row in rows
                if row["case_id"] == case["case_id"]
                and row["parameter_space"] == parameter_space
            )
            _, _, gain = running_gain(history, initial)
            iteration_curves[parameter_space].append(gain)
        minimum = min(len(value) for value in iteration_curves[parameter_space])
        stacked = np.stack(
            [value[:minimum] for value in iteration_curves[parameter_space]]
        )
        x_iteration = np.arange(minimum)
        axes[0, 0].fill_between(
            x_iteration,
            np.quantile(stacked, 0.10, axis=0),
            np.quantile(stacked, 0.90, axis=0),
            color=COLORS[parameter_space],
            alpha=0.08,
        )
        axes[0, 0].fill_between(
            x_iteration,
            np.quantile(stacked, 0.25, axis=0),
            np.quantile(stacked, 0.75, axis=0),
            color=COLORS[parameter_space],
            alpha=0.18,
        )
        axes[0, 0].plot(
            x_iteration,
            np.median(stacked, axis=0),
            color=COLORS[parameter_space],
            lw=2.6,
            label=LABELS[parameter_space],
        )

    for parameter_space, values in (("latent", latent_gain), ("data", data_gain)):
        ordered = np.sort(values)
        probability = np.arange(1, ordered.size + 1) / ordered.size
        axes[0, 1].step(
            ordered,
            probability,
            where="post",
            color=COLORS[parameter_space],
            lw=2.4,
            label=(
                f"{LABELS[parameter_space]} "
                f"(zero gain {np.mean(values <= 1.0e-10):.0%})"
            ),
        )

    order = np.argsort(difference)
    rank = np.arange(len(paired))
    for y, index in zip(rank, order, strict=True):
        axes[1, 0].plot(
            [data_gain[index], latent_gain[index]],
            [y, y],
            color="#B8B8B8",
            lw=1.2,
            zorder=1,
        )
    axes[1, 0].scatter(
        data_gain[order],
        rank,
        color=COLORS["data"],
        s=30,
        label=LABELS["data"],
        zorder=2,
    )
    axes[1, 0].scatter(
        latent_gain[order],
        rank,
        color=COLORS["latent"],
        s=30,
        label=LABELS["latent"],
        zorder=2,
    )

    axes[1, 1].axhline(0.0, color="#555555", ls="--", lw=1)
    scatter = axes[1, 1].scatter(
        initial_scores,
        gain_difference,
        c=np.asarray([row["n_base_coils"] for row in paired]),
        cmap="viridis",
        s=43,
        edgecolor="white",
        linewidth=0.5,
    )
    colorbar = fig.colorbar(scatter, ax=axes[1, 1], pad=0.02)
    colorbar.set_label("Base coils")

    axes[0, 0].set(
        xlabel="Adam iteration",
        ylabel="Running-best score gain",
        title="Median optimization progress (bands: IQR and P10-P90)",
    )
    axes[0, 1].set(
        xlabel="Best score gain after 100 steps",
        ylabel="Empirical cumulative probability",
        title="Full gain distribution",
    )
    axes[1, 0].set(
        xlabel="Best score gain after 100 steps",
        ylabel="Pairs sorted by Flow advantage",
        title="Matched outcomes from the same initial coil",
        yticks=[],
    )
    axes[1, 1].set(
        xlabel="Initial score after 32-to-1 screening",
        ylabel="Flow gain minus data-space gain",
        title="Does the advantage depend on starting quality?",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    axes[0, 1].legend(frameon=False)
    axes[1, 0].legend(frameon=False, loc="lower right")
    fig.savefig(
        args.output_dir / "flow_vs_data_optimization.png",
        dpi=220,
        facecolor="white",
    )
    plt.close(fig)
    print(json.dumps(aggregate))


if __name__ == "__main__":
    main()
