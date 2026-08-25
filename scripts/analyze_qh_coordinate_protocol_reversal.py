from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def summarize(values: pd.Series | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    data_wins = int(np.count_nonzero(array > 0.0))
    latent_wins = int(np.count_nonzero(array < 0.0))
    return {
        "count": int(array.size),
        "data_wins": data_wins,
        "latent_wins": latent_wins,
        "ties": int(array.size - data_wins - latent_wins),
        "mean": float(np.mean(array)),
        "p10": float(np.percentile(array, 10)),
        "p50": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "sign_test_p": float(stats.binomtest(data_wins, data_wins + latent_wins).pvalue),
        "wilcoxon_p": float(stats.wilcoxon(array).pvalue),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose the apparent reversal between the 32- and 309-case coordinate controls."
    )
    parser.add_argument(
        "--old-pairs",
        type=Path,
        default=Path(
            "reports/summary1/assets/flow_pairs_true_starts_20260824/"
            "flow_pairs/analysis_review2/pair_metrics.csv"
        ),
    )
    parser.add_argument(
        "--new-pairs",
        type=Path,
        default=Path(
            "reports/assets/qh_data_space_large_validation_20260825/"
            "comparison_analysis/paired_optimizer_trajectories.csv"
        ),
    )
    parser.add_argument(
        "--history-summary",
        type=Path,
        default=Path(
            "reports/assets/qh_data_space_large_validation_20260825/"
            "protocol_diagnosis/history_same32.jsonl"
        ),
    )
    parser.add_argument(
        "--latent-trajectory-summary",
        type=Path,
        default=Path(
            "reports/assets/qh_adam_trajectory_dataset_pilot_20260813/"
            "trajectory_summary.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "reports/assets/qh_data_space_large_validation_20260825/"
            "protocol_diagnosis"
        ),
    )
    parser.add_argument("--resamples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    old = pd.read_csv(args.old_pairs)
    new = pd.read_csv(args.new_pairs)
    history = pd.DataFrame(
        json.loads(line)
        for line in args.history_summary.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    )
    history = history.pivot(index="trajectory_id", columns="method")
    history.columns = ["_".join(column) for column in history.columns]

    matched = old.merge(history, left_on="trajectory_id", right_index=True, validate="one_to_one")
    if len(matched) != 32:
        raise RuntimeError(f"expected 32 historical pairs, found {len(matched)}")
    if set(old["trajectory_id"]) - set(new["trajectory_id"]):
        raise RuntimeError("the 309-case table is missing one or more historical pair IDs")

    matched["delta_old_2dir_100"] = matched["data_best"] - matched["latent_best"]
    matched["delta_64dir_100"] = matched["best100_data"] - matched["best100_latent"]
    matched["delta_64dir_200"] = matched["best200_data"] - matched["best200_latent"]
    matched["data_protocol_gain_at_100"] = matched["best100_data"] - matched["data_best"]
    matched["latent_protocol_gain_at_100"] = matched["best100_latent"] - matched["latent_best"]
    matched["data_extra_100_steps"] = matched["best200_data"] - matched["best100_data"]
    matched["latent_extra_100_steps"] = matched["best200_latent"] - matched["best100_latent"]

    comparisons = {
        "old_2_directions_100_steps_equal_lr": summarize(matched["delta_old_2dir_100"]),
        "new_64_directions_100_steps_asymmetric_lr": summarize(matched["delta_64dir_100"]),
        "new_64_directions_200_steps_asymmetric_lr": summarize(matched["delta_64dir_200"]),
    }

    latent_summary = pd.read_csv(args.latent_trajectory_summary)
    rescored = matched[["trajectory_id", "best200_data"]].merge(
        latent_summary[["trajectory_id", "best_online_score", "best_global_score"]],
        on="trajectory_id",
        validate="one_to_one",
    )
    online_delta = rescored["best200_data"] - rescored["best_online_score"]
    global_delta = rescored["best200_data"] - rescored["best_global_score"]

    old = old.copy()
    new = new.copy()
    old["condition"] = (
        "nfp" + old["nfp"].astype(str) + "_coils" + old["n_base_coils"].astype(str)
    )
    new["condition"] = (
        "nfp" + new["nfp"].astype(str) + "_coils" + new["n_base_coils"].astype(str)
    )
    template = old["condition"].value_counts().to_dict()
    groups = {
        condition: new.loc[
            new["condition"] == condition, "best_difference_data_minus_latent"
        ].to_numpy(dtype=float)
        for condition in template
    }
    missing = [condition for condition, values in groups.items() if not len(values)]
    if missing:
        raise RuntimeError(f"the 309-case table lacks historical conditions: {missing}")
    rng = np.random.default_rng(args.seed)
    resampled_wins = np.empty(args.resamples, dtype=int)
    resampled_medians = np.empty(args.resamples, dtype=float)
    for index in range(args.resamples):
        values = np.concatenate(
            [
                rng.choice(group, size=count, replace=len(group) < count)
                for condition, count in template.items()
                for group in [groups[condition]]
            ]
        )
        resampled_wins[index] = int(np.count_nonzero(values > 0.0))
        resampled_medians[index] = float(np.median(values))

    observed_old_wins = int(np.count_nonzero(matched["delta_old_2dir_100"] > 0.0))
    observed_old_median = float(np.median(matched["delta_old_2dir_100"]))
    summary = {
        "format": "qh_coordinate_protocol_reversal_diagnosis_v1",
        "same_id_count": int(len(matched)),
        "comparisons": comparisons,
        "within_coordinate_protocol_change_at_100": {
            "data": {
                "positive_count": int(np.count_nonzero(matched["data_protocol_gain_at_100"] > 0.0)),
                "median": float(np.median(matched["data_protocol_gain_at_100"])),
            },
            "latent": {
                "positive_count": int(np.count_nonzero(matched["latent_protocol_gain_at_100"] > 0.0)),
                "median": float(np.median(matched["latent_protocol_gain_at_100"])),
                "warning": "This change combines 2-to-64 directions with latent LR 0.01-to-0.02.",
            },
        },
        "extra_steps_100_to_200": {
            "data_median_best_gain": float(np.median(matched["data_extra_100_steps"])),
            "latent_median_best_gain": float(np.median(matched["latent_extra_100_steps"])),
        },
        "historical_latent_online_vs_independent_rescore": {
            "online_comparison": summarize(online_delta),
            "independent_rescore_comparison": summarize(global_delta),
            "conclusion": "Replacing historical online best with its independent rescore does not change the 11-to-21 outcome.",
        },
        "condition_stratified_resampling_under_new_protocol": {
            "resamples": int(args.resamples),
            "historical_condition_template_size": int(sum(template.values())),
            "observed_old_data_wins": observed_old_wins,
            "observed_old_median_delta": observed_old_median,
            "count_with_at_least_observed_wins": int(np.count_nonzero(resampled_wins >= observed_old_wins)),
            "count_with_at_least_observed_median": int(np.count_nonzero(resampled_medians >= observed_old_median)),
            "data_win_count_percentiles": [
                float(value) for value in np.percentile(resampled_wins, [1, 5, 50, 95, 99])
            ],
            "median_delta_percentiles": [
                float(value) for value in np.percentile(resampled_medians, [1, 5, 50, 95, 99])
            ],
        },
        "diagnosis": (
            "The 309-case result is numerically real but is not a replication of the "
            "32-case coordinate control. Direction count, latent learning rate, "
            "iteration budget, and random-direction seeds changed together."
        ),
    }

    matched.to_csv(args.output_dir / "same32_protocol_comparison.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    labels = ["2 dirs, 100 steps\nequal LR", "64 dirs, 100 steps\nasymmetric LR", "64 dirs, 200 steps\nasymmetric LR"]
    delta_columns = ["delta_old_2dir_100", "delta_64dir_100", "delta_64dir_200"]
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    x = np.arange(len(labels))
    for _, row in matched.iterrows():
        axes[0].plot(x, [row[column] for column in delta_columns], color="#A7A7A7", alpha=0.32, lw=0.8)
    medians = [float(matched[column].median()) for column in delta_columns]
    axes[0].plot(x, medians, color="#D1495B", marker="o", lw=2.4, label="median")
    axes[0].axhline(0.0, color="black", ls="--", lw=1)
    axes[0].set_yscale("symlog", linthresh=1.0)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("best score difference: data - latent")
    axes[0].set_title("The same 32 starts reverse when the protocol changes")
    axes[0].legend(frameon=False)

    data_gain = matched["data_protocol_gain_at_100"].to_numpy(dtype=float)
    latent_gain = matched["latent_protocol_gain_at_100"].to_numpy(dtype=float)
    axes[1].boxplot([data_gain, latent_gain], tick_labels=["data", "latent"], showfliers=False)
    jitter = np.linspace(-0.07, 0.07, len(matched))
    axes[1].scatter(1.0 + jitter, np.sort(data_gain), s=16, alpha=0.6, color="#4C78A8")
    axes[1].scatter(2.0 + jitter, np.sort(latent_gain), s=16, alpha=0.6, color="#F58518")
    axes[1].axhline(0.0, color="black", ls="--", lw=1)
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[1].set_ylabel("best-score change at step 100")
    axes[1].set_title("Protocol change benefits both; latent changes more")
    figure.savefig(args.output_dir / "same32_protocol_reversal.png", dpi=190)
    plt.close(figure)


if __name__ == "__main__":
    main()
