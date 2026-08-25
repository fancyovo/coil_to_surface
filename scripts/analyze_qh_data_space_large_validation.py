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


STAGES = [0, 10, 25, 50, 75, 100, 150, 200]


def _finite(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _distribution(values: pd.Series | np.ndarray) -> dict[str, Any]:
    array = _finite(values)
    if not array.size:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
    }


def _bootstrap_median_ci(
    values: pd.Series | np.ndarray, *, seed: int = 20260825, repeats: int = 20_000
) -> list[float]:
    array = _finite(values)
    rng = np.random.default_rng(seed)
    medians = np.empty(repeats, dtype=float)
    for start in range(0, repeats, 1_000):
        stop = min(start + 1_000, repeats)
        samples = rng.choice(array, size=(stop - start, array.size), replace=True)
        medians[start:stop] = np.median(samples, axis=1)
    return [float(value) for value in np.percentile(medians, [2.5, 97.5])]


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().eq("true")


def _geometric_median(values: pd.Series | np.ndarray) -> float:
    array = _finite(values)
    array = array[array > 0.0]
    return float(np.exp(np.median(np.log(array)))) if array.size else float("nan")


def _paired_test(delta: np.ndarray, *, positive_is_data_win: bool) -> dict[str, Any]:
    delta = _finite(delta)
    nonzero = delta[delta != 0.0]
    data_wins = int(np.count_nonzero(nonzero > 0.0))
    latent_wins = int(np.count_nonzero(nonzero < 0.0))
    if not positive_is_data_win:
        data_wins, latent_wins = latent_wins, data_wins
    wilcoxon = stats.wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
    return {
        "count": int(delta.size),
        "data_wins": data_wins,
        "latent_wins": latent_wins,
        "ties": int(delta.size - nonzero.size),
        "data_win_fraction_non_ties": float(data_wins / nonzero.size),
        "two_sided_sign_test_p": float(
            stats.binomtest(data_wins, int(nonzero.size), 0.5).pvalue
        ),
        "wilcoxon_two_sided_p": float(wilcoxon.pvalue),
    }


def analyze_score_space(
    data_path: Path, latent_path: Path, output_dir: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = pd.read_csv(data_path)
    latent = pd.read_csv(latent_path)
    paired = data.merge(
        latent[
            [
                "trajectory_id",
                "initial_online_score",
                "best_online_score",
                "optimization_wall_s",
            ]
        ],
        on="trajectory_id",
        how="inner",
        suffixes=("_data", "_latent"),
        validate="one_to_one",
    )
    if len(paired) != len(data) or len(paired) != len(latent):
        raise RuntimeError("The optimization tables are not an exact trajectory pairing.")

    reference_error = (
        paired["reference_best_score"] - paired["best_online_score_latent"]
    ).abs()
    if float(reference_error.max()) > 1e-9:
        raise RuntimeError("The data-space references do not match the latent corpus.")

    paired["best_difference_data_minus_latent"] = (
        paired["best_online_score_data"] - paired["best_online_score_latent"]
    )
    paired["gain_data"] = (
        paired["best_online_score_data"] - paired["initial_online_score_data"]
    )
    paired["gain_latent"] = (
        paired["best_online_score_latent"] - paired["initial_online_score_latent"]
    )
    paired["runtime_ratio_data_over_latent"] = (
        paired["optimization_wall_s_data"] / paired["optimization_wall_s_latent"]
    )

    difference = paired["best_difference_data_minus_latent"].to_numpy()
    conditions = []
    for (nfp, n_coils), group in paired.groupby(["nfp", "n_base_coils"]):
        conditions.append(
            {
                "nfp": int(nfp),
                "n_base_coils": int(n_coils),
                "count": int(len(group)),
                "data_best_median": float(group["best_online_score_data"].median()),
                "latent_best_median": float(group["best_online_score_latent"].median()),
                "difference_median": float(
                    group["best_difference_data_minus_latent"].median()
                ),
                "data_win_fraction": float(
                    (group["best_difference_data_minus_latent"] > 0.0).mean()
                ),
            }
        )
    condition_frame = pd.DataFrame(conditions).sort_values(["nfp", "n_base_coils"])

    thresholds = {}
    for threshold in [85.0, 90.0, 92.0]:
        thresholds[f"score_ge_{threshold:g}"] = {
            "data_count": int((paired["best_online_score_data"] >= threshold).sum()),
            "data_fraction": float(
                (paired["best_online_score_data"] >= threshold).mean()
            ),
            "latent_count": int((paired["best_online_score_latent"] >= threshold).sum()),
            "latent_fraction": float(
                (paired["best_online_score_latent"] >= threshold).mean()
            ),
        }

    summary = {
        "pairing": {
            "trajectory_count": int(len(paired)),
            "max_reference_best_score_error": float(reference_error.max()),
            "initial_roundtrip_difference": _distribution(
                paired["initial_online_score_data"]
                - paired["initial_online_score_latent"]
            ),
        },
        "best_score": {
            "data": _distribution(paired["best_online_score_data"]),
            "latent": _distribution(paired["best_online_score_latent"]),
            "data_minus_latent": _distribution(difference),
            "median_difference_bootstrap_95": _bootstrap_median_ci(difference),
            "paired_test": _paired_test(difference, positive_is_data_win=True),
            "pearson": float(
                stats.pearsonr(
                    paired["best_online_score_latent"],
                    paired["best_online_score_data"],
                ).statistic
            ),
            "spearman": float(
                stats.spearmanr(
                    paired["best_online_score_latent"],
                    paired["best_online_score_data"],
                ).statistic
            ),
            "thresholds": thresholds,
        },
        "gain": {
            "data": _distribution(paired["gain_data"]),
            "latent": _distribution(paired["gain_latent"]),
        },
        "optimization_wall_s": {
            "data": _distribution(paired["optimization_wall_s_data"]),
            "latent": _distribution(paired["optimization_wall_s_latent"]),
            "data_over_latent": _distribution(
                paired["runtime_ratio_data_over_latent"]
            ),
            "latent_over_data_median_speedup": float(
                paired["optimization_wall_s_latent"].median()
                / paired["optimization_wall_s_data"].median()
            ),
        },
    }

    paired.to_csv(output_dir / "paired_optimizer_trajectories.csv", index=False)
    condition_frame.to_csv(output_dir / "optimizer_condition_summary.csv", index=False)
    plot_score_space(paired, condition_frame, output_dir)
    return paired, summary


def plot_score_space(
    paired: pd.DataFrame, conditions: pd.DataFrame, output_dir: Path
) -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12})
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), constrained_layout=True)

    axis = axes[0]
    scatter = axis.scatter(
        paired["best_online_score_latent"],
        paired["best_online_score_data"],
        c=paired["n_base_coils"],
        cmap="viridis",
        s=24,
        alpha=0.72,
        linewidths=0,
    )
    lower = min(
        paired["best_online_score_latent"].min(),
        paired["best_online_score_data"].min(),
    )
    upper = max(
        paired["best_online_score_latent"].max(),
        paired["best_online_score_data"].max(),
    )
    axis.plot([lower, upper], [lower, upper], color="black", lw=1, ls="--")
    axis.set(xlabel="Latent-space best score", ylabel="Data-space best score")
    axis.set_title("309 matched starts")
    fig.colorbar(scatter, ax=axis, label="Base coils")

    axis = axes[1]
    difference = paired["best_difference_data_minus_latent"]
    axis.hist(difference, bins=32, color="#4C78A8", alpha=0.82)
    axis.axvline(0.0, color="black", lw=1)
    axis.axvline(difference.median(), color="#E45756", lw=2, label="median")
    axis.axvline(difference.mean(), color="#F2CF5B", lw=2, label="mean")
    axis.set(
        xlabel="Best-score difference (data - latent)", ylabel="Trajectories"
    )
    axis.set_title("Latent wins more often; data has a positive tail")
    axis.legend(frameon=False)

    axis = axes[2]
    axis.scatter(
        paired["optimization_wall_s_latent"] / 60.0,
        paired["optimization_wall_s_data"] / 60.0,
        c="#54A24B",
        s=22,
        alpha=0.65,
        linewidths=0,
    )
    lower = min(
        paired["optimization_wall_s_latent"].min(),
        paired["optimization_wall_s_data"].min(),
    ) / 60.0
    upper = max(
        paired["optimization_wall_s_latent"].max(),
        paired["optimization_wall_s_data"].max(),
    ) / 60.0
    axis.plot([lower, upper], [lower, upper], color="black", lw=1, ls="--")
    axis.set(
        xlabel="Latent optimization wall time (min)",
        ylabel="Data optimization wall time (min)",
    )
    axis.set_title("Optimization only; screening excluded")
    fig.savefig(output_dir / "score_space_comparison.png", dpi=190)
    plt.close(fig)

    conditions = conditions[conditions["count"] >= 6].copy()
    conditions["label"] = conditions.apply(
        lambda row: f"NFP {int(row.nfp)}, coils {int(row.n_base_coils)} (n={int(row['count'])})",
        axis=1,
    )
    conditions = conditions.sort_values("difference_median")
    height = max(6.0, 0.31 * len(conditions))
    fig, axis = plt.subplots(figsize=(9.0, height), constrained_layout=True)
    colors = np.where(conditions["difference_median"] >= 0.0, "#54A24B", "#4C78A8")
    axis.barh(conditions["label"], conditions["difference_median"], color=colors)
    axis.axvline(0.0, color="black", lw=1)
    axis.set_xlabel("Median best-score difference (data - latent)")
    axis.set_title("Condition-level result (groups with at least six trajectories)")
    fig.savefig(output_dir / "score_condition_breakdown.png", dpi=190)
    plt.close(fig)


def analyze_face_space(
    data_path: Path, latent_path: Path, output_dir: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = pd.read_csv(data_path)
    latent = pd.read_csv(latent_path)
    data = data[_as_bool(data["scheduled_snapshot"])].copy()

    data_ids = set(data["trajectory_id"])
    latent_ids = set(latent["trajectory_id"])
    common_ids = sorted(data_ids & latent_ids)
    data_only = sorted(data_ids - latent_ids)
    latent_only = sorted(latent_ids - data_ids)

    keys = ["trajectory_id", "iteration", "surface_name"]
    fields = [
        "face_qh",
        "accepted",
        "solved_regular",
        "native_qh",
        "native_score",
        "optimizer_wall_s",
        "face_iota",
        "native_iota_mid",
    ]
    paired = data[keys + fields].merge(
        latent[keys + fields],
        on=keys,
        how="inner",
        suffixes=("_data", "_latent"),
        validate="one_to_one",
    )
    for method in ["data", "latent"]:
        paired[f"accepted_{method}"] = _as_bool(paired[f"accepted_{method}"])
        paired[f"solved_regular_{method}"] = _as_bool(
            paired[f"solved_regular_{method}"]
        )
    paired["both_strict"] = paired["accepted_data"] & paired["accepted_latent"]
    paired["both_solved"] = (
        paired["solved_regular_data"] & paired["solved_regular_latent"]
    )
    paired["face_qh_ratio_data_over_latent"] = (
        paired["face_qh_data"] / paired["face_qh_latent"]
    )

    fixed = paired[paired["surface_name"] == "fixed_probe"].copy()
    start = fixed[(fixed["iteration"] == 0) & fixed["both_strict"]].copy()
    end = fixed[(fixed["iteration"] == 200) & fixed["both_strict"]].copy()

    best_rows = []
    for trajectory_id in common_ids:
        data_group = data[
            (data["trajectory_id"] == trajectory_id)
            & (data["surface_name"] == "fixed_probe")
            & _as_bool(data["accepted"])
        ]
        latent_group = latent[
            (latent["trajectory_id"] == trajectory_id)
            & (latent["surface_name"] == "fixed_probe")
            & _as_bool(latent["accepted"])
        ]
        if data_group.empty or latent_group.empty:
            continue
        data_best = data_group.loc[data_group["face_qh"].idxmin()]
        latent_best = latent_group.loc[latent_group["face_qh"].idxmin()]
        best_rows.append(
            {
                "trajectory_id": trajectory_id,
                "data_best_face_qh": float(data_best["face_qh"]),
                "data_best_iteration": int(data_best["iteration"]),
                "latent_best_face_qh": float(latent_best["face_qh"]),
                "latent_best_iteration": int(latent_best["iteration"]),
            }
        )
    best = pd.DataFrame(best_rows)
    if not best.empty:
        best["ratio_data_over_latent"] = (
            best["data_best_face_qh"] / best["latent_best_face_qh"]
        )

    def face_comparison(frame: pd.DataFrame) -> dict[str, Any]:
        if frame.empty:
            return {"count": 0}
        log_delta = np.log10(frame["face_qh_data"]) - np.log10(
            frame["face_qh_latent"]
        )
        return {
            "count": int(len(frame)),
            "data_face_qh": _distribution(frame["face_qh_data"]),
            "latent_face_qh": _distribution(frame["face_qh_latent"]),
            "data_over_latent": _distribution(
                frame["face_qh_ratio_data_over_latent"]
            ),
            "paired_test_lower_is_better": _paired_test(
                log_delta.to_numpy(), positive_is_data_win=False
            ),
        }

    strict_counts = fixed.groupby("trajectory_id")["both_strict"].sum()
    complete_ids = set(strict_counts[strict_counts == len(STAGES)].index)
    complete_fixed = fixed[fixed["trajectory_id"].isin(complete_ids)]
    common_curve = []
    for iteration in STAGES:
        group = complete_fixed[complete_fixed["iteration"] == iteration]
        common_curve.append(
            {
                "iteration": iteration,
                "paired_count": int(len(group)),
                "data_geometric_median_face_qh": _geometric_median(
                    group["face_qh_data"]
                ),
                "latent_geometric_median_face_qh": _geometric_median(
                    group["face_qh_latent"]
                ),
            }
        )
    curve = pd.DataFrame(common_curve)

    if best.empty:
        best_summary = {"count": 0}
    else:
        log_delta = np.log10(best["data_best_face_qh"]) - np.log10(
            best["latent_best_face_qh"]
        )
        best_summary = {
            "count": int(len(best)),
            "data_best_face_qh": _distribution(best["data_best_face_qh"]),
            "latent_best_face_qh": _distribution(best["latent_best_face_qh"]),
            "data_over_latent": _distribution(best["ratio_data_over_latent"]),
            "paired_test_lower_is_better": _paired_test(
                log_delta.to_numpy(), positive_is_data_win=False
            ),
        }

    summary = {
        "selection_audit": {
            "data_selected_trajectories": len(data_ids),
            "latent_selected_trajectories": len(latent_ids),
            "common_trajectories": len(common_ids),
            "data_only_trajectories": len(data_only),
            "latent_only_trajectories": len(latent_only),
            "strict_method_comparison_scope": "common trajectories only",
        },
        "paired_rows": {
            "all_surfaces": int(len(paired)),
            "fixed_probe": int(len(fixed)),
            "fixed_probe_both_strict": int(fixed["both_strict"].sum()),
            "trajectories_strict_at_all_eight_stages": int(len(complete_ids)),
        },
        "start_fixed_probe": face_comparison(start),
        "iteration_200_fixed_probe": face_comparison(end),
        "best_of_eight_scheduled_fixed_probe": best_summary,
    }

    paired.to_csv(output_dir / "paired_common_face_records.csv", index=False)
    curve.to_csv(output_dir / "paired_common_face_curve.csv", index=False)
    best.to_csv(output_dir / "paired_common_face_best.csv", index=False)
    plot_face_space(start, end, best, curve, output_dir)
    return paired, summary


def plot_face_space(
    start: pd.DataFrame,
    end: pd.DataFrame,
    best: pd.DataFrame,
    curve: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.2), constrained_layout=True)

    def scatter_panel(axis: plt.Axes, frame: pd.DataFrame, title: str) -> None:
        if frame.empty:
            axis.text(0.5, 0.5, "No strict pairs", ha="center", va="center")
            return
        axis.scatter(
            frame["face_qh_latent"],
            frame["face_qh_data"],
            c="#4C78A8",
            alpha=0.75,
            s=34,
            linewidths=0,
        )
        values = np.concatenate(
            [frame["face_qh_latent"].to_numpy(), frame["face_qh_data"].to_numpy()]
        )
        lower, upper = float(values.min()), float(values.max())
        axis.plot([lower, upper], [lower, upper], color="black", lw=1, ls="--")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set(xlabel="Latent face QH", ylabel="Data face QH")
        axis.set_title(f"{title} (n={len(frame)})")

    scatter_panel(axes[0, 0], start, "Same-start check, iteration 0")
    scatter_panel(axes[0, 1], end, "Iteration 200, strict pairs")

    axis = axes[1, 0]
    if not best.empty:
        axis.scatter(
            best["latent_best_face_qh"],
            best["data_best_face_qh"],
            c="#F58518",
            alpha=0.75,
            s=34,
            linewidths=0,
        )
        values = np.concatenate(
            [best["latent_best_face_qh"].to_numpy(), best["data_best_face_qh"].to_numpy()]
        )
        lower, upper = float(values.min()), float(values.max())
        axis.plot([lower, upper], [lower, upper], color="black", lw=1, ls="--")
        axis.set_xscale("log")
        axis.set_yscale("log")
    axis.set(xlabel="Latent best sampled face QH", ylabel="Data best sampled face QH")
    axis.set_title(f"Best of eight scheduled stages (n={len(best)})")

    axis = axes[1, 1]
    axis.plot(
        curve["iteration"],
        curve["latent_geometric_median_face_qh"],
        marker="o",
        label="Latent",
        color="#4C78A8",
    )
    axis.plot(
        curve["iteration"],
        curve["data_geometric_median_face_qh"],
        marker="o",
        label="Data",
        color="#F58518",
    )
    axis.set_yscale("log")
    axis.set(xlabel="Iteration", ylabel="Paired geometric-median face QH")
    pair_count = int(curve["paired_count"].min()) if not curve.empty else 0
    axis.set_title(f"Same strict trajectories at all eight stages (n={pair_count})")
    axis.legend(frameon=False)
    fig.savefig(output_dir / "paired_face_qh_comparison.png", dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the final paired analysis for the large data-space rerun."
    )
    parser.add_argument("--data-optimizer", type=Path, required=True)
    parser.add_argument("--latent-optimizer", type=Path, required=True)
    parser.add_argument("--data-face", type=Path, required=True)
    parser.add_argument("--latent-face", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _, score_summary = analyze_score_space(
        args.data_optimizer, args.latent_optimizer, args.output_dir
    )
    _, face_summary = analyze_face_space(
        args.data_face, args.latent_face, args.output_dir
    )
    summary = {
        "format": "qh_data_space_large_validation_final_analysis_v1",
        "score_space": score_summary,
        "face_space": face_summary,
    }
    (args.output_dir / "method_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
