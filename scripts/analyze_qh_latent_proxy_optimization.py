from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import mannwhitneyu, pearsonr, spearmanr


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    score = np.asarray([row["score"] for row in rows], dtype=float)
    status = np.asarray([row["status"] for row in rows], dtype="U32")
    ok_score = score[status == "ok"]
    raw_logit = np.asarray([row.get("proxy_raw_logit", np.nan) for row in rows], dtype=float)
    finite = np.isfinite(raw_logit) & np.isfinite(score)
    if np.sum(finite) >= 3 and np.std(raw_logit[finite]) > 0.0 and np.std(score[finite]) > 0.0:
        proxy_score_correlation = {
            "count": int(np.sum(finite)),
            "pearson": float(pearsonr(raw_logit[finite], score[finite]).statistic),
            "spearman": float(spearmanr(raw_logit[finite], score[finite]).statistic),
        }
    else:
        proxy_score_correlation = {"count": int(np.sum(finite)), "pearson": None, "spearman": None}
    movement = [row.get("metadata", {}).get("latent_l2_from_initial") for row in rows]
    movement = np.asarray([value for value in movement if value is not None], dtype=float)
    return {
        "count": len(rows),
        "score": {
            "mean": float(np.mean(score)),
            "median": float(np.median(score)),
            "p10": float(np.percentile(score, 10)),
            "p25": float(np.percentile(score, 25)),
            "p75": float(np.percentile(score, 75)),
            "p90": float(np.percentile(score, 90)),
            "p95": float(np.percentile(score, 95)),
            "max": float(np.max(score)),
        },
        "status_counts": {str(value): int(np.sum(status == value)) for value in np.unique(status)},
        "status_ok_rate": float(np.mean(status == "ok")),
        "status_ok_score": {
            "count": int(len(ok_score)),
            "mean": float(np.mean(ok_score)) if len(ok_score) else None,
            "median": float(np.median(ok_score)) if len(ok_score) else None,
        },
        "score_exceedance_rate": {
            str(threshold): float(np.mean(score >= threshold)) for threshold in (5, 10, 20, 30)
        },
        "proxy_probability": {
            "median": float(np.median([row["proxy_probability"] for row in rows])),
            "min": float(np.min([row["proxy_probability"] for row in rows])),
            "max": float(np.max([row["proxy_probability"] for row in rows])),
        },
        "proxy_raw_logit": {
            "median": float(np.median([row.get("proxy_raw_logit", np.nan) for row in rows])),
            "min": float(np.nanmin([row.get("proxy_raw_logit", np.nan) for row in rows])),
            "max": float(np.nanmax([row.get("proxy_raw_logit", np.nan) for row in rows])),
        },
        "proxy_raw_logit_vs_score": proxy_score_correlation,
        "latent_rms": {
            "median": float(np.median([row["latent_rms"] for row in rows])),
            "p90": float(np.percentile([row["latent_rms"] for row in rows], 90)),
            "max": float(np.max([row["latent_rms"] for row in rows])),
        },
        "latent_l2_from_initial": {
            "count": int(len(movement)),
            "median": float(np.median(movement)) if len(movement) else None,
            "p90": float(np.percentile(movement, 90)) if len(movement) else None,
        },
    }


def bootstrap_difference(
    candidate: list[dict[str, Any]],
    control: list[dict[str, Any]],
    *,
    seed: int,
    replicates: int = 10000,
) -> dict[str, list[float] | float]:
    rng = np.random.default_rng(seed)
    candidate_score = np.asarray([row["score"] for row in candidate], dtype=float)
    control_score = np.asarray([row["score"] for row in control], dtype=float)
    candidate_ok = np.asarray([row["status"] == "ok" for row in candidate], dtype=float)
    control_ok = np.asarray([row["status"] == "ok" for row in control], dtype=float)
    values = {"mean_score": [], "median_score": [], "status_ok_rate": []}
    for _ in range(replicates):
        x_index = rng.integers(0, len(candidate), len(candidate))
        y_index = rng.integers(0, len(control), len(control))
        values["mean_score"].append(float(np.mean(candidate_score[x_index]) - np.mean(control_score[y_index])))
        values["median_score"].append(float(np.median(candidate_score[x_index]) - np.median(control_score[y_index])))
        values["status_ok_rate"].append(float(np.mean(candidate_ok[x_index]) - np.mean(control_ok[y_index])))
    output: dict[str, list[float] | float] = {}
    for name, samples in values.items():
        array = np.asarray(samples)
        output[name] = {
            "estimate": float(np.mean(array)),
            "ci95": [float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))],
        }
    return output


def compare(candidate: list[dict[str, Any]], control: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    x = np.asarray([row["score"] for row in candidate], dtype=float)
    y = np.asarray([row["score"] for row in control], dtype=float)
    test = mannwhitneyu(x, y, alternative="greater", method="auto")
    return {
        "mann_whitney_u": float(test.statistic),
        "one_sided_p_greater": float(test.pvalue),
        "rank_biserial": float(2.0 * test.statistic / (len(x) * len(y)) - 1.0),
        "bootstrap_candidate_minus_control": bootstrap_difference(candidate, control, seed=seed),
    }


def latent_diversity(latent: np.ndarray) -> dict[str, Any]:
    values = np.asarray(latent, dtype=np.float64)
    if values.ndim != 3 or len(values) < 2:
        raise ValueError("latent diversity requires at least two rank-3 samples")
    flat = values.reshape(len(values), -1)
    dimension = flat.shape[1]
    square = np.sum(flat * flat, axis=1)
    square_distance = np.maximum(square[:, None] + square[None, :] - 2.0 * flat @ flat.T, 0.0)
    rms_distance = np.sqrt(square_distance / dimension)
    np.fill_diagonal(rms_distance, np.inf)
    nearest = np.min(rms_distance, axis=1)
    pairwise = rms_distance[np.triu_indices(len(values), k=1)]
    norm = np.sqrt(square).clip(min=1.0e-12)
    cosine = (flat @ flat.T) / (norm[:, None] * norm[None, :])
    pairwise_cosine = cosine[np.triu_indices(len(values), k=1)]
    return {
        "count": len(values),
        "rounded_1e4_unique_count": int(len(np.unique(np.round(flat, 4), axis=0))),
        "nearest_neighbor_rms_distance": {
            "min": float(np.min(nearest)),
            "median": float(np.median(nearest)),
            "p90": float(np.percentile(nearest, 90)),
        },
        "pairwise_rms_distance": {
            "p10": float(np.percentile(pairwise, 10)),
            "median": float(np.median(pairwise)),
            "p90": float(np.percentile(pairwise, 90)),
        },
        "pairwise_cosine_similarity": {
            "p10": float(np.percentile(pairwise_cosine, 10)),
            "median": float(np.median(pairwise_cosine)),
            "p90": float(np.percentile(pairwise_cosine, 90)),
            "max": float(np.max(pairwise_cosine)),
        },
    }


def plot(groups: dict[str, list[dict[str, Any]]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "iid_control_29824": "IID Gaussian control (29824)",
        "free": "free Adam",
        "projected": "radius-projected Adam",
    }
    colors = {"iid_control_29824": "#486a88", "free": "#9a4d42", "projected": "#237a57"}
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for name, rows in groups.items():
        score = np.sort([row["score"] for row in rows])
        axes[0, 0].step(score, np.arange(1, len(score) + 1) / len(score), where="post", color=colors[name], label=labels[name])
    axes[0, 0].set(xlabel="native score", ylabel="empirical CDF", title="Native-score distributions", xlim=(-1, None))
    axes[0, 0].legend()

    axes[0, 1].boxplot(
        [[row["score"] for row in groups[name]] for name in groups],
        tick_labels=[labels[name] for name in groups],
        showfliers=False,
    )
    axes[0, 1].tick_params(axis="x", rotation=12)
    axes[0, 1].set(ylabel="native score", title="Central score distribution")

    statuses = sorted({row["status"] for rows in groups.values() for row in rows})
    bottom = np.zeros(len(groups))
    for status in statuses:
        rate = np.asarray([np.mean([row["status"] == status for row in rows]) for rows in groups.values()])
        axes[1, 0].bar(np.arange(len(groups)), rate, bottom=bottom, label=status)
        bottom += rate
    axes[1, 0].set_xticks(np.arange(len(groups)), [labels[name] for name in groups], rotation=12)
    axes[1, 0].set(ylabel="fraction", title="Physical-gate status", ylim=(0, 1))
    axes[1, 0].legend(fontsize=8)

    for name, rows in groups.items():
        axes[1, 1].scatter(
            [row.get("proxy_raw_logit", np.nan) for row in rows],
            [row["score"] for row in rows],
            s=13,
            alpha=0.45,
            color=colors[name],
            label=labels[name],
        )
    axes[1, 1].set(xlabel="raw proxy logit", ylabel="native score", title="Proxy objective after active optimization")
    axes[1, 1].legend()
    figure.savefig(output_path, dpi=190)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare actively optimized proxy candidates with existing IID controls.")
    parser.add_argument("--optimized-scored", type=Path, required=True)
    parser.add_argument("--control-scored", type=Path, required=True)
    parser.add_argument("--selected-latents", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    optimized = load_jsonl(args.optimized_scored)
    control_all = load_jsonl(args.control_scored)
    control = [row for row in control_all if "iid_prior" in row["sampling_modes"]]
    free = [row for row in optimized if "optimized_free" in row["sampling_modes"]]
    projected = [row for row in optimized if "optimized_projected" in row["sampling_modes"]]
    if not control or not free or not projected:
        raise ValueError("control, free-Adam, and projected-Adam groups must all be nonempty")
    groups = {"iid_control_29824": control, "free": free, "projected": projected}
    with np.load(args.selected_latents, allow_pickle=False) as payload:
        selected_latent = np.asarray(payload["latent"])
        selected_variant = np.asarray(payload["variant"])
    diversity = {
        variant: latent_diversity(selected_latent[selected_variant == variant])
        for variant in ("free", "projected")
    }
    summary = {
        "format": "qh_latent_proxy_optimization_comparison_v1",
        "groups": {name: summarize(rows) for name, rows in groups.items()},
        "comparisons_to_iid_control": {
            name: compare(rows, control, seed=args.seed + index)
            for index, (name, rows) in enumerate((("free", free), ("projected", projected)))
        },
        "selected_latent_diversity": diversity,
        "control_reused": {
            "path": str(args.control_scored.resolve()),
            "selection": "sampling_modes contains iid_prior",
            "count": len(control),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "optimized_vs_iid_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    plot(groups, args.output_dir / "optimized_vs_iid_score_distribution.png")
    print(json.dumps({"event": "complete", **summary}, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
