from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stellarator_eval.serialization import write_json


def _finite_pairs(rows, x_key, y_key):
    pairs = []
    for row in rows:
        try:
            x = float(x_key(row))
            y = float(y_key(row))
        except (TypeError, ValueError, KeyError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            pairs.append((x, y))
    if not pairs:
        return np.empty(0), np.empty(0)
    return np.asarray(pairs, dtype=float).T


def _correlation(x, y):
    if len(x) < 3 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return {"count": int(len(x)), "spearman": None, "pearson": None}
    return {
        "count": int(len(x)),
        "spearman": float(spearmanr(x, y).statistic),
        "pearson": float(pearsonr(x, y).statistic),
    }


def _distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    batch = json.loads(args.batch_summary.read_text(encoding="utf-8"))
    rows = batch["rows"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    valid = [row for row in rows if row.get("score") is not None]
    successful = [row for row in valid if row.get("status") == "volume_qs"]
    scores = [float(row["score"]) for row in valid]
    x_meta, y_score = _finite_pairs(
        valid,
        lambda row: -np.log10(float(row["metadata_qs_error"])),
        lambda row: row["score"],
    )
    x_volume, y_meta = _finite_pairs(
        successful,
        lambda row: np.log10(float(row["details"]["volume_qs_global_error"])),
        lambda row: np.log10(float(row["metadata_qs_error"])),
    )

    status_score = {}
    for status in sorted({str(row["status"]) for row in valid}):
        status_score[status] = _distribution(
            [float(row["score"]) for row in valid if row["status"] == status]
        )
    deciles = []
    order = sorted(
        valid,
        key=lambda row: float(row.get("metadata_qs_error", np.inf)),
    )
    for index, group in enumerate(np.array_split(np.asarray(order, dtype=object), 10)):
        group = list(group)
        if not group:
            continue
        deciles.append(
            {
                "metadata_rank_decile": index + 1,
                "count": len(group),
                "metadata_qs_error_median": float(
                    np.median([float(row["metadata_qs_error"]) for row in group])
                ),
                "score_mean": float(np.mean([float(row["score"]) for row in group])),
                "score_median": float(np.median([float(row["score"]) for row in group])),
                "volume_qs_success_fraction": float(
                    np.mean([row["status"] == "volume_qs" for row in group])
                ),
            }
        )

    analysis = {
        "sample_count": len(rows),
        "finite_score_count": len(valid),
        "status_counts": dict(Counter(str(row["status"]) for row in rows)),
        "score_distribution": _distribution(scores),
        "status_score_distribution": status_score,
        "score_vs_better_metadata_qs": _correlation(x_meta, y_score),
        "volume_metric_vs_metadata_qs": _correlation(x_volume, y_meta),
        "metadata_rank_deciles": deciles,
        "surface_inverse_aspect_ratio": _distribution(
            [
                row.get("details", {}).get("surface_inverse_aspect_ratio")
                for row in successful
            ]
        ),
        "volume_qs_global_error": _distribution(
            [row.get("details", {}).get("volume_qs_global_error") for row in successful]
        ),
        "timing": batch.get("timing"),
    }
    write_json(args.output_dir / "analysis.json", analysis)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    colors = {0: "#2978a0", 1: "#d1495b"}
    for helicity in (0, 1):
        subset = [row for row in valid if int(row.get("helicity", -1)) == helicity]
        x, y = _finite_pairs(
            subset,
            lambda row: row["metadata_qs_error"],
            lambda row: row["score"],
        )
        axes[0].scatter(x, y, s=10, alpha=0.45, color=colors[helicity], label="QA" if helicity == 0 else "QH")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("QUASR metadata QS error (validation only)")
    axes[0].set_ylabel("volume quality score")
    axes[0].legend(frameon=False)

    labels = list(status_score)
    data = [
        [float(row["score"]) for row in valid if row["status"] == status]
        for status in labels
    ]
    axes[1].boxplot(data, labels=labels, showfliers=False)
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_ylabel("score")
    axes[1].set_title("Physical-stage separation")

    axes[2].plot(
        [row["metadata_rank_decile"] for row in deciles],
        [row["score_mean"] for row in deciles],
        marker="o",
        color="#2a9d5b",
        label="mean score",
    )
    axes[2].set_xlabel("metadata QS rank decile (1 = best)")
    axes[2].set_ylabel("mean score", color="#2a9d5b")
    twin = axes[2].twinx()
    twin.plot(
        [row["metadata_rank_decile"] for row in deciles],
        [row["volume_qs_success_fraction"] for row in deciles],
        marker="s",
        color="#6d597a",
        label="success fraction",
    )
    twin.set_ylabel("volume-QS success fraction", color="#6d597a")
    fig.tight_layout()
    fig.savefig(args.output_dir / "score_discrimination.png", dpi=180)
    plt.close(fig)
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
