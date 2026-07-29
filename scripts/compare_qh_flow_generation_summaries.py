from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def score_block(summary: dict[str, Any], name: str) -> dict[str, Any]:
    if name in summary:
        return summary[name]
    legacy_name = f"new_{name}"
    if legacy_name in summary:
        return summary[legacy_name]
    raise KeyError(f"missing {name!r} score block")


def status_counts(summary: dict[str, Any]) -> dict[str, int]:
    if "status_counts" in summary:
        return {str(key): int(value) for key, value in summary["status_counts"].items()}
    counts: dict[str, int] = {}
    for transition, count in summary["status_transitions"].items():
        destination = transition.rsplit("->", 1)[-1]
        counts[destination] = counts.get(destination, 0) + int(count)
    return counts


def comparison_summary(
    old: dict[str, Any], new: dict[str, Any]
) -> dict[str, Any]:
    old_all = score_block(old, "score_all")
    new_all = score_block(new, "score_all")
    old_ok = score_block(old, "score_ok")
    new_ok = score_block(new, "score_ok")
    old_count = int(old_all["count"])
    new_count = int(new_all["count"])
    old_status = status_counts(old)
    new_status = status_counts(new)

    def relative_change(key: str, before: dict[str, Any], after: dict[str, Any]) -> float:
        return float(after[key]) / float(before[key]) - 1.0

    return {
        "old_count": old_count,
        "new_count": new_count,
        "old_geometry_eligible_rate": 1.0
        - old_status.get("geometry_rejected", 0) / old_count,
        "new_geometry_eligible_rate": 1.0
        - new_status.get("geometry_rejected", 0) / new_count,
        "old_ok_rate": old_status.get("ok", 0) / old_count,
        "new_ok_rate": new_status.get("ok", 0) / new_count,
        "score_all_relative_change": {
            key: relative_change(key, old_all, new_all)
            for key in ("mean", "median", "p90", "max")
        },
        "score_ok_relative_change": {
            key: relative_change(key, old_ok, new_ok)
            for key in ("mean", "median", "p90", "max")
        },
        "old_score_all": old_all,
        "new_score_all": new_all,
        "old_score_ok": old_ok,
        "new_score_ok": new_ok,
        "old_status_counts": old_status,
        "new_status_counts": new_status,
    }


def plot_comparison(
    old: dict[str, Any],
    new: dict[str, Any],
    new_top: dict[str, Any],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ("original loss", "physical loss")
    colors = ("#35617a", "#b5543a")
    old_all = score_block(old, "score_all")
    new_all = score_block(new, "score_all")
    old_ok = score_block(old, "score_ok")
    new_ok = score_block(new, "score_ok")
    counts = comparison_summary(old, new)

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    statistics = ("mean", "median", "p90", "max")
    x = np.arange(len(statistics))
    width = 0.36
    for index, (label, color, values) in enumerate(
        zip(labels, colors, (old_all, new_all), strict=True)
    ):
        axes[0, 0].bar(
            x + (index - 0.5) * width,
            [values[key] for key in statistics],
            width,
            label=label,
            color=color,
        )
    axes[0, 0].set(
        title="Score distribution, all 2,048 samples",
        xticks=x,
        xticklabels=statistics,
        ylabel="score",
    )
    axes[0, 0].legend()

    for index, (label, color, values) in enumerate(
        zip(labels, colors, (old_ok, new_ok), strict=True)
    ):
        axes[0, 1].bar(
            x + (index - 0.5) * width,
            [values[key] for key in statistics],
            width,
            label=label,
            color=color,
        )
    axes[0, 1].set(
        title="Score distribution, status=ok",
        xticks=x,
        xticklabels=statistics,
        ylabel="score",
    )
    axes[0, 1].legend()

    rates = ("geometry eligible", "status=ok")
    old_rates = (
        counts["old_geometry_eligible_rate"],
        counts["old_ok_rate"],
    )
    new_rates = (
        counts["new_geometry_eligible_rate"],
        counts["new_ok_rate"],
    )
    rate_x = np.arange(len(rates))
    for index, (label, color, values) in enumerate(
        zip(labels, colors, (old_rates, new_rates), strict=True)
    ):
        axes[1, 0].bar(
            rate_x + (index - 0.5) * width,
            values,
            width,
            label=label,
            color=color,
        )
    axes[1, 0].set(
        title="Generated validity",
        xticks=rate_x,
        xticklabels=rates,
        ylabel="fraction",
        ylim=(0.0, 1.05),
    )
    axes[1, 0].legend()

    old_components = old["best_new"]["components"]
    new_components = new_top["flow_evaluation"]["native_score"]["components"]
    components = ("axis", "psi", "surface", "coordinate", "volume_qs", "iota", "coil")
    component_x = np.arange(len(components))
    for index, (label, color, values) in enumerate(
        zip(labels, colors, (old_components, new_components), strict=True)
    ):
        axes[1, 1].bar(
            component_x + (index - 0.5) * width,
            [values[key] for key in components],
            width,
            label=label,
            color=color,
        )
    axes[1, 1].set(
        title="Highest-score sample components",
        xticks=component_x,
        xticklabels=components,
        ylabel="component score",
        ylim=(0.0, 105.0),
    )
    axes[1, 1].tick_params(axis="x", rotation=30)
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp" + output.suffix)
    figure.savefig(temporary, dpi=160)
    plt.close(figure)
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--new-top", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    old = json.loads(args.old.read_text(encoding="utf-8"))
    new = json.loads(args.new.read_text(encoding="utf-8"))
    new_top = json.loads(args.new_top.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = comparison_summary(old, new)
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    plot_comparison(old, new, new_top, args.output_dir / "generation_comparison.png")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
