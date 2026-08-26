from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark_summary1_evaluator_modes import summarize


COLORS = {
    "independent": "#315A78",
    "strict_continuation": "#D17A45",
    "neighborhood_proxy": "#4D8B6A",
}


def load_records(root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(root.glob("records_shard_*.jsonl")):
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not rows:
        raise FileNotFoundError(f"no benchmark records below {root}")
    return rows


def first_candidate_rows(records: list[dict]) -> dict[tuple[str, int, str], dict]:
    output: dict[tuple[str, int, str], dict] = {}
    for row in records:
        if row.get("candidate_index") is None:
            continue
        key = (row["case_id"], int(row["candidate_index"]), row["mode"])
        if key not in output or int(row["repeat"]) < int(output[key]["repeat"]):
            output[key] = row
    return output


def paired_strict_proxy(records: list[dict]) -> list[tuple[dict, dict]]:
    first = first_candidate_rows(records)
    pairs: list[tuple[dict, dict]] = []
    for case_id, candidate, mode in first:
        if mode != "strict_continuation":
            continue
        strict = first[(case_id, candidate, mode)]
        proxy = first.get((case_id, candidate, "neighborhood_proxy"))
        if proxy and strict["status"] == proxy["status"] == "ok":
            pairs.append((strict, proxy))
    return pairs


def coil_component_summary(records: list[dict]) -> dict[str, float | int]:
    pairs = paired_strict_proxy(records)
    strict = np.asarray([row[0]["components"]["coil"] for row in pairs], dtype=float)
    proxy = np.asarray([row[1]["components"]["coil"] for row in pairs], dtype=float)
    error = proxy - strict
    absolute = np.abs(error)
    return {
        "count": int(len(pairs)),
        "spearman": float(spearmanr(strict, proxy).statistic),
        "signed_bias": float(np.mean(error)),
        "mae": float(np.mean(absolute)),
        "absolute_error_p50": float(np.median(absolute)),
        "absolute_error_p95": float(np.quantile(absolute, 0.95)),
        "absolute_error_max": float(np.max(absolute)),
        "weighted_score_error_p95": float(0.08 * np.quantile(absolute, 0.95)),
    }


def plot_overview(records: list[dict], destination: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.0), constrained_layout=True)
    axes = axes.ravel()
    formal = ["independent", "strict_continuation"]
    values = [
        [row["wall_s"] for row in records if row["mode"] == mode]
        for mode in formal
    ]
    boxes = axes[0].boxplot(
        values,
        tick_labels=["Independent", "Strict continuation"],
        patch_artist=True,
    )
    for patch, mode in zip(boxes["boxes"], formal, strict=True):
        patch.set_facecolor(COLORS[mode])
        patch.set_alpha(0.78)
    axes[0].set_ylabel("Wall time per formal evaluation [s]")
    axes[0].set_title("Formal evaluator latency")
    axes[0].grid(axis="y", alpha=0.2)

    grouped: dict[int, list[float]] = defaultdict(list)
    for row in records:
        if row["mode"] == "neighborhood_batch":
            grouped[int(row["batch_size"])].append(row["wall_s"] / int(row["batch_size"]))
    sizes = sorted(grouped)
    median = [float(np.median(grouped[size])) for size in sizes]
    low = [float(np.quantile(grouped[size], 0.10)) for size in sizes]
    high = [float(np.quantile(grouped[size], 0.90)) for size in sizes]
    axes[1].plot(sizes, median, marker="o", color=COLORS["neighborhood_proxy"], lw=2)
    axes[1].fill_between(sizes, low, high, color=COLORS["neighborhood_proxy"], alpha=0.18)
    axes[1].set(xscale="log", yscale="log", xlabel="Candidates in one CUDA batch", ylabel="Amortized wall time [s/candidate]")
    axes[1].set_title("Neighborhood proxy throughput")
    axes[1].grid(alpha=0.2, which="both")

    pairs = paired_strict_proxy(records)
    x = [float(strict["score"]) for strict, _ in pairs]
    y = [float(proxy["score"]) for _, proxy in pairs]
    axes[2].scatter(x, y, s=22, alpha=0.72, color=COLORS["neighborhood_proxy"], edgecolors="none")
    if x:
        lower = min(x + y)
        upper = max(x + y)
        axes[2].plot([lower, upper], [lower, upper], color="#555555", ls="--", lw=1)
    axes[2].set(xlabel="Strict-continuation score", ylabel="Neighborhood proxy score")
    axes[2].set_title("Local proxy fidelity")
    axes[2].grid(alpha=0.2)

    strict_coil = [float(strict["components"]["coil"]) for strict, _ in pairs]
    proxy_coil = [float(proxy["components"]["coil"]) for _, proxy in pairs]
    axes[3].scatter(
        strict_coil,
        proxy_coil,
        s=22,
        alpha=0.72,
        color="#B85C38",
        edgecolors="none",
    )
    if strict_coil:
        lower = min(strict_coil + proxy_coil)
        upper = max(strict_coil + proxy_coil)
        axes[3].plot([lower, upper], [lower, upper], color="#555555", ls="--", lw=1)
    axes[3].set(
        xlabel="Exactly recomputed coil component",
        ylabel="First-order coil component",
        title="Coil-engineering linearization",
    )
    axes[3].grid(alpha=0.2)
    fig.savefig(destination, dpi=220, facecolor="white")
    plt.close(fig)


def plot_stage_timing(records: list[dict], destination: Path) -> dict[str, dict]:
    groups = (
        ("Input and coil geometry", ("field_create_s", "coil_geometry_s")),
        ("Magnetic axis", ("axis_search_s", "axis_trace_s")),
        ("$s$ fit and validation", ("psi_points_s", "psi_fit_s", "psi_validate_s")),
        ("Surface and flux", ("surface_screen_s", "flux_s")),
        (
            "$\\alpha/\\iota$ fit",
            ("volume_points_s", "field_volume_s", "alpha_assemble_s", "alpha_solve_s"),
        ),
        ("QS and score", ("qs_metrics_s", "score_s")),
    )
    modes = ("independent", "strict_continuation")
    grouped_means: dict[str, dict[str, float]] = {}
    total_means: dict[str, float] = {}
    for mode in modes:
        rows = [row for row in records if row["mode"] == mode]
        per_call: dict[str, list[float]] = {label: [] for label, _ in groups}
        per_call["Other and synchronization"] = []
        totals: list[float] = []
        for row in rows:
            timing = row.get("timing", {})
            measured = 0.0
            for label, keys in groups:
                value = sum(float(timing.get(key, 0.0)) for key in keys)
                per_call[label].append(value)
                measured += value
            total = float(timing.get("total_s", row["wall_s"]))
            totals.append(total)
            per_call["Other and synchronization"].append(max(total - measured, 0.0))
        grouped_means[mode] = {
            label: float(np.mean(values)) for label, values in per_call.items()
        }
        total_means[mode] = float(np.mean(totals))

    labels = [label for label, _ in groups] + ["Other and synchronization"]
    fig, ax = plt.subplots(figsize=(9.2, 4.3), constrained_layout=True)
    left = np.zeros(2)
    palette = plt.get_cmap("tab20c")
    for index, label in enumerate(labels):
        values = np.asarray([grouped_means[mode][label] for mode in modes])
        ax.barh([0, 1], values, left=left, label=label, color=palette(index))
        left += values
    ax.set_yticks([0, 1], ["Independent", "Strict continuation"])
    ax.invert_yaxis()
    for row, mode in enumerate(modes):
        ax.text(
            total_means[mode] + 0.03,
            row,
            f"{total_means[mode]:.2f} s mean",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Mean non-overlapping native stage time [s]")
    ax.set_title("Formal evaluator time decomposition")
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    ax.grid(axis="x", alpha=0.2)
    fig.savefig(destination, dpi=220, facecolor="white")
    plt.close(fig)
    return {
        mode: {
            "count": len([row for row in records if row["mode"] == mode]),
            "total_mean_s": total_means[mode],
            "stage_mean_s": grouped_means[mode],
            "stage_fraction": {
                label: value / total_means[mode]
                for label, value in grouped_means[mode].items()
            },
        }
        for mode in modes
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.input_dir)
    summary = {
        "format": "summary1_evaluator_modes_benchmark_aggregate_v1",
        "record_count": len(records),
        "case_count": len({row["case_id"] for row in records}),
        **summarize(records),
        "coil_component_proxy": coil_component_summary(records),
    }
    plot_overview(records, args.output_dir / "evaluator_modes_overview.png")
    summary["formal_stage_timing"] = plot_stage_timing(
        records, args.output_dir / "formal_stage_timing.png"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"record_count": len(records), "case_count": summary["case_count"]}))


if __name__ == "__main__":
    main()
