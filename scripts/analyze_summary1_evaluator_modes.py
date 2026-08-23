from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

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


def plot_overview(records: list[dict], destination: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), constrained_layout=True)
    formal = ["independent", "strict_continuation"]
    values = [
        [row["wall_s"] for row in records if row["mode"] == mode]
        for mode in formal
    ]
    boxes = axes[0].boxplot(values, labels=["Independent", "Strict continuation"], patch_artist=True)
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

    first = first_candidate_rows(records)
    x: list[float] = []
    y: list[float] = []
    for case_id, candidate, mode in first:
        if mode != "strict_continuation":
            continue
        strict = first[(case_id, candidate, mode)]
        proxy = first.get((case_id, candidate, "neighborhood_proxy"))
        if proxy and strict["status"] == proxy["status"] == "ok":
            x.append(float(strict["score"]))
            y.append(float(proxy["score"]))
    axes[2].scatter(x, y, s=22, alpha=0.72, color=COLORS["neighborhood_proxy"], edgecolors="none")
    if x:
        lower = min(x + y)
        upper = max(x + y)
        axes[2].plot([lower, upper], [lower, upper], color="#555555", ls="--", lw=1)
    axes[2].set(xlabel="Strict-continuation score", ylabel="Neighborhood proxy score")
    axes[2].set_title("Local proxy fidelity")
    axes[2].grid(alpha=0.2)
    fig.savefig(destination, dpi=220, facecolor="white")
    plt.close(fig)


def plot_stage_timing(records: list[dict], destination: Path) -> None:
    ignored = {"total_s", "score_s"}
    medians: dict[str, dict[str, float]] = {}
    for mode in ("independent", "strict_continuation"):
        rows = [row for row in records if row["mode"] == mode]
        keys = sorted({key for row in rows for key in row.get("timing", {}) if key not in ignored})
        medians[mode] = {
            key: float(np.median([float(row["timing"].get(key, 0.0)) for row in rows]))
            for key in keys
        }
    totals = defaultdict(float)
    for mode in medians:
        for key, value in medians[mode].items():
            totals[key] += value
    selected = [key for key, _ in sorted(totals.items(), key=lambda item: item[1], reverse=True)[:8]]
    fig, ax = plt.subplots(figsize=(9.2, 4.3), constrained_layout=True)
    left = np.zeros(2)
    palette = plt.get_cmap("tab20c")
    for index, key in enumerate(selected):
        values = np.asarray([medians[mode].get(key, 0.0) for mode in ("independent", "strict_continuation")])
        ax.barh([0, 1], values, left=left, label=key.removesuffix("_s"), color=palette(index))
        left += values
    ax.set_yticks([0, 1], ["Independent", "Strict continuation"])
    ax.invert_yaxis()
    ax.set_xlabel("Median native stage time [s]")
    ax.set_title("Where the formal evaluator spends time")
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    ax.grid(axis="x", alpha=0.2)
    fig.savefig(destination, dpi=220, facecolor="white")
    plt.close(fig)


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
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    plot_overview(records, args.output_dir / "evaluator_modes_overview.png")
    plot_stage_timing(records, args.output_dir / "formal_stage_timing.png")
    print(json.dumps({"record_count": len(records), "case_count": summary["case_count"]}))


if __name__ == "__main__":
    main()
