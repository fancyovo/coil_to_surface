from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STATUS_ORDER = (
    "ok",
    "flux_rejected",
    "drift_rejected",
    "no_surface",
    "no_axis",
    "alpha_failed",
    "internal_error",
)


def finite(values):
    return np.asarray([value for value in values if value is not None and math.isfinite(value)])


def statistics(values) -> dict:
    values = finite(values)
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
    }


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for index in range(len(unique)):
            members = inverse == index
            ranks[members] = np.mean(ranks[members])
    return ranks


def spearman(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(keep) < 3:
        return float("nan")
    return float(np.corrcoef(rank(x[keep]), rank(y[keep]))[0, 1])


def load_rows(input_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(input_dir.glob("worker_*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.input_dir)
    metadata = {
        int(row["ID"]): row
        for row in json.loads(args.metadata.read_text(encoding="utf-8"))
    }
    for row in rows:
        row["metadata"] = metadata.get(int(row["case_id"]), {})
    args.output_dir.mkdir(parents=True, exist_ok=True)

    statuses = [row["native_score"]["status"] for row in rows]
    scores = [row["native_score"]["score"] for row in rows]
    quality = [-math.log10(max(float(row["metadata"]["qs_error"]), 1.0e-300)) for row in rows]
    ok_rows = [row for row in rows if row["native_score"]["status"] == "ok"]
    summary = {
        "count": len(rows),
        "status_counts": dict(Counter(statuses)),
        "finite_score_count": int(np.count_nonzero(np.isfinite(scores))),
        "score": statistics(scores),
        "score_by_status": {
            status: statistics(
                row["native_score"]["score"]
                for row in rows
                if row["native_score"]["status"] == status
            )
            for status in STATUS_ORDER
            if status in statuses
        },
        "components": {
            name: statistics(row["native_score"]["components"][name] for row in rows)
            for name in ("axis", "psi", "surface", "coordinate", "volume_qs", "iota", "coil")
        },
        "by_helicity": {},
        "correlation": {
            "score_vs_minus_log10_metadata_qs_spearman": spearman(scores, quality),
            "ok_score_vs_minus_log10_metadata_qs_spearman": spearman(
                [row["native_score"]["score"] for row in ok_rows],
                [-math.log10(max(float(row["metadata"]["qs_error"]), 1.0e-300)) for row in ok_rows],
            ),
            "ok_volume_component_vs_minus_log10_metadata_qs_spearman": spearman(
                [row["native_score"]["components"]["volume_qs"] for row in ok_rows],
                [-math.log10(max(float(row["metadata"]["qs_error"]), 1.0e-300)) for row in ok_rows],
            ),
        },
    }
    for helicity in (0, 1):
        group = [row for row in rows if int(row["helicity"]) == helicity]
        summary["by_helicity"][str(helicity)] = {
            "count": len(group),
            "status_counts": dict(Counter(row["native_score"]["status"] for row in group)),
            "score": statistics(row["native_score"]["score"] for row in group),
        }

    iota_errors = [
        abs(
            0.5 * (
                row["native_score"]["diagnostics"]["iota_min"]
                + row["native_score"]["diagnostics"]["iota_max"]
            )
            - float(row["metadata"]["mean_iota"])
        )
        for row in ok_rows
    ]
    summary["ok_iota_absolute_error"] = statistics(iota_errors)
    summary["ok_qs_global_error"] = statistics(
        row["native_score"]["diagnostics"]["qs_global_error"] for row in ok_rows
    )
    summary["ok_surface_inverse_aspect_ratio"] = statistics(
        row["native_score"]["diagnostics"]["surface_inverse_aspect_ratio"] for row in ok_rows
    )
    for name in (
        "volume_valid_fraction",
        "volume_weight_effective_fraction",
        "edge_weight_effective_fraction",
        "volume_candidate_count",
        "volume_available_count",
        "volume_point_count",
    ):
        summary[f"ok_{name}"] = statistics(
            row["native_score"]["diagnostics"][name] for row in ok_rows
        )
    summary["flux_attempt_count"] = statistics(
        row["native_score"]["diagnostics"]["flux_attempt_count"] for row in rows
    )
    summary["correlation"]["ok_score_vs_minus_log10_native_qs_spearman"] = spearman(
        [row["native_score"]["score"] for row in ok_rows],
        [-math.log10(max(row["native_score"]["diagnostics"]["qs_global_error"], 1.0e-300)) for row in ok_rows],
    )
    summary["correlation"]["ok_score_vs_surface_size_spearman"] = spearman(
        [row["native_score"]["score"] for row in ok_rows],
        [row["native_score"]["diagnostics"]["surface_inverse_aspect_ratio"] for row in ok_rows],
    )
    summary["correlation"]["ok_volume_component_vs_minus_log10_native_qs_spearman"] = spearman(
        [row["native_score"]["components"]["volume_qs"] for row in ok_rows],
        [-math.log10(max(row["native_score"]["diagnostics"]["qs_global_error"], 1.0e-300)) for row in ok_rows],
    )
    summary["correlation"]["ok_volume_component_vs_volume_weight_ess_spearman"] = spearman(
        [row["native_score"]["components"]["volume_qs"] for row in ok_rows],
        [row["native_score"]["diagnostics"]["volume_weight_effective_fraction"] for row in ok_rows],
    )
    summary["correlation"]["ok_native_qs_vs_volume_weight_ess_spearman"] = spearman(
        [row["native_score"]["diagnostics"]["qs_global_error"] for row in ok_rows],
        [row["native_score"]["diagnostics"]["volume_weight_effective_fraction"] for row in ok_rows],
    )
    qh_ok_rows = [row for row in ok_rows if int(row["helicity"]) == 1]
    summary["correlation"]["qh_ok_score_vs_iota_score_spearman"] = spearman(
        [row["native_score"]["score"] for row in qh_ok_rows],
        [row["native_score"]["diagnostics"]["score_iota"] for row in qh_ok_rows],
    )
    summary["correlation"]["qh_ok_score_vs_abs_iota_spearman"] = spearman(
        [row["native_score"]["score"] for row in qh_ok_rows],
        [
            min(
                abs(row["native_score"]["diagnostics"]["iota_min"]),
                abs(row["native_score"]["diagnostics"]["iota_max"]),
            )
            for row in qh_ok_rows
        ],
    )
    summary["qh_ok_iota_score"] = statistics(
        row["native_score"]["diagnostics"]["score_iota"] for row in qh_ok_rows
    )
    summary["qh_ok_below_unit_iota_count"] = sum(
        min(
            abs(row["native_score"]["diagnostics"]["iota_min"]),
            abs(row["native_score"]["diagnostics"]["iota_max"]),
        ) < 1.0
        for row in qh_ok_rows
    )
    summary["component_p90_p10_spread"] = {
        name: values.get("p90", float("nan")) - values.get("p10", float("nan"))
        for name, values in summary["components"].items()
    }
    one_point_bins = Counter(min(99, max(0, int(math.floor(score)))) for score in scores if math.isfinite(score))
    summary["largest_one_point_score_bin"] = {
        "count": max(one_point_bins.values(), default=0),
        "fraction": max(one_point_bins.values(), default=0) / max(len(scores), 1),
    }

    timing_names = list(rows[0]["native_score"]["timing"]) if rows else []
    summary["timing"] = {
        name: statistics(row["native_score"]["timing"][name] for row in rows)
        for name in timing_names
    }
    summary["wall_s"] = statistics(row["wall_s"] for row in rows)
    summary["under_10s_fraction"] = float(np.mean([row["wall_s"] < 10.0 for row in rows])) if rows else 0.0
    summary["runtime_over_10s"] = [
        {
            "case_id": int(row["case_id"]),
            "helicity": int(row["helicity"]),
            "nfp": int(row["nfp"]),
            "status": row["native_score"]["status"],
            "wall_s": float(row["wall_s"]),
            "axis_search_s": float(row["native_score"]["timing"]["axis_search_s"]),
        }
        for row in sorted(rows, key=lambda item: item["wall_s"], reverse=True)
        if row["wall_s"] >= 10.0
    ]
    job_file = args.input_dir / "job.json"
    if job_file.is_file():
        job = json.loads(job_file.read_text(encoding="utf-8"))
        wall = float(job["wall_finished"]) - float(job["wall_started"])
        summary["batch"] = {
            "wall_s": wall,
            "samples_per_second": len(rows) / wall,
            "amortized_s_per_sample": wall / max(len(rows), 1),
            **job,
        }

    top = sorted(rows, key=lambda row: row["native_score"]["score"], reverse=True)[:20]
    summary["top20_audit"] = {
        "minimum_inverse_aspect_ratio": float(min(
            row["native_score"]["diagnostics"]["surface_inverse_aspect_ratio"] for row in top
        )),
        "maximum_qs_global_error": float(max(
            row["native_score"]["diagnostics"]["qs_global_error"] for row in top
        )),
        "minimum_volume_qs_component": float(min(
            row["native_score"]["components"]["volume_qs"] for row in top
        )),
        "minimum_iota_component": float(min(
            row["native_score"]["components"]["iota"] for row in top
        )),
        "minimum_volume_valid_fraction": float(min(
            row["native_score"]["diagnostics"]["volume_valid_fraction"] for row in top
        )),
        "minimum_volume_weight_effective_fraction": float(min(
            row["native_score"]["diagnostics"]["volume_weight_effective_fraction"] for row in top
        )),
        "case_ids": [int(row["case_id"]) for row in top],
    } if top and all(row["native_score"]["status"] == "ok" for row in top) else {
        "all_top20_ok": False,
        "case_ids": [int(row["case_id"]) for row in top],
    }

    with (args.output_dir / "rows.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "case_id", "helicity", "nfp", "status", "score", "wall_s",
            "metadata_qs_error", "metadata_mean_iota", "surface_level",
            "inverse_aspect_ratio", "iota", "qs_global_error", "qs_edge_error",
            "iota_score", "qs_residual_score", "surface_size_score",
            "volume_qs_size_factor", "volume_qs_iota_factor",
            "flux_attempt_count", "volume_valid_fraction",
            "volume_weight_effective_fraction", "edge_weight_effective_fraction",
            "volume_candidate_count", "volume_available_count", "volume_point_count",
        ))
        writer.writeheader()
        for row in sorted(rows, key=lambda item: int(item["case_id"])):
            native = row["native_score"]
            diagnostics = native["diagnostics"]
            writer.writerow({
                "case_id": row["case_id"],
                "helicity": row["helicity"],
                "nfp": row["nfp"],
                "status": native["status"],
                "score": native["score"],
                "wall_s": row["wall_s"],
                "metadata_qs_error": row["metadata_qs_error"],
                "metadata_mean_iota": row["metadata_mean_iota"],
                "surface_level": diagnostics["surface_level"],
                "inverse_aspect_ratio": diagnostics["surface_inverse_aspect_ratio"],
                "iota": 0.5 * (diagnostics["iota_min"] + diagnostics["iota_max"]),
                "qs_global_error": diagnostics["qs_global_error"],
                "qs_edge_error": diagnostics["qs_edge_error"],
                "iota_score": diagnostics["score_iota"],
                "qs_residual_score": diagnostics["score_qs_residual"],
                "surface_size_score": diagnostics["score_surface_size"],
                "volume_qs_size_factor": diagnostics["score_volume_qs_size_factor"],
                "volume_qs_iota_factor": diagnostics["score_volume_qs_iota_factor"],
                "flux_attempt_count": diagnostics["flux_attempt_count"],
                "volume_valid_fraction": diagnostics["volume_valid_fraction"],
                "volume_weight_effective_fraction": diagnostics["volume_weight_effective_fraction"],
                "edge_weight_effective_fraction": diagnostics["edge_weight_effective_fraction"],
                "volume_candidate_count": diagnostics["volume_candidate_count"],
                "volume_available_count": diagnostics["volume_available_count"],
                "volume_point_count": diagnostics["volume_point_count"],
            })

    colors = plt.get_cmap("tab10")
    figure, axis = plt.subplots(figsize=(9, 5.2))
    bins = np.linspace(0.0, 100.0, 41)
    for index, status in enumerate(STATUS_ORDER):
        values = [row["native_score"]["score"] for row in rows if row["native_score"]["status"] == status]
        if values:
            axis.hist(values, bins=bins, alpha=0.65, label=f"{status} ({len(values)})", color=colors(index))
    axis.set(xlabel="Native score", ylabel="Samples", xlim=(0, 100))
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(args.output_dir / "score_histogram_by_status.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5.5))
    for index, status in enumerate(STATUS_ORDER):
        group = [row for row in rows if row["native_score"]["status"] == status]
        if group:
            axis.scatter(
                [float(row["metadata"]["qs_error"]) for row in group],
                [row["native_score"]["score"] for row in group],
                s=14, alpha=0.62, label=status, color=colors(index), edgecolors="none",
            )
    axis.set_xscale("log")
    axis.set(xlabel="QUASR metadata QS error (validation only)", ylabel="Native score", ylim=(0, 100))
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(args.output_dir / "score_vs_metadata_qs.png", dpi=180)
    plt.close(figure)

    stage_names = [name for name in timing_names if name not in {"total_s", "score_s"}]
    means = [summary["timing"][name].get("mean", 0.0) for name in stage_names]
    p95 = [summary["timing"][name].get("p95", 0.0) for name in stage_names]
    order = np.argsort(means)
    figure, axis = plt.subplots(figsize=(9, 6.2))
    positions = np.arange(len(stage_names))
    axis.barh(positions, np.asarray(p95)[order], color="#d7dadd", label="P95")
    axis.barh(positions, np.asarray(means)[order], color="#146c94", label="Mean")
    axis.set_yticks(positions, [stage_names[index].removesuffix("_s") for index in order])
    axis.set(xlabel="Seconds per sample", ylabel="Native stage")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(args.output_dir / "timing_breakdown.png", dpi=180)
    plt.close(figure)

    if ok_rows:
        figure, axis = plt.subplots(figsize=(8.5, 5.5))
        scatter = axis.scatter(
            [row["native_score"]["diagnostics"]["surface_inverse_aspect_ratio"] for row in ok_rows],
            [row["native_score"]["diagnostics"]["qs_global_error"] for row in ok_rows],
            c=[row["native_score"]["score"] for row in ok_rows], cmap="viridis",
            s=18, alpha=0.7, edgecolors="none", vmin=0, vmax=100,
        )
        axis.set_yscale("log")
        axis.set(xlabel="Selected surface inverse aspect ratio", ylabel="Volume QS error")
        figure.colorbar(scatter, ax=axis, label="Native score")
        figure.tight_layout()
        figure.savefig(args.output_dir / "surface_size_vs_volume_qs.png", dpi=180)
        plt.close(figure)

    decile_rows = defaultdict(list)
    for helicity in (0, 1):
        group = sorted(
            (row for row in rows if int(row["helicity"]) == helicity),
            key=lambda row: float(row["metadata"]["qs_error"]),
            reverse=True,
        )
        for index, row in enumerate(group):
            decile = min(9, 10 * index // max(len(group), 1))
            decile_rows[decile].append(row)
    deciles = np.arange(1, 11)
    decile_scores = [np.median([row["native_score"]["score"] for row in decile_rows[index]]) for index in range(10)]
    decile_ok = [np.mean([row["native_score"]["status"] == "ok" for row in decile_rows[index]]) for index in range(10)]
    summary["metadata_qs_deciles_worst_to_best"] = {
        "median_score": [float(value) for value in decile_scores],
        "ok_fraction": [float(value) for value in decile_ok],
    }
    score_decile_rows = defaultdict(list)
    for index, row in enumerate(sorted(rows, key=lambda item: item["native_score"]["score"])):
        decile = min(9, 10 * index // max(len(rows), 1))
        score_decile_rows[decile].append(row)
    score_decile_ok = [
        [row for row in score_decile_rows[index] if row["native_score"]["status"] == "ok"]
        for index in range(10)
    ]
    score_decile_summary = {
        "median_score": [
            float(np.median([row["native_score"]["score"] for row in score_decile_rows[index]]))
            for index in range(10)
        ],
        "ok_fraction": [
            float(np.mean([row["native_score"]["status"] == "ok" for row in score_decile_rows[index]]))
            for index in range(10)
        ],
        "median_surface_inverse_aspect_ratio": [
            statistics(
                row["native_score"]["diagnostics"]["surface_inverse_aspect_ratio"]
                for row in score_decile_rows[index]
            ).get("median", float("nan"))
            for index in range(10)
        ],
        "median_ok_qs_global_error": [
            statistics(
                row["native_score"]["diagnostics"]["qs_global_error"]
                for row in score_decile_ok[index]
            ).get("median", float("nan"))
            for index in range(10)
        ],
    }
    summary["score_deciles_low_to_high"] = score_decile_summary
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.plot(deciles, decile_scores, marker="o", color="#146c94", label="Median score")
    axis.set(xlabel="Metadata QS decile (1 worst, 10 best)", ylabel="Median native score", xticks=deciles)
    second = axis.twinx()
    second.plot(deciles, decile_ok, marker="s", color="#c84b31", label="Full-chain success")
    second.set(ylabel="Full-chain success fraction", ylim=(0, 1.02))
    handles = axis.lines + second.lines
    axis.legend(handles, [line.get_label() for line in handles], frameon=False, loc="best")
    figure.tight_layout()
    figure.savefig(args.output_dir / "quality_decile_gradient.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(8.5, 7.2), sharex=True)
    axes[0].plot(
        deciles, score_decile_summary["median_surface_inverse_aspect_ratio"],
        marker="o", color="#146c94", label="Median selected $a/R$",
    )
    axes[0].set_ylabel("Selected surface $a/R$")
    success_axis = axes[0].twinx()
    success_axis.plot(
        deciles, score_decile_summary["ok_fraction"], marker="s",
        color="#c84b31", label="Full-chain success",
    )
    success_axis.set(ylabel="Success fraction", ylim=(0, 1.02))
    axes[0].legend(axes[0].lines + success_axis.lines,
                   [line.get_label() for line in axes[0].lines + success_axis.lines],
                   frameon=False, loc="best")
    axes[1].plot(
        deciles, score_decile_summary["median_ok_qs_global_error"],
        marker="o", color="#5b8c5a",
    )
    axes[1].set_yscale("log")
    axes[1].set(
        xlabel="Native score decile (1 lowest, 10 highest)",
        ylabel="Median volume QS error\n(successful cases)",
        xticks=deciles,
    )
    figure.tight_layout()
    figure.savefig(args.output_dir / "score_decile_physics.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
