from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {0: "#167c80", 1: "#d1495b"}
LABELS = {0: "QA", 1: "QH"}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rankdata(values):
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def plot_sample_validation(summary, output: Path) -> None:
    rows = [row for row in summary["rows"] if row["status"] == "ok"]
    metadata = np.asarray([row["metadata_qs_error"] for row in rows])
    metric = np.asarray([row["target_f_C_over_B3_rms"] for row in rows])
    helicity = np.asarray([row["helicity"] for row in rows])
    spearman = np.corrcoef(rankdata(np.log(metadata)), rankdata(np.log(metric)))[0, 1]

    fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
    for value in (0, 1):
        selected = helicity == value
        ax.scatter(
            metadata[selected],
            metric[selected],
            s=48,
            color=COLORS[value],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.9,
            label=f"{LABELS[value]} ({selected.sum()})",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("QUASR metadata QS error")
    ax.set_ylabel(r"Volume differential $f_C/B^3$ RMS")
    ax.set_title(f"Independent ranking check, Spearman r = {spearman:.3f}")
    ax.grid(True, which="both", color="#d8d8d3", linewidth=0.6, alpha=0.7)
    ax.legend(frameon=False)
    fig.savefig(output / "quasr_cross_validation.png", dpi=180)
    plt.close(fig)


def load_case_timings(batch_path: Path, rows):
    stages = {
        name: []
        for name in (
            "axis",
            "psi",
            "screen",
            "volume",
            "flux_calibration",
            "volume_points",
            "B_grad_B",
            "alpha_fit",
            "alpha_qr",
            "qs_metrics",
            "total",
        )
    }
    for row in rows:
        if row["status"] != "ok":
            continue
        case_path = batch_path.parent / f"id_{int(row['id']):07d}" / "summary.json"
        if not case_path.exists():
            continue
        case = load_json(case_path)
        timing = case["timing"]
        downstream = case["volume_qs"]
        inner = downstream["timing"]
        stages["axis"].append(timing["axis_s"])
        stages["psi"].append(timing["psi_fit_s"])
        stages["screen"].append(timing["surface_screen_s"])
        stages["volume"].append(timing["volume_qs_s"])
        stages["flux_calibration"].append(
            sum(attempt["time_s"] for attempt in downstream["flux_attempts"])
        )
        stages["volume_points"].append(inner["volume_points_s"])
        stages["B_grad_B"].append(inner["B_grad_B_s"])
        stages["alpha_fit"].append(inner["alpha_total_s"])
        stages["alpha_qr"].append(inner["alpha_qr_s"])
        stages["qs_metrics"].append(inner["qs_metrics_s"])
        stages["total"].append(timing["total_s"])
    return stages


def plot_timing(summary, batch_path: Path, output: Path) -> None:
    stages = load_case_timings(batch_path, summary["rows"])
    labels = ["Axis", r"Fit $s$", "Screen", "Volume QS", "Total"]
    values = [stages[key] for key in ("axis", "psi", "screen", "volume", "total")]
    if not all(values):
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    box = ax.boxplot(values, tick_labels=labels, patch_artist=True, showfliers=True)
    for index, patch in enumerate(box["boxes"]):
        patch.set_facecolor("#3a7d8c" if index < 4 else "#d1495b")
        patch.set_alpha(0.82)
    for median in box["medians"]:
        median.set_color("white")
        median.set_linewidth(1.8)
    ax.axhline(10.0, color="#202124", linestyle="--", linewidth=1.1, label="10 s target")
    ax.set_ylabel("Wall time per successful sample (s)")
    ax.set_title("End-to-end timing on one idle GPU")
    ax.grid(True, axis="y", color="#d8d8d3", linewidth=0.6)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.96))
    fig.savefig(output / "end_to_end_timing.png", dpi=180)
    plt.close(fig)


def distribution(values):
    values = np.asarray(values, dtype=float)
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def write_statistics(summary, batch_path: Path, output: Path) -> None:
    rows = summary["rows"]
    successful = [row for row in rows if row["status"] == "ok"]
    metadata = np.asarray([row["metadata_qs_error"] for row in successful])
    metric = np.asarray([row["target_f_C_over_B3_rms"] for row in successful])
    stages = load_case_timings(batch_path, rows)
    by_helicity = {}
    for helicity in (0, 1):
        selected = [row for row in rows if row.get("helicity") == helicity]
        by_helicity[LABELS[helicity]] = {
            "requested": len(selected),
            "success": sum(row["status"] == "ok" for row in selected),
        }
    case_quality = {"alpha_relative_l2": [], "normal_B_relative_l2": []}
    for row in successful:
        case_path = batch_path.parent / f"id_{int(row['id']):07d}" / "summary.json"
        if not case_path.exists():
            continue
        diagnostics = load_json(case_path)["volume_qs"]["alpha"]["diagnostics"]
        case_quality["alpha_relative_l2"].append(diagnostics["relative_l2"])
        case_quality["normal_B_relative_l2"].append(diagnostics["normal_B_relative_l2"])
    statistics = {
        "success_count": len(successful),
        "failure_count": len(rows) - len(successful),
        "by_helicity": by_helicity,
        "failure_reasons": dict(Counter(row.get("reason") or row["status"] for row in rows if row["status"] != "ok")),
        "correlation": {
            "spearman_log": float(np.corrcoef(rankdata(np.log(metadata)), rankdata(np.log(metric)))[0, 1]),
            "pearson_log": float(np.corrcoef(np.log(metadata), np.log(metric))[0, 1]),
        },
        "target_metric": distribution(metric),
        "selected_s_edge": distribution([row["s_edge"] for row in successful]),
        "timing_s": {key: distribution(value) for key, value in stages.items() if value},
        "quality": {key: distribution(value) for key, value in case_quality.items() if value},
    }
    (output / "validation_statistics.json").write_text(
        json.dumps(statistics, indent=2), encoding="utf-8"
    )


def plot_numerical_audits(audit_dir: Path, output: Path) -> None:
    fp64 = load_json(audit_dir / "fp64_audit.json")
    points = load_json(audit_dir / "point_convergence_audit.json")
    trace = load_json(audit_dir / "long_trace_iota_audit.json")
    desc = load_json(audit_dir / "desc_fC_formula_audit.json")

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9), constrained_layout=True)
    ids = [str(row["id"]) for row in fp64]
    fp64_difference = 100.0 * np.abs([row["metric_relative"] for row in fp64])
    point_difference = 100.0 * np.abs([row["relative"] for row in points])
    x = np.arange(len(ids))
    axes[0].bar(x - 0.18, fp64_difference, width=0.36, color="#167c80", label="FP32 vs FP64")
    axes[0].bar(x + 0.18, point_difference, width=0.36, color="#e09f3e", label="30k vs 100k")
    axes[0].set_xticks(x, ids, rotation=45, ha="right")
    axes[0].set_ylabel("Absolute metric difference (%)")
    axes[0].set_title("Precision and point convergence")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(True, axis="y", color="#d8d8d3", linewidth=0.6)

    trace_ids = [str(row["id"]) for row in trace]
    trace_difference = 100.0 * np.abs([row["relative_difference"] for row in trace])
    trace_drift = 100.0 * np.asarray([row["s_drift_relative_p95"] for row in trace])
    axes[1].bar(trace_ids, trace_difference, color="#d1495b", label=r"$\iota$ difference")
    axes[1].plot(trace_ids, trace_drift, color="#202124", marker="o", label=r"32-period $s$ drift")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Absolute relative difference (%)")
    axes[1].set_title("Independent long field-line trace")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(True, axis="y", which="both", color="#d8d8d3", linewidth=0.6)

    axes[2].bar(["Relative L2", "Max abs"], [desc["relative_l2"], desc["max_abs"]], color=["#167c80", "#e09f3e"])
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Formula discrepancy")
    axes[2].set_title(f"DESC $f_C$ audit ({desc['points']} points)")
    axes[2].grid(True, axis="y", which="both", color="#d8d8d3", linewidth=0.6)

    fig.savefig(output / "numerical_cross_checks.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-summary", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = load_json(args.batch_summary)
    write_statistics(summary, args.batch_summary, args.output_dir)
    plot_sample_validation(summary, args.output_dir)
    plot_timing(summary, args.batch_summary, args.output_dir)
    plot_numerical_audits(args.audit_dir, args.output_dir)


if __name__ == "__main__":
    main()
