from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_rows(path: Path) -> list[dict]:
    rows = []
    for worker in sorted(path.glob("worker_*.jsonl")):
        with worker.open(encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    return rows


def finite(values) -> np.ndarray:
    return np.asarray([value for value in values if math.isfinite(float(value))], dtype=float)


def statistics(values) -> dict:
    values = finite(values)
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    for index in np.flatnonzero(counts > 1):
        members = inverse == index
        ranks[members] = np.mean(ranks[members])
    return ranks


def spearman(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(keep) < 3:
        return float("nan")
    if np.ptp(x[keep]) == 0.0 or np.ptp(y[keep]) == 0.0:
        return float("nan")
    return float(np.corrcoef(rank(x[keep]), rank(y[keep]))[0, 1])


def diagnostics(row: dict) -> dict:
    return row["native_score"]["diagnostics"]


def component(row: dict) -> float:
    return float(row["native_score"]["components"]["volume_qs"])


def q_down(value: float, scale: float, power: float = 0.9) -> float:
    return 1.0 / (1.0 + (max(float(value), 0.0) / scale) ** power)


def q_up(value: float, scale: float = 0.04, power: float = 2.0) -> float:
    return 1.0 / (1.0 + (scale / max(float(value), 1.0e-300)) ** power)


def score_factors(row: dict) -> tuple[float, float]:
    diag = diagnostics(row)
    helicity_norm = math.hypot(1.0, float(row["nfp"])) if int(row["helicity"]) == 1 else 1.0
    global_score = q_down(diag["qs_global_error"], 0.05 * helicity_norm)
    edge_score = q_down(diag["qs_edge_error"], 0.07 * helicity_norm)
    residual_score = 0.8 * global_score + 0.2 * edge_score
    size_factor = 0.35 + 0.65 * q_up(diag["surface_inverse_aspect_ratio"])
    return residual_score, size_factor


def group_audit(rows: list[dict]) -> dict:
    raw_qs = np.asarray([diagnostics(row)["qs_global_error"] for row in rows])
    volume_component = np.asarray([component(row) for row in rows])
    size = np.asarray([diagnostics(row)["surface_inverse_aspect_ratio"] for row in rows])
    valid = np.asarray([diagnostics(row)["volume_valid_fraction"] for row in rows])
    ess = np.asarray([diagnostics(row)["volume_weight_effective_fraction"] for row in rows])
    points = np.asarray([diagnostics(row)["volume_point_count"] for row in rows])
    residual_score = np.asarray([score_factors(row)[0] for row in rows])
    size_factor = np.asarray([score_factors(row)[1] for row in rows])
    effective_points = ess * points
    metadata_qs = np.asarray([row["metadata_qs_error"] for row in rows], dtype=float)
    quality = -np.log10(np.maximum(raw_qs, 1.0e-300))
    metadata_quality = -np.log10(np.maximum(metadata_qs, 1.0e-300))
    return {
        "count": len(rows),
        "raw_qs_global_error": statistics(raw_qs),
        "metadata_qs_error": statistics(metadata_qs),
        "volume_qs_component": statistics(volume_component),
        "surface_inverse_aspect_ratio": statistics(size),
        "volume_valid_fraction": statistics(valid),
        "volume_weight_effective_fraction": statistics(ess),
        "volume_weight_effective_point_count": statistics(effective_points),
        "volume_point_count": statistics(points),
        "residual_soft_score": statistics(residual_score),
        "surface_size_factor": statistics(size_factor),
        "spearman": {
            "component_vs_minus_log10_raw_qs": spearman(volume_component, quality),
            "raw_qs_vs_metadata_qs": spearman(raw_qs, metadata_qs),
            "component_vs_minus_log10_metadata_qs": spearman(
                volume_component, metadata_quality
            ),
            "component_vs_residual_soft_score": spearman(volume_component, residual_score),
            "component_vs_surface_size_factor": spearman(volume_component, size_factor),
            "component_vs_surface_size": spearman(volume_component, size),
            "component_vs_weight_ess": spearman(volume_component, ess),
            "raw_qs_vs_weight_ess": spearman(raw_qs, ess),
            "residual_soft_score_vs_weight_ess": spearman(residual_score, ess),
            "component_vs_valid_fraction": spearman(volume_component, valid),
            "component_vs_point_count": spearman(volume_component, points),
        },
    }


def compare_runs(previous: list[dict], current: list[dict]) -> dict:
    previous_by_id = {
        int(row["case_id"]): row
        for row in previous
        if row["native_score"]["status"] == "ok"
    }
    current_by_id = {
        int(row["case_id"]): row
        for row in current
        if row["native_score"]["status"] == "ok"
    }
    common = sorted(previous_by_id.keys() & current_by_id.keys())
    if not common:
        return {"common_ok_count": 0}
    old_qs = np.asarray([diagnostics(previous_by_id[key])["qs_global_error"] for key in common])
    new_qs = np.asarray([diagnostics(current_by_id[key])["qs_global_error"] for key in common])
    relative = new_qs / old_qs - 1.0
    return {
        "common_ok_count": len(common),
        "raw_qs_old_vs_corrected_spearman": spearman(old_qs, new_qs),
        "raw_qs_relative_change": statistics(relative),
    }


def top_audit(rows: list[dict], key, count: int = 20) -> dict:
    top = sorted(rows, key=key, reverse=True)[:count]
    return {
        "case_ids": [int(row["case_id"]) for row in top],
        "minimum_valid_fraction": min(diagnostics(row)["volume_valid_fraction"] for row in top),
        "minimum_weight_effective_fraction": min(
            diagnostics(row)["volume_weight_effective_fraction"] for row in top
        ),
        "maximum_raw_qs_global_error": max(
            diagnostics(row)["qs_global_error"] for row in top
        ),
    }


def convergence_audit(path: Path) -> tuple[dict, list[dict]]:
    by_case: dict[int, dict[int, dict]] = {}
    for worker in sorted(path.glob("points_*.jsonl")):
        point_count = int(worker.stem.rsplit("_", 1)[1])
        with worker.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    by_case.setdefault(int(row["case_id"]), {})[point_count] = row

    details = []
    for case_id, variants in sorted(by_case.items()):
        reference_count = max(variants)
        reference_qs = diagnostics(variants[reference_count])["qs_global_error"]
        reference_component = component(variants[reference_count])
        for point_count, row in sorted(variants.items()):
            native = row["native_score"]
            qs = diagnostics(row)["qs_global_error"]
            details.append({
                "case_id": case_id,
                "helicity": int(row["helicity"]),
                "point_count": point_count,
                "reference_point_count": reference_count,
                "status": native["status"],
                "qs_global_error": qs,
                "qs_relative_to_reference": qs / reference_qs - 1.0,
                "volume_qs_component": component(row),
                "component_delta_to_reference": component(row) - reference_component,
                "weight_effective_fraction": diagnostics(row)[
                    "volume_weight_effective_fraction"
                ],
            })

    non_reference = [row for row in details if row["point_count"] != row["reference_point_count"]]
    return {
        "case_count": len(by_case),
        "evaluation_count": len(details),
        "all_status_ok": all(row["status"] == "ok" for row in details),
        "maximum_absolute_qs_relative_change": max(
            abs(row["qs_relative_to_reference"]) for row in non_reference
        ),
        "maximum_absolute_component_change": max(
            abs(row["component_delta_to_reference"]) for row in non_reference
        ),
        "by_point_count": {
            str(point_count): {
                "maximum_absolute_qs_relative_change": max(
                    abs(row["qs_relative_to_reference"])
                    for row in details if row["point_count"] == point_count
                ),
                "maximum_absolute_component_change": max(
                    abs(row["component_delta_to_reference"])
                    for row in details if row["point_count"] == point_count
                ),
            }
            for point_count in sorted({row["point_count"] for row in details})
        },
        "cases": details,
    }, details


def plot_convergence(output: Path, details: list[dict]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for case_id in sorted({row["case_id"] for row in details}):
        group = sorted(
            (row for row in details if row["case_id"] == case_id),
            key=lambda row: row["point_count"],
        )
        label = f"{case_id} ({'QA' if group[0]['helicity'] == 0 else 'QH'})"
        axes[0].plot(
            [row["point_count"] for row in group],
            [100.0 * row["qs_relative_to_reference"] for row in group],
            marker="o", label=label,
        )
        axes[1].plot(
            [row["point_count"] for row in group],
            [row["component_delta_to_reference"] for row in group],
            marker="o", label=label,
        )
    axes[0].set(xlabel="Volume point count", ylabel="Raw QS change vs 200k [%]")
    axes[1].set(xlabel="Volume point count", ylabel="QS component change vs 200k")
    for axis in axes:
        axis.axhline(0.0, color="#555555", linewidth=1.0)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("current_dir", type=Path)
    parser.add_argument("--previous-dir", type=Path)
    parser.add_argument("--convergence-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    current = load_rows(args.current_dir)
    ok = [row for row in current if row["native_score"]["status"] == "ok"]
    summary = {
        "count": len(current),
        "status_counts": dict(Counter(row["native_score"]["status"] for row in current)),
        "successful": group_audit(ok),
        "by_helicity": {
            str(helicity): group_audit([row for row in ok if int(row["helicity"]) == helicity])
            for helicity in (0, 1)
        },
        "top20_by_total_score": top_audit(
            ok, lambda row: row["native_score"]["score"]
        ),
        "top20_by_volume_qs_component": top_audit(ok, component),
    }
    if args.previous_dir:
        summary["previous_comparison"] = compare_runs(load_rows(args.previous_dir), current)
    convergence_details = None
    if args.convergence_dir:
        summary["point_count_convergence"], convergence_details = convergence_audit(
            args.convergence_dir
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "qs_sampling_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    colors = {0: "#146c94", 1: "#d16f32"}
    labels = {0: "QA", 1: "QH"}
    for helicity in (0, 1):
        group = [row for row in ok if int(row["helicity"]) == helicity]
        axes[0].scatter(
            [diagnostics(row)["qs_global_error"] for row in group],
            [component(row) for row in group],
            s=15, alpha=0.55, color=colors[helicity], label=labels[helicity], edgecolors="none",
        )
        axes[1].scatter(
            [diagnostics(row)["volume_weight_effective_fraction"] for row in group],
            [component(row) for row in group],
            s=15, alpha=0.55, color=colors[helicity], edgecolors="none",
        )
    axes[0].set_xscale("log")
    axes[0].set(xlabel="Raw differential QS error", ylabel="Volume QS component")
    axes[0].legend(frameon=False)
    axes[1].set(xlabel="Physical-weight effective fraction", ylabel="Volume QS component")

    scatter = axes[2].scatter(
        [diagnostics(row)["surface_inverse_aspect_ratio"] for row in ok],
        [component(row) for row in ok],
        c=[-math.log10(max(diagnostics(row)["qs_global_error"], 1.0e-300)) for row in ok],
        cmap="viridis", s=16, alpha=0.62, edgecolors="none",
    )
    axes[2].set(xlabel="Selected surface inverse aspect ratio", ylabel="Volume QS component")
    figure.colorbar(scatter, ax=axes[2], label=r"$-\log_{10}$ raw QS error")
    for axis in axes:
        axis.grid(alpha=0.18)
    figure.tight_layout()
    figure.savefig(args.output_dir / "qs_sampling_audit.png", dpi=180)
    plt.close(figure)
    if convergence_details:
        plot_convergence(args.output_dir / "qs_sampling_convergence.png", convergence_details)


if __name__ == "__main__":
    main()
