from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def q_down(value: float, scale: float, power: float = 0.9) -> float:
    return 1.0 / (1.0 + (max(float(value), 0.0) / scale) ** power)


def q_up(value: float, scale: float = 0.04, power: float = 2.0) -> float:
    return 1.0 / (1.0 + (scale / max(float(value), 1.0e-300)) ** power)


def helicity_norm(helicity: int, nfp: int) -> float:
    return math.hypot(1.0, float(nfp)) if int(helicity) == 1 else 1.0


def volume_qs_component(row: dict[str, Any], *, normalize_helicity: bool) -> float:
    norm = helicity_norm(int(row["helicity"]), int(row["nfp"])) if normalize_helicity else 1.0
    global_score = q_down(float(row["qs_global_error"]), 0.05 * norm)
    edge_score = q_down(float(row["qs_edge_error"]), 0.07 * norm)
    residual_score = 0.8 * global_score + 0.2 * edge_score
    size_score = q_up(float(row["inverse_aspect_ratio"]))
    return 100.0 * residual_score * (0.35 + 0.65 * size_score)


def statistics(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
        "fraction_ge_50": float(np.mean(array >= 50.0)),
        "fraction_ge_70": float(np.mean(array >= 70.0)),
    }


def load_quasr_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [row for row in rows if row["status"] == "ok"]


def load_candidates(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def diagnostic_row(candidate: dict[str, Any], helicity: int) -> dict[str, Any]:
    diagnostics = candidate["native_score"]["diagnostics"]
    return {
        "helicity": helicity,
        "nfp": 3,
        "qs_global_error": diagnostics["qs_global_error"],
        "qs_edge_error": diagnostics["qs_edge_error"],
        "inverse_aspect_ratio": diagnostics["surface_inverse_aspect_ratio"],
    }


def candidate_audit(path: Path, helicity: int) -> dict[str, Any]:
    candidates = [candidate for candidate in load_candidates(path) if candidate["status"] == "ok"]
    rows = [diagnostic_row(candidate, helicity) for candidate in candidates]
    old_components = [float(candidate["native_score"]["components"]["volume_qs"]) for candidate in candidates]
    corrected_components = [
        volume_qs_component(row, normalize_helicity=True) for row in rows
    ]
    rescored = []
    for candidate, old_component, corrected_component in zip(
        candidates, old_components, corrected_components, strict=True
    ):
        rescored.append(
            {
                "generation": int(candidate["generation"]),
                "candidate": int(candidate["candidate"]),
                "old_total": float(candidate["score"]),
                "corrected_total": float(candidate["score"] + 0.20 * (corrected_component - old_component)),
                "old_volume_qs": old_component,
                "corrected_volume_qs": corrected_component,
                "qs_global_error": float(candidate["native_score"]["diagnostics"]["qs_global_error"]),
            }
        )
    return {
        "successful_candidates": len(candidates),
        "old_volume_qs": statistics(old_components),
        "corrected_volume_qs": statistics(corrected_components),
        "old_best": max(rescored, key=lambda row: row["old_total"]),
        "corrected_best_within_saved_candidates": max(
            rescored, key=lambda row: row["corrected_total"]
        ),
        "old_component_values": old_components,
        "corrected_component_values": corrected_components,
    }


def plot_audit(
    output: Path,
    quasr_groups: dict[str, dict[str, list[float]]],
    cem_qh: dict[str, Any],
    quasr_qh_median: float,
    cem_qh_best: float,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    figure.patch.set_facecolor("#f6f5f1")
    for axis in axes:
        axis.set_facecolor("#fbfaf7")
        axis.spines[["top", "right"]].set_visible(False)

    box_values = [
        quasr_groups["QA"]["old"],
        quasr_groups["QA"]["corrected"],
        quasr_groups["QH"]["old"],
        quasr_groups["QH"]["corrected"],
    ]
    boxes = axes[0].boxplot(box_values, whis=(5, 95), showfliers=False, patch_artist=True)
    for patch, color in zip(boxes["boxes"], ["#397c8c", "#82aeb8", "#b14c3d", "#df8f78"], strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)
    axes[0].set_xticklabels(["QA old", "QA norm", "QH old", "QH norm"], rotation=18)
    axes[0].set_ylabel("volume_qs component")
    axes[0].set_title("QUASR successful samples")
    axes[0].set_ylim(0, 60)

    error = np.logspace(-3, 1.5, 400)
    axes[1].semilogx(error, [q_down(value, 0.05) for value in error], color="#397c8c", label="old / QA")
    qh_scale = 0.05 * math.sqrt(10.0)
    axes[1].semilogx(error, [q_down(value, qh_scale) for value in error], color="#b14c3d", label="QH, NFP=3 norm")
    axes[1].axvline(
        quasr_qh_median, color="#777777", linestyle=":", linewidth=1.2,
        label="QUASR QH median",
    )
    axes[1].axvline(
        cem_qh_best, color="#222222", linestyle="--", linewidth=1.2,
        label="best saved CEM QH",
    )
    axes[1].set_xlabel(r"raw differential $\epsilon_{C,V}$")
    axes[1].set_ylabel("residual soft score")
    axes[1].set_title("Soft-scale response")
    axes[1].legend(frameon=False, fontsize=8)

    boxes = axes[2].boxplot(
        [cem_qh["old_component_values"], cem_qh["corrected_component_values"]],
        whis=(5, 95),
        showfliers=False,
        patch_artist=True,
    )
    for patch, color in zip(boxes["boxes"], ["#b14c3d", "#df8f78"], strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)
    axes[2].set_xticklabels(["old", "helicity norm"])
    axes[2].set_ylabel("volume_qs component")
    axes[2].set_title("CEM QH successful candidates")
    axes[2].set_ylim(0, 16)

    figure.tight_layout()
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit native QS score scaling without rerunning physics.")
    parser.add_argument("--quasr-rows", type=Path, required=True)
    parser.add_argument("--qh-candidates", type=Path, required=True)
    parser.add_argument("--qa-candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = load_quasr_rows(args.quasr_rows)
    groups = {
        "QA": [row for row in rows if int(row["helicity"]) == 0],
        "QH": [row for row in rows if int(row["helicity"]) == 1],
    }
    component_values = {
        name: {
            "old": [volume_qs_component(row, normalize_helicity=False) for row in group],
            "corrected": [volume_qs_component(row, normalize_helicity=True) for row in group],
        }
        for name, group in groups.items()
    }
    native = np.asarray([float(row["qs_global_error"]) for row in rows])
    metadata = np.asarray([float(row["metadata_qs_error"]) for row in rows])
    log10_ratio = np.log10(native) - np.log10(metadata)
    cem_qh = candidate_audit(args.qh_candidates, helicity=1)
    cem_qa = candidate_audit(args.qa_candidates, helicity=0)
    summary = {
        "formula": {
            "helicity_norm": "sqrt(M^2 + N^2)",
            "raw_qs_diagnostics_changed": False,
            "global_scale": "0.05 * helicity_norm",
            "edge_scale": "0.07 * helicity_norm",
        },
        "quasr": {
            "successful_samples": len(rows),
            "native_to_metadata_log10_ratio": statistics(log10_ratio.tolist()),
            "raw_qs_global_error": {
                name: statistics([float(row["qs_global_error"]) for row in group])
                for name, group in groups.items()
            },
            "volume_qs_component": {
                name: {
                    key: statistics(values) for key, values in variants.items()
                }
                for name, variants in component_values.items()
            },
        },
        "cem": {"QH": cem_qh, "QA": cem_qa},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_audit(
        args.output_dir / "qs_score_scale_audit.png",
        component_values,
        cem_qh,
        float(np.median([float(row["qs_global_error"]) for row in groups["QH"]])),
        float(cem_qh["corrected_best_within_saved_candidates"]["qs_global_error"]),
    )
    for target in summary["cem"].values():
        target.pop("old_component_values")
        target.pop("corrected_component_values")

    (args.output_dir / "qs_score_scale_audit.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
