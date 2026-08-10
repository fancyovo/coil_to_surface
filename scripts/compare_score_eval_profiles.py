from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def baseline_rows(rows: list[dict]) -> dict[tuple[int, int], dict]:
    return {
        (int(row["case_id"]), int(row["repeat"])): row
        for row in rows if row["variant"] == "baseline"
    }


def medians_by_variant(rows: list[dict]) -> dict[str, float]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(float(row["caller_wall_s"]))
    return {name: float(np.median(values)) for name, values in grouped.items()}


def native_stage_medians(rows: dict[tuple[int, int], dict]) -> dict[str, float]:
    values = defaultdict(list)
    for row in rows.values():
        timing = row["result"]["timing"]
        known = {
            "coil geometry": timing["coil_geometry_s"],
            "field creation": timing["field_create_s"],
            "axis (hint)": timing["axis_search_s"] + timing["axis_trace_s"],
            "psi points": timing["psi_points_s"],
            "psi fit": timing["psi_fit_s"],
            "psi validation": timing["psi_validate_s"],
            "surface screen": timing["surface_screen_s"],
        }
        accounted = sum(known.values()) + timing["score_s"]
        known["downstream and other"] = max(0.0, timing["total_s"] - accounted)
        for name, value in known.items():
            values[name].append(float(value))
    return {name: float(np.median(stage_values)) for name, stage_values in values.items()}


def load_nvtx_medians(path: Path) -> dict[str, float]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = next(index for index, line in enumerate(lines) if line.startswith("Time (%)"))
    reader = csv.DictReader(lines[header:])
    return {row["Range"].lstrip(":"): float(row["Med (ns)"]) * 1.0e-9 for row in reader}


def flatten_numeric(prefix: str, value, output: dict[str, float]) -> None:
    if isinstance(value, dict):
        for name, child in value.items():
            flatten_numeric(f"{prefix}.{name}" if prefix else name, child, output)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix] = float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--after-nvtx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    before_all = load_jsonl(args.before)
    after_all = load_jsonl(args.after)
    before = baseline_rows(before_all)
    after = baseline_rows(after_all)
    keys = sorted(set(before) & set(after))
    if len(keys) != len(before) or len(keys) != len(after):
        raise ValueError("before/after baseline keys do not match")

    differences = defaultdict(float)
    for key in keys:
        left, right = {}, {}
        flatten_numeric("result", before[key]["result"], left)
        flatten_numeric("result", after[key]["result"], right)
        for name in set(left) & set(right):
            if name.startswith("result.timing"):
                continue
            if np.isfinite(left[name]) and np.isfinite(right[name]):
                differences[name] = max(differences[name], abs(left[name] - right[name]))

    before_variants = medians_by_variant(before_all)
    after_variants = medians_by_variant(after_all)
    before_stage = native_stage_medians(before)
    after_stage = native_stage_medians(after)
    summary = {
        "matched_baseline_calls": len(keys),
        "score_max_abs_difference": differences["result.score"],
        "component_max_abs_difference": max(
            value for name, value in differences.items() if name.startswith("result.components.")
        ),
        "diagnostic_max_abs_differences": {
            name.removeprefix("result.diagnostics."): value
            for name, value in differences.items()
            if name.startswith("result.diagnostics.") and value > 0.0
        },
        "baseline_wall_before_p50_s": before_variants["baseline"],
        "baseline_wall_after_p50_s": after_variants["baseline"],
        "baseline_speedup": before_variants["baseline"] / after_variants["baseline"],
        "before_stage_p50_s": before_stage,
        "after_stage_p50_s": after_stage,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )

    variants = list(after_variants)
    x = np.arange(len(variants))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - width / 2, [before_variants[name] for name in variants], width, label="before")
    ax.bar(x + width / 2, [after_variants[name] for name in variants], width, label="after")
    ax.set_xticks(x, variants, rotation=30, ha="right")
    ax.set_ylabel("Median strict-hint wall time (s)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "variant_runtime_before_after.png", dpi=180)
    plt.close(fig)

    stages = list(after_stage)
    x = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width / 2, [before_stage[name] for name in stages], width, label="before")
    ax.bar(x + width / 2, [after_stage[name] for name in stages], width, label="after")
    ax.set_xticks(x, stages, rotation=30, ha="right")
    ax.set_ylabel("Median native stage time (s)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "baseline_stage_before_after.png", dpi=180)
    plt.close(fig)

    nvtx = load_nvtx_medians(args.after_nvtx)
    categories = {
        "CPU FP64": sum(
            nvtx.get(name, 0.0) for name in (
                "score.coil_geometry.cpu", "axis.domain.cpu", "axis.hint.topology.cpu",
                "psi.points.cpu", "psi.validate.points.cpu", "psi.validate.reduce.cpu",
                "surface.ray_roots.cpu", "surface.reduce.cpu", "surface.confidence.cpu",
                "score.finalize.cpu",
            )
        ),
        "GPU FP64": nvtx["axis.hint.fp64_verify"],
        "GPU mixed FP32/FP64": sum(
            nvtx[name] for name in ("axis.hint.refine", "axis.samples", "surface.trace.mixed")
        ),
        "GPU FP32-dominant": sum(
            nvtx[name] for name in ("psi.fullgpu.total", "psi.validate.field.fp32", "score.downstream")
        ),
    }
    categories["API / allocation / other"] = max(0.0, nvtx["score.total"] - sum(categories.values()))
    fig, ax = plt.subplots(figsize=(11, 2.8))
    left = 0.0
    colors = ["#577590", "#f94144", "#f8961e", "#43aa8b", "#adb5bd"]
    for (name, value), color in zip(categories.items(), colors):
        ax.barh([0], [value], left=left, label=f"{name}: {value * 1000:.1f} ms", color=color)
        left += value
    ax.set_yticks([])
    ax.set_xlabel("Representative steady strict-hint score time (s)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "precision_path_breakdown.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
