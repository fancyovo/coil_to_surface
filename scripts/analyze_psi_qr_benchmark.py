#!/usr/bin/env python3
"""Summarize the fixed-matrix single-GPU psi least-squares benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def estimated_flops(method: str, m: int, n: int) -> float | None:
    householder = 2.0 * m * n**2 - (2.0 / 3.0) * n**3
    if (
        method in {"legacy", "generic", "magma", "magma3"}
        or method.startswith("legacy_pad")
        or method.startswith("legacy_bf16x9")
    ):
        return householder
    if method.startswith("tsqr"):
        packed = method.startswith("tsqrp")
        blocks = int(method[5 if packed else 4 :])
        return 2.0 * m * n**2 + ((4.0 * blocks - 2.0) / 3.0) * n**3
    if method.startswith("shifted_") or method.startswith("normal_"):
        return 2.0 * m * n**2 + (1.0 / 3.0) * n**3 + 2.0 * m * n
    if method.startswith("pcgls"):
        iterations = int(method[5:])
        gram = 2.0 * m * n**2 + (1.0 / 3.0) * n**3 + 2.0 * m * n
        return gram + iterations * (4.0 * m * n + 2.0 * n**2)
    if method.startswith("lsqr"):
        iterations = int(method[4:])
        return iterations * 4.0 * m * n
    return None


def family(method: str) -> str:
    if method in {"legacy", "generic"} or method.startswith("legacy_"):
        return "Householder"
    if method.startswith("magma"):
        return "MAGMA"
    if method.startswith("tsqr"):
        return "TSQR"
    if method.startswith("bcgs"):
        return "block GS"
    if method.startswith("ir_"):
        return "iterative refinement"
    if method.startswith("lsqr"):
        return "LSQR"
    if method.startswith("pcgls"):
        return "PCGLS"
    if method.startswith("shifted_") or method.startswith("normal_") or method == "choleskyqr2":
        return "Gram/Cholesky"
    return "other"


def finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def load_records(input_dir: Path) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    errors: list[dict] = []
    for path in sorted(input_dir.glob("*.json")):
        try:
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                raise ValueError("empty result file")
            # The C++ benchmark prints lowercase nan for non-finite diagnostics.
            # Preserve those failed runs instead of silently dropping them.
            record = json.loads(text.replace(":nan", ":NaN"))
            if "method" not in record:
                raise ValueError("result has no method field")
            record["source_file"] = path.name
            records.append(record)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"source_file": path.name, "error": str(exc)})
    return records, errors


def annotate(records: list[dict]) -> dict:
    baseline = next(record for record in records if record["method"] == "legacy")
    base_time = float(baseline["solve_ms_p50"])
    base_physical = float(baseline["physical_residual_relative"])
    base_augmented = float(baseline["augmented_residual_relative"])

    for record in records:
        method = str(record["method"])
        solve_ms = float(record["solve_ms_p50"])
        physical = record.get("physical_residual_relative")
        augmented = record.get("augmented_residual_relative")
        coefficient = record.get("coefficient_relative_error")
        normal = record.get("normal_residual_relative")
        finite = all(finite_number(value) for value in (physical, augmented, coefficient, normal))
        equivalent = bool(
            finite
            and float(coefficient) <= 1.0e-3
            and float(physical) <= 1.001 * base_physical
            and float(augmented) <= 1.001 * base_augmented
            and float(normal) <= 1.0e-4
        )
        exact = bool(
            finite
            and float(coefficient) <= 1.0e-7
            and abs(float(physical) / base_physical - 1.0) <= 1.0e-7
            and abs(float(augmented) / base_augmented - 1.0) <= 1.0e-7
        )
        flops = estimated_flops(method, int(record["rows"]), int(record["cols"]))
        record.update(
            family=family(method),
            speedup_vs_legacy=base_time / solve_ms,
            physical_residual_ratio= float(physical) / base_physical if finite else math.nan,
            reference_equivalent=equivalent,
            bitwise_metric_equivalent=exact,
            significant_integration_candidate=bool(equivalent and base_time / solve_ms >= 1.5),
            estimated_algorithm_flops=flops,
            estimated_algorithm_tflops=(flops / (solve_ms * 1.0e9)) if flops is not None else None,
        )
    return baseline


def write_tables(records: list[dict], errors: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda item: float(item["solve_ms_p50"]))
    fields = [
        "method",
        "family",
        "solve_ms_p50",
        "solve_ms_p95",
        "factor_ms_p50",
        "apply_ms_p50",
        "equivalent_tflops_p50",
        "estimated_algorithm_tflops",
        "speedup_vs_legacy",
        "coefficient_relative_error",
        "physical_residual_relative",
        "physical_residual_ratio",
        "augmented_residual_relative",
        "normal_residual_relative",
        "reference_equivalent",
        "bitwise_metric_equivalent",
        "significant_integration_candidate",
        "source_file",
    ]
    with (output_dir / "benchmark_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps({"records": ordered, "load_errors": errors}, indent=2, allow_nan=True),
        encoding="utf-8",
    )


def selected_records(records: list[dict]) -> list[dict]:
    preferred = [
        "legacy",
        "legacy_pad256",
        "legacy_nondeterministic",
        "generic",
        "magma",
        "magma3",
        "tsqrp2",
        "bcgs2_128",
        "bcgs2tf32_128",
        "ir_sh",
        "lsqr50",
        "shifted_8e-4",
        "pcgls5",
        "pcgls20",
    ]
    by_method = {record["method"]: record for record in records}
    return [by_method[method] for method in preferred if method in by_method]


def plot_runtime(records: list[dict], output_dir: Path) -> None:
    chosen = selected_records(records)
    colors = [
        "#238636" if record["reference_equivalent"] else "#d97706"
        if finite_number(record["physical_residual_relative"]) else "#b42318"
        for record in chosen
    ]
    fig, ax = plt.subplots(figsize=(10.5, 6.3))
    y = list(range(len(chosen)))
    ax.barh(y, [record["solve_ms_p50"] for record in chosen], color=colors)
    ax.set_yticks(y, [record["method"] for record in chosen])
    ax.invert_yaxis()
    ax.axvline(64.5, color="#7c3aed", linestyle="--", linewidth=1.5, label="30 TFLOP/s HH target")
    ax.axvline(181.932807, color="#374151", linestyle=":", linewidth=1.5, label="cuSOLVER baseline")
    ax.set_xscale("log")
    ax.set_xlim(25.0, 4000.0)
    ax.set_xlabel("Single-GPU least-squares time, P50 (ms)")
    ax.set_title("Fixed psi matrix: speed and reference-equivalent accuracy")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "selected_runtime.png", dpi=180)
    plt.close(fig)


def plot_tradeoff(records: list[dict], output_dir: Path) -> None:
    finite = [
        record
        for record in records
        if finite_number(record.get("physical_residual_relative"))
        and finite_number(record.get("coefficient_relative_error"))
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.7))
    palette = {
        "Householder": "#2563eb",
        "MAGMA": "#0891b2",
        "TSQR": "#16a34a",
        "block GS": "#65a30d",
        "iterative refinement": "#7c3aed",
        "LSQR": "#db2777",
        "PCGLS": "#ea580c",
        "Gram/Cholesky": "#dc2626",
        "other": "#6b7280",
    }
    for record in finite:
        color = palette.get(record["family"], "#6b7280")
        coefficient = max(float(record["coefficient_relative_error"]), 1.0e-9)
        axes[0].scatter(record["solve_ms_p50"], coefficient, color=color, s=34)
        axes[1].scatter(record["equivalent_tflops_p50"], record["physical_residual_ratio"], color=color, s=34)
        if record in selected_records(records):
            axes[0].annotate(record["method"], (record["solve_ms_p50"], coefficient), fontsize=7, xytext=(3, 3), textcoords="offset points")
            axes[1].annotate(record["method"], (record["equivalent_tflops_p50"], record["physical_residual_ratio"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axes[0].axvline(64.5, color="#7c3aed", linestyle="--", linewidth=1.2)
    axes[0].axhline(1.0e-3, color="#374151", linestyle=":", linewidth=1.2)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Single-GPU least-squares time, P50 (ms)")
    axes[0].set_ylabel("Coefficient error vs Householder")
    axes[0].set_title("Runtime vs solution change")
    axes[0].grid(alpha=0.22, which="both")
    axes[1].axvline(30.0, color="#7c3aed", linestyle="--", linewidth=1.2)
    axes[1].axhline(1.001, color="#374151", linestyle=":", linewidth=1.2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Householder-equivalent throughput (TFLOP/s)")
    axes[1].set_ylabel("Physical residual / Householder residual")
    axes[1].set_title("High throughput is not sufficient")
    axes[1].grid(alpha=0.22, which="both")
    fig.tight_layout()
    fig.savefig(output_dir / "speed_accuracy_tradeoff.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    records, errors = load_records(args.input_dir)
    if not records:
        raise SystemExit("no benchmark records found")
    annotate(records)
    write_tables(records, errors, args.output_dir)
    plot_runtime(records, args.output_dir)
    plot_tradeoff(records, args.output_dir)
    print(json.dumps({"records": len(records), "load_errors": len(errors)}, indent=2))


if __name__ == "__main__":
    main()
