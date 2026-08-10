from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TOP_LEVEL_STAGES = (
    ("field_create_s", "Field object"),
    ("coil_geometry_s", "Coil geometry"),
    ("axis_search_s", "Axis continuation"),
    ("axis_trace_s", "Axis curve sampling"),
    ("psi_points_s", "Psi point generation"),
    ("psi_fit_s", "Psi least squares"),
    ("psi_validate_s", "Psi validation"),
    ("surface_screen_s", "Continuous surface screen"),
    ("flux_s", "Flux calibration"),
    ("volume_points_s", "Volume point generation"),
    ("field_volume_s", "Volume magnetic field"),
    ("alpha_assemble_s", "Alpha/iota assembly"),
    ("alpha_solve_s", "Alpha/iota least squares"),
    ("qs_metrics_s", "Volume QS metrics"),
    ("score_s", "Final score formula"),
)

CATEGORIES = (
    ("Magnetic axis", ("axis_search_s", "axis_trace_s")),
    ("Psi fit and validation", ("psi_points_s", "psi_fit_s", "psi_validate_s")),
    ("Continuous surface screen", ("surface_screen_s",)),
    (
        "Alpha/iota fit",
        ("volume_points_s", "field_volume_s", "alpha_assemble_s", "alpha_solve_s"),
    ),
    ("Flux calibration", ("flux_s",)),
    ("Coil and field setup", ("field_create_s", "coil_geometry_s")),
    ("QS metrics and final score", ("qs_metrics_s", "score_s")),
)

AXIS_STAGES = (
    ("axis_domain_s", "Axis domain setup"),
    ("axis_candidate_refine_s", "Mixed-precision Newton refinement"),
    ("axis_fp64_verify_s", "Five-line FP64 verification"),
    ("axis_topology_s", "Topology reduction"),
    ("axis_trace_s", "Axis curve sampling"),
)

SURFACE_STAGES = (
    ("surface_ray_roots_s", "Ray/root setup"),
    ("surface_mixed_trace_s", "One-period mixed-precision trace"),
    ("surface_mixed_reduce_s", "Drift reduction"),
    ("surface_fp64_trace_s", "Optional FP64 trace"),
    ("surface_fp64_reduce_s", "Optional FP64 reduction"),
    ("surface_long_trace_s", "Optional long trace"),
    ("surface_long_reduce_s", "Optional long-trace reduction"),
    ("surface_confidence_s", "Confidence aggregation"),
)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def summarize_values(values: list[float], totals: list[float]) -> dict[str, float]:
    shares = [100.0 * value / total for value, total in zip(values, totals)]
    return {
        "p50_ms": 1000.0 * percentile(values, 50),
        "p95_ms": 1000.0 * percentile(values, 95),
        "max_ms": 1000.0 * max(values),
        "median_call_share_percent": percentile(shares, 50),
    }


def summarize_named_stages(
    rows: list[dict],
    definitions: tuple[tuple[str, str], ...],
    totals: list[float],
) -> list[dict]:
    records = []
    for key, label in definitions:
        values = [float(row["result"]["timing"][key]) for row in rows]
        records.append({"key": key, "label": label, **summarize_values(values, totals)})
    return records


def summarize_categories(rows: list[dict], totals: list[float]) -> list[dict]:
    records = []
    for label, keys in CATEGORIES:
        values = [
            sum(float(row["result"]["timing"][key]) for key in keys)
            for row in rows
        ]
        records.append(
            {
                "label": label,
                "keys": list(keys),
                **summarize_values(values, totals),
            }
        )
    return records


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def draw_horizontal_bars(records: list[dict], path: Path, title: str) -> None:
    visible = [record for record in records if record["p50_ms"] >= 0.01]
    visible.sort(key=lambda record: record["p50_ms"])
    labels = [record["label"] for record in visible]
    values = [record["p50_ms"] for record in visible]
    shares = [record["median_call_share_percent"] for record in visible]

    fig, ax = plt.subplots(figsize=(10.5, max(4.8, 0.48 * len(visible) + 1.6)))
    bars = ax.barh(labels, values, color="#2f6f8f")
    right = max(values) * 1.18
    ax.set_xlim(0.0, right)
    for bar, value, share in zip(bars, values, shares):
        ax.text(
            value + right * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f} ms  ({share:.1f}%)",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("P50 time per score call (ms)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", default="psi_grid48")
    args = parser.parse_args()

    all_rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line
    ]
    rows = [
        row
        for row in all_rows
        if row.get("variant") == args.variant and row["result"]["status"] == "ok"
    ]
    if not rows:
        raise ValueError(f"no successful rows for variant {args.variant!r}")

    totals = [float(row["result"]["timing"]["total_s"]) for row in rows]
    caller_walls = [float(row["caller_wall_s"]) for row in rows]
    top_level = summarize_named_stages(rows, TOP_LEVEL_STAGES, totals)
    categories = summarize_categories(rows, totals)
    axis = summarize_named_stages(rows, AXIS_STAGES, totals)
    surface = summarize_named_stages(rows, SURFACE_STAGES, totals)
    accounted = [
        sum(float(row["result"]["timing"][key]) for key, _ in TOP_LEVEL_STAGES)
        for row in rows
    ]

    summary = {
        "variant": args.variant,
        "calls": len(rows),
        "case_count": len({int(row["case_id"]) for row in rows}),
        "native_total": summarize_values(totals, totals),
        "caller_wall": summarize_values(caller_walls, caller_walls),
        "top_level_accounted": summarize_values(accounted, totals),
        "categories": categories,
        "top_level_stages": top_level,
        "axis_stages": axis,
        "surface_stages": surface,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "grid48_stage_breakdown.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_csv(args.output_dir / "grid48_categories.csv", categories)
    write_csv(args.output_dir / "grid48_top_level_stages.csv", top_level)
    write_csv(args.output_dir / "grid48_axis_stages.csv", axis)
    write_csv(args.output_dir / "grid48_surface_stages.csv", surface)
    draw_horizontal_bars(
        categories,
        args.output_dir / "grid48_category_breakdown.png",
        "Grid48 score-call bottlenecks (69 cases, 138 calls)",
    )
    draw_horizontal_bars(
        axis,
        args.output_dir / "grid48_axis_breakdown.png",
        "Magnetic-axis path breakdown",
    )
    draw_horizontal_bars(
        surface,
        args.output_dir / "grid48_surface_breakdown.png",
        "Continuous-surface path breakdown",
    )


if __name__ == "__main__":
    main()
