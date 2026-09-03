from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.axisflip_coil_scale import (
    coil_score_decomposition,
    load_full_axis_points,
    tokens_from_raw,
)
from scripts.native_score_runtime import write_json
from scripts.prepare_axisflip_case23_coil_shrink import (
    full_surface_points,
    geometry_metrics,
    periodic_envelope,
    write_geometry_html,
    write_geometry_png,
)


def repair_argument(values: list[str]) -> tuple[str, float, Path, Path]:
    label, scale_text, start_text, output_text = values
    return label, float(scale_text), Path(start_text), Path(output_text)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def native_endpoint(native: dict[str, Any]) -> dict[str, float]:
    if native.get("status") != "ok":
        raise ValueError(f"native endpoint is not valid: {native.get('status')}")
    diagnostics = native["diagnostics"]
    pieces = coil_score_decomposition(native)
    return {
        "total": float(native["score"]),
        "volume_qs": float(native["components"]["volume_qs"]),
        "coil": float(native["components"]["coil"]),
        "coil_length_mean_m": float(diagnostics["coil_length_mean"]),
        "curvature_p95_per_m": float(diagnostics["coil_curvature_p95"]),
        "curvature_max_per_m": float(diagnostics["coil_curvature_max"]),
        "min_intercoil_distance_m": float(diagnostics["coil_min_intercoil_distance"]),
        "min_z_axis_distance_m": float(diagnostics["coil_min_axis_distance"]),
        **pieces,
    }


def combine_scan_rows(*summaries: dict[str, Any]) -> list[dict[str, Any]]:
    rows: dict[float, dict[str, Any]] = {}
    for summary in summaries:
        for row in summary["candidates"]:
            rows[float(row["scale"])] = row
    return [rows[key] for key in sorted(rows, reverse=True)]


def load_history(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize case-23 coil-shrink scan and Adam200 repairs.")
    parser.add_argument("--coarse-summary", type=Path, required=True)
    parser.add_argument("--refine-summary", type=Path, required=True)
    parser.add_argument("--axis-data", type=Path, required=True)
    parser.add_argument("--surface-mesh", type=Path, required=True)
    parser.add_argument(
        "--repair",
        action="append",
        nargs=4,
        metavar=("LABEL", "SCALE", "START_JSON", "OUTPUT_DIR"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repairs = [repair_argument(value) for value in args.repair]
    if len(repairs) != 2:
        raise ValueError("this registered experiment requires exactly two repairs")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    coarse = read_json(args.coarse_summary)
    refine = read_json(args.refine_summary)
    scan_rows = combine_scan_rows(coarse, refine)
    with np.load(args.axis_data) as axis_data:
        axis_points = load_full_axis_points(axis_data)
    one_period_surface, surface_points, surface_nfp = full_surface_points(args.surface_mesh)
    if surface_nfp != 6:
        raise ValueError(f"expected nfp=6 reference surface, got {surface_nfp}")
    surface_flat = surface_points.reshape(-1, 3)
    axis_tree = cKDTree(axis_points)
    surface_tree = cKDTree(surface_flat)
    surface_axis_distance, surface_axis_index = axis_tree.query(surface_flat, workers=1)
    surface_envelope = periodic_envelope(surface_axis_index, surface_axis_distance, len(axis_points))

    results: list[dict[str, Any]] = []
    for label, scale, start_path, repair_root in repairs:
        start_payload = read_json(start_path)
        best_payload = read_json(repair_root / "optimization" / "best.json")
        manifest = read_json(repair_root / "optimization" / "manifest.json")
        history = load_history(repair_root / "optimization" / "history.jsonl")
        start_native = start_payload["data_prior_screening"]["native_score"]
        best_record = best_payload["original_space_local_gradient_adam"]
        best_native = best_record["native_score"]
        start = native_endpoint(start_native)
        best = native_endpoint(best_native)
        start_tokens = tokens_from_raw(start_payload["raw"])
        best_tokens = tokens_from_raw(best_payload["raw"])
        start_geometry = geometry_metrics(
            start_tokens,
            nfp=6,
            axis_points=axis_points,
            axis_tree=axis_tree,
            surface_tree=surface_tree,
            surface_envelope=surface_envelope,
        )
        best_geometry = geometry_metrics(
            best_tokens,
            nfp=6,
            axis_points=axis_points,
            axis_tree=axis_tree,
            surface_tree=surface_tree,
            surface_envelope=surface_envelope,
        )
        for endpoint, geometry in ((start, start_geometry), (best, best_geometry)):
            endpoint.update(geometry)
        row = {
            "label": label,
            "scale": scale,
            "best_iteration": int(best_record["best_iteration"]),
            "completed_iteration": int(history[-1]["iteration"]),
            "final_score": float(history[-1]["current_score"]),
            "initial_consistency_gate": manifest["initial_consistency_gate"],
            "start": start,
            "best": best,
            "delta": {key: best[key] - start[key] for key in start},
            "history": history,
            "manifest": manifest,
        }
        results.append(row)
        safe = label.replace(".", "p")
        write_geometry_png(
            output_dir / f"{safe}_start.png",
            one_period_surface=one_period_surface,
            tokens=start_tokens,
            axis_points=axis_points,
            nfp=6,
            scale=scale,
            label=f"Scale {scale:.3f} start",
        )
        write_geometry_html(
            output_dir / f"{safe}_start.html",
            one_period_surface=one_period_surface,
            tokens=start_tokens,
            axis_points=axis_points,
            nfp=6,
            scale=scale,
            label=f"Scale {scale:.3f} start",
        )
        write_geometry_png(
            output_dir / f"{safe}_repair_best.png",
            one_period_surface=one_period_surface,
            tokens=best_tokens,
            axis_points=axis_points,
            nfp=6,
            scale=scale,
            label=f"Scale {scale:.3f} repair best",
        )
        write_geometry_html(
            output_dir / f"{safe}_repair_best.html",
            one_period_surface=one_period_surface,
            tokens=best_tokens,
            axis_points=axis_points,
            nfp=6,
            scale=scale,
            label=f"Scale {scale:.3f} repair best",
        )

    flat_rows = []
    for row in results:
        flat: dict[str, Any] = {
            "label": row["label"],
            "scale": row["scale"],
            "best_iteration": row["best_iteration"],
            "completed_iteration": row["completed_iteration"],
            "final_score": row["final_score"],
        }
        for phase in ("start", "best", "delta"):
            flat.update({f"{phase}_{key}": value for key, value in row[phase].items()})
        flat_rows.append(flat)
    with (output_dir / "repair_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    machine_summary = {
        "format": "axisflip_case23_coil_shrink_results_v1",
        "protocol_id": coarse["protocol_id"],
        "source": coarse["source"],
        "score_library_sha256": coarse["score_library_sha256"],
        "coarse_scan_code_commit": coarse["code_commit"],
        "refine_scan_code_commit": refine["code_commit"],
        "scan": scan_rows,
        "repairs": [{key: value for key, value in row.items() if key not in {"history", "manifest"}} for row in results],
    }
    write_json(output_dir / "results_summary.json", machine_summary)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    scales = np.asarray([float(row["scale"]) for row in scan_rows])
    valid = np.asarray([row["status"] == "ok" for row in scan_rows])
    scores = np.asarray([float(row["score"]) for row in scan_rows])
    radius = np.asarray([float(row["geometry"]["effective_radius_m"]) for row in scan_rows])
    axis_mean = np.asarray([float(row["geometry"]["reference_axis_distance_mean_m"]) for row in scan_rows])
    margin = np.asarray([float(row["geometry"]["reference_surface_tube_margin_min_m"]) for row in scan_rows])
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.5), constrained_layout=True)
    axes[0].plot(scales[valid], scores[valid], "o-", color="#176b87", label="valid")
    axes[0].scatter(scales[~valid], scores[~valid], marker="x", color="#aa3c2f", label="invalid")
    axes[0].set(xlabel="Axis-centered scale", ylabel="ABI-11 total score")
    axes[0].legend()
    axes[1].plot(scales, radius, "o-", label="effective coil radius")
    axes[1].plot(scales, axis_mean, "s-", label="mean coil-axis distance")
    axes[1].set(xlabel="Axis-centered scale", ylabel="Distance [m]")
    axes[1].legend()
    axes[2].plot(scales, margin, "o-", color="#27824a")
    axes[2].axhline(0.0, color="#555555", linewidth=1)
    axes[2].set(xlabel="Axis-centered scale", ylabel="Minimum tube margin [m]")
    for axis in axes:
        axis.grid(alpha=0.22)
    figure.suptitle("Case 23 axis-centered coil-shrink scan")
    figure.savefig(output_dir / "shrink_scan_tradeoff.png", dpi=185)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), constrained_layout=True)
    for axis, row in zip(axes, results, strict=True):
        history = row["history"]
        iteration = np.asarray([0] + [int(item["iteration"]) for item in history])
        current = np.asarray([row["start"]["total"]] + [float(item["current_score"]) for item in history])
        best_curve = np.asarray([row["start"]["total"]] + [float(item["best_score"]) for item in history])
        axis.plot(iteration, current, color="#8c8c8c", linewidth=1.1, label="current")
        axis.plot(iteration, best_curve, color="#176b87", linewidth=2.0, label="best so far")
        axis.set(title=f"{row['label']} (scale {row['scale']:.2f})", xlabel="Adam update", ylabel="ABI-11 total score")
        axis.grid(alpha=0.22)
        axis.legend()
    figure.suptitle("Original-space Adam200 repair trajectories")
    figure.savefig(output_dir / "repair_trajectories.png", dpi=185)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), constrained_layout=True)
    width = 0.34
    colors = ("#b8c7cc", "#176b87")
    for index, row in enumerate(results):
        offset = (index - 0.5) * width
        axes[0, 0].bar(
            np.arange(3) + offset,
            [row["best"][key] for key in ("total", "volume_qs", "coil")],
            width,
            color=colors[index],
            label=row["label"],
        )
        axes[0, 1].bar(
            np.arange(2) + offset,
            [row["best"][key] for key in ("points_curvature", "points_distance")],
            width,
            color=colors[index],
            label=row["label"],
        )
    axes[0, 0].set_xticks(np.arange(3), ("Total", "Volume-QS", "Coil"))
    axes[0, 0].set_ylabel("Best score [points]")
    axes[0, 0].legend()
    axes[0, 1].set_xticks(np.arange(2), ("Curvature", "Distance"))
    axes[0, 1].set_ylabel("Best engineering contribution [points]")
    axes[0, 1].legend()
    for axis, row in zip(axes[1], results, strict=True):
        names = ("effective_radius_m", "reference_axis_distance_mean_m", "reference_surface_nearest_distance_m", "min_intercoil_distance_m")
        labels = ("Effective radius", "Mean magnetic-axis distance", "Nearest reference surface", "Min intercoil")
        x = np.arange(len(names))
        axis.bar(x - width / 2, [row["start"][name] for name in names], width, color="#b8c7cc", label="shrunk start")
        axis.bar(x + width / 2, [row["best"][name] for name in names], width, color="#176b87", label="repair best")
        axis.set_xticks(x, labels, rotation=16, ha="right")
        axis.set_ylabel("Distance [m]")
        axis.set_title(f"{row['label']} geometry")
        axis.legend()
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.22)
    figure.suptitle("Case 23 repair endpoints")
    figure.savefig(output_dir / "repair_endpoint_comparison.png", dpi=185)
    plt.close(figure)

    print(json.dumps({"event": "complete", "repairs": len(results), "output_dir": str(output_dir)}))


if __name__ == "__main__":
    main()
