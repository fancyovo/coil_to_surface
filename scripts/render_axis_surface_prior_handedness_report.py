#!/usr/bin/env python3
"""Render the signed-helicity audit and negative-target Adam200 results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COMPONENTS = ("axis", "psi", "surface", "coordinate", "volume_qs", "iota", "coil")
COMPONENT_LABELS = ("Axis", "Psi", "Surface", "Coordinate", "Volume QS", "Iota", "Coil")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_history(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sample_id_from_trajectory(trajectory_id: str) -> str:
    suffix = "_continue_adam200"
    if not trajectory_id.endswith(suffix):
        raise ValueError(f"unexpected trajectory id: {trajectory_id}")
    return trajectory_id[: -len(suffix)]


def best_score_payload(best: dict[str, Any]) -> dict[str, Any]:
    node = best["original_space_local_gradient_adam"]
    native_score = node["native_score"]
    return {
        "score": float(native_score["score"]),
        "iteration": int(node["best_iteration"]),
        "components": {name: float(native_score["components"][name]) for name in COMPONENTS},
        "diagnostics": {
            "iota_min": float(native_score["diagnostics"]["iota_min"]),
            "iota_max": float(native_score["diagnostics"]["iota_max"]),
            "qs_global_error": float(native_score["diagnostics"]["qs_global_error"]),
        },
    }


def collect_results(negative_run_root: Path, mirror_path: Path) -> dict[str, Any]:
    mirror = load_json(mirror_path)
    mirror_by_sample = {row["sample_id"]: row for row in mirror["cases"]}
    trajectories: list[dict[str, Any]] = []

    manifest_paths = sorted(
        (negative_run_root / "trajectories").glob("*/trajectory_manifest.json")
    )
    if not manifest_paths:
        raise FileNotFoundError(f"no trajectory manifests below {negative_run_root}")

    for manifest_path in manifest_paths:
        trajectory_root = manifest_path.parent
        manifest = load_json(manifest_path)
        trajectory_id = str(manifest["trajectory_id"])
        sample_id = sample_id_from_trajectory(trajectory_id)
        history = load_history(trajectory_root / "optimization" / "history.jsonl")
        best = best_score_payload(load_json(trajectory_root / "optimization" / "best.json"))
        mirror_case = mirror_by_sample[sample_id]
        positive = mirror_case["scores"]["identity__target_positive"]
        negative = mirror_case["scores"]["identity__target_negative"]
        best_iteration = int(manifest["optimization"]["best_iteration"])
        best_history = next(row for row in history if int(row["iteration"]) == best_iteration)
        recorded_iota = [float(row["current_iota"]) for row in history]

        trajectories.append(
            {
                "sample_id": sample_id,
                "trajectory_id": trajectory_id,
                "case_id": int(manifest["case"]["case_id"]),
                "nfp": int(manifest["case"]["nfp"]),
                "n_base_coils": int(manifest["case"]["n_base_coils"]),
                "target_helicity": list(manifest["target_helicity"]),
                "historical_positive_target_continuation_score": float(
                    manifest["case"]["source_score"]
                ),
                "standalone_same_geometry": {
                    "positive_target": {
                        "score": float(positive["score"]),
                        "components": {
                            name: float(positive["components"][name]) for name in COMPONENTS
                        },
                    },
                    "negative_target": {
                        "score": float(negative["score"]),
                        "components": {
                            name: float(negative["components"][name]) for name in COMPONENTS
                        },
                    },
                },
                "mirror_checks": mirror_case["comparisons"],
                "negative_target_adam200": {
                    "status": manifest["optimization"]["status"],
                    "initial_score": float(manifest["optimization"]["initial_score"]),
                    "final_score": float(manifest["optimization"]["final_score"]),
                    "best_score": float(manifest["optimization"]["best_score"]),
                    "best_iteration": best_iteration,
                    "gain_from_initial": float(manifest["optimization"]["best_score"])
                    - float(manifest["optimization"]["initial_score"]),
                    "completed_iterations": int(
                        manifest["optimization"]["completed_iterations"]
                    ),
                    "wall_s": float(manifest["timing"]["trajectory_wall_s"]),
                    "first_recorded_iota": recorded_iota[0],
                    "best_iteration_iota": float(best_history["current_iota"]),
                    "final_iota": recorded_iota[-1],
                    "all_recorded_iota_negative": all(value < 0.0 for value in recorded_iota),
                    "best_native_score": best,
                },
                "history": history,
                "provenance": manifest["provenance"],
            }
        )

    protocol = load_json(negative_run_root / "protocol.json")
    runtime_manifest = load_json(negative_run_root / "runtime_manifest.json")
    return {
        "format": "axis_surface_prior_handedness_report_assets_v1",
        "negative_run_root": str(negative_run_root),
        "signed_mirror_source": str(mirror_path),
        "protocol": protocol,
        "runtime_manifest": runtime_manifest,
        "trajectories": trajectories,
    }


def write_csv(results: dict[str, Any], path: Path) -> None:
    fields = [
        "sample_id",
        "nfp",
        "n_base_coils",
        "historical_positive_target_continuation_score",
        "standalone_positive_score",
        "standalone_negative_score",
        "negative_adam_initial_score",
        "negative_adam_best_score",
        "negative_adam_best_iteration",
        "negative_adam_gain",
        "negative_adam_final_score",
        "best_iteration_iota",
        "wall_s",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results["trajectories"]:
            standalone = row["standalone_same_geometry"]
            adam = row["negative_target_adam200"]
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "nfp": row["nfp"],
                    "n_base_coils": row["n_base_coils"],
                    "historical_positive_target_continuation_score": row[
                        "historical_positive_target_continuation_score"
                    ],
                    "standalone_positive_score": standalone["positive_target"]["score"],
                    "standalone_negative_score": standalone["negative_target"]["score"],
                    "negative_adam_initial_score": adam["initial_score"],
                    "negative_adam_best_score": adam["best_score"],
                    "negative_adam_best_iteration": adam["best_iteration"],
                    "negative_adam_gain": adam["gain_from_initial"],
                    "negative_adam_final_score": adam["final_score"],
                    "best_iteration_iota": adam["best_iteration_iota"],
                    "wall_s": adam["wall_s"],
                }
            )


def summary_without_history(results: dict[str, Any]) -> dict[str, Any]:
    summary = dict(results)
    summary["trajectories"] = []
    for row in results["trajectories"]:
        compact = {key: value for key, value in row.items() if key != "history"}
        compact["history_jsonl"] = str(
            Path(results["negative_run_root"])
            / "trajectories"
            / row["trajectory_id"]
            / "optimization"
            / "history.jsonl"
        )
        summary["trajectories"].append(compact)
    return summary


def plot_trajectories(results: dict[str, Any], path: Path) -> None:
    rows = results["trajectories"]
    fig, axes = plt.subplots(len(rows), 2, figsize=(12.0, 7.2), constrained_layout=True)
    if len(rows) == 1:
        axes = [axes]

    for row_index, row in enumerate(rows):
        history = row["history"]
        adam = row["negative_target_adam200"]
        iterations = [0] + [int(item["iteration"]) for item in history]
        current = [adam["initial_score"]] + [float(item["current_score"]) for item in history]
        best = [adam["initial_score"]] + [float(item["best_score"]) for item in history]
        iota_iterations = [int(item["iteration"]) for item in history]
        iota = [float(item["current_iota"]) for item in history]
        label = f"{row['sample_id']}  (nfp={row['nfp']}, nc={row['n_base_coils']})"

        score_ax = axes[row_index][0]
        score_ax.plot(iterations, current, color="#9aa0a6", linewidth=0.9, alpha=0.8, label="Current")
        score_ax.plot(iterations, best, color="#1769aa", linewidth=2.0, label="Best so far")
        score_ax.axhline(70.0, color="#c62828", linewidth=1.0, linestyle="--", label="Score 70")
        score_ax.scatter(
            [adam["best_iteration"]],
            [adam["best_score"]],
            color="#1769aa",
            s=28,
            zorder=3,
        )
        score_ax.set_title(label)
        score_ax.set_xlabel("Adam update")
        score_ax.set_ylabel("ABI-11 score, target (1,-nfp)")
        score_ax.grid(alpha=0.2)
        score_ax.legend(loc="lower right", fontsize=8)

        iota_ax = axes[row_index][1]
        iota_ax.plot(iota_iterations, iota, color="#00897b", linewidth=1.2)
        iota_ax.axhline(0.0, color="#c62828", linewidth=1.0, linestyle="--")
        iota_ax.set_title(f"{label}: recorded midpoint iota")
        iota_ax.set_xlabel("Adam update")
        iota_ax.set_ylabel("Midpoint iota")
        iota_ax.grid(alpha=0.2)

    fig.suptitle("Explicit negative-helicity continuation", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_components(results: dict[str, Any], path: Path) -> None:
    rows = results["trajectories"]
    fig, axes = plt.subplots(len(rows), 1, figsize=(12.0, 7.2), constrained_layout=True)
    if len(rows) == 1:
        axes = [axes]
    x_values = list(range(len(COMPONENTS)))
    width = 0.24

    for axis, row in zip(axes, rows):
        standalone = row["standalone_same_geometry"]
        optimized = row["negative_target_adam200"]["best_native_score"]
        series = (
            (
                "Source geometry, +(nfp) target",
                standalone["positive_target"]["components"],
                "#9aa0a6",
            ),
            (
                "Source geometry, -(nfp) target",
                standalone["negative_target"]["components"],
                "#ef6c00",
            ),
            ("Negative-target Adam200 best", optimized["components"], "#1769aa"),
        )
        for offset, (label, components, color) in zip((-width, 0.0, width), series):
            axis.bar(
                [value + offset for value in x_values],
                [components[name] for name in COMPONENTS],
                width=width,
                label=label,
                color=color,
            )
        axis.set_title(
            f"{row['sample_id']}  (nfp={row['nfp']}, nc={row['n_base_coils']})"
        )
        axis.set_xticks(x_values, COMPONENT_LABELS)
        axis.set_ylim(0.0, 105.0)
        axis.set_ylabel("Component score")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(loc="lower left", ncol=3, fontsize=8)

    fig.suptitle("Signed target and optimized component scores", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-run-root", type=Path, required=True)
    parser.add_argument("--mirror-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = collect_results(args.negative_run_root, args.mirror_json)

    summary_path = args.output_dir / "negative_hand_results.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            summary_without_history(results),
            handle,
            ensure_ascii=True,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
    write_csv(results, args.output_dir / "negative_hand_summary.csv")
    plot_trajectories(results, args.output_dir / "negative_hand_adam200.png")
    plot_components(results, args.output_dir / "signed_score_components.png")
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
