from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MILESTONE_GENERATIONS = (1, 8, 16, 32, 48, 64, 80, 94, 96)
LARGE_SURFACE_THRESHOLD = 0.03


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_candidate(row: dict) -> dict:
    native = row["native_score"]
    diagnostics = native["diagnostics"]
    return {
        "generation": row["generation"],
        "candidate": row["candidate"],
        "score": row["score"],
        "status": row["status"],
        "components": native["components"],
        "qs_global_error": diagnostics["qs_global_error"],
        "qs_edge_error": diagnostics["qs_edge_error"],
        "iota": 0.5 * (diagnostics["iota_min"] + diagnostics["iota_max"]),
        "iota_score": diagnostics.get("score_iota"),
        "surface_inverse_aspect_ratio": diagnostics[
            "surface_inverse_aspect_ratio"
        ],
        "surface_volume": diagnostics["surface_volume"],
        "volume_valid_fraction": diagnostics["volume_valid_fraction"],
        "volume_weight_effective_fraction": diagnostics[
            "volume_weight_effective_fraction"
        ],
    }


def load_candidates(path: Path) -> tuple[list[dict], Counter[str]]:
    successful: list[dict] = []
    statuses: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            statuses[row["status"]] += 1
            if row["status"] == "ok":
                successful.append(compact_candidate(row))
    return successful, statuses


def select_cases(rows: list[dict]) -> dict[str, dict]:
    large_rows = [
        row
        for row in rows
        if row["surface_inverse_aspect_ratio"] >= LARGE_SURFACE_THRESHOLD
    ]
    return {
        "best_total": max(rows, key=lambda row: row["score"]),
        "best_volume_qs": max(
            rows, key=lambda row: row["components"]["volume_qs"]
        ),
        "minimum_raw_qs": min(rows, key=lambda row: row["qs_global_error"]),
        "minimum_raw_qs_large_surface": (
            min(large_rows, key=lambda row: row["qs_global_error"])
            if large_rows
            else None
        ),
    }


def generation_champions(rows: list[dict]) -> dict[int, dict]:
    champions: dict[int, dict] = {}
    for row in rows:
        generation = row["generation"]
        if generation not in champions or row["score"] > champions[generation]["score"]:
            champions[generation] = row
    return champions


def plot_convergence(
    summary: dict,
    champions: dict[int, dict],
    output: Path,
    quasr_reference_score: float,
) -> None:
    generations = summary["generations"]
    x = np.asarray([row["generation"] for row in generations])
    best = np.asarray([row["best_score"] for row in generations])
    median = np.asarray([row["score_median"] for row in generations])
    ok_fraction = np.asarray(
        [row["statuses"].get("ok", 0) / summary["manifest"]["popsize"] for row in generations]
    )
    sigma = np.asarray([row["sigma_mean"] for row in generations])
    wall = np.asarray([row["wall_s"] for row in generations])

    figure, axes = plt.subplots(2, 2, figsize=(11.4, 7.6))
    axes[0, 0].plot(x, best, color="#b43b2f", linewidth=2.2, label="best so far")
    axes[0, 0].plot(x, median, color="#277c83", linewidth=1.6, label="generation median")
    axes[0, 0].axhline(
        quasr_reference_score,
        color="#555555",
        linestyle=":",
        label="QUASR QH reference max",
    )
    axes[0, 0].axvline(8, color="#888888", linestyle="--", linewidth=1.0)
    axes[0, 0].set_ylabel("native score")
    axes[0, 0].legend(fontsize=8)

    axis_sigma = axes[0, 1].twinx()
    axes[0, 1].plot(x, ok_fraction, color="#277c83", linewidth=2.0, label="success fraction")
    axis_sigma.plot(x, sigma, color="#b43b2f", linewidth=1.6, label="mean sigma")
    axes[0, 1].set_ylabel("successful fraction", color="#277c83")
    axis_sigma.set_ylabel("CEM mean sigma", color="#b43b2f")
    axes[0, 1].set_ylim(0, 1.03)
    lines = axes[0, 1].lines + axis_sigma.lines
    axes[0, 1].legend(lines, [line.get_label() for line in lines], fontsize=8, loc="center right")

    component_colors = {
        "surface": "#227c9d",
        "coordinate": "#17a398",
        "volume_qs": "#d95d39",
        "iota": "#e09f3e",
        "coil": "#6a4c93",
    }
    for component, color in component_colors.items():
        values = [champions[int(g)]["components"][component] for g in x]
        axes[1, 0].plot(x, values, color=color, linewidth=1.6, label=component)
    axes[1, 0].set_ylabel("generation champion component")
    axes[1, 0].set_xlabel("generation")
    axes[1, 0].legend(fontsize=8, ncol=2)

    axes[1, 1].plot(x, wall, color="#4f5d75", linewidth=1.6)
    axes[1, 1].axhline(np.mean(wall), color="#b43b2f", linestyle=":", label=f"mean {np.mean(wall):.1f} s")
    axes[1, 1].set_ylabel("wall time per generation [s]")
    axes[1, 1].set_xlabel("generation")
    axes[1, 1].legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.suptitle("Single-seed long QH CEM run")
    figure.tight_layout()
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def plot_pareto(rows: list[dict], selected: dict[str, dict], output: Path) -> None:
    size = np.asarray([row["surface_inverse_aspect_ratio"] for row in rows])
    raw_qs = np.asarray([row["qs_global_error"] for row in rows])
    score = np.asarray([row["score"] for row in rows])

    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.7))
    scatter = axes[0].scatter(size, raw_qs, c=score, s=9, alpha=0.45, cmap="viridis", rasterized=True)
    axes[0].set_yscale("log")
    axes[0].axvline(0.03779, color="#555555", linestyle=":", label="QUASR QH max size")
    axes[0].set_xlabel("surface inverse aspect ratio")
    axes[0].set_ylabel("raw differential QH error (lower is better)")
    axes[0].legend(fontsize=8)
    figure.colorbar(scatter, ax=axes[0], label="total score")

    axes[1].scatter(raw_qs, score, c=size, s=9, alpha=0.45, cmap="plasma", rasterized=True)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("raw differential QH error (lower is better)")
    axes[1].set_ylabel("total score")
    for label, row in selected.items():
        if label == "minimum_raw_qs" or row is None:
            continue
        axes[0].scatter(
            row["surface_inverse_aspect_ratio"],
            row["qs_global_error"],
            marker="*",
            s=150,
            edgecolor="black",
            linewidth=0.7,
            label=label.replace("_", " "),
        )
        axes[1].scatter(
            row["qs_global_error"],
            row["score"],
            marker="*",
            s=150,
            edgecolor="black",
            linewidth=0.7,
        )
    axes[0].legend(fontsize=7, loc="upper right")
    figure.colorbar(axes[1].collections[0], ax=axes[1], label="surface inverse aspect ratio")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(f"Size-QS tradeoff across {len(rows):,} successful candidates")
    figure.tight_layout()
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def audit_full_evaluation(full: dict) -> dict:
    rows = []
    surface_keys = (
        "psi_level",
        "radius_min",
        "radius_mean",
        "radius_max",
        "initial_volume",
        "initial_boozer_residual_norm",
        "ls_time_s",
        "ls_success",
        "ls_iota",
        "ls_residual_norm",
        "newton_time_s",
        "newton_success",
        "newton_iter",
        "iota",
        "volume",
        "newton_residual_norm",
        "qs_error_QA_1_0",
        "qs_error_QH_1_1",
        "qs_error_QP_0_1",
        "total_time_s",
    )
    for row in full["rows"]:
        compact = {
            "a": row["a"],
            "total_time_s": row["total_time_s"],
            "reused": row.get("reused", False),
        }
        if row.get("best_surface"):
            compact["best_surface"] = {
                key: row["best_surface"].get(key) for key in surface_keys
            }
        else:
            compact["best_surface"] = None
            compact["error"] = row.get("error")
        rows.append(compact)

    visualization = full["visualization"]
    desc = full["desc"]
    desc_keys = (
        "toroidal_flux",
        "nested_initial",
        "initial_force_compute_success",
        "initial_force_compute_time_s",
        "initial_force_mean_abs_normalized",
        "initial_force_p95_abs_normalized",
        "initial_force_max_abs_normalized",
        "solve_call_success",
        "optimizer_success",
        "optimizer_message",
        "optimizer_cost",
        "optimizer_nit",
        "optimizer_nfev",
        "optimizer_njev",
        "optimizer_optimality",
        "solve_time_s",
        "nested_final",
        "final_force_compute_success",
        "final_force_compute_time_s",
        "final_force_mean_abs_normalized",
        "final_force_p95_abs_normalized",
        "final_force_max_abs_normalized",
    )
    return {
        "target": full["target"],
        "status": full["status"],
        "rows": rows,
        "selected_largest_surface_a": full["best"]["a"],
        "total_sweep_time_s": full["total_sweep_time_s"],
        "total_time_s": full["total_time_s"],
        "visualization": {
            "surface_meta": visualization["surface_meta"],
            "b_abs_min": visualization["b_abs_min"],
            "b_abs_mean": visualization["b_abs_mean"],
            "b_abs_max": visualization["b_abs_max"],
            "poincare_hit_counts": visualization["poincare"]["hit_counts"],
        },
        "desc": {key: desc.get(key) for key in desc_keys},
    }


def plot_surface_sweep(full: dict, output: Path) -> None:
    rows = [row for row in full["rows"] if row.get("best_surface")]
    a_values = np.asarray([row["a"] for row in rows])
    volumes = np.asarray([row["best_surface"]["volume"] for row in rows])
    levels = np.asarray([row["best_surface"]["psi_level"] for row in rows])

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    axes[0].plot(a_values, volumes, "o-", color="#277c83", linewidth=2.1)
    for a_value, volume, level in zip(a_values, volumes, levels, strict=True):
        axes[0].annotate(
            f"psi={level:g}",
            (a_value, volume),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    axes[0].set_xlabel("psi fit radius parameter a [m]")
    axes[0].set_ylabel("largest Boozer-solvable volume [m$^3$]")
    axes[0].set_xlim(a_values.min() - 0.015, a_values.max() + 0.015)
    axes[0].set_ylim(0.0, volumes.max() * 1.14)

    for label, key, color in (
        ("QA", "qs_error_QA_1_0", "#277c83"),
        ("QH", "qs_error_QH_1_1", "#b43b2f"),
        ("QP", "qs_error_QP_0_1", "#6a4c93"),
    ):
        values = 100.0 * np.asarray([row["best_surface"][key] for row in rows])
        axes[1].plot(a_values, values, "o-", linewidth=1.8, color=color, label=label)
    axes[1].set_xlabel("psi fit radius parameter a [m]")
    axes[1].set_ylabel("Simsopt single-surface QS error [%]")
    axes[1].legend()

    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle("Independent stable-path surface sweep")
    figure.tight_layout()
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a long native-score CEM run.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--best-case", type=Path)
    parser.add_argument("--full-summary", type=Path)
    parser.add_argument("--quasr-reference-score", type=float, default=74.00629662106351)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = load_json(args.summary)
    successful, statuses = load_candidates(args.candidates)
    selected = select_cases(successful)
    champions = generation_champions(successful)
    milestones = [
        row
        for row in summary["generations"]
        if row["generation"] in MILESTONE_GENERATIONS
    ]
    audit = {
        "configuration": {
            key: summary["manifest"][key]
            for key in (
                "target",
                "nfp",
                "n_base_coils",
                "seed",
                "iterations",
                "popsize",
                "elite",
                "gpu_ids",
            )
        },
        "total_candidates": sum(statuses.values()),
        "successful_candidates": len(successful),
        "status_counts": dict(statuses),
        "total_wall_s": summary["total_wall_s"],
        "start_score": summary["start_score"],
        "best_score": summary["best_score"],
        "milestones": milestones,
        "selected_cases": selected,
        "large_surface_threshold": LARGE_SURFACE_THRESHOLD,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.best_case:
        best_case = load_json(args.best_case)
        manifest = best_case["cem"]["manifest"]
        public_manifest_keys = (
            "target",
            "nfp",
            "n_base_coils",
            "latent_dim_per_coil",
            "seed",
            "iterations",
            "popsize",
            "elite",
            "sigma",
            "smoothing",
            "latent_limit",
            "current_l1_a",
            "pca_sha256",
            "native_lib_sha256",
            "gpu_ids",
        )
        public_case = {
            "nfp": best_case["nfp"],
            "raw": best_case["raw"],
            "cem": {
                **{
                    key: value
                    for key, value in best_case["cem"].items()
                    if key != "manifest"
                },
                "manifest": {
                    key: manifest[key]
                    for key in public_manifest_keys
                    if key in manifest
                },
            },
        }
        audit["best_native_timing"] = best_case["cem"]["native_score"]["timing"]
        (args.output_dir / "best_case.json").write_text(
            json.dumps(public_case, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    (args.output_dir / "long_cem_audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    plot_convergence(
        summary,
        champions,
        args.output_dir / "long_cem_convergence.png",
        args.quasr_reference_score,
    )
    plot_pareto(successful, selected, args.output_dir / "long_cem_pareto.png")
    if args.full_summary:
        full = load_json(args.full_summary)
        full_audit = audit_full_evaluation(full)
        (args.output_dir / "full_evaluation_audit.json").write_text(
            json.dumps(full_audit, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        plot_surface_sweep(full, args.output_dir / "full_surface_sweep.png")


if __name__ == "__main__":
    main()
