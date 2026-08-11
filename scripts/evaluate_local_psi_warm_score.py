from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_PYTHON = REPO_ROOT / "gpu_backend" / "python"
for path in (REPO_ROOT, GPU_PYTHON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from stellarator_gpu import score_coils_native, score_coils_psi_warm_batch_native


COMPONENTS = ("axis", "psi", "surface", "coordinate", "volume_qs", "iota", "coil")


def score_arguments(tokens: np.ndarray) -> tuple[np.ndarray, ...]:
    values = np.asarray(tokens, dtype=np.float64)
    return values[..., :33], values[..., 33:66], values[..., 66:99], values[..., 99]


def exact_config(center: dict[str, Any]) -> dict[str, Any]:
    return {
        "iota_degree": 3,
        "surface_selection_mode": 1,
        "surface_confidence_periods": 1,
        "surface_theta_count": 128,
        "surface_trace_steps": 400,
        "surface_flux_bisection_iters": 6,
        "axis_hint_enabled": 1,
        "axis_hint_require_continuation": 2,
        "axis_hint_R": center["axis_R"],
        "axis_hint_Z": center["axis_Z"],
    }


def compact(result: dict[str, Any]) -> dict[str, Any]:
    diagnostics = result["diagnostics"]
    return {
        "score": float(result["score"]),
        "status": result["status"],
        "components": {name: float(result["components"][name]) for name in COMPONENTS},
        "psi_train_rms": float(diagnostics["psi_train_rms"]),
        "psi_angle_p95": float(diagnostics["psi_angle_p95"]),
        "surface_level": float(diagnostics["surface_level"]),
        "surface_confidence_risk": float(diagnostics["surface_confidence_risk"]),
        "alpha_relative_l2": float(diagnostics["alpha_relative_l2"]),
        "qs_global_error": float(diagnostics["qs_global_error"]),
        "psi_fit_s": float(result["timing"]["psi_fit_s"]),
        "total_s": float(result["timing"]["total_s"]),
    }


def finite_difference(values: dict[str, dict[str, Any]], field: str, scale: float) -> np.ndarray:
    output = []
    for direction in range(4):
        minus = values[f"direction_{direction:03d}_minus"]
        plus = values[f"direction_{direction:03d}_plus"]
        if field == "score":
            minus_value, plus_value = minus["score"], plus["score"]
        else:
            minus_value = minus["components"][field]
            plus_value = plus["components"][field]
        output.append((plus_value - minus_value) / (2.0 * scale))
    return np.asarray(output, dtype=np.float64)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0.0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=0.005)
    parser.add_argument("--iterations", default="0,2,4,8,16")
    args = parser.parse_args()
    iterations = [int(value) for value in args.iterations.split(",")]
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    manifest = json.loads((args.candidate_dir / "candidates.json").read_text(encoding="utf-8"))
    arrays = np.load(args.candidate_dir / "candidates.npz")
    tokens = np.asarray(arrays["tokens"], dtype=np.float64)
    x, y, z, current = score_arguments(tokens)
    if len(manifest["centers"]) != 1:
        raise ValueError("psi warm-score calibration requires exactly one center")
    center = manifest["centers"][0]
    center_row = next(row for row in manifest["candidates"] if row["kind"] == "center")
    endpoint_rows = [
        row for row in manifest["candidates"]
        if row["kind"] == "endpoint"
        and int(row["direction_index"]) < 4
        and np.isclose(float(row["scale"]), args.scale, rtol=0.0, atol=1.0e-15)
    ]
    endpoint_rows.sort(key=lambda row: (int(row["direction_index"]), int(row["sign"])))
    if len(endpoint_rows) != 8:
        raise ValueError(f"expected 8 endpoints, found {len(endpoint_rows)}")
    center_index = int(center_row["candidate_index"])
    endpoint_indices = np.asarray([int(row["candidate_index"]) for row in endpoint_rows], dtype=np.int64)
    labels = [
        f"direction_{int(row['direction_index']):03d}_{'plus' if int(row['sign']) > 0 else 'minus'}"
        for row in endpoint_rows
    ]
    config = exact_config(center)

    exact: dict[str, dict[str, Any]] = {}
    exact_wall_started = time.perf_counter()
    for label, index in zip(labels, endpoint_indices, strict=True):
        exact[label] = compact(score_coils_native(
            args.lib, x[index], y[index], z[index], current[index], int(center["nfp"]),
            device_id=0, target_helicity=(1, int(center["nfp"])), config_overrides=config,
        ))
    exact_wall_s = time.perf_counter() - exact_wall_started

    variants: dict[str, Any] = {}
    for count in iterations:
        started = time.perf_counter()
        payload = score_coils_psi_warm_batch_native(
            args.lib,
            x[center_index], y[center_index], z[center_index], current[center_index],
            x[endpoint_indices], y[endpoint_indices], z[endpoint_indices], current[endpoint_indices],
            int(center["nfp"]), count, device_id=0,
            target_helicity=(1, int(center["nfp"])), config_overrides=config,
        )
        query = {
            label: compact(result)
            for label, result in zip(labels, payload["query_score_results"], strict=True)
        }
        variants[str(count)] = {
            "wall_s": time.perf_counter() - started,
            "center": compact(payload["center_score_result"]),
            "query": query,
        }

    exact_score_gradient = finite_difference(exact, "score", args.scale)
    summary_rows = []
    component_cosines: dict[str, dict[str, float]] = {}
    for count in iterations:
        query = variants[str(count)]["query"]
        exact_scores = np.asarray([exact[label]["score"] for label in labels])
        warm_scores = np.asarray([query[label]["score"] for label in labels])
        warm_gradient = finite_difference(query, "score", args.scale)
        component_cosines[str(count)] = {
            name: cosine(
                finite_difference(exact, name, args.scale),
                finite_difference(query, name, args.scale),
            )
            for name in COMPONENTS
        }
        summary_rows.append({
            "iterations": count,
            "all_status_ok": all(query[label]["status"] == "ok" for label in labels),
            "score_max_abs_error": float(np.max(np.abs(warm_scores - exact_scores))),
            "score_rmse": float(np.sqrt(np.mean(np.square(warm_scores - exact_scores)))),
            "score_gradient_cosine": cosine(exact_score_gradient, warm_gradient),
            "score_gradient_norm_ratio": float(np.linalg.norm(warm_gradient) / np.linalg.norm(exact_score_gradient)),
            "score_gradient_sign_fraction": float(np.mean(np.sign(warm_gradient) == np.sign(exact_score_gradient))),
            "psi_angle_p95_ratio_p50": float(np.median([
                query[label]["psi_angle_p95"] / exact[label]["psi_angle_p95"] for label in labels
            ])),
            "psi_fit_s_p50": float(np.median([query[label]["psi_fit_s"] for label in labels])),
            "total_s_p50": float(np.median([query[label]["total_s"] for label in labels])),
            "batch_wall_s": float(variants[str(count)]["wall_s"]),
        })

    output = {
        "format": "local_psi_warm_score_v1",
        "scale": args.scale,
        "direction_count": 4,
        "center": center,
        "exact_wall_s": exact_wall_s,
        "exact": exact,
        "variants": variants,
        "summary": summary_rows,
        "component_gradient_cosines": component_cosines,
    }
    (args.output_dir / "results.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    counts = np.asarray(iterations)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    for count in iterations:
        query = variants[str(count)]["query"]
        axes[0].scatter(
            [exact[label]["score"] for label in labels],
            [query[label]["score"] for label in labels],
            label=f"{count} iter", s=28,
        )
    limits = axes[0].get_xlim()
    axes[0].plot(limits, limits, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("endpoint QR score")
    axes[0].set_ylabel("warm-start score")
    axes[0].legend(fontsize=8)
    axes[1].plot(counts, [row["score_gradient_cosine"] for row in summary_rows], marker="o")
    axes[1].axhline(0.9, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("4-direction score-gradient cosine")
    axes[1].set_ylim(-1.05, 1.05)
    axes[2].plot(counts, [1000.0 * row["psi_fit_s_p50"] for row in summary_rows], marker="o", label="psi fit")
    axes[2].plot(counts, [1000.0 * row["total_s_p50"] for row in summary_rows], marker="o", label="complete score")
    axes[2].set_ylabel("endpoint median time (ms)")
    axes[2].legend()
    for axis in axes:
        axis.set_xlabel("warm CGLS iterations")
        axis.grid(True, alpha=0.25)
    fig.suptitle("Full-score effect of center-warm full-grid48 psi fits (h=0.005)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "warm_score_comparison.png", dpi=180)
    plt.close(fig)

    matrix = np.asarray([[component_cosines[str(count)][name] for count in iterations] for name in COMPONENTS])
    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    image = axis.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")
    axis.set_xticks(range(len(iterations)), labels=iterations)
    axis.set_yticks(range(len(COMPONENTS)), labels=COMPONENTS)
    axis.set_xlabel("warm CGLS iterations")
    axis.set_title("Component finite-difference gradient cosine vs endpoint QR")
    fig.colorbar(image, ax=axis, label="cosine")
    fig.tight_layout()
    fig.savefig(args.output_dir / "warm_component_gradient_cosines.png", dpi=180)
    plt.close(fig)

    print(json.dumps({"summary": summary_rows, "component_gradient_cosines": component_cosines}, indent=2))


if __name__ == "__main__":
    main()
