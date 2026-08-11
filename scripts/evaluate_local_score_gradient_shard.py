from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_PYTHON = REPO_ROOT / "gpu_backend" / "python"
for path in (REPO_ROOT, GPU_PYTHON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from stellarator_gpu import (
    score_coils_g4_fixed_branch_batch_native,
    score_coils_native,
)


def score_arguments(tokens: np.ndarray) -> tuple[np.ndarray, ...]:
    values = np.asarray(tokens, dtype=np.float64)
    return values[..., :33], values[..., 33:66], values[..., 66:99], values[..., 99]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def surface_levels(level: float) -> tuple[float, ...]:
    return (float(level),) * 16


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


def proxy_config(center: dict[str, Any], *, small: bool, normal: bool) -> dict[str, Any]:
    point_count = 8192 if small else 16384
    grid = 24 if small else 32
    return {
        "iota_degree": 3,
        "axis_hint_enabled": 1,
        "axis_hint_require_continuation": 2,
        "axis_hint_R": center["axis_R"],
        "axis_hint_Z": center["axis_Z"],
        "axis_newton_iters": 2,
        "axis_fallback_newton_iters": 2,
        "axis_trace_steps": 256,
        "axis_sample_count": 120,
        "psi_n_r": grid,
        "psi_n_z": grid,
        "psi_n_phi": grid,
        "psi_validation_points": 64,
        "psi_solver_mode": 1 if normal else 2,
        "psi_precision_mode": 2,
        "surface_level_count": 1,
        "surface_levels": surface_levels(center["surface_level"]),
        "surface_selection_mode": 1,
        "surface_confidence_periods": 1,
        "surface_theta_count": 64,
        "surface_trace_steps": 128,
        "surface_newton_iters": 8,
        "surface_long_trace_periods": 1,
        "surface_flux_bisection_iters": 2,
        "flux_level_count": 5,
        "flux_phi_count": 4,
        "flux_theta_count": 64,
        "flux_radial_quadrature": 12,
        "volume_point_count": point_count,
        "volume_phi_count": 32 if small else 48,
        "volume_theta_count": 16 if small else 24,
        "alpha_fit_point_count": point_count,
        "alpha_solver_mode": 1 if normal else 2,
    }


def compact(result: dict[str, Any]) -> dict[str, Any]:
    diagnostics = result["diagnostics"]
    return {
        "score": float(result["score"]),
        "status": str(result["status"]),
        "components": result["components"],
        "timing": result["timing"],
        "diagnostics": {
            name: diagnostics.get(name)
            for name in (
                "axis_R", "axis_Z", "axis_residual", "axis_topology_trace",
                "psi_angle_p95", "surface_level", "surface_inverse_aspect_ratio",
                "surface_confidence_risk", "flux_edge", "alpha_relative_l2",
                "alpha_normal_B_relative_l2", "iota_min", "iota_max",
                "qs_global_error", "qs_qa_global_error", "qs_qp_global_error",
                "volume_candidate_count", "volume_available_count", "volume_point_count",
                "error_message",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index")

    manifest = json.loads(
        (args.candidate_dir / "candidates.json").read_text(encoding="utf-8")
    )
    arrays = np.load(args.candidate_dir / "candidates.npz")
    tokens = np.asarray(arrays["tokens"], dtype=np.float64)
    x, y, z, current = score_arguments(tokens)
    centers = {int(row["center_index"]): row for row in manifest["centers"]}
    center_candidate_indices = {
        int(row["center_index"]): int(row["candidate_index"])
        for row in manifest["candidates"] if row["kind"] == "center"
    }
    selected = [
        row for row in manifest["candidates"]
        if int(row["candidate_index"]) % args.shard_count == args.shard_index
    ]

    started = time.perf_counter()
    rows: dict[int, dict[str, Any]] = {}
    for metadata in selected:
        index = int(metadata["candidate_index"])
        center = centers[int(metadata["center_index"])]
        row = {**metadata, "center_label": center["label"], "variants": {}}
        for name, overrides in (
            ("exact", exact_config(center)),
            ("axis_qr16k", proxy_config(center, small=False, normal=False)),
            ("axis_ne16k", proxy_config(center, small=False, normal=True)),
            ("axis_ne8k", proxy_config(center, small=True, normal=True)),
        ):
            call_started = time.perf_counter()
            result = score_coils_native(
                args.lib, x[index], y[index], z[index], current[index],
                int(center["nfp"]), device_id=0,
                target_helicity=(1, int(center["nfp"])),
                config_overrides=overrides,
            )
            row["variants"][name] = {
                **compact(result),
                "call_wall_s": time.perf_counter() - call_started,
            }
        rows[index] = row

    for center_index, center in centers.items():
        group = [
            row for row in selected if int(row["center_index"]) == center_index
        ]
        if not group:
            continue
        indices = np.asarray([int(row["candidate_index"]) for row in group], dtype=np.int64)
        call_started = time.perf_counter()
        center_candidate_index = center_candidate_indices[center_index]
        payload = score_coils_g4_fixed_branch_batch_native(
            args.lib,
            x[center_candidate_index], y[center_candidate_index],
            z[center_candidate_index], current[center_candidate_index],
            x[indices], y[indices], z[indices], current[indices],
            int(center["nfp"]), device_id=0,
            target_helicity=(1, int(center["nfp"])),
            config_overrides=proxy_config(center, small=True, normal=True),
        )
        batch_wall = time.perf_counter() - call_started
        for index, result in zip(indices, payload["query_score_results"], strict=True):
            rows[int(index)]["variants"]["fixed_ne8k"] = {
                **compact(result),
                "call_wall_s": float(result["timing"]["total_s"]),
                "batch_wall_s": batch_wall,
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index in sorted(rows):
            handle.write(json.dumps(json_safe(rows[index]), allow_nan=False) + "\n")
    summary = {
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "candidate_count": len(rows),
        "wall_s": time.perf_counter() - started,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
