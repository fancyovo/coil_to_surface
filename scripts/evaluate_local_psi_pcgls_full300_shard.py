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

from stellarator_gpu import score_coils_native, score_coils_psi_warm_batch_native
from scripts.score_gradient_proxy import coordinate_omitted_gradient_score


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
    components = {name: float(result["components"][name]) for name in COMPONENTS}
    return {
        "score": float(result["score"]),
        "gradient_proxy_score": coordinate_omitted_gradient_score(
            result["score"], components
        ),
        "status": result["status"],
        "components": components,
        "psi_angle_p95": float(diagnostics["psi_angle_p95"]),
        "surface_level": float(diagnostics["surface_level"]),
        "alpha_relative_l2": float(diagnostics["alpha_relative_l2"]),
        "qs_global_error": float(diagnostics["qs_global_error"]),
        "psi_fit_s": float(result["timing"]["psi_fit_s"]),
        "total_s": float(result["timing"]["total_s"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--scale", type=float, default=0.005)
    parser.add_argument("--iterations", default="2,4")
    args = parser.parse_args()
    iterations = [int(value) for value in args.iterations.split(",")]

    manifest = json.loads((args.candidate_dir / "candidates.json").read_text(encoding="utf-8"))
    arrays = np.load(args.candidate_dir / "candidates.npz")
    tokens = np.asarray(arrays["tokens"], dtype=np.float64)
    x, y, z, current = score_arguments(tokens)
    if len(manifest["centers"]) != 1:
        raise ValueError("full300 PCGLS calibration requires exactly one center")
    center = manifest["centers"][0]
    center_row = next(row for row in manifest["candidates"] if row["kind"] == "center")
    center_index = int(center_row["candidate_index"])
    endpoint_rows = [
        row for row in manifest["candidates"]
        if row["kind"] == "endpoint"
        and np.isclose(float(row["scale"]), args.scale, rtol=0.0, atol=1.0e-15)
        and int(row["direction_index"]) % args.shard_count == args.shard_index
    ]
    endpoint_rows.sort(key=lambda row: (int(row["direction_index"]), int(row["sign"])))
    endpoint_indices = np.asarray([int(row["candidate_index"]) for row in endpoint_rows], dtype=np.int64)
    config = exact_config(center)

    started = time.perf_counter()
    records = [{"metadata": row, "variants": {}} for row in endpoint_rows]
    for record, index in zip(records, endpoint_indices, strict=True):
        record["variants"]["exact"] = compact(score_coils_native(
            args.lib, x[index], y[index], z[index], current[index], int(center["nfp"]),
            device_id=0, target_helicity=(1, int(center["nfp"])), config_overrides=config,
        ))

    for count in iterations:
        payload = score_coils_psi_warm_batch_native(
            args.lib,
            x[center_index], y[center_index], z[center_index], current[center_index],
            x[endpoint_indices], y[endpoint_indices], z[endpoint_indices], current[endpoint_indices],
            int(center["nfp"]), count, use_center_qr_preconditioner=True,
            device_id=0, target_helicity=(1, int(center["nfp"])), config_overrides=config,
        )
        for record, result in zip(records, payload["query_score_results"], strict=True):
            record["variants"][f"pcgls{count}"] = compact(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    summary = {
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "endpoint_count": len(records),
        "iterations": iterations,
        "wall_s": time.perf_counter() - started,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
