from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_PYTHON = REPO_ROOT / "gpu_backend" / "python"
for path in (REPO_ROOT, GPU_PYTHON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from stellarator_gpu import BatchCoilFieldGpu, CoilFieldGpu


def score_arguments(tokens: np.ndarray) -> tuple[np.ndarray, ...]:
    values = np.asarray(tokens, dtype=np.float64)
    return values[..., :33], values[..., 33:66], values[..., 66:99], values[..., 99]


def relative_rms(actual: np.ndarray, expected: np.ndarray) -> float:
    difference = np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    denominator = max(float(np.linalg.norm(expected)), 1.0e-30)
    return float(np.linalg.norm(difference) / denominator)


def timed(function):
    started = time.perf_counter()
    result = function()
    return result, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-count", type=int, default=600)
    parser.add_argument("--point-count", type=int, default=256)
    parser.add_argument("--segments-per-coil", type=int, default=256)
    parser.add_argument("--trace-steps", type=int, default=400)
    parser.add_argument("--axis-integration-steps", type=int, default=960)
    parser.add_argument("--axis-samples", type=int, default=240)
    parser.add_argument("--reference-count", type=int, default=4)
    args = parser.parse_args()

    manifest = json.loads((args.candidate_dir / "candidates.json").read_text(encoding="utf-8"))
    arrays = np.load(args.candidate_dir / "candidates.npz")
    tokens = np.asarray(arrays["tokens"], dtype=np.float64)
    center = manifest["centers"][0]
    endpoint_rows = [
        row for row in manifest["candidates"]
        if row["kind"] == "endpoint" and np.isclose(row["scale"], 0.005)
    ]
    endpoint_rows.sort(key=lambda row: (int(row["direction_index"]), int(row["sign"])))
    if len(endpoint_rows) < args.query_count:
        raise ValueError(f"only {len(endpoint_rows)} matching endpoints are available")
    endpoint_indices = np.asarray([
        int(row["candidate_index"]) for row in endpoint_rows[: args.query_count]
    ])
    x, y, z, current = score_arguments(tokens[endpoint_indices])
    nfp = int(center["nfp"])

    rng = np.random.default_rng(2026081201)
    radius = rng.uniform(0.005, 0.06, size=(args.query_count, args.point_count))
    angle = rng.uniform(0.0, 2.0 * np.pi, size=(args.query_count, args.point_count))
    phi = rng.uniform(0.0, 2.0 * np.pi / nfp, size=(args.query_count, args.point_count))
    cylindrical_R = float(center["axis_R"]) + radius * np.cos(angle)
    cylindrical_Z = float(center["axis_Z"]) + radius * np.sin(angle)
    points = np.stack(
        (cylindrical_R * np.cos(phi), cylindrical_R * np.sin(phi), cylindrical_Z), axis=-1
    ).astype(np.float32)
    line_offsets = np.asarray([-0.02, -0.01, 0.0, 0.01, 0.02], dtype=np.float64)
    R0 = np.broadcast_to(float(center["axis_R"]) + line_offsets, (args.query_count, 5)).copy()
    Z0 = np.broadcast_to(float(center["axis_Z"]) + line_offsets[::-1], (args.query_count, 5)).copy()
    axis_R0 = np.full(args.query_count, float(center["axis_R"]), dtype=np.float64)
    axis_Z0 = np.full(args.query_count, float(center["axis_Z"]), dtype=np.float64)

    batch, create_wall_s = timed(lambda: BatchCoilFieldGpu(
        args.lib, x, y, z, current, nfp,
        segments_per_coil=args.segments_per_coil, device_id=0,
    ))
    try:
        batch_B, eval_B_wall_s = timed(lambda: batch.eval_B(points))
        (batch_B_grad, batch_gradient), eval_B_grad_wall_s = timed(
            lambda: batch.eval_B_grad(points)
        )
        (batch_R1, batch_Z1), trace_wall_s = timed(
            lambda: batch.trace_period(R0, Z0, steps=args.trace_steps)
        )
        refined_axis, refine_axis_wall_s = timed(lambda: batch.refine_axis_hint(
            axis_R0, axis_Z0,
            trace_steps=args.axis_integration_steps,
            newton_iterations=6,
            finite_difference_step=2.0e-4,
            maximum_newton_step=0.25,
            residual_tolerance=1.0e-7,
            hint_max_distance=0.08,
        ))
        batch_axis, axis_wall_s = timed(lambda: batch.trace_axis_samples(
            refined_axis["R"], refined_axis["Z"],
            integration_steps=args.axis_integration_steps,
            sample_count=args.axis_samples,
        ))

        references = np.linspace(
            0, args.query_count - 1, min(args.reference_count, args.query_count), dtype=int
        )
        errors = []
        reference_wall_s = 0.0
        for query in references:
            started = time.perf_counter()
            field = CoilFieldGpu(
                args.lib, x[query], y[query], z[query], current[query], nfp,
                segments_per_coil=args.segments_per_coil, device_id=0,
            )
            try:
                single_B = field.eval_B(points[query], precision="fp32")
                single_B_grad, single_gradient = field.eval_B_grad(
                    points[query], precision="fp32"
                )
                single_R1, single_Z1 = field.trace_period_blockline_mixed(
                    R0[query], Z0[query], steps=args.trace_steps,
                    threads_per_line=256, mode="bf32_state64",
                )
                single_axis = field.trace_axis_samples(
                    refined_axis["R"][query], refined_axis["Z"][query], nfp=nfp,
                    integration_steps=args.axis_integration_steps,
                    sample_count=args.axis_samples,
                )
            finally:
                field.close()
            reference_wall_s += time.perf_counter() - started
            errors.append({
                "query": int(query),
                "B_relative_rms": relative_rms(batch_B[query], single_B),
                "B_from_grad_relative_rms": relative_rms(batch_B_grad[query], single_B_grad),
                "grad_B_relative_rms": relative_rms(batch_gradient[query], single_gradient),
                "trace_R_relative_rms": relative_rms(batch_R1[query], single_R1),
                "trace_Z_absolute_rms": float(np.sqrt(np.mean(np.square(
                    batch_Z1[query] - single_Z1
                )))),
                "axis_R_relative_rms": relative_rms(batch_axis[0][query], single_axis[0]),
                "axis_Z_absolute_rms": float(np.sqrt(np.mean(np.square(
                    batch_axis[1][query] - single_axis[1]
                )))),
                "axis_R_phi_relative_rms": relative_rms(batch_axis[2][query], single_axis[2]),
                "axis_Z_phi_relative_rms": relative_rms(batch_axis[3][query], single_axis[3]),
            })
    finally:
        batch.close()

    output = {
        "format": "query_batch_field_benchmark_v1",
        "query_count": args.query_count,
        "point_count": args.point_count,
        "segments_per_coil": args.segments_per_coil,
        "nfp": nfp,
        "trace_steps": args.trace_steps,
        "axis_integration_steps": args.axis_integration_steps,
        "axis_samples": args.axis_samples,
        "timing_s": {
            "field_create": create_wall_s,
            "eval_B": eval_B_wall_s,
            "eval_B_grad": eval_B_grad_wall_s,
            "trace_period_5_lines": trace_wall_s,
            "refine_axis_hint": refine_axis_wall_s,
            "trace_axis_samples": axis_wall_s,
            "tested_stage_total": create_wall_s + eval_B_wall_s + eval_B_grad_wall_s +
                trace_wall_s + refine_axis_wall_s + axis_wall_s,
            "sequential_reference_subset": reference_wall_s,
        },
        "reference_errors": errors,
        "axis_refinement": {
            "valid_count": int(np.count_nonzero(refined_axis["valid"])),
            "residual_p50_p95_max": [
                float(value) for value in np.quantile(
                    refined_axis["residual"], [0.5, 0.95, 1.0]
                )
            ],
            "hint_distance_p50_p95_max": [
                float(value) for value in np.quantile(np.hypot(
                    refined_axis["R"] - axis_R0,
                    refined_axis["Z"] - axis_Z0,
                ), [0.5, 0.95, 1.0])
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
