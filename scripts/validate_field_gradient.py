from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
GPU_PYTHON = REPO_ROOT / "gpu_backend" / "python"
if str(GPU_PYTHON) not in sys.path:
    sys.path.insert(0, str(GPU_PYTHON))

from stellarator_eval.field import build_field, load_case_file, normalize_currents
from stellarator_gpu import (
    CoilFieldGpu,
    eval_B_grad_segments_cpu,
    make_segments_cpu,
)


def relative_l2(value, reference) -> float:
    value = np.asarray(value, dtype=float)
    reference = np.asarray(reference, dtype=float)
    return float(np.linalg.norm(value - reference) / max(np.linalg.norm(reference), 1e-300))


def error_metrics(value, reference) -> dict[str, float]:
    value = np.asarray(value, dtype=float)
    reference = np.asarray(reference, dtype=float)
    scale = np.linalg.norm(reference.reshape(len(reference), -1), axis=1)
    difference = np.linalg.norm((value - reference).reshape(len(reference), -1), axis=1)
    point_relative = difference / np.maximum(scale, 1e-300)
    return {
        "relative_l2": relative_l2(value, reference),
        "point_relative_mean": float(np.mean(point_relative)),
        "point_relative_p95": float(np.percentile(point_relative, 95)),
        "point_relative_max": float(np.max(point_relative)),
        "max_abs": float(np.max(np.abs(value - reference))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=Path("examples/01.json"))
    parser.add_argument("--key", default="raw")
    parser.add_argument("--current-unit", default="MA")
    parser.add_argument("--gpu-lib", type=Path, default=Path("gpu_backend/build_volume_qs/libstellarator_gpu.so"))
    parser.add_argument("--segments", type=int, default=256)
    parser.add_argument("--points", type=int, default=128)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    field_input = load_case_file(args.case, args.key)
    built = build_field(field_input, args.current_unit)
    currents_a = normalize_currents(field_input.currents, args.current_unit)
    rng = np.random.default_rng(20260726)
    phi = rng.uniform(0.0, 2.0 * np.pi / field_input.nfp, args.points)
    radius = 0.72 * built.coil_r0 + rng.uniform(-0.025, 0.025, args.points)
    z = rng.uniform(-0.035, 0.035, args.points)
    xyz = np.column_stack([radius * np.cos(phi), radius * np.sin(phi), z])

    t0 = time.perf_counter()
    built.field.set_points(xyz)
    simsopt_B = np.asarray(built.field.B())
    simsopt_grad = np.asarray(built.field.dB_by_dX())
    simsopt_time = time.perf_counter() - t0

    segment_position, segment_weight = make_segments_cpu(
        field_input.coeffs_x,
        field_input.coeffs_y,
        field_input.coeffs_z,
        currents_a,
        field_input.nfp,
        args.segments,
    )
    t0 = time.perf_counter()
    cpu_B, cpu_grad = eval_B_grad_segments_cpu(xyz, segment_position, segment_weight)
    cpu_time = time.perf_counter() - t0

    gpu = CoilFieldGpu(
        args.gpu_lib,
        field_input.coeffs_x,
        field_input.coeffs_y,
        field_input.coeffs_z,
        currents_a,
        field_input.nfp,
        segments_per_coil=args.segments,
        device_id=args.device,
    )
    try:
        timings = {}
        outputs = {}
        pure_field = {}
        for precision in ("fp64", "fp32"):
            gpu.eval_B_grad(xyz, precision=precision)
            t0 = time.perf_counter()
            outputs[precision] = gpu.eval_B_grad(xyz, precision=precision)
            timings[f"cuda_{precision}_s"] = float(time.perf_counter() - t0)
            gpu.eval_B(xyz, precision=precision)
            t0 = time.perf_counter()
            pure_field[precision] = gpu.eval_B(xyz, precision=precision)
            timings[f"cuda_B_only_{precision}_s"] = float(time.perf_counter() - t0)
    finally:
        gpu.close()

    payload = {
        "case": str(args.case),
        "point_count": int(args.points),
        "segment_count": int(len(segment_position)),
        "segments_per_coil": int(args.segments),
        "timing": {
            "simsopt_s": float(simsopt_time),
            "cpu_segment_s": float(cpu_time),
            **timings,
        },
        "segment_vs_simsopt": {
            "B": error_metrics(cpu_B, simsopt_B),
            "grad_B": error_metrics(cpu_grad, simsopt_grad),
        },
        "cuda_fp64_vs_segment": {
            "B": error_metrics(outputs["fp64"][0], cpu_B),
            "grad_B": error_metrics(outputs["fp64"][1], cpu_grad),
        },
        "cuda_fp32_vs_fp64": {
            "B": error_metrics(outputs["fp32"][0], outputs["fp64"][0]),
            "grad_B": error_metrics(outputs["fp32"][1], outputs["fp64"][1]),
        },
        "B_only_vs_fused": {
            precision: error_metrics(pure_field[precision], outputs[precision][0])
            for precision in ("fp64", "fp32")
        },
        "divergence": {
            precision: {
                "rms": float(np.sqrt(np.mean(np.trace(outputs[precision][1], axis1=1, axis2=2) ** 2))),
                "max_abs": float(np.max(np.abs(np.trace(outputs[precision][1], axis1=1, axis2=2)))),
            }
            for precision in ("fp64", "fp32")
        },
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
