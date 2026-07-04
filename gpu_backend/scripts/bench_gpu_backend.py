from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "python"))

from stellarator_gpu import CoilFieldGpu, eval_B_segments_cpu, load_case, make_segments_cpu


def relerr(a, b):
    num = np.linalg.norm(a - b, axis=-1)
    den = np.maximum(np.linalg.norm(b, axis=-1), 1e-300)
    return num / den


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lib", default="build/libstellarator_gpu.so")
    p.add_argument("--case-file", default="../examples/01.json")
    p.add_argument("--key", default="raw")
    p.add_argument("--segments", type=int, default=256)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--points", type=int, default=4096)
    p.add_argument("--lines", type=int, default=256)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--output", default="bench_result.json")
    args = p.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.device))
    cx, cy, cz, currents, nfp = load_case(args.case_file, args.key)

    t0 = time.perf_counter()
    field = CoilFieldGpu(args.lib, cx, cy, cz, currents, nfp=nfp, segments_per_coil=args.segments, device_id=0)
    create_time = time.perf_counter() - t0
    print(f"segments={field.segment_count} create_time={create_time:.4f}s")

    rng = np.random.default_rng(20260705)
    R = rng.uniform(0.8, 1.5, args.points)
    phi = rng.uniform(0.0, 2.0 * np.pi / nfp, args.points)
    Z = rng.uniform(-0.25, 0.25, args.points)
    xyz = np.column_stack([R * np.cos(phi), R * np.sin(phi), Z])

    seg_pos, seg_wdl = make_segments_cpu(cx, cy, cz, currents, nfp, args.segments)
    ref_n = min(args.points, 256)
    t0 = time.perf_counter()
    B_cpu = eval_B_segments_cpu(xyz[:ref_n], seg_pos, seg_wdl)
    cpu_ref_time = time.perf_counter() - t0
    B_gpu = field.eval_B(xyz[:ref_n])
    e = relerr(B_gpu, B_cpu)
    print(f"B correctness mean={e.mean():.3e} p95={np.percentile(e,95):.3e} max={e.max():.3e}")

    # Warm up.
    field.eval_B(xyz)
    eval_times = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        field.eval_B(xyz)
        eval_times.append(time.perf_counter() - t0)
    print(f"eval_B n={args.points} median={np.median(eval_times):.6f}s")

    R0 = rng.uniform(1.0, 1.3, args.lines)
    Z0 = rng.uniform(-0.05, 0.05, args.lines)
    field.trace_period(R0, Z0, args.steps, nfp=nfp)
    trace_times = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        R1, Z1 = field.trace_period(R0, Z0, args.steps, nfp=nfp)
        trace_times.append(time.perf_counter() - t0)
    disp = np.sqrt((R1 - R0) ** 2 + (Z1 - Z0) ** 2)
    print(f"trace n={args.lines} steps={args.steps} median={np.median(trace_times):.6f}s disp_min={disp.min():.3e}")

    result = {
        "case_file": args.case_file,
        "key": args.key,
        "nfp": nfp,
        "segments_per_coil": args.segments,
        "segment_count": field.segment_count,
        "create_time_s": create_time,
        "B_correctness_vs_cpu_segments": {
            "ref_points": ref_n,
            "cpu_ref_time_s": cpu_ref_time,
            "mean_rel": float(e.mean()),
            "p95_rel": float(np.percentile(e, 95)),
            "max_rel": float(e.max()),
        },
        "eval_B": {
            "points": args.points,
            "times_s": eval_times,
            "median_s": float(np.median(eval_times)),
            "points_per_s": float(args.points / np.median(eval_times)),
        },
        "trace_period": {
            "lines": args.lines,
            "steps": args.steps,
            "times_s": trace_times,
            "median_s": float(np.median(trace_times)),
            "line_steps_per_s": float(args.lines * args.steps / np.median(trace_times)),
            "disp_min": float(disp.min()),
            "disp_median": float(np.median(disp)),
        },
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    field.close()


if __name__ == "__main__":
    main()
