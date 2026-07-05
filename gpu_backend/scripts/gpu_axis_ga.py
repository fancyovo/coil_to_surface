from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from stellarator_gpu import CoilFieldGpu, load_case


def initial_grid(rc, zc, span, grid):
    rs = np.linspace(rc - span, rc + span, grid)
    zs = np.linspace(zc - span, zc + span, grid)
    rg, zg = np.meshgrid(rs, zs, indexing="xy")
    return rg.ravel(), zg.ravel()


def pairwise_midpoints(r, z):
    return (0.5 * (r[:, None] + r[None, :])).ravel(), (0.5 * (z[:, None] + z[None, :])).ravel()


def unique_points(r, z, decimals=14):
    pts = np.column_stack([r, z])
    _, idx = np.unique(np.round(pts, decimals=decimals), axis=0, return_index=True)
    idx = np.sort(idx)
    return r[idx], z[idx]


def estimate_coil_r0(cx, cy, cz, samples=512):
    t = np.linspace(0.0, 1.0, samples, endpoint=False)
    vals = []
    for xcoef, ycoef in zip(cx, cy):
        order = (len(xcoef) - 1) // 2
        x = np.full(samples, xcoef[0])
        y = np.full(samples, ycoef[0])
        for m in range(1, order + 1):
            s = np.sin(2 * np.pi * m * t)
            c = np.cos(2 * np.pi * m * t)
            x += xcoef[2 * m - 1] * s + xcoef[2 * m] * c
            y += ycoef[2 * m - 1] * s + ycoef[2 * m] * c
        vals.append(np.mean(np.sqrt(x * x + y * y)))
    return float(np.mean(vals))


def run_ga(field, r_center, nfp, args):
    r, z = initial_grid(r_center, args.z_center, args.span, args.grid)
    history = []
    best = None
    for gen in range(args.max_generations + 1):
        t0 = time.perf_counter()
        if args.trace_mode == "auto":
            re, ze = field.trace_period_blockline_precision(
                r,
                z,
                steps=args.steps,
                precision=args.trace_precision,
                threads_per_line=args.blockline_threads,
                nfp=nfp,
            )
        elif args.trace_mode == "warp":
            re, ze = field.trace_period(r, z, steps=args.steps, nfp=nfp)
        elif args.trace_mode == "blockline":
            re, ze = field.trace_period_blockline(
                r, z, steps=args.steps, threads_per_line=args.blockline_threads, nfp=nfp
            )
        elif args.trace_mode == "blockline_mixed64":
            re, ze = field.trace_period_blockline_mixed(
                r, z, steps=args.steps, threads_per_line=args.blockline_threads, mode="bf32_state64", nfp=nfp
            )
        elif args.trace_mode == "blockline_f32":
            re, ze = field.trace_period_blockline_mixed(
                r, z, steps=args.steps, threads_per_line=args.blockline_threads, mode="f32", nfp=nfp
            )
        elif args.trace_mode == "blockline_f16state":
            re, ze = field.trace_period_blockline_mixed(
                r, z, steps=args.steps, threads_per_line=args.blockline_threads, mode="f32_state16", nfp=nfp
            )
        else:
            raise ValueError(f"unknown trace_mode={args.trace_mode}")
        dt = time.perf_counter() - t0
        residual = np.sqrt((re - r) ** 2 + (ze - z) ** 2)
        order = np.argsort(residual)
        top = order[: args.keep]
        row = {
            "generation": gen,
            "population": int(len(r)),
            "best_R": float(r[order[0]]),
            "best_Z": float(z[order[0]]),
            "best_residual": float(residual[order[0]]),
            "top_median_residual": float(np.median(residual[top])),
            "top_max_residual": float(np.max(residual[top])),
            "top_R_span": float(np.max(r[top]) - np.min(r[top])),
            "top_Z_span": float(np.max(z[top]) - np.min(z[top])),
            "trace_time_s": dt,
        }
        history.append(row)
        best = row
        if row["best_residual"] <= args.tol or gen == args.max_generations:
            break
        r, z = pairwise_midpoints(r[top], z[top])
        r, z = unique_points(r, z)
        target = args.grid * args.grid
        if len(r) < target:
            local_span = max(row["top_R_span"], row["top_Z_span"], args.span * 2 ** (-(gen + 4)), 1e-10)
            rg, zg = initial_grid(row["best_R"], row["best_Z"], local_span, args.grid)
            r, z = unique_points(np.r_[r, rg], np.r_[z, zg])
        if len(r) > target:
            r = r[:target]
            z = z[:target]
    return best, history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lib", default="build/libstellarator_gpu.so")
    p.add_argument("--case-file", default="../examples/01.json")
    p.add_argument("--key", default="raw")
    p.add_argument("--segments", type=int, default=256)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--grid", type=int, default=16)
    p.add_argument("--keep", type=int, default=16)
    p.add_argument("--span", type=float, default=0.5)
    p.add_argument("--z-center", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--max-generations", type=int, default=32)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument(
        "--trace-mode",
        choices=["auto", "warp", "blockline", "blockline_mixed64", "blockline_f32", "blockline_f16state"],
        default="auto",
        help="legacy kernel selector; default auto uses --trace-precision",
    )
    p.add_argument("--trace-precision", choices=["mixed64", "fp64", "fp32"], default="mixed64")
    p.add_argument("--blockline-threads", type=int, default=256)
    p.add_argument("--output", default="gpu_axis_ga.json")
    args = p.parse_args()

    cx, cy, cz, currents, nfp = load_case(args.case_file, args.key)
    r0 = estimate_coil_r0(cx, cy, cz)
    t0 = time.perf_counter()
    field = CoilFieldGpu(args.lib, cx, cy, cz, currents, nfp=nfp, segments_per_coil=args.segments, device_id=args.device)
    create_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    best, history = run_ga(field, r0, nfp, args)
    total = time.perf_counter() - t0
    result = {
        "case_file": args.case_file,
        "key": args.key,
        "nfp": nfp,
        "segments_per_coil": args.segments,
        "segment_count": field.segment_count,
        "trace_mode": args.trace_mode,
        "trace_precision": args.trace_precision if args.trace_mode == "auto" else None,
        "effective_trace": args.trace_precision if args.trace_mode == "auto" else args.trace_mode,
        "blockline_threads": args.blockline_threads if args.trace_mode != "warp" else None,
        "create_time_s": create_time,
        "ga_time_s": total,
        "converged": bool(best["best_residual"] <= args.tol),
        "best": best,
        "history": history,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    field.close()


if __name__ == "__main__":
    main()
