from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import AxisGAConfig, BoozerConfig, EvalConfig, PsiFitConfig, SurfaceScanConfig
from .pipeline import evaluate_case_file
from .serialization import jsonable


def parse_levels(text: str):
    return tuple(float(x) for x in text.replace(",", " ").split() if x.strip())


def build_parser():
    p = argparse.ArgumentParser(description="Evaluate local stellarator magnetic surfaces from Fourier coil data.")
    p.add_argument("--case-file", default="examples/01.json")
    p.add_argument("--key", default="raw")
    p.add_argument("--output-dir", default="runs/01_raw")
    p.add_argument("--current-unit", choices=["MA", "A"], default="MA")
    p.add_argument("--omp-threads", type=int, default=1)

    p.add_argument("--a", type=float, default=0.05)
    p.add_argument("--psi-poly-degree", type=int, default=10)
    p.add_argument("--psi-m-tor", type=int, default=12)
    p.add_argument("--psi-n-r", type=int, default=80)
    p.add_argument("--psi-n-z", type=int, default=80)
    p.add_argument("--psi-n-phi", type=int, default=80)
    p.add_argument("--psi-validation-points", type=int, default=4000)

    p.add_argument("--axis-max-generations", type=int, default=32)
    p.add_argument("--axis-rk4-steps", type=int, default=800)
    p.add_argument("--axis-tol", type=float, default=1e-8)
    p.add_argument("--axis-backend", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--axis-span", type=float, default=0.5)
    p.add_argument("--axis-gpu-lib", default="gpu_backend/build_mixed/libstellarator_gpu.so")
    p.add_argument("--axis-gpu-precision", choices=["mixed64", "fp64", "fp32"], default="mixed64")
    p.add_argument("--axis-gpu-verify-precision", choices=["mixed64", "fp64", "fp32", "none"], default="fp64")
    p.add_argument("--axis-gpu-segments", type=int, default=256)
    p.add_argument("--axis-gpu-device", type=int, default=0)
    p.add_argument("--axis-staged", action="store_true")
    p.add_argument("--axis-switch-tol", type=float, default=1e-6)

    p.add_argument("--levels", default="0.001,0.002,0.004,0.008,0.012,0.02,0.04,0.08,0.12,0.16")
    p.add_argument("--screen-n-alpha", type=int, default=256)
    p.add_argument("--screen-trace-steps", type=int, default=800)
    p.add_argument("--drift-rel-tol", type=float, default=0.30)
    p.add_argument("--drift-abs-tol", type=float, default=5e-4)
    p.add_argument("--max-boozer-candidates", type=int, default=3)
    p.add_argument("--screen-trace-backend", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--screen-gpu-lib", default="gpu_backend/build_mixed/libstellarator_gpu.so")
    p.add_argument("--screen-gpu-precision", choices=["mixed64", "fp64", "fp32"], default="mixed64")
    p.add_argument("--screen-gpu-verify-precision", choices=["mixed64", "fp64", "fp32", "none"], default="fp64")
    p.add_argument("--screen-gpu-verify-candidates", type=int, default=3)
    p.add_argument("--screen-gpu-segments", type=int, default=256)
    p.add_argument("--screen-gpu-device", type=int, default=0)

    p.add_argument("--surface-order", type=int, default=6)
    p.add_argument("--initial-iota", type=float, default=-2.0)
    p.add_argument("--ls-maxiter", type=int, default=100)
    p.add_argument("--newton-maxiter", type=int, default=30)
    p.add_argument("--qs-sdim", type=int, default=16)
    return p


def config_from_args(args) -> EvalConfig:
    return EvalConfig(
        axis=AxisGAConfig(
            backend=args.axis_backend,
            span=args.axis_span,
            rk4_steps=args.axis_rk4_steps,
            max_generations=args.axis_max_generations,
            tol=args.axis_tol,
            gpu_lib_path=args.axis_gpu_lib,
            gpu_trace_precision=args.axis_gpu_precision,
            gpu_verify_precision="" if args.axis_gpu_verify_precision == "none" else args.axis_gpu_verify_precision,
            gpu_segments_per_coil=args.axis_gpu_segments,
            gpu_device=args.axis_gpu_device,
            staged=args.axis_staged,
            switch_tol=args.axis_switch_tol,
        ),
        psi=PsiFitConfig(
            a=args.a,
            poly_degree=args.psi_poly_degree,
            m_tor=args.psi_m_tor,
            n_r=args.psi_n_r,
            n_z=args.psi_n_z,
            n_phi=args.psi_n_phi,
            validation_points=args.psi_validation_points,
        ),
        scan=SurfaceScanConfig(
            levels=parse_levels(args.levels),
            n_alpha=args.screen_n_alpha,
            trace_steps=args.screen_trace_steps,
            drift_rel_tol=args.drift_rel_tol,
            drift_abs_tol=args.drift_abs_tol,
            max_boozer_candidates=args.max_boozer_candidates,
            trace_backend=args.screen_trace_backend,
            gpu_lib_path=args.screen_gpu_lib,
            gpu_trace_precision=args.screen_gpu_precision,
            gpu_verify_precision="" if args.screen_gpu_verify_precision == "none" else args.screen_gpu_verify_precision,
            gpu_verify_candidates=args.screen_gpu_verify_candidates,
            gpu_segments_per_coil=args.screen_gpu_segments,
            gpu_device=args.screen_gpu_device,
        ),
        boozer=BoozerConfig(
            surface_order=args.surface_order,
            initial_iota=args.initial_iota,
            ls_maxiter=args.ls_maxiter,
            newton_maxiter=args.newton_maxiter,
            qs_sdim=args.qs_sdim,
        ),
        current_unit=args.current_unit,
        omp_threads=args.omp_threads,
    )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    result = evaluate_case_file(args.case_file, key=args.key, config=config_from_args(args), output_dir=Path(args.output_dir))
    best = result.get("best_surface")
    print(f"summary: {Path(args.output_dir) / 'summary.json'}")
    print(f"axis residual: {result['axis']['best_residual']:.3e}, has_axis={result['axis']['has_axis']}")
    if best:
        print(
            "best surface: "
            f"psi={best['psi_level']:.6g}, iota={best['iota']:.8g}, "
            f"volume={best['volume']:.6g}, G={best['G']:.6g}"
        )
    else:
        print("best surface: none")


if __name__ == "__main__":
    main()
