from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import AxisGAConfig, BoozerConfig, DiagnosticsConfig, EvalConfig, PsiFitConfig, SurfaceScanConfig
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
    p.add_argument("--psi-batch-size", type=int, default=20000)
    p.add_argument("--psi-validation-points", type=int, default=4000)
    p.add_argument("--psi-backend", choices=["cpu", "gpu", "fullgpu"], default="fullgpu")
    p.add_argument("--psi-linear-solver", choices=["normal_eq", "qr"], default="qr")
    p.add_argument("--psi-normal-eq-backend", choices=["auto", "cpu", "gpu"], default="auto")
    p.add_argument("--psi-normal-eq-precision", choices=["fp64", "fp32"], default="fp32")
    p.add_argument("--psi-gpu-lib", default="gpu_backend/build_mixed/libstellarator_gpu.so")
    p.add_argument("--psi-gpu-segments", type=int, default=256)
    p.add_argument("--psi-gpu-device", type=int, default=0)

    p.add_argument("--axis-max-generations", type=int, default=32)
    p.add_argument("--axis-method", choices=["fixed_point", "ga"], default="fixed_point")
    p.add_argument("--axis-rk4-steps", type=int, default=800)
    p.add_argument("--axis-tol", type=float, default=1e-7)
    p.add_argument("--axis-backend", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--axis-span", type=float, default=0.5)
    p.add_argument("--axis-gpu-lib", default="gpu_backend/build_mixed/libstellarator_gpu.so")
    p.add_argument("--axis-gpu-precision", choices=["mixed64", "fp64", "fp32"], default="mixed64")
    p.add_argument("--axis-gpu-verify-precision", choices=["mixed64", "fp64", "fp32", "none"], default="fp64")
    p.add_argument("--axis-gpu-segments", type=int, default=256)
    p.add_argument("--axis-gpu-device", type=int, default=0)
    p.add_argument("--axis-staged", action="store_true")
    p.add_argument("--axis-switch-tol", type=float, default=1e-6)
    p.add_argument("--axis-fp-grid", type=int, default=48)
    p.add_argument("--axis-fp-max-candidates", type=int, default=16)
    p.add_argument("--axis-fp-newton-iters", type=int, default=6)
    p.add_argument("--axis-fp-fallback-grid", type=int, default=96)
    p.add_argument("--axis-fp-fallback-max-candidates", type=int, default=96)
    p.add_argument("--axis-fp-fallback-newton-iters", type=int, default=8)
    p.add_argument("--axis-fp-r-floor", type=float, default=1e-4)
    p.add_argument("--axis-disable-topology-filter", action="store_true")
    p.add_argument("--axis-allow-non-elliptic", action="store_true")
    p.add_argument("--axis-topology-fd-rel", type=float, default=2e-4)
    p.add_argument("--axis-topology-fd-abs", type=float, default=2e-6)
    p.add_argument("--axis-topology-margin", type=float, default=2e-2)
    p.add_argument("--axis-residual-first", action="store_true")

    p.add_argument("--levels", default="0.001,0.002,0.004,0.008,0.012,0.02,0.04,0.08,0.12,0.16")
    p.add_argument("--screen-n-alpha", type=int, default=256)
    p.add_argument("--screen-trace-steps", type=int, default=800)
    p.add_argument("--drift-rel-tol", type=float, default=0.30)
    p.add_argument("--drift-abs-tol", type=float, default=5e-4)
    p.add_argument("--max-boozer-candidates", type=int, default=3)
    p.add_argument("--screen-trace-backend", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--screen-gpu-lib", default="gpu_backend/build_mixed/libstellarator_gpu.so")
    p.add_argument("--screen-gpu-precision", choices=["mixed64", "fp64", "fp32"], default="mixed64")
    p.add_argument("--screen-gpu-verify-precision", choices=["mixed64", "fp64", "fp32", "none"], default="fp64")
    p.add_argument("--screen-gpu-verify-candidates", type=int, default=3)
    p.add_argument("--screen-gpu-segments", type=int, default=256)
    p.add_argument("--screen-gpu-device", type=int, default=0)

    p.add_argument("--surface-order", type=int, default=6)
    p.add_argument("--surface-extract-backend", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--surface-gpu-lib", default="gpu_backend/build_mixed/libstellarator_gpu.so")
    p.add_argument("--surface-gpu-device", type=int, default=0)
    p.add_argument("--initial-iota", type=float, default=-2.0)
    p.add_argument("--ls-maxiter", type=int, default=100)
    p.add_argument("--newton-maxiter", type=int, default=30)
    p.add_argument("--qs-sdim", type=int, default=16)

    p.add_argument("--export-axis-heatmap", action="store_true", help="Export a high-resolution one-period axis residual heatmap.")
    p.add_argument("--axis-heatmap-grid", type=int, default=256)
    p.add_argument("--axis-heatmap-file", default="axis_residual_heatmap.png")
    p.add_argument("--export-psi-slices", action="store_true", help="Export fine local psi slices after psi fitting.")
    p.add_argument("--psi-slice-grid", type=int, default=241)
    p.add_argument("--psi-slice-phi-count", type=int, default=17)
    p.add_argument("--psi-slice-file", default="psi_slices.png")
    p.add_argument("--plot-dpi", type=int, default=170)
    return p


def config_from_args(args) -> EvalConfig:
    return EvalConfig(
        axis=AxisGAConfig(
            method=args.axis_method,
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
            fixed_point_grid=args.axis_fp_grid,
            fixed_point_max_candidates=args.axis_fp_max_candidates,
            fixed_point_newton_iters=args.axis_fp_newton_iters,
            fixed_point_fallback_grid=args.axis_fp_fallback_grid,
            fixed_point_fallback_max_candidates=args.axis_fp_fallback_max_candidates,
            fixed_point_fallback_newton_iters=args.axis_fp_fallback_newton_iters,
            fixed_point_r_floor=args.axis_fp_r_floor,
            fixed_point_topology_filter=not args.axis_disable_topology_filter,
            fixed_point_require_elliptic=not args.axis_allow_non_elliptic,
            fixed_point_topology_fd_rel=args.axis_topology_fd_rel,
            fixed_point_topology_fd_abs=args.axis_topology_fd_abs,
            fixed_point_topology_margin=args.axis_topology_margin,
            fixed_point_prefer_round_elliptic=not args.axis_residual_first,
        ),
        psi=PsiFitConfig(
            backend=args.psi_backend,
            linear_solver=args.psi_linear_solver,
            normal_eq_backend=args.psi_normal_eq_backend,
            normal_eq_precision=args.psi_normal_eq_precision,
            a=args.a,
            poly_degree=args.psi_poly_degree,
            m_tor=args.psi_m_tor,
            n_r=args.psi_n_r,
            n_z=args.psi_n_z,
            n_phi=args.psi_n_phi,
            batch_size=args.psi_batch_size,
            validation_points=args.psi_validation_points,
            gpu_lib_path=args.psi_gpu_lib,
            gpu_segments_per_coil=args.psi_gpu_segments,
            gpu_device=args.psi_gpu_device,
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
            surface_extract_backend=args.surface_extract_backend,
            gpu_lib_path=args.surface_gpu_lib,
            gpu_device=args.surface_gpu_device,
            initial_iota=args.initial_iota,
            ls_maxiter=args.ls_maxiter,
            newton_maxiter=args.newton_maxiter,
            qs_sdim=args.qs_sdim,
        ),
        diagnostics=DiagnosticsConfig(
            export_axis_heatmap=args.export_axis_heatmap,
            axis_heatmap_grid=args.axis_heatmap_grid,
            axis_heatmap_filename=args.axis_heatmap_file,
            export_psi_slices=args.export_psi_slices,
            psi_slice_grid=args.psi_slice_grid,
            psi_slice_phi_count=args.psi_slice_phi_count,
            psi_slice_filename=args.psi_slice_file,
            plot_dpi=args.plot_dpi,
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
