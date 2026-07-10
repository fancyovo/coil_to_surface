from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simsopt.geo import SurfaceXYZTensorFourier, ToroidalFlux

from stellarator_eval.axis import rk4_period_samples
from stellarator_eval.config import BoozerConfig, SurfaceScanConfig
from stellarator_eval.field import build_field
from stellarator_eval.serialization import write_json
from stellarator_eval.surface import level_curve_phi0

from scripts.desc_psi_volume_initial_guess_experiment import (
    boundary_layer_points,
    build_equilibrium,
    dataclass_from_dict,
    fit_axis_curve,
    force_stats,
    load_field_input,
    load_json,
    load_psi_model,
    make_xyz_surface,
    psi_layer_points,
    write_vmec_input_from_surface,
)

TWOPI = 2.0 * np.pi


def now() -> float:
    return time.perf_counter()


def _axis_angle(model, r, z, phi):
    ra, za, _, _ = model.axis_at(np.asarray(phi, dtype=float))
    return np.arctan2(z - za, r - ra)


def fieldline_joint_points(
    field,
    model,
    psi_edge: float,
    rho_values,
    cfg: SurfaceScanConfig,
    *,
    n_alpha: int,
    n_zeta: int,
    trace_steps: int,
    iota_hint: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    t0 = now()
    all_rho = []
    all_theta = []
    all_zeta = []
    all_R = []
    all_Z = []
    all_lambda_plus = []
    all_lambda_minus = []
    all_lambda_hint = []
    layer_stats = []
    for rho_desc in rho_values:
        level = float(psi_edge * rho_desc * rho_desc)
        theta0, r0, z0, radii = level_curve_phi0(model, level, n_alpha, cfg)
        t_trace = now()
        phi_hist, r_hist, z_hist, r_end, z_end = rk4_period_samples(
            field, r0, z0, model.nfp, n_zeta=n_zeta, steps=trace_steps
        )
        trace_s = now() - t_trace
        phi_grid = np.broadcast_to(phi_hist[None, :], r_hist.shape)
        theta_raw = _axis_angle(model, r_hist, z_hist, phi_grid)
        theta_node = np.mod(theta_raw, TWOPI)
        theta_end = _axis_angle(
            model,
            r_end,
            z_end,
            np.full_like(r_end, TWOPI / model.nfp),
        )
        delta = np.angle(np.exp(1j * (theta_end - theta0)))
        mean_delta = np.angle(np.mean(np.exp(1j * delta)))
        period = TWOPI / model.nfp
        rot_plus = mean_delta / period
        rot_minus = -mean_delta / period
        rot_hint = float(iota_hint) if iota_hint is not None else rot_plus

        def phase_lambda(rot):
            lam = np.angle(np.exp(1j * (theta0[:, None] + rot * phi_hist[None, :] - theta_raw)))
            lam -= float(np.mean(lam))
            return lam

        lam_plus = phase_lambda(rot_plus)
        lam_minus = phase_lambda(rot_minus)
        lam_hint = phase_lambda(rot_hint)

        all_rho.append(np.full(r_hist.size, float(rho_desc)))
        all_theta.append(theta_node.ravel())
        all_zeta.append(phi_grid.ravel())
        all_R.append(r_hist.ravel())
        all_Z.append(z_hist.ravel())
        all_lambda_plus.append(lam_plus.ravel())
        all_lambda_minus.append(lam_minus.ravel())
        all_lambda_hint.append(lam_hint.ravel())
        layer_stats.append(
            {
                "rho": float(rho_desc),
                "psi_level": level,
                "line_count": int(n_alpha),
                "zeta_count": int(n_zeta),
                "point_count": int(r_hist.size),
                "trace_time_s": float(trace_s),
                "radius_min": float(np.min(radii)),
                "radius_mean": float(np.mean(radii)),
                "radius_max": float(np.max(radii)),
                "mean_delta_rad": float(mean_delta),
                "iota_geom_plus": float(rot_plus),
                "iota_geom_minus": float(rot_minus),
                "iota_hint": float(rot_hint),
                "delta_std_rad": float(np.std(np.angle(np.exp(1j * (delta - mean_delta))))),
                "lambda_plus_rms": float(np.sqrt(np.mean(lam_plus**2))),
                "lambda_minus_rms": float(np.sqrt(np.mean(lam_minus**2))),
                "lambda_hint_rms": float(np.sqrt(np.mean(lam_hint**2))),
            }
        )

    nodes = np.column_stack(
        [
            np.concatenate(all_rho),
            np.concatenate(all_theta),
            np.concatenate(all_zeta),
        ]
    )
    return (
        nodes,
        np.concatenate(all_R),
        np.concatenate(all_Z),
        np.concatenate(all_lambda_plus),
        np.concatenate(all_lambda_minus),
        np.concatenate(all_lambda_hint),
        {"time_s": float(now() - t0), "layers": layer_stats},
    )


def append_boundary(nodes, R, Z, lam, surface, nfp: int, nphi: int, ntheta: int, *, lambda_value: float = 0.0):
    nodes_b, R_b, Z_b = boundary_layer_points(surface, 1.0, nfp, nphi, ntheta)
    lam_b = np.full_like(R_b, float(lambda_value), dtype=float)
    return (
        np.vstack([nodes, nodes_b]),
        np.concatenate([R, R_b]),
        np.concatenate([Z, Z_b]),
        np.concatenate([lam, lam_b]),
    )


def boundary_mismatch(eq, input_surface, nfp: int, *, ntheta: int = 64, nzeta: int = 32) -> dict:
    from desc.grid import Grid

    theta = np.linspace(0.0, TWOPI, ntheta, endpoint=False)
    zeta = np.linspace(0.0, TWOPI / nfp, nzeta, endpoint=False)
    tg, zg = np.meshgrid(theta, zeta, indexing="ij")
    nodes = np.column_stack(
        [np.ones(tg.size), tg.ravel(), zg.ravel()]
    )
    grid = Grid(nodes, NFP=nfp, sort=False)
    ref = input_surface.compute(["R", "Z"], grid=grid)
    bnd = eq.surface.compute(["R", "Z"], grid=grid)
    body = eq.compute(["R", "Z"], grid=grid)

    def stats(prefix, data):
        dr = np.asarray(data["R"]) - np.asarray(ref["R"])
        dz = np.asarray(data["Z"]) - np.asarray(ref["Z"])
        d = np.sqrt(dr * dr + dz * dz)
        return {
            f"{prefix}_rms": float(np.sqrt(np.mean(d * d))),
            f"{prefix}_max": float(np.max(d)),
        }

    out = {}
    out.update(stats("surface_boundary_mismatch", bnd))
    out.update(stats("body_rho1_mismatch", body))
    return out


def run_initial_variant(
    *,
    name: str,
    input_path: Path,
    psi: float,
    L: int,
    M: int,
    N: int,
    axis_curve,
    point_guess,
    nfp: int,
    run_solve: bool,
    solve_maxiter: int | None,
    solve_ftol: float | None,
    out_dir: Path,
) -> dict:
    from desc.geometry import FourierRZToroidalSurface
    from desc.grid import Grid

    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"variant": name}
    desc_surface = FourierRZToroidalSurface.from_input_file(str(input_path))
    t0 = now()
    eq = build_equilibrium(
        desc_surface,
        psi,
        L,
        M,
        N,
        axis_curve=axis_curve,
        constructor_ensure_nested=False,
    )
    result["desc_construct_time_s"] = float(now() - t0)
    result["nested_after_construct"] = bool(eq.is_nested())
    result.update(boundary_mismatch(eq, desc_surface, nfp))

    nodes, R, Z, lam = point_guess
    t0 = now()
    grid = Grid(nodes, NFP=nfp, sort=False)
    eq.set_initial_guess(grid, R, Z, lam, ensure_nested=False)
    result["desc_set_initial_guess_time_s"] = float(now() - t0)
    result["point_guess_count"] = int(len(R))
    result["lambda_rms_target"] = float(np.sqrt(np.mean(np.asarray(lam) ** 2)))
    result["lambda_max_abs_target"] = float(np.max(np.abs(lam)))
    result["nested_after_point_guess"] = bool(eq.is_nested())
    result.update(boundary_mismatch(eq, desc_surface, nfp))
    result.update(force_stats(eq, "initial"))

    if run_solve:
        solve_kwargs: dict[str, Any] = {"verbose": 1}
        if solve_maxiter is not None:
            solve_kwargs["maxiter"] = int(solve_maxiter)
        if solve_ftol is not None:
            solve_kwargs["ftol"] = float(solve_ftol)
        result["solve_kwargs"] = solve_kwargs
        t0 = now()
        try:
            eq_solved, opt = eq.solve(**solve_kwargs)
            result["desc_solve_call_success"] = True
            result["desc_optimizer_success"] = bool(getattr(opt, "success", False))
            result["desc_optimizer_message"] = str(getattr(opt, "message", ""))
            for attr in ("cost", "nit", "nfev", "njev", "optimality"):
                if hasattr(opt, attr):
                    result[f"desc_optimizer_{attr}"] = float(getattr(opt, attr))
            if hasattr(opt, "fun"):
                fun = np.abs(np.asarray(opt.fun, dtype=float).ravel())
                fun = fun[np.isfinite(fun)]
                result["desc_optimizer_fun_mean_abs"] = float(np.mean(fun)) if fun.size else float("nan")
                result["desc_optimizer_fun_p95_abs"] = float(np.percentile(fun, 95)) if fun.size else float("nan")
                result["desc_optimizer_fun_max_abs"] = float(np.max(fun)) if fun.size else float("nan")
            eq = eq_solved
        except Exception as exc:
            result["desc_solve_call_success"] = False
            result["desc_solve_error"] = repr(exc)
        result["desc_solve_time_s"] = float(now() - t0)
        result["nested_after_solve"] = bool(eq.is_nested())
        result.update(force_stats(eq, "final"))
    try:
        eq.save(str(out_dir / "equilibrium.h5"))
        result["equilibrium_path"] = str(out_dir / "equilibrium.h5")
    except Exception as exc:
        result["equilibrium_save_error"] = repr(exc)
    write_json(out_dir / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Test joint R/Z/lambda/iota-style DESC initial guesses from field-line geometry.")
    parser.add_argument("--case-file", required=True)
    parser.add_argument("--key", default="raw")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--level-dir", default=None)
    parser.add_argument("--surface-npz", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--current-unit", default=None)
    parser.add_argument("--surface-order", type=int, default=None)
    parser.add_argument("--surface-nphi", type=int, default=96)
    parser.add_argument("--surface-ntheta", type=int, default=144)
    parser.add_argument("--axis-order", type=int, default=8)
    parser.add_argument("--desc-L", type=int, default=6)
    parser.add_argument("--desc-M", type=int, default=6)
    parser.add_argument("--desc-N", type=int, default=6)
    parser.add_argument("--psi-layer-order", type=int, default=8)
    parser.add_argument("--psi-backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--psi-rho-min", type=float, default=0.18)
    parser.add_argument("--psi-rho-max", type=float, default=0.88)
    parser.add_argument("--psi-layers", type=int, default=4)
    parser.add_argument("--trace-alpha", type=int, default=12)
    parser.add_argument("--trace-zeta", type=int, default=9)
    parser.add_argument("--trace-steps", type=int, default=360)
    parser.add_argument("--append-boundary-points", action="store_true")
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--gpu-lib-path", default="gpu_backend/build_mixed/libstellarator_gpu.so")
    parser.add_argument("--run-solve", action="store_true")
    parser.add_argument("--solve-maxiter", type=int, default=20)
    parser.add_argument("--solve-ftol", type=float, default=1e-2)
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_device))
    total0 = now()
    case_file = Path(args.case_file)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = load_json(run_dir / "summary.json")
    config_dict = summary.get("config", {})
    nfp = int(summary["nfp"])
    current_unit = args.current_unit or config_dict.get("current_unit") or "A"
    surface_order = args.surface_order or int(config_dict.get("boozer", {}).get("surface_order", 6))
    stellsym = bool(config_dict.get("boozer", {}).get("stellsym", True))
    scan_cfg = dataclass_from_dict(SurfaceScanConfig, config_dict.get("scan"))
    boozer_cfg = dataclass_from_dict(BoozerConfig, config_dict.get("boozer"))
    scan_cfg = replace(scan_cfg, gpu_lib_path=args.gpu_lib_path, gpu_device=args.gpu_device)
    boozer_cfg = replace(boozer_cfg, gpu_lib_path=args.gpu_lib_path, gpu_device=args.gpu_device)

    if args.surface_npz:
        surface_npz = Path(args.surface_npz)
        level_dir = surface_npz.parent
    elif args.level_dir:
        level_dir = Path(args.level_dir)
        surface_npz = level_dir / "boozer_surface.npz"
    else:
        best = summary.get("best_surface") or {}
        level = float(best["psi_level"])
        level_dir = run_dir / f"level_{level:.6g}".replace(".", "p")
        surface_npz = level_dir / "boozer_surface.npz"

    timings: dict[str, float] = {}
    t0 = now()
    field_input = load_field_input(case_file, args.key)
    built = build_field(field_input, current_unit=current_unit)
    timings["field_build_s"] = float(now() - t0)

    t0 = now()
    surface, surface_meta = make_xyz_surface(
        surface_npz,
        nfp=nfp,
        order=surface_order,
        stellsym=stellsym,
        nphi=args.surface_nphi,
        ntheta=args.surface_ntheta,
    )
    timings["simsopt_surface_reconstruct_s"] = float(now() - t0)

    t0 = now()
    toroidal_flux = float(ToroidalFlux(surface, built.field).J())
    timings["toroidal_flux_s"] = float(now() - t0)
    input_path = out_dir / "boundary_input.check"
    t0 = now()
    input_info = write_vmec_input_from_surface(surface, toroidal_flux, input_path)
    timings["write_boundary_input_s"] = float(now() - t0)

    t0 = now()
    axis_curve, axis_info = fit_axis_curve(run_dir / "axis_data.npz", nfp=nfp, axis_order=args.axis_order)
    timings["axis_curve_fit_s"] = float(now() - t0)

    t0 = now()
    model = load_psi_model(run_dir / "psi_model.npz")
    timings["psi_model_load_s"] = float(now() - t0)
    psi_edge = float(surface_meta.get("psi_level") or 0.0)
    if psi_edge <= 0:
        raise ValueError("surface npz does not contain a positive psi_level")

    rho_values = np.linspace(args.psi_rho_min, args.psi_rho_max, args.psi_layers)
    nphi_layer = 2 * args.psi_layer_order + 1
    ntheta_layer = 2 * args.psi_layer_order + 1

    t0 = now()
    nodes_psi, R_psi, Z_psi, psi_extract_info = psi_layer_points(
        model, psi_edge, rho_values, args.psi_layer_order, scan_cfg, boozer_cfg, backend=args.psi_backend
    )
    lam0_psi = np.zeros_like(R_psi)
    if args.append_boundary_points:
        psi_guess = append_boundary(nodes_psi, R_psi, Z_psi, lam0_psi, surface, nfp, nphi_layer, ntheta_layer)
    else:
        psi_guess = (nodes_psi, R_psi, Z_psi, lam0_psi)
    timings["psi_point_guess_s"] = float(now() - t0)

    t0 = now()
    iota_hint = surface_meta.get("iota")
    t0 = now()
    nodes_tr, R_tr, Z_tr, lam_plus, lam_minus, lam_hint, trace_info = fieldline_joint_points(
        built.field,
        model,
        psi_edge,
        rho_values,
        scan_cfg,
        n_alpha=args.trace_alpha,
        n_zeta=args.trace_zeta,
        trace_steps=args.trace_steps,
        iota_hint=float(iota_hint) if iota_hint is not None else None,
    )
    if args.append_boundary_points:
        trace_guess_zero = append_boundary(nodes_tr, R_tr, Z_tr, np.zeros_like(R_tr), surface, nfp, nphi_layer, ntheta_layer)
        trace_guess_plus = append_boundary(nodes_tr, R_tr, Z_tr, lam_plus, surface, nfp, nphi_layer, ntheta_layer)
        trace_guess_minus = append_boundary(nodes_tr, R_tr, Z_tr, lam_minus, surface, nfp, nphi_layer, ntheta_layer)
        trace_guess_hint = append_boundary(nodes_tr, R_tr, Z_tr, lam_hint, surface, nfp, nphi_layer, ntheta_layer)
    else:
        trace_guess_zero = (nodes_tr, R_tr, Z_tr, np.zeros_like(R_tr))
        trace_guess_plus = (nodes_tr, R_tr, Z_tr, lam_plus)
        trace_guess_minus = (nodes_tr, R_tr, Z_tr, lam_minus)
        trace_guess_hint = (nodes_tr, R_tr, Z_tr, lam_hint)
    timings["fieldline_joint_points_s"] = float(now() - t0)

    variants = {
        "psi_RZ_lambda0": psi_guess,
        "trace_RZ_lambda0": trace_guess_zero,
        "trace_RZ_lambda_sfl_plus": trace_guess_plus,
        "trace_RZ_lambda_sfl_minus": trace_guess_minus,
        "trace_RZ_lambda_iota_hint": trace_guess_hint,
    }
    variant_results = []
    for name, guess in variants.items():
        print(f"[variant] {name}", flush=True)
        variant_results.append(
            run_initial_variant(
                name=name,
                input_path=input_path,
                psi=toroidal_flux,
                L=args.desc_L,
                M=args.desc_M,
                N=args.desc_N,
                axis_curve=axis_curve,
                point_guess=guess,
                nfp=nfp,
                run_solve=args.run_solve,
                solve_maxiter=args.solve_maxiter,
                solve_ftol=args.solve_ftol,
                out_dir=out_dir / name,
            )
        )

    result = {
        "case_file": str(case_file),
        "run_dir": str(run_dir),
        "level_dir": str(level_dir),
        "surface_npz": str(surface_npz),
        "output_dir": str(out_dir),
        "nfp": nfp,
        "current_unit": current_unit,
        "surface_order": int(surface_order),
        "desc_resolution": {"L": args.desc_L, "M": args.desc_M, "N": args.desc_N},
        "surface_meta": surface_meta,
        "toroidal_flux": toroidal_flux,
        "boundary_input": input_info,
        "axis_fit": axis_info,
        "rho_values": [float(x) for x in rho_values],
        "psi_extract": psi_extract_info,
        "fieldline_trace": trace_info,
        "append_boundary_points": bool(args.append_boundary_points),
        "timings": timings,
        "variants": variant_results,
        "total_time_s": float(now() - total0),
    }
    write_json(out_dir / "summary.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
