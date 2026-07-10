from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import replace
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

from simsopt.geo import ToroidalFlux

from stellarator_eval.config import BoozerConfig, SurfaceScanConfig
from stellarator_eval.desc_joint_ls import (
    JointSpectralFitConfig,
    PhaseConstraintSet,
    WeightedSampleSet,
    apply_joint_fit,
    fit_joint_rzl_data,
)
from stellarator_eval.serialization import write_json
from stellarator_eval.surface import fit_xyz_tensor_surface, level_curve_phi0

from scripts.desc_joint_rzl_initial_guess_experiment import (
    _axis_angle,
    append_boundary,
    boundary_mismatch,
    build_equilibrium,
    dataclass_from_dict,
    fieldline_joint_points,
    fit_axis_curve,
    force_stats,
    load_field_input,
    load_json,
    load_psi_model,
    make_xyz_surface,
    now,
    psi_layer_points,
    rk4_period_samples,
    write_vmec_input_from_surface,
)
from stellarator_eval.field import build_field

TWOPI = 2.0 * np.pi


def axis_sample_points(model, *, nfp: int, nzeta: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zeta = np.linspace(0.0, TWOPI / nfp, int(nzeta), endpoint=False)
    R_axis, Z_axis, _, _ = model.axis_at(zeta)
    nodes = np.column_stack([np.zeros_like(zeta), np.zeros_like(zeta), zeta])
    return nodes, np.asarray(R_axis, dtype=float), np.asarray(Z_axis, dtype=float)


def _dataset(nodes, values, weight: float, label: str) -> WeightedSampleSet:
    return WeightedSampleSet(
        nodes=np.asarray(nodes, dtype=float),
        values=np.asarray(values, dtype=float),
        weight=float(weight),
        label=label,
    )


def _parse_int_list(text: str) -> tuple[int, ...]:
    items = [item.strip() for item in str(text).split(",") if item.strip()]
    if not items:
        raise ValueError("expected at least one integer")
    return tuple(int(item) for item in items)


def fieldline_phase_constraint_data(
    field,
    model,
    psi_edge: float,
    rho_values,
    cfg: SurfaceScanConfig,
    *,
    n_alpha: int,
    n_zeta: int,
    trace_steps: int,
    weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, PhaseConstraintSet, dict]:
    t0 = now()
    all_rho = []
    all_theta_node = []
    all_zeta = []
    all_R = []
    all_Z = []
    all_theta_unwrapped = []
    all_beta_ids = []
    beta_groups = []
    layer_stats = []
    beta_offset = 0
    for layer_idx, rho_desc in enumerate(rho_values):
        level = float(psi_edge * rho_desc * rho_desc)
        theta0, r0, z0, radii = level_curve_phi0(model, level, n_alpha, cfg)
        t_trace = now()
        phi_hist, r_hist, z_hist, _, _ = rk4_period_samples(
            field, r0, z0, model.nfp, n_zeta=n_zeta, steps=trace_steps
        )
        trace_s = now() - t_trace
        phi_grid = np.broadcast_to(phi_hist[None, :], r_hist.shape)
        theta_raw = _axis_angle(model, r_hist, z_hist, phi_grid)
        theta_unwrapped = np.unwrap(theta_raw, axis=1)
        theta_node = np.mod(theta_raw, TWOPI)
        beta_local = np.broadcast_to(np.arange(n_alpha, dtype=int)[:, None], r_hist.shape).ravel()

        all_rho.append(np.full(r_hist.size, float(rho_desc)))
        all_theta_node.append(theta_node.ravel())
        all_zeta.append(phi_grid.ravel())
        all_R.append(r_hist.ravel())
        all_Z.append(z_hist.ravel())
        all_theta_unwrapped.append(theta_unwrapped.ravel())
        all_beta_ids.append(beta_local + beta_offset)
        beta_groups.extend([layer_idx] * int(n_alpha))
        beta_offset += int(n_alpha)
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
                "theta0_mean": float(np.mean(theta0)),
                "theta_unwrapped_rms": float(np.sqrt(np.mean(theta_unwrapped**2))),
            }
        )

    nodes = np.column_stack(
        [
            np.concatenate(all_rho),
            np.concatenate(all_theta_node),
            np.concatenate(all_zeta),
        ]
    )
    phase = PhaseConstraintSet(
        nodes=nodes,
        theta=np.concatenate(all_theta_unwrapped),
        beta_ids=np.concatenate(all_beta_ids),
        beta_groups=np.asarray(beta_groups, dtype=int),
        weight=float(weight),
        label="trace_phase_constraints",
    )
    return (
        nodes,
        np.concatenate(all_R),
        np.concatenate(all_Z),
        phase,
        {
            "time_s": float(now() - t0),
            "layers": layer_stats,
            "beta_count": int(len(beta_groups)),
            "beta_group_count": int(len(rho_values)),
        },
    )


def _flip_phase_theta(phase: PhaseConstraintSet, label: str) -> PhaseConstraintSet:
    return PhaseConstraintSet(
        nodes=np.asarray(phase.nodes, dtype=float),
        theta=-np.asarray(phase.theta, dtype=float),
        beta_ids=np.asarray(phase.beta_ids, dtype=int),
        beta_groups=None if phase.beta_groups is None else np.asarray(phase.beta_groups, dtype=int),
        weight=phase.weight,
        label=label,
    )


def _variant_specs(
    *,
    nodes_psi,
    R_psi,
    Z_psi,
    nodes_tr,
    R_tr,
    Z_tr,
    lam_plus,
    lam_minus,
    lam_hint,
) -> dict[str, dict[str, Any]]:
    return {
        "psi_joint_lambda0": {"nodes": nodes_psi, "R": R_psi, "Z": Z_psi, "lambda": np.zeros_like(R_psi)},
        "trace_joint_lambda0": {"nodes": nodes_tr, "R": R_tr, "Z": Z_tr, "lambda": np.zeros_like(R_tr)},
        "trace_joint_lambda_sfl_plus": {"nodes": nodes_tr, "R": R_tr, "Z": Z_tr, "lambda": lam_plus},
        "trace_joint_lambda_sfl_minus": {"nodes": nodes_tr, "R": R_tr, "Z": Z_tr, "lambda": lam_minus},
        "trace_joint_lambda_iota_hint": {"nodes": nodes_tr, "R": R_tr, "Z": Z_tr, "lambda": lam_hint},
    }


def run_external_variant(
    *,
    name: str,
    input_path: Path,
    psi: float,
    L: int,
    M: int,
    N: int,
    axis_curve,
    nfp: int,
    model,
    point_guess: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None],
    phase_constraints: PhaseConstraintSet | None,
    surface,
    boundary_source: str,
    fit_cfg: JointSpectralFitConfig,
    interior_weight: float,
    lambda_weight: float,
    boundary_weight: float,
    axis_weight: float,
    axis_zeta_count: int,
    run_solve: bool,
    solve_maxiter: int | None,
    solve_ftol: float | None,
    out_dir: Path,
) -> dict:
    from desc.geometry import FourierRZToroidalSurface

    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"variant": name}
    try:
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

        nodes_fit, R_fit, Z_fit, lam_fit = point_guess
        nphi_fit = 2 * M + 1
        ntheta_fit = 2 * M + 1
        boundary_nodes, boundary_R, boundary_Z = append_boundary(
            np.empty((0, 3), dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
            surface,
            nfp,
            nphi_fit,
            ntheta_fit,
        )[:3]
        result["boundary_source"] = boundary_source
        axis_nodes, axis_R, axis_Z = axis_sample_points(model, nfp=nfp, nzeta=axis_zeta_count)

        R_sets = [
            _dataset(nodes_fit, R_fit, interior_weight, "interior_R"),
            _dataset(boundary_nodes, boundary_R, boundary_weight, "boundary_R"),
            _dataset(axis_nodes, axis_R, axis_weight, "axis_R"),
        ]
        Z_sets = [
            _dataset(nodes_fit, Z_fit, interior_weight, "interior_Z"),
            _dataset(boundary_nodes, boundary_Z, boundary_weight, "boundary_Z"),
            _dataset(axis_nodes, axis_Z, axis_weight, "axis_Z"),
        ]
        L_sets = []
        if lam_fit is not None:
            L_sets.append(_dataset(nodes_fit, lam_fit, lambda_weight, "interior_lambda"))

        t0 = now()
        fit = fit_joint_rzl_data(
            eq,
            R_datasets=R_sets,
            Z_datasets=Z_sets,
            L_datasets=L_sets,
            phase_datasets=[phase_constraints] if phase_constraints is not None else None,
            config=fit_cfg,
        )
        result["external_fit_time_s"] = float(now() - t0)
        result["external_fit_matrix_shape"] = list(fit.matrix_shape)
        result["external_fit"] = fit.diagnostics
        if lam_fit is not None:
            result["lambda_rms_target"] = float(np.sqrt(np.mean(np.asarray(lam_fit) ** 2)))
            result["lambda_max_abs_target"] = float(np.max(np.abs(lam_fit))) if len(lam_fit) else 0.0
        if phase_constraints is not None:
            result["phase_theta_rms_target"] = float(np.sqrt(np.mean(np.asarray(phase_constraints.theta) ** 2)))
            result["phase_beta_count"] = int(len(np.unique(np.asarray(phase_constraints.beta_ids, dtype=int))))

        apply_joint_fit(eq, fit)
        full_L = np.asarray(eq.L_lmn, dtype=float).copy()
        eq.L_lmn = np.zeros_like(full_L)
        result["nested_after_rz_fit_lambda0"] = bool(eq.is_nested())
        lambda_scale_scan = []
        for scale in np.linspace(0.0, 1.0, 11):
            eq.L_lmn = scale * full_L
            lambda_scale_scan.append(
                {"scale": float(scale), "nested": bool(eq.is_nested())}
            )
        eq.L_lmn = full_L
        result["lambda_scale_nested_scan"] = lambda_scale_scan
        pre_sync_mismatch = boundary_mismatch(eq, desc_surface, nfp)
        result.update({f"pre_sync_{key}": value for key, value in pre_sync_mismatch.items()})
        if boundary_source == "psi-ray":
            eq.surface = eq.get_surface_at(rho=1.0)
            result["boundary_synchronized_to_volume"] = True
        else:
            result["boundary_synchronized_to_volume"] = False
        result["nested_after_external_fit"] = bool(eq.is_nested())
        result.update(boundary_mismatch(eq, eq.surface, nfp))
        result.update(force_stats(eq, "initial"))

        np.savez(
            out_dir / "joint_fit_coeffs.npz",
            R_lmn=fit.R_lmn,
            Z_lmn=fit.Z_lmn,
            L_lmn=fit.L_lmn,
            beta=fit.beta,
            iota_coeffs=fit.iota_coeffs,
            iota_powers=np.asarray(fit.iota_powers, dtype=int),
        )

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
                result["desc_solve_traceback"] = traceback.format_exc()
            result["desc_solve_time_s"] = float(now() - t0)
            result["nested_after_solve"] = bool(eq.is_nested())
            result.update(force_stats(eq, "final"))

        try:
            eq.save(str(out_dir / "equilibrium.h5"))
            result["equilibrium_path"] = str(out_dir / "equilibrium.h5")
        except Exception as exc:
            result["equilibrium_save_error"] = repr(exc)
    except Exception as exc:
        result["variant_error"] = repr(exc)
        result["variant_traceback"] = traceback.format_exc()
    write_json(out_dir / "summary.json", result)
    return result


def parse_variants(text: str) -> list[str]:
    if text.strip().lower() == "all":
        return [
            "psi_joint_lambda0",
            "trace_joint_lambda0",
            "trace_joint_lambda_sfl_plus",
            "trace_joint_lambda_sfl_minus",
            "trace_joint_lambda_iota_hint",
            "trace_phase_joint_iota",
            "trace_phase_joint_iota_theta_flip",
            "psi_trace_phase_joint_iota",
            "psi_trace_phase_joint_iota_theta_flip",
        ]
    return [item.strip() for item in text.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="External weighted linear LS fit for DESC R/Z/L initial guess.")
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
    parser.add_argument(
        "--desc-boundary-source",
        choices=["psi-ray", "boozer-native"],
        default="psi-ray",
        help="Use a boundary parameterized consistently with the interior psi layers by default.",
    )
    parser.add_argument("--axis-order", type=int, default=8)
    parser.add_argument("--desc-L", type=int, default=6)
    parser.add_argument("--desc-M", type=int, default=6)
    parser.add_argument("--desc-N", type=int, default=6)
    parser.add_argument("--psi-layer-order", type=int, default=8)
    parser.add_argument("--psi-backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--psi-rho-min", type=float, default=0.12)
    parser.add_argument("--psi-rho-max", type=float, default=0.88)
    parser.add_argument("--psi-layers", type=int, default=8)
    parser.add_argument("--trace-alpha", type=int, default=24)
    parser.add_argument("--trace-zeta", type=int, default=17)
    parser.add_argument("--trace-steps", type=int, default=720)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--gpu-lib-path", default="gpu_backend/build_mixed/libstellarator_gpu.so")
    parser.add_argument("--variants", default="all")
    parser.add_argument("--interior-weight", type=float, default=1.0)
    parser.add_argument("--lambda-weight", type=float, default=1.0)
    parser.add_argument("--boundary-weight", type=float, default=40.0)
    parser.add_argument("--axis-weight", type=float, default=20.0)
    parser.add_argument("--axis-zeta-count", type=int, default=32)
    parser.add_argument("--rz-ridge", type=float, default=1e-8)
    parser.add_argument("--l-ridge", type=float, default=1e-8)
    parser.add_argument("--beta-ridge", type=float, default=1e-8)
    parser.add_argument("--iota-ridge", type=float, default=1e-8)
    parser.add_argument("--beta-gauge-weight", type=float, default=10.0)
    parser.add_argument("--iota-powers", default="0,2,4")
    parser.add_argument("--penalty-power", type=float, default=1.5)
    parser.add_argument("--radial-mode-scale", type=float, default=0.35)
    parser.add_argument("--poloidal-mode-scale", type=float, default=0.20)
    parser.add_argument("--toroidal-mode-scale", type=float, default=0.20)
    parser.add_argument("--normalize-columns", action="store_true")
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
    boozer_surface, surface_meta = make_xyz_surface(
        surface_npz,
        nfp=nfp,
        order=surface_order,
        stellsym=stellsym,
        nphi=args.surface_nphi,
        ntheta=args.surface_ntheta,
    )
    timings["simsopt_surface_reconstruct_s"] = float(now() - t0)

    t0 = now()
    axis_curve, axis_info = fit_axis_curve(run_dir / "axis_data.npz", nfp=nfp, axis_order=args.axis_order)
    timings["axis_curve_fit_s"] = float(now() - t0)

    t0 = now()
    model = load_psi_model(run_dir / "psi_model.npz")
    timings["psi_model_load_s"] = float(now() - t0)
    psi_edge = float(surface_meta.get("psi_level") or 0.0)
    if psi_edge <= 0:
        raise ValueError("surface npz does not contain a positive psi_level")

    boundary_fit_rms = None
    if args.desc_boundary_source == "psi-ray":
        boundary_order = max(args.surface_order or surface_order, args.desc_M, args.psi_layer_order)
        boundary_nodes_raw, boundary_R_raw, boundary_Z_raw, _ = psi_layer_points(
            model,
            psi_edge,
            np.asarray([1.0]),
            boundary_order,
            scan_cfg,
            boozer_cfg,
            backend=args.psi_backend,
        )
        nangle = 2 * boundary_order + 1
        zeta_raw = boundary_nodes_raw[:, 2]
        xyz = np.column_stack(
            [
                boundary_R_raw * np.cos(zeta_raw),
                boundary_R_raw * np.sin(zeta_raw),
                boundary_Z_raw,
            ]
        ).reshape(nangle, nangle, 3)
        surface, boundary_fit_rms = fit_xyz_tensor_surface(
            xyz, nfp, max(surface_order, args.desc_M), stellsym
        )
    else:
        surface = boozer_surface

    t0 = now()
    toroidal_flux = float(ToroidalFlux(surface, built.field).J())
    timings["toroidal_flux_s"] = float(now() - t0)
    input_path = out_dir / "boundary_input.check"
    t0 = now()
    input_info = write_vmec_input_from_surface(surface, toroidal_flux, input_path)
    timings["write_boundary_input_s"] = float(now() - t0)

    rho_values = np.linspace(args.psi_rho_min, args.psi_rho_max, args.psi_layers)
    t0 = now()
    nodes_psi, R_psi, Z_psi, psi_extract_info = psi_layer_points(
        model, psi_edge, rho_values, args.psi_layer_order, scan_cfg, boozer_cfg, backend=args.psi_backend
    )
    timings["psi_point_guess_s"] = float(now() - t0)

    t0 = now()
    iota_hint = surface_meta.get("iota")
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
    timings["fieldline_joint_points_s"] = float(now() - t0)

    t0 = now()
    nodes_phase, R_phase, Z_phase, phase_constraints, phase_info = fieldline_phase_constraint_data(
        built.field,
        model,
        psi_edge,
        rho_values,
        scan_cfg,
        n_alpha=args.trace_alpha,
        n_zeta=args.trace_zeta,
        trace_steps=args.trace_steps,
        weight=args.lambda_weight,
    )
    timings["fieldline_phase_constraints_s"] = float(now() - t0)

    variant_specs = _variant_specs(
        nodes_psi=nodes_psi,
        R_psi=R_psi,
        Z_psi=Z_psi,
        nodes_tr=nodes_tr,
        R_tr=R_tr,
        Z_tr=Z_tr,
        lam_plus=lam_plus,
        lam_minus=lam_minus,
        lam_hint=lam_hint,
    )
    variant_specs["trace_phase_joint_iota"] = {
        "nodes": nodes_phase,
        "R": R_phase,
        "Z": Z_phase,
        "lambda": None,
        "phase": phase_constraints,
    }
    variant_specs["trace_phase_joint_iota_theta_flip"] = {
        "nodes": nodes_phase,
        "R": R_phase,
        "Z": Z_phase,
        "lambda": None,
        "phase": _flip_phase_theta(phase_constraints, "trace_phase_constraints_theta_flip"),
    }
    variant_specs["psi_trace_phase_joint_iota"] = {
        "nodes": nodes_psi,
        "R": R_psi,
        "Z": Z_psi,
        "lambda": None,
        "phase": phase_constraints,
    }
    variant_specs["psi_trace_phase_joint_iota_theta_flip"] = {
        "nodes": nodes_psi,
        "R": R_psi,
        "Z": Z_psi,
        "lambda": None,
        "phase": _flip_phase_theta(
            phase_constraints, "trace_phase_constraints_theta_flip"
        ),
    }
    fit_cfg = JointSpectralFitConfig(
        rz_ridge=args.rz_ridge,
        l_ridge=args.l_ridge,
        beta_ridge=args.beta_ridge,
        iota_ridge=args.iota_ridge,
        beta_gauge_weight=args.beta_gauge_weight,
        radial_mode_scale=args.radial_mode_scale,
        poloidal_mode_scale=args.poloidal_mode_scale,
        toroidal_mode_scale=args.toroidal_mode_scale,
        penalty_power=args.penalty_power,
        iota_powers=_parse_int_list(args.iota_powers),
        normalize_columns=args.normalize_columns,
    )

    variant_results = []
    for name in parse_variants(args.variants):
        print(f"[variant] {name}", flush=True)
        spec = variant_specs[name]
        guess = (spec["nodes"], spec["R"], spec["Z"], spec.get("lambda"))
        variant_results.append(
            run_external_variant(
                name=name,
                input_path=input_path,
                psi=toroidal_flux,
                L=args.desc_L,
                M=args.desc_M,
                N=args.desc_N,
                axis_curve=axis_curve,
                nfp=nfp,
                model=model,
                point_guess=guess,
                phase_constraints=spec.get("phase"),
                surface=surface,
                boundary_source=args.desc_boundary_source,
                fit_cfg=fit_cfg,
                interior_weight=args.interior_weight,
                lambda_weight=args.lambda_weight,
                boundary_weight=args.boundary_weight,
                axis_weight=args.axis_weight,
                axis_zeta_count=args.axis_zeta_count,
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
        "desc_boundary_source": args.desc_boundary_source,
        "desc_boundary_fit_rms": boundary_fit_rms,
        "toroidal_flux": toroidal_flux,
        "boundary_input": input_info,
        "axis_fit": axis_info,
        "rho_values": [float(x) for x in rho_values],
        "psi_extract": psi_extract_info,
        "fieldline_trace": trace_info,
        "fieldline_phase_constraints": phase_info,
        "fit_config": vars(args),
        "timings": timings,
        "variants": variant_results,
        "total_time_s": float(now() - total0),
    }
    write_json(out_dir / "summary.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
