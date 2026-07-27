from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simsopt.geo import ToroidalFlux

from stellarator_eval.alpha_clebsch import (
    evaluate_lambda,
    load_alpha_fit,
    load_flux_calibration,
)
from stellarator_eval.config import BoozerConfig, SurfaceScanConfig
from stellarator_eval.field import build_field
from stellarator_eval.serialization import write_json
from scripts.desc_psi_volume_initial_guess_experiment import (
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
from scripts.diagnose_desc_rz_nesting import (
    jacobian_stats,
    plot_jacobian_map,
    plot_rz_fit_sections,
)
from scripts.desc_external_rzl_data_ls_experiment import axis_sample_points

TWOPI = 2.0 * np.pi


def parse_ints(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def to_straight_nodes(nodes: np.ndarray, alpha_fit, nfp: int) -> tuple[np.ndarray, dict]:
    nodes = np.asarray(nodes, dtype=float)
    rho = nodes[:, 0]
    theta_clockwise = -nodes[:, 1]
    zeta = nodes[:, 2]
    lam, lambda_theta, _ = evaluate_lambda(
        alpha_fit, rho, theta_clockwise, zeta, nfp
    )
    theta_straight_clockwise = theta_clockwise + lam
    # DESC expects a right-handed poloidal angle, opposite to the clockwise
    # angle used to make grad(psi_T) x grad(theta) match the physical field.
    theta_desc = np.mod(-theta_straight_clockwise, TWOPI)
    transformed = np.column_stack([rho, theta_desc, zeta])
    return transformed, {
        "lambda_rms": float(np.sqrt(np.mean(lam * lam))),
        "lambda_max_abs": float(np.max(np.abs(lam))),
        "one_plus_lambda_theta_min": float(np.min(1.0 + lambda_theta)),
        "noninvertible_fraction": float(np.mean(1.0 + lambda_theta <= 0.0)),
    }


def fit_basis_gpu_qr(basis, nodes, values, weights) -> tuple[np.ndarray, dict]:
    import torch

    started = time.perf_counter()
    matrix = np.asarray(basis.evaluate(nodes), dtype=float)
    values = np.asarray(values, dtype=float)
    sqrt_weight = np.sqrt(np.asarray(weights, dtype=float))
    device = "cuda"
    A = torch.as_tensor(matrix * sqrt_weight[:, None], dtype=torch.float64, device=device)
    b = torch.as_tensor(values * sqrt_weight, dtype=torch.float64, device=device)[:, None]
    scale = torch.linalg.vector_norm(A, dim=0).clamp_min(1e-30)
    A /= scale[None, :]
    torch.cuda.synchronize()
    solve_started = time.perf_counter()
    scaled_coeffs = torch.linalg.lstsq(A, b, driver="gels").solution[:, 0]
    torch.cuda.synchronize()
    solve_s = time.perf_counter() - solve_started
    coeffs = (scaled_coeffs / scale).cpu().numpy()
    residual = matrix @ coeffs - values
    diagnostics = {
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
        "solve_s": float(solve_s),
        "total_s": float(time.perf_counter() - started),
        "rms": float(np.sqrt(np.mean(residual * residual))),
        "max_abs": float(np.max(np.abs(residual))),
        "weighted_rms": float(np.sqrt(np.mean(weights * residual * residual))),
        "column_scale_min": float(torch.min(scale).item()),
        "column_scale_max": float(torch.max(scale).item()),
    }
    del A, b, scale, scaled_coeffs
    torch.cuda.empty_cache()
    return coeffs, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--alpha-dir", type=Path, required=True)
    parser.add_argument("--alpha-fit", default="alpha_fit_L12_M12_N16.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--key", default="raw")
    parser.add_argument("--resolutions", default="6,8,10,12")
    parser.add_argument("--rho-layers", type=int, default=21)
    parser.add_argument("--psi-order", type=int, default=18)
    parser.add_argument("--rz-ridge", type=float, default=1e-10)
    parser.add_argument("--boundary-weight", type=float, default=20.0)
    parser.add_argument("--axis-weight", type=float, default=10.0)
    parser.add_argument("--solve-resolution", type=int, default=0)
    parser.add_argument("--solve-maxiter", type=int, default=50)
    parser.add_argument("--solve-ftol", type=float, default=1e-8)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    summary = load_json(args.run_dir / "summary.json")
    config = summary["config"]
    nfp = int(summary["nfp"])
    current_unit = config.get("current_unit") or "A"
    s_edge = float(summary["best_surface"]["psi_level"])
    level_dir = args.run_dir / f"level_{s_edge:.6g}".replace(".", "p")
    surface_order = int(config["boozer"]["surface_order"])
    stellsym = bool(config["boozer"]["stellsym"])

    field_input = load_field_input(args.case_file, args.key)
    built = build_field(field_input, current_unit=current_unit)
    model = load_psi_model(args.run_dir / "psi_model.npz")
    surface, _ = make_xyz_surface(
        level_dir / "boozer_surface.npz",
        nfp=nfp,
        order=surface_order,
        stellsym=stellsym,
        nphi=96,
        ntheta=144,
    )
    toroidal_flux_surface = float(ToroidalFlux(surface, built.field).J())
    calibration = load_flux_calibration(args.alpha_dir / "flux_calibration.npz")
    toroidal_flux_calibrated = float(TWOPI * calibration.psi_edge)
    toroidal_flux_desc = float(
        np.copysign(abs(toroidal_flux_calibrated), toroidal_flux_surface)
    )
    alpha_fit = load_alpha_fit(args.alpha_dir / args.alpha_fit)

    input_path = args.output_dir / "boundary_input.check"
    write_vmec_input_from_surface(surface, toroidal_flux_desc, input_path)
    axis_curve, axis_info = fit_axis_curve(
        args.run_dir / "axis_data.npz", nfp=nfp, axis_order=10
    )
    scan_cfg = dataclass_from_dict(SurfaceScanConfig, config.get("scan"))
    boozer_cfg = dataclass_from_dict(BoozerConfig, config.get("boozer"))

    rho_values = np.linspace(0.06, 0.96, int(args.rho_layers))
    raw_nodes, target_R, target_Z, layer_info = psi_layer_points(
        model,
        s_edge,
        rho_values,
        args.psi_order,
        scan_cfg,
        boozer_cfg,
        backend="gpu",
    )
    interior_nodes, interior_transform = to_straight_nodes(raw_nodes, alpha_fit, nfp)
    raw_boundary_nodes, boundary_R, boundary_Z, boundary_info = psi_layer_points(
        model,
        s_edge,
        np.asarray([1.0]),
        args.psi_order,
        scan_cfg,
        boozer_cfg,
        backend="gpu",
    )
    boundary_nodes, boundary_transform = to_straight_nodes(
        raw_boundary_nodes, alpha_fit, nfp
    )

    rows = []
    snapshots = []
    from desc.geometry import FourierRZToroidalSurface

    input_surface = FourierRZToroidalSurface.from_input_file(str(input_path))
    for resolution in parse_ints(args.resolutions):
        eq = build_equilibrium(
            input_surface,
            toroidal_flux_desc,
            resolution,
            resolution,
            resolution,
            axis_curve=axis_curve,
            constructor_ensure_nested=False,
        )
        axis_nodes, axis_R, axis_Z = axis_sample_points(
            model, nfp=nfp, nzeta=max(32, 4 * resolution)
        )
        fit_nodes = np.vstack([interior_nodes, boundary_nodes, axis_nodes])
        fit_R = np.concatenate([target_R, boundary_R, axis_R])
        fit_Z = np.concatenate([target_Z, boundary_Z, axis_Z])
        weights = np.concatenate(
            [
                np.ones(len(interior_nodes)),
                np.full(len(boundary_nodes), args.boundary_weight),
                np.full(len(axis_nodes), args.axis_weight),
            ]
        )
        fit_started = time.perf_counter()
        R_lmn, R_fit_info = fit_basis_gpu_qr(eq.R_basis, fit_nodes, fit_R, weights)
        Z_lmn, Z_fit_info = fit_basis_gpu_qr(eq.Z_basis, fit_nodes, fit_Z, weights)
        eq.R_lmn = R_lmn
        eq.Z_lmn = Z_lmn
        eq.L_lmn = np.zeros_like(eq.L_lmn)
        eq.surface = eq.get_surface_at(rho=1.0)
        synchronized_jac = jacobian_stats(eq)
        boundary_R_fit = np.asarray(eq.R_basis.evaluate(boundary_nodes)) @ R_lmn
        boundary_Z_fit = np.asarray(eq.Z_basis.evaluate(boundary_nodes)) @ Z_lmn
        axis_R_fit = np.asarray(eq.R_basis.evaluate(axis_nodes)) @ R_lmn
        axis_Z_fit = np.asarray(eq.Z_basis.evaluate(axis_nodes)) @ Z_lmn

        def vector_rms(Ra, Za, Rb, Zb):
            return float(np.sqrt(np.mean((Ra - Rb) ** 2 + (Za - Zb) ** 2)))

        result = {
            "name": f"alpha_straight_LMN{resolution}",
            "resolution": int(resolution),
            "fit_time_s": float(time.perf_counter() - fit_started),
            "R_fit": R_fit_info,
            "Z_fit": Z_fit_info,
            "R_fit_rms": R_fit_info["rms"],
            "Z_fit_rms": Z_fit_info["rms"],
            "boundary_parametric_rms": vector_rms(
                boundary_R_fit, boundary_Z_fit, boundary_R, boundary_Z
            ),
            "axis_parametric_rms": vector_rms(
                axis_R_fit, axis_Z_fit, axis_R, axis_Z
            ),
            "synchronized_jacobian": {
                key: value
                for key, value in synchronized_jac.items()
                if key not in {"grid_nodes", "sqrtg_values"}
            },
        }
        result["nested_after_boundary_sync"] = bool(eq.is_nested())
        result.update(force_stats(eq, "initial"))
        snapshot_eq = copy.deepcopy(eq)
        result_dir = args.output_dir / f"LMN{resolution}"
        result_dir.mkdir()
        eq.save(str(result_dir / "equilibrium_initial.h5"))
        if resolution == args.solve_resolution:
            solve_started = time.perf_counter()
            try:
                solved, optimizer = eq.solve(
                    maxiter=args.solve_maxiter,
                    ftol=args.solve_ftol,
                    verbose=2,
                )
                result["solve_call_success"] = True
                result["optimizer_success"] = bool(getattr(optimizer, "success", False))
                result["optimizer_message"] = str(getattr(optimizer, "message", ""))
                result["optimizer_cost"] = float(getattr(optimizer, "cost", np.nan))
                result["optimizer_iterations"] = int(getattr(optimizer, "nit", -1))
                result["nested_after_solve"] = bool(solved.is_nested())
                result.update(force_stats(solved, "final"))
                solved.save(str(result_dir / "equilibrium_solved.h5"))
            except Exception as exc:
                result["solve_call_success"] = False
                result["solve_error"] = repr(exc)
            result["solve_time_s"] = float(time.perf_counter() - solve_started)
        write_json(result_dir / "summary.json", result)
        rows.append(result)
        snapshots.append(
            {
                "case": f"alpha_straight_LMN{resolution}",
                "eq": snapshot_eq,
                "jac": synchronized_jac,
                "target_nodes": np.vstack([interior_nodes, boundary_nodes]),
                "target_R": np.concatenate([target_R, boundary_R]),
                "target_Z": np.concatenate([target_Z, boundary_Z]),
            }
        )

    nested_rows = [row for row in rows if row["nested_after_boundary_sync"]]
    best_row = min(
        nested_rows or rows,
        key=lambda row: row["R_fit_rms"] + row["Z_fit_rms"],
    )
    best_index = rows.index(best_row)
    plot_rz_fit_sections(snapshots[best_index], args.output_dir / "rz_straight_fit_sections.png")
    plot_jacobian_map(snapshots[best_index], args.output_dir / "rz_straight_jacobian.png")
    output = {
        "s_edge": s_edge,
        "alpha_fit": args.alpha_fit,
        "alpha_iota_coeffs": alpha_fit.iota_coeffs.tolist(),
        "toroidal_flux_calibrated": toroidal_flux_calibrated,
        "toroidal_flux_surface": toroidal_flux_surface,
        "toroidal_flux_desc": toroidal_flux_desc,
        "toroidal_flux_magnitude_relative_difference": float(
            (abs(toroidal_flux_calibrated) - abs(toroidal_flux_surface))
            / abs(toroidal_flux_surface)
        ),
        "axis_fit": axis_info,
        "interior_layers": layer_info,
        "boundary_layer": boundary_info,
        "interior_transform": interior_transform,
        "boundary_transform": boundary_transform,
        "rows": rows,
        "best_resolution": best_row["resolution"],
        "total_time_s": float(time.perf_counter() - started),
    }
    write_json(args.output_dir / "summary.json", output)


if __name__ == "__main__":
    main()
