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

from stellarator_eval.config import VolumeQSConfig
from stellarator_eval.field import input_from_packed_vector, load_case_file, normalize_currents
from stellarator_eval.quasr import load_quasr_field_input
from stellarator_eval.volume_qs import (
    apply_flux_coordinates,
    calibrate_toroidal_flux_gpu,
    compute_f_c,
    evaluate_straight_field_alpha,
    fit_alpha_vector_gpu_qr,
    fit_flux_scale,
    fit_straight_field_alpha,
    load_psi_model,
    sample_volume_points,
    summarize_volume_qs,
    subset_volume_points,
    vacuum_G,
)
from stellarator_gpu import CoilFieldGpu


def load_input(path: Path, key: str):
    data = json.loads(path.read_text(encoding="utf-8"))
    if key in data:
        return load_case_file(path, key)
    if "input" in data and "packed_values" in data["input"]:
        return input_from_packed_vector(data["input"]["packed_values"], coeff_count=int(data["input"]["coeff_count"]))
    raise ValueError(f"cannot find {key!r} or input.packed_values in {path}")


def reference_values(boozer_path: Path | None, flux_path: Path | None):
    result = {}
    if boozer_path and boozer_path.exists():
        data = np.load(boozer_path)
        result["boozer_iota"] = float(data["iota"])
        result["boozer_G"] = float(data["G"])
    if flux_path and flux_path.exists():
        data = np.load(flux_path)
        coefficients = np.asarray(data["polynomial_coeffs"], dtype=float)
        s_edge = float(np.asarray(data["s_knots"])[-1])
        result["calibrated_psi_edge"] = float(sum(c * s_edge ** (k + 1) for k, c in enumerate(coefficients)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path)
    parser.add_argument("--quasr-root", type=Path)
    parser.add_argument("--id", type=int)
    parser.add_argument("--key", default="raw")
    parser.add_argument("--current-unit", default="A")
    parser.add_argument("--psi-model", type=Path, required=True)
    parser.add_argument("--boozer-reference", type=Path)
    parser.add_argument("--flux-reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-lib", type=Path, default=Path("gpu_backend/build_volume_qs/libstellarator_gpu.so"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--precision", choices=("fp32", "fp64"), default="fp32")
    parser.add_argument("--points", type=int, default=100000)
    parser.add_argument("--alpha-fit-points", type=int, default=30000)
    parser.add_argument("--alpha-validation-points", type=int, default=0)
    parser.add_argument("--grid-xy", type=int, default=144)
    parser.add_argument("--grid-phi", type=int, default=96)
    parser.add_argument("--s-edge", type=float, default=0.16)
    parser.add_argument("--rho-min", type=float, default=0.08)
    parser.add_argument("--alpha-order", type=int, default=12)
    parser.add_argument("--alpha-toroidal-order", type=int)
    parser.add_argument("--iota-degree", type=int, default=0)
    parser.add_argument("--alpha-ridge", type=float, default=1e-7)
    parser.add_argument(
        "--alpha-method", choices=("hybrid", "vector", "direction"), default="hybrid"
    )
    parser.add_argument("--flux-levels", type=int, default=11)
    parser.add_argument("--flux-phis", type=int, default=8)
    parser.add_argument("--flux-thetas", type=int, default=256)
    parser.add_argument("--flux-radial", type=int, default=24)
    parser.add_argument("--flux-degree", type=int, default=3)
    parser.add_argument("--require-flux-quality", action="store_true")
    args = parser.parse_args()

    config = VolumeQSConfig(
        s_edge=args.s_edge,
        rho_min=args.rho_min,
        point_count=args.points,
        alpha_fit_point_count=args.alpha_fit_points,
        grid_xy=args.grid_xy,
        grid_phi=args.grid_phi,
        alpha_radial_order=args.alpha_order,
        alpha_poloidal_order=args.alpha_order,
        alpha_toroidal_order=args.alpha_toroidal_order or args.alpha_order,
        iota_degree=args.iota_degree,
        alpha_ridge=args.alpha_ridge,
        flux_level_count=args.flux_levels,
        flux_phi_count=args.flux_phis,
        flux_theta_count=args.flux_thetas,
        flux_radial_quadrature=args.flux_radial,
        flux_degree=args.flux_degree,
        precision=args.precision,
        gpu_device=args.device,
        gpu_lib_path=str(args.gpu_lib),
    )
    total_start = time.perf_counter()
    model = load_psi_model(args.psi_model)
    if args.id is not None:
        if args.quasr_root is None:
            raise ValueError("--id requires --quasr-root")
        field_input, _ = load_quasr_field_input(args.quasr_root, args.id)
        case_name = f"quasr_{args.id:07d}"
    else:
        if args.case is None:
            raise ValueError("pass either --case or --id with --quasr-root")
        field_input = load_input(args.case, args.key)
        case_name = str(args.case)
    currents_a = normalize_currents(field_input.currents, args.current_unit)
    timings = {}

    start = time.perf_counter()
    points = sample_volume_points(model, config, device=f"cuda:{args.device}")
    points["nfp"] = np.asarray(model.nfp)
    timings["volume_points_s"] = float(time.perf_counter() - start)

    start = time.perf_counter()
    gpu = CoilFieldGpu(
        args.gpu_lib,
        field_input.coeffs_x,
        field_input.coeffs_y,
        field_input.coeffs_z,
        currents_a,
        field_input.nfp,
        segments_per_coil=config.gpu_segments_per_coil,
        device_id=args.device,
    )
    timings["gpu_field_create_s"] = float(time.perf_counter() - start)
    try:
        calibration = None
        if args.alpha_method in {"hybrid", "vector"}:
            start = time.perf_counter()
            calibration = calibrate_toroidal_flux_gpu(model, gpu, config)
            if args.require_flux_quality and not calibration.diagnostics["quality_ok"]:
                raise RuntimeError(
                    "flux calibration quality gate failed: "
                    f"boundary={calibration.diagnostics['boundary_residual_max']:.3e}, "
                    f"section_edge={calibration.diagnostics['section_relative_std_edge']:.3e}"
                )
            apply_flux_coordinates(points, calibration)
            timings["flux_calibration_s"] = float(time.perf_counter() - start)
        start = time.perf_counter()
        B, grad_B = gpu.eval_B_grad(points["xyz"], precision=args.precision)
        timings["B_grad_B_s"] = float(time.perf_counter() - start)
    finally:
        gpu.close()

    fit_count = min(config.alpha_fit_point_count, len(B))
    fit_indices = np.floor(np.linspace(0, len(B), fit_count, endpoint=False)).astype(int)
    fit_points = subset_volume_points(points, fit_indices)
    fit_B = B[fit_indices]
    if args.alpha_method == "vector":
        alpha = fit_alpha_vector_gpu_qr(fit_points, fit_B, config, device=f"cuda:{args.device}")
        flux = calibration
    elif args.alpha_method == "hybrid":
        alpha = fit_straight_field_alpha(fit_points, fit_B, config, device=f"cuda:{args.device}")
        flux = calibration
    else:
        alpha = fit_straight_field_alpha(fit_points, fit_B, config, device=f"cuda:{args.device}")
        flux = fit_flux_scale(fit_points, fit_B, alpha, config)
    timings["alpha_total_s"] = float(alpha.diagnostics["total_s"])
    timings["alpha_qr_s"] = float(alpha.diagnostics["solve_s"])
    if args.alpha_method == "direction":
        timings["flux_scale_s"] = float(flux.diagnostics["total_s"])
    fit_mask = np.zeros(len(B), dtype=bool)
    fit_mask[fit_indices] = True
    validation_indices = np.flatnonzero(~fit_mask)
    if len(validation_indices) > args.alpha_validation_points > 0:
        choose = np.floor(
            np.linspace(0, len(validation_indices), args.alpha_validation_points, endpoint=False)
        ).astype(int)
        validation_indices = validation_indices[choose]
    alpha_validation = (
        evaluate_straight_field_alpha(
            subset_volume_points(points, validation_indices), B[validation_indices], alpha
        )
        if args.alpha_validation_points > 0 and len(validation_indices)
        else None
    )
    G = vacuum_G(currents_a, field_input.nfp, flux.psi_edge)
    metrics = {}
    start = time.perf_counter()
    for name, M, N in (
        ("QA", 1, 0),
        ("QH_plus", 1, field_input.nfp),
        ("QH_minus", 1, -field_input.nfp),
        ("QP", 0, field_input.nfp),
    ):
        fields = compute_f_c(points, B, grad_B, alpha, flux, M=M, N=N, G=G)
        metrics[name] = summarize_volume_qs(points, fields, config.radial_bins)
    timings["qs_metrics_s"] = float(time.perf_counter() - start)
    timings["downstream_total_s"] = float(time.perf_counter() - total_start)

    reference = reference_values(args.boozer_reference, args.flux_reference)
    iota_mid = float(alpha.iota(np.asarray([0.5]))[0])
    comparisons = {}
    if "boozer_iota" in reference:
        comparisons["iota_mid_minus_boozer"] = iota_mid - reference["boozer_iota"]
    if "boozer_G" in reference:
        comparisons["G_relative_to_boozer"] = (G - reference["boozer_G"]) / reference["boozer_G"]
    if "calibrated_psi_edge" in reference:
        comparisons["psi_edge_relative_to_flux_calibration"] = (
            flux.psi_edge - reference["calibrated_psi_edge"]
        ) / reference["calibrated_psi_edge"]

    if args.alpha_method in {"hybrid", "vector"}:
        flux_coefficients = np.asarray(flux.polynomial_coeffs).tolist()
    else:
        flux_coefficients = np.asarray(flux.coefficients).tolist()

    payload = {
        "case": case_name,
        "psi_model": str(args.psi_model),
        "config": vars(config),
        "alpha_method": args.alpha_method,
        "sampling": {
            "point_count": int(len(points["s"])),
            "candidate_count": int(points["candidate_count"][0]),
            "available_count": int(points["available_count"][0]),
            "s_min": float(np.min(points["s"])),
            "s_max": float(np.max(points["s"])),
        },
        "timing": timings,
        "alpha": {
            "iota_coefficients": alpha.iota_coeffs.tolist(),
            "diagnostics": alpha.diagnostics,
            "validation": alpha_validation,
        },
        "flux": {
            "coefficients": flux_coefficients,
            "diagnostics": flux.diagnostics,
        },
        "G": G,
        "reference": reference,
        "comparisons": comparisons,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
