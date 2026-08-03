from __future__ import annotations

import argparse
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

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stellarator_eval.alpha_clebsch import (
    AlphaFitResult,
    FluxCalibration,
    alpha_coordinates_from_volume_points,
    calibrate_toroidal_flux,
    disjoint_train_validation_indices,
    evaluate_alpha_fit,
    evaluate_lambda,
    fit_alpha_gpu_qr,
    physical_coordinate_data,
    pchip_flux_diagnostic,
    sample_uniform_volume,
)
from stellarator_eval.axis import rk4_period_samples
from stellarator_eval.config import PsiFitConfig, SurfaceScanConfig, VolumeQSConfig
from stellarator_eval.field import build_field
from stellarator_eval.psi import _b_components_gpu
from stellarator_eval.serialization import write_json
from stellarator_eval.surface import level_curve_phi0
from stellarator_eval.volume_qs import (
    apply_flux_coordinates,
    calibrate_toroidal_flux_gpu,
    sample_volume_points,
)
from scripts.desc_psi_volume_initial_guess_experiment import load_field_input, load_psi_model

TWOPI = 2.0 * np.pi


def resolve_project_path(path: Path, root: Path = ROOT) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def parse_orders(text: str) -> list[tuple[int, int, int]]:
    orders = []
    for item in text.split(","):
        fields = [int(value) for value in item.strip().split(":")]
        if len(fields) != 3:
            raise ValueError("orders must use radial:poloidal:toroidal triples")
        orders.append(tuple(fields))
    return orders


def sample_bfield(gpu_field, R, Z, phi, *, precision: str) -> np.ndarray:
    br, bphi, bz = _b_components_gpu(
        gpu_field, R, Z, phi, precision=precision
    )
    return np.column_stack([br, bphi, bz])


def save_fit(path: Path, result: AlphaFitResult) -> None:
    np.savez(
        path,
        lambda_coeffs=result.lambda_coeffs,
        iota_coeffs=result.iota_coeffs,
        mode_l=np.asarray([mode.l for mode in result.modes], dtype=int),
        mode_m=np.asarray([mode.m for mode in result.modes], dtype=int),
        mode_n=np.asarray([mode.n for mode in result.modes], dtype=int),
        mode_kind=np.asarray([mode.kind for mode in result.modes]),
        radial_order=result.radial_order,
        poloidal_order=result.poloidal_order,
        toroidal_order=result.toroidal_order,
        iota_degree=result.iota_degree,
    )


def plot_flux_calibration(calibration: FluxCalibration, path: Path) -> None:
    s = np.linspace(0.0, calibration.s_knots[-1], 500)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for values in calibration.psi_by_section:
        axes[0].plot(calibration.s_knots, values, color="#9aa0a6", alpha=0.35, lw=0.8)
    axes[0].plot(calibration.s_knots, calibration.psi_knots, "o", color="#b3261e", label="section mean")
    axes[0].plot(s, calibration.evaluate(s), color="#174ea6", lw=2, label="polynomial calibration")
    axes[0].set_xlabel(r"fitted invariant $s$")
    axes[0].set_ylabel(r"physical $\psi_T=\Phi_T/(2\pi)$ [Wb/rad]")
    axes[0].legend(frameon=False)
    derivative = calibration.derivative(s)
    axes[1].plot(s, derivative, color="#188038", lw=2)
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_xlabel(r"fitted invariant $s$")
    axes[1].set_ylabel(r"$d\psi_T/ds$")
    figure.suptitle("Toroidal-flux calibration")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_order_scan(rows: list[dict], path: Path) -> None:
    labels = [row["label"] for row in rows]
    x = np.arange(len(rows))
    validation = [row["validation"]["relative_l2"] for row in rows]
    floor = [row["validation"]["normal_floor_relative_l2"] for row in rows]
    folds = [row["validation"]["noninvertible_fraction"] for row in rows]
    columns = [row["fit"]["column_count"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    axes[0].semilogy(x, validation, "o-", label="full vector residual")
    axes[0].semilogy(x, floor, "s--", label=r"normal floor from $B\cdot\nabla\psi$")
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].set_ylabel("relative L2")
    axes[0].legend(frameon=False)
    axes[1].plot(x, folds, "o-", color="#b3261e", label="non-invertible fraction")
    axes[1].set_xticks(x, [f"{label}\n{column} cols" for label, column in zip(labels, columns)], rotation=25, ha="right")
    axes[1].set_ylabel(r"fraction with $1+\partial_\theta\lambda\leq0$")
    axes[1].set_ylim(bottom=-0.01)
    axes[1].legend(frameon=False)
    figure.suptitle("Unconstrained GPU QR order scan")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_coordinate_invertibility(result: AlphaFitResult, nfp: int, path: Path) -> None:
    rho = np.linspace(0.06, 1.0, 180)
    theta = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    rg, tg = np.meshgrid(rho, theta, indexing="ij")
    phi = np.zeros_like(rg)
    _, lambda_theta, _ = evaluate_lambda(result, rg.ravel(), tg.ravel(), phi.ravel(), nfp)
    jacobian = (1.0 + lambda_theta).reshape(rg.shape)
    figure, axis = plt.subplots(figsize=(8.8, 4.4))
    image = axis.pcolormesh(theta, rho, jacobian, shading="auto", cmap="RdBu_r")
    if np.min(jacobian) <= 0.0 <= np.max(jacobian):
        axis.contour(theta, rho, jacobian, levels=[0.0], colors="black", linewidths=1.0)
    axis.set_xlabel(r"clockwise geometric $\theta$ at $\phi=0$")
    axis.set_ylabel(r"$\rho$")
    axis.set_title(r"Coordinate Jacobian $1+\partial_\theta\lambda$")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def trace_straightness(
    field,
    model,
    calibration: FluxCalibration,
    result: AlphaFitResult,
    s_edge: float,
    *,
    periods: int,
    lines_per_surface: int,
    samples_per_period: int,
    steps_per_period: int,
    path: Path,
) -> dict:
    scan = SurfaceScanConfig(max_radius_scale=1.0)
    rho_levels = [0.3, 0.6, 0.9]
    records = []
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(rho_levels)))
    for color, rho0 in zip(colors, rho_levels):
        _, R, Z, _ = level_curve_phi0(
            model,
            s_edge * rho0 * rho0,
            int(lines_per_surface),
            scan,
        )
        R_current = R.copy()
        Z_current = Z.copy()
        all_phi = []
        all_R = []
        all_Z = []
        period_length = TWOPI / model.nfp
        for period_index in range(int(periods)):
            phi_local, R_hist, Z_hist, R_current, Z_current = rk4_period_samples(
                field,
                R_current,
                Z_current,
                model.nfp,
                n_zeta=int(samples_per_period),
                steps=int(steps_per_period),
            )
            all_phi.append(phi_local + period_index * period_length)
            all_R.append(R_hist)
            all_Z.append(Z_hist)
        phi_global = np.concatenate(all_phi)
        R_trace = np.concatenate(all_R, axis=1)
        Z_trace = np.concatenate(all_Z, axis=1)
        for line_index in range(int(lines_per_surface)):
            phi_line = phi_global
            R_line = R_trace[line_index]
            Z_line = Z_trace[line_index]
            coordinates = physical_coordinate_data(
                model,
                calibration,
                R_line,
                Z_line,
                np.mod(phi_line, period_length),
            )
            lam, _, _ = evaluate_lambda(
                result,
                coordinates["rho"],
                coordinates["theta"],
                phi_line,
                model.nfp,
            )
            theta_raw = np.unwrap(coordinates["theta"])
            theta_corrected = np.unwrap(coordinates["theta"] + lam)
            iota0 = float(result.iota(np.asarray([rho0]))[0])
            regression = np.column_stack([np.ones_like(phi_line), phi_line])
            raw_line_coeffs, *_ = np.linalg.lstsq(regression, theta_raw, rcond=None)
            corrected_line_coeffs, *_ = np.linalg.lstsq(
                regression, theta_corrected, rcond=None
            )
            raw_intercept = float(np.mean(theta_raw - iota0 * phi_line))
            corrected_intercept = float(np.mean(theta_corrected - iota0 * phi_line))
            raw_residual = theta_raw - (raw_intercept + iota0 * phi_line)
            corrected_residual = theta_corrected - (corrected_intercept + iota0 * phi_line)
            corrected_best_residual = theta_corrected - regression @ corrected_line_coeffs
            alpha_drift = theta_corrected - iota0 * phi_line
            records.append(
                {
                    "rho": rho0,
                    "line": line_index,
                    "iota": iota0,
                    "raw_regression_iota": float(raw_line_coeffs[1]),
                    "corrected_regression_iota": float(corrected_line_coeffs[1]),
                    "raw_line_rms_rad": float(np.sqrt(np.mean(raw_residual**2))),
                    "corrected_line_rms_rad": float(np.sqrt(np.mean(corrected_residual**2))),
                    "corrected_best_slope_rms_rad": float(
                        np.sqrt(np.mean(corrected_best_residual**2))
                    ),
                    "alpha_drift_peak_to_peak_rad": float(np.ptp(alpha_drift)),
                    "rho_drift_peak_to_peak": float(np.ptp(coordinates["rho"])),
                }
            )
            label = rf"$\rho={rho0:.1f}$" if line_index == 0 else None
            axes[0].plot(phi_line, theta_raw, color=color, alpha=0.75, lw=1.0, label=label)
            axes[1].plot(phi_line, theta_corrected, color=color, alpha=0.75, lw=1.0, label=label)
            axes[1].plot(
                phi_line,
                corrected_intercept + iota0 * phi_line,
                color="black",
                alpha=0.22,
                lw=0.7,
                ls="--",
            )
    axes[0].set_title(r"Before: geometric $\theta$")
    axes[1].set_title(r"After: $\vartheta=\theta+\lambda$")
    for axis in axes:
        axis.set_xlabel(r"unwrapped toroidal angle $\phi$")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    axes[0].set_ylabel("unwrapped poloidal angle")
    figure.suptitle("Magnetic-field-line straightening")
    figure.tight_layout()
    figure.savefig(path, dpi=190)
    plt.close(figure)
    raw = np.asarray([row["raw_line_rms_rad"] for row in records])
    corrected = np.asarray([row["corrected_line_rms_rad"] for row in records])
    return {
        "lines": records,
        "raw_rms_mean_rad": float(np.mean(raw)),
        "corrected_rms_mean_rad": float(np.mean(corrected)),
        "improvement_factor": float(np.mean(raw) / max(np.mean(corrected), 1e-30)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--case-file", type=Path, default=Path("examples/cem_qh03.json"))
    parser.add_argument("--case-key", default="raw")
    parser.add_argument("--current-unit", default="A")
    parser.add_argument("--s-edge", type=float, default=0.16)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--orders", default="4:4:4,6:6:6,8:8:8,10:10:12")
    parser.add_argument("--iota-degree", type=int, default=3)
    parser.add_argument("--train-points", type=int, default=120000)
    parser.add_argument("--validation-points", type=int, default=60000)
    parser.add_argument("--grid-xy", type=int, default=144)
    parser.add_argument("--grid-phi", type=int, default=96)
    parser.add_argument("--rho-min", type=float, default=0.06)
    parser.add_argument(
        "--minimum-candidate-valid-fraction",
        type=float,
        default=0.95,
        help=(
            "Minimum valid fraction of the generated ray candidates. The fixed "
            "train+validation point budget is always required independently."
        ),
    )
    parser.add_argument(
        "--sampling-backend",
        choices=("gpu-ray", "legacy-cartesian"),
        default="gpu-ray",
    )
    parser.add_argument("--relative-weighting", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp64"), default="fp32")
    parser.add_argument(
        "--gpu-lib",
        type=Path,
        default=Path("gpu_backend/build_mixed/libstellarator_gpu.so"),
    )
    parser.add_argument("--skip-fieldline-plot", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    model = load_psi_model(args.run_dir / "psi_model.npz")
    field_input = load_field_input(args.case_file, args.case_key)
    psi_cfg = PsiFitConfig(
        backend="gpu",
        gpu_lib_path=str(resolve_project_path(args.gpu_lib)),
        gpu_device=0,
        gpu_segments_per_coil=256,
    )
    from stellarator_eval.psi import _make_gpu_field

    gpu_field = _make_gpu_field(field_input, model.nfp, psi_cfg, args.current_unit)
    try:
        stage_timings = {}
        b_sampler = lambda r, z, p: _b_components_gpu(
            gpu_field, r, z, p, precision=args.precision
        )
        calibration_levels = args.s_edge * np.asarray(
            [0.015625, 0.03125, 0.0625, 0.10, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81, 1.0]
        )
        stage_start = time.perf_counter()
        if args.sampling_backend == "gpu-ray":
            volume_config = VolumeQSConfig(
                s_edge=args.s_edge,
                rho_min=args.rho_min,
                point_count=args.train_points + args.validation_points,
                alpha_fit_point_count=args.train_points + args.validation_points,
                minimum_candidate_valid_fraction=(
                    args.minimum_candidate_valid_fraction
                ),
                grid_xy=args.grid_xy,
                grid_phi=args.grid_phi,
                precision=args.precision,
                gpu_device=0,
                gpu_lib_path=str(resolve_project_path(args.gpu_lib)),
            )
            calibration = calibrate_toroidal_flux_gpu(
                model,
                gpu_field,
                volume_config,
                levels=calibration_levels,
            )
        else:
            calibration = calibrate_toroidal_flux(
                model,
                calibration_levels,
                b_sampler,
                phi_count=8,
                theta_count=256,
                radial_quadrature=24,
                polynomial_degree=4,
            )
        stage_timings["flux_calibration_s"] = float(time.perf_counter() - stage_start)
        calibration.diagnostics["pchip_comparison"] = pchip_flux_diagnostic(calibration)
        plot_flux_calibration(calibration, args.out_dir / "flux_calibration.png")
        np.savez(
            args.out_dir / "flux_calibration.npz",
            s_knots=calibration.s_knots,
            psi_knots=calibration.psi_knots,
            psi_by_section=calibration.psi_by_section,
            phi_sections=calibration.phi_sections,
            polynomial_coeffs=calibration.polynomial_coeffs,
        )

        if args.sampling_backend == "gpu-ray":
            stage_start = time.perf_counter()
            points = sample_volume_points(model, volume_config, device=args.device)
            stage_timings["volume_sampling_s"] = float(time.perf_counter() - stage_start)
            stage_start = time.perf_counter()
            apply_flux_coordinates(points, calibration)
            coordinates = alpha_coordinates_from_volume_points(points)
            stage_timings["coordinate_construction_s"] = float(
                time.perf_counter() - stage_start
            )
            stage_start = time.perf_counter()
            sampled_B = gpu_field.eval_B(points["xyz"], precision=args.precision)
            stage_timings["field_sampling_s"] = float(time.perf_counter() - stage_start)
            train_indices, validation_indices = disjoint_train_validation_indices(
                len(sampled_B), args.train_points, args.validation_points
            )
            train_coordinates = {
                key: value[train_indices] for key, value in coordinates.items()
            }
            validation_coordinates = {
                key: value[validation_indices] for key, value in coordinates.items()
            }
            train_B = sampled_B[train_indices]
            validation_B = sampled_B[validation_indices]
            sampling_diagnostics = {
                "candidate_points": int(points["candidate_count"][0]),
                "available_points": int(points["available_count"][0]),
                "minimum_required_points": int(points["minimum_count"][0]),
                "candidate_valid_fraction": float(
                    points["candidate_valid_fraction"][0]
                ),
                "sampled_points": int(len(sampled_B)),
            }
        else:
            stage_start = time.perf_counter()
            train = sample_uniform_volume(
                model,
                args.s_edge,
                count=args.train_points,
                grid_xy=args.grid_xy,
                grid_phi=args.grid_phi,
                rho_min=args.rho_min,
                offset_seed=1,
            )
            validation = sample_uniform_volume(
                model,
                args.s_edge,
                count=args.validation_points,
                grid_xy=args.grid_xy + 1,
                grid_phi=args.grid_phi + 1,
                rho_min=args.rho_min,
                offset_seed=7,
            )
            stage_timings["volume_sampling_s"] = float(time.perf_counter() - stage_start)
            stage_start = time.perf_counter()
            train_coordinates = physical_coordinate_data(model, calibration, *train)
            validation_coordinates = physical_coordinate_data(model, calibration, *validation)
            stage_timings["coordinate_construction_s"] = float(
                time.perf_counter() - stage_start
            )
            stage_start = time.perf_counter()
            train_B = sample_bfield(
                gpu_field, train[0], train[1], train[2], precision=args.precision
            )
            validation_B = sample_bfield(
                gpu_field,
                validation[0],
                validation[1],
                validation[2],
                precision=args.precision,
            )
            stage_timings["field_sampling_s"] = float(time.perf_counter() - stage_start)
            sampling_diagnostics = {}

        baseline = AlphaFitResult(
            modes=[],
            lambda_coeffs=np.zeros(0),
            iota_coeffs=np.zeros(1),
            radial_order=0,
            poloidal_order=0,
            toroidal_order=0,
            iota_degree=0,
            diagnostics={},
        )
        baseline_training, _ = evaluate_alpha_fit(
            baseline, train_coordinates, train_B, model.nfp
        )
        baseline_validation, _ = evaluate_alpha_fit(
            baseline, validation_coordinates, validation_B, model.nfp
        )

        order_rows = []
        fitted_results: list[AlphaFitResult] = []
        for radial, poloidal, toroidal in parse_orders(args.orders):
            label = f"L{radial}_M{poloidal}_N{toroidal}"
            result = fit_alpha_gpu_qr(
                train_coordinates,
                train_B,
                nfp=model.nfp,
                radial_order=radial,
                poloidal_order=poloidal,
                toroidal_order=toroidal,
                iota_degree=args.iota_degree,
                relative_weighting=args.relative_weighting,
                device=args.device,
                precision=args.precision,
            )
            training_metrics, _ = evaluate_alpha_fit(result, train_coordinates, train_B, model.nfp)
            validation_metrics, _ = evaluate_alpha_fit(
                result, validation_coordinates, validation_B, model.nfp
            )
            save_fit(args.out_dir / f"alpha_fit_{label}.npz", result)
            order_rows.append(
                {
                    "label": label,
                    "orders": [radial, poloidal, toroidal],
                    "fit": result.diagnostics,
                    "training": training_metrics,
                    "validation": validation_metrics,
                    "iota_coeffs": result.iota_coeffs.tolist(),
                }
            )
            fitted_results.append(result)

        best_index = int(np.argmin([row["validation"]["relative_l2"] for row in order_rows]))
        best = fitted_results[best_index]
        plot_order_scan(order_rows, args.out_dir / "order_scan.png")
        plot_coordinate_invertibility(best, model.nfp, args.out_dir / "coordinate_invertibility.png")

        straightness = None
        if not args.skip_fieldline_plot:
            built = build_field(field_input, current_unit=args.current_unit)
            straightness = trace_straightness(
                built.field,
                model,
                calibration,
                best,
                args.s_edge,
                periods=8,
                lines_per_surface=4,
                samples_per_period=64,
                steps_per_period=480,
                path=args.out_dir / "fieldline_straightening.png",
            )

        summary = {
            "case": args.case_file.name,
            "run_dir": str(args.run_dir),
            "s_edge": args.s_edge,
            "nfp": model.nfp,
            "sampling": {
                "backend": args.sampling_backend,
                "kind": (
                    "equal_area_ray_lattice_gpu_psi"
                    if args.sampling_backend == "gpu-ray"
                    else "shifted_uniform_cartesian_lattice_filtered_by_s"
                ),
                "train_points": args.train_points,
                "validation_points": args.validation_points,
                "grid_xy": args.grid_xy,
                "grid_phi": args.grid_phi,
                "rho_min": args.rho_min,
                **sampling_diagnostics,
            },
            "backends": {
                "flux_calibration_field": (
                    f"vectorized C++/CUDA {args.precision}"
                    if args.sampling_backend == "gpu-ray"
                    else f"batched C++/CUDA {args.precision}"
                ),
                "training_and_validation_field": f"C++/CUDA {args.precision}",
                "alpha_design_and_qr": f"PyTorch CUDA {args.precision} gels",
                "coordinate_sampling_and_diagnostics": (
                    f"PyTorch CUDA {args.precision} psi + NumPy geometry"
                    if args.sampling_backend == "gpu-ray"
                    else "NumPy/SciPy CPU"
                ),
            },
            "stage_timings": stage_timings,
            "calibration": calibration.diagnostics,
            "calibration_polynomial_coeffs": calibration.polynomial_coeffs.tolist(),
            "baseline_lambda0_iota0": {
                "training": baseline_training,
                "validation": baseline_validation,
            },
            "orders": order_rows,
            "best_index": best_index,
            "best_label": order_rows[best_index]["label"],
            "straightness": straightness,
            "total_time_s": float(time.perf_counter() - started),
        }
        write_json(args.out_dir / "summary.json", summary)
    finally:
        gpu_field.close()


if __name__ == "__main__":
    main()
