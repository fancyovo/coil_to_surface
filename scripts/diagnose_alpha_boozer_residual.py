from __future__ import annotations

import argparse
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

from simsopt.geo import SurfaceXYZTensorFourier, boozer_surface_residual

from stellarator_eval.alpha_clebsch import evaluate_lambda, load_alpha_fit
from stellarator_eval.config import BoozerConfig, SurfaceScanConfig
from stellarator_eval.field import build_field
from stellarator_eval.serialization import write_json
from stellarator_eval.surface import surface_points_from_level_gpu
from scripts.desc_psi_volume_initial_guess_experiment import (
    dataclass_from_dict,
    load_field_input,
    load_json,
    load_psi_model,
    make_xyz_surface,
)

TWOPI = 2.0 * np.pi
MU0 = 4.0e-7 * np.pi


def parse_floats(text: str) -> np.ndarray:
    return np.asarray([float(value.strip()) for value in text.split(",") if value.strip()])


def parse_ints(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def periodic_resample(parameter, values, target):
    parameter = np.mod(np.asarray(parameter, dtype=float), TWOPI)
    order = np.argsort(parameter)
    parameter = parameter[order]
    values = np.asarray(values)[order]
    keep = np.r_[True, np.diff(parameter) > 1e-12]
    parameter = parameter[keep]
    values = values[keep]
    xp = np.r_[parameter[-1] - TWOPI, parameter, parameter[0] + TWOPI]
    result = np.empty((len(target), values.shape[1]))
    for component in range(values.shape[1]):
        fp = np.r_[values[-1, component], values[:, component], values[0, component]]
        result[:, component] = np.interp(target, xp, fp)
    return result


def reparameterize_surface(xyz, rho: float, nfp: int, alpha_fit, variant: str):
    nphi, ntheta, _ = xyz.shape
    phi = np.arange(nphi) * TWOPI / (nfp * nphi)
    theta_ccw = np.arange(ntheta) * TWOPI / ntheta
    target = theta_ccw.copy()
    output = np.empty_like(xyz)
    jacobian_min = float("nan")
    jacobian_fold_fraction = float("nan")
    jacobian_values = []
    for iphi, phi_value in enumerate(phi):
        if variant == "pipeline_ccw":
            parameter = theta_ccw
        else:
            theta_clockwise = -theta_ccw
            if variant == "geometric_clockwise":
                lam = np.zeros_like(theta_clockwise)
                lambda_theta = np.zeros_like(theta_clockwise)
            elif variant == "alpha_clockwise":
                lam, lambda_theta, _ = evaluate_lambda(
                    alpha_fit,
                    np.full(ntheta, rho),
                    theta_clockwise,
                    np.full(ntheta, phi_value),
                    nfp,
                )
            else:
                raise ValueError(f"unknown parameterization variant {variant!r}")
            parameter = theta_clockwise + lam
            jacobian_values.append(1.0 + lambda_theta)
        output[iphi] = periodic_resample(parameter, xyz[iphi], target)
    if jacobian_values:
        values = np.concatenate(jacobian_values)
        jacobian_min = float(np.min(values))
        jacobian_fold_fraction = float(np.mean(values <= 0.0))
    return output, {
        "one_plus_lambda_theta_min": jacobian_min,
        "noninvertible_fraction": jacobian_fold_fraction,
    }


def fit_tensor_surface(xyz, nfp: int, order: int, stellsym: bool):
    nphi, ntheta, _ = xyz.shape
    surface = SurfaceXYZTensorFourier(
        mpol=order,
        ntor=order,
        stellsym=stellsym,
        nfp=nfp,
        quadpoints_phi=np.arange(nphi) / (nfp * nphi),
        quadpoints_theta=np.arange(ntheta) / ntheta,
    )
    surface.least_squares_fit(xyz)
    distance = np.linalg.norm(surface.gamma() - xyz, axis=2)
    return surface, {
        "spectral_fit_rms_m": float(np.sqrt(np.mean(distance * distance))),
        "spectral_fit_max_m": float(np.max(distance)),
    }


def validation_surface(surface, *, nphi: int, ntheta: int):
    evaluated = SurfaceXYZTensorFourier(
        mpol=surface.mpol,
        ntor=surface.ntor,
        stellsym=surface.stellsym,
        nfp=surface.nfp,
        quadpoints_phi=(np.arange(nphi) + 0.371) / (surface.nfp * nphi),
        quadpoints_theta=(np.arange(ntheta) + 0.413) / ntheta,
    )
    evaluated.set_dofs(surface.get_dofs())
    return evaluated


class GpuBOnlyFieldAdapter:
    """Expose the subset of the Simsopt field API needed by residual checks."""

    def __init__(self, gpu_field, *, precision: str = "fp64"):
        self.gpu_field = gpu_field
        self.precision = precision
        self._points = None
        self._field = None

    def set_points(self, points) -> None:
        self._points = np.ascontiguousarray(points, dtype=float).reshape((-1, 3))
        self._field = None

    def compute(self, derivatives: int = 0) -> None:
        if derivatives != 0:
            raise ValueError("GpuBOnlyFieldAdapter supports B evaluation only")
        self._evaluate()

    def B(self):
        return self._evaluate()

    def _evaluate(self):
        if self._points is None:
            raise RuntimeError("set_points must be called before B")
        if self._field is None:
            self._field = np.asarray(
                self.gpu_field.eval_B(self._points, precision=self.precision),
                dtype=float,
            )
        return self._field


def residual_for_iota_G(surface, field, iota: float, G: float):
    xyz = surface.gamma()
    xphi = surface.gammadash1()
    xtheta = surface.gammadash2()
    field.set_points(xyz.reshape((-1, 3)))
    B = field.B().reshape(xyz.shape)
    B2 = np.sum(B * B, axis=2)
    Bnorm = np.sqrt(B2)
    tangent = xphi + iota * xtheta
    residual = G * B - B2[..., None] * tangent
    weighted = residual / Bnorm[..., None]
    point_relative = np.linalg.norm(residual, axis=2) / np.maximum(abs(G) * Bnorm, 1e-30)

    tangent_parallel = (
        np.sum(tangent * B, axis=2) / np.maximum(B2, 1e-30)
    )[..., None] * B
    tangent_perpendicular = tangent - tangent_parallel
    tangent_norm = np.linalg.norm(tangent, axis=2)
    direction_sine = np.linalg.norm(tangent_perpendicular, axis=2) / np.maximum(
        tangent_norm, 1e-30
    )
    local_G = np.sum(B * tangent, axis=2)
    normal = surface.normal()
    normal_sine = np.abs(np.sum(B * normal, axis=2)) / np.maximum(
        Bnorm * np.linalg.norm(normal, axis=2), 1e-30
    )
    return {
        "iota": float(iota),
        "G": float(G),
        "simsopt_residual_norm": float(
            np.linalg.norm(boozer_surface_residual(surface, iota, G, field, derivatives=0)[0])
        ),
        "relative_l2": float(
            np.linalg.norm(weighted) / max(abs(G) * np.sqrt(Bnorm.size), 1e-30)
        ),
        "point_relative_mean": float(np.mean(point_relative)),
        "point_relative_p95": float(np.percentile(point_relative, 95)),
        "point_relative_max": float(np.max(point_relative)),
        "direction_sine_l2": float(
            np.linalg.norm(tangent_perpendicular) / max(np.linalg.norm(tangent), 1e-30)
        ),
        "direction_angle_p95_deg": float(
            np.degrees(np.arcsin(np.clip(np.percentile(direction_sine, 95), 0.0, 1.0)))
        ),
        "local_G_mean": float(np.mean(local_G)),
        "local_G_relative_std": float(np.std(local_G) / max(abs(np.mean(local_G)), 1e-30)),
        "local_G_relative_deviation_p95": float(
            np.percentile(np.abs(local_G - np.mean(local_G)), 95)
            / max(abs(np.mean(local_G)), 1e-30)
        ),
        "normal_B_sine_mean": float(np.mean(normal_sine)),
        "normal_B_sine_p95": float(np.percentile(normal_sine, 95)),
    }


def optimize_iota_G_on_fixed_surface(surface, field):
    xyz = surface.gamma()
    xphi = surface.gammadash1()
    xtheta = surface.gammadash2()
    field.set_points(xyz.reshape((-1, 3)))
    B = field.B().reshape(xyz.shape)
    B2 = np.sum(B * B, axis=2)
    Bnorm = np.sqrt(B2)
    matrix = np.column_stack(
        [
            (B / Bnorm[..., None]).reshape(-1),
            (-B2[..., None] * xtheta / Bnorm[..., None]).reshape(-1),
        ]
    )
    rhs = (B2[..., None] * xphi / Bnorm[..., None]).reshape(-1)
    values, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    return residual_for_iota_G(surface, field, iota=float(values[1]), G=float(values[0]))


def fixed_iota_best_G(surface, field, iota: float):
    xyz = surface.gamma()
    xphi = surface.gammadash1()
    xtheta = surface.gammadash2()
    field.set_points(xyz.reshape((-1, 3)))
    B = field.B().reshape(xyz.shape)
    B2 = np.sum(B * B, axis=2)
    Bnorm = np.sqrt(B2)
    column = (B / Bnorm[..., None]).reshape(-1)
    rhs = (B2[..., None] * (xphi + iota * xtheta) / Bnorm[..., None]).reshape(-1)
    G = float(np.dot(column, rhs) / np.dot(column, column))
    return residual_for_iota_G(surface, field, iota=iota, G=G)


def plot_residual_sweep(rows, output: Path):
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))
    styles = {
        ("geometric_clockwise", 6): ("#7f8c8d", "--", "geometric, order 6"),
        ("alpha_clockwise", 6): ("#d55e00", "-", "alpha, order 6"),
        ("geometric_clockwise", 12): ("#4d4d4d", ":", "geometric, order 12"),
        ("alpha_clockwise", 12): ("#0072b2", "-", "alpha, order 12"),
    }
    for key, (color, linestyle, label) in styles.items():
        selected = sorted(
            [row for row in rows if (row["variant"], row["surface_order"]) == key],
            key=lambda row: row["rho"],
        )
        if not selected:
            continue
        rho = [row["rho"] for row in selected]
        axes[0].semilogy(
            rho,
            [
                row["fixed_reference_iota_best_G"]["relative_l2"]
                for row in selected
            ],
            marker="o",
            ms=4,
            color=color,
            ls=linestyle,
            label=label,
        )
        axes[1].semilogy(
            rho,
            [
                max(row["fixed_reference_iota_best_G"]["direction_angle_p95_deg"], 1e-8)
                for row in selected
            ],
            marker="o",
            ms=4,
            color=color,
            ls=linestyle,
        )
        axes[2].semilogy(
            rho,
            [row["fixed_reference_iota_best_G"]["local_G_relative_std"] for row in selected],
            marker="o",
            ms=4,
            color=color,
            ls=linestyle,
        )
    axes[0].set_ylabel("normalized Simsopt Boozer residual")
    axes[1].set_ylabel("direction mismatch p95 [deg]")
    axes[2].set_ylabel(r"std$(G_{local})/|mean(G_{local})|$")
    for axis in axes:
        axis.set_xlabel(r"$\rho$")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle("Fixed-surface Boozer diagnostics: no surface optimization")
    figure.tight_layout()
    figure.savefig(output, dpi=190)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument("--alpha-dir", type=Path, required=True)
    parser.add_argument("--alpha-fit", default="alpha_fit_L12_M12_N16.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rho-values", default="0.12,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--surface-orders", default="6,12")
    parser.add_argument("--extract-order", type=int, default=24)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    summary = load_json(args.run_dir / "summary.json")
    config = summary["config"]
    nfp = int(summary["nfp"])
    current_unit = config.get("current_unit") or "A"
    s_edge = float(summary["best_surface"]["psi_level"])
    model = load_psi_model(args.run_dir / "psi_model.npz")
    alpha_fit = load_alpha_fit(args.alpha_dir / args.alpha_fit)
    field_input = load_field_input(args.case_file, "raw")
    built = build_field(field_input, current_unit=current_unit)
    field = built.field
    scan_cfg = dataclass_from_dict(SurfaceScanConfig, config.get("scan"))
    boozer_cfg = dataclass_from_dict(BoozerConfig, config.get("boozer"))
    stellsym = bool(boozer_cfg.stellsym)
    theoretical_G = float(MU0 * sum(abs(coil.current.get_value()) for coil in field.coils))

    rows = []
    rho_values = parse_floats(args.rho_values)
    orders = parse_ints(args.surface_orders)
    for rho in rho_values:
        xyz, radii, extraction = surface_points_from_level_gpu(
            model,
            s_edge * rho * rho,
            args.extract_order,
            scan_cfg,
            boozer_cfg,
        )
        for variant in ("pipeline_ccw", "geometric_clockwise", "alpha_clockwise"):
            parameterized, transform = reparameterize_surface(
                xyz, float(rho), nfp, alpha_fit, variant
            )
            for order in orders:
                fitted, fit_info = fit_tensor_surface(
                    parameterized, nfp, order, stellsym
                )
                evaluated = validation_surface(
                    fitted, nphi=4 * order + 5, ntheta=4 * order + 7
                )
                optimized = optimize_iota_G_on_fixed_surface(evaluated, field)
                alpha_iota = float(alpha_fit.iota(np.asarray([rho]))[0])
                reference_iota = -alpha_iota if variant == "pipeline_ccw" else alpha_iota
                fixed_reference_iota = fixed_iota_best_G(
                    evaluated, field, reference_iota
                )
                theoretical = residual_for_iota_G(
                    evaluated, field, iota=reference_iota, G=theoretical_G
                )
                rows.append(
                    {
                        "rho": float(rho),
                        "s_level": float(s_edge * rho * rho),
                        "variant": variant,
                        "surface_order": int(order),
                        "radius_min_m": float(np.min(radii)),
                        "radius_mean_m": float(np.mean(radii)),
                        "radius_max_m": float(np.max(radii)),
                        "extraction": {key: float(value) for key, value in extraction.items()},
                        "transform": transform,
                        "spectral_fit": fit_info,
                        "optimized": optimized,
                        "reference_iota": float(reference_iota),
                        "fixed_reference_iota_best_G": fixed_reference_iota,
                        "fixed_reference_iota_theoretical_G": theoretical,
                    }
                )

    level_dir = args.run_dir / f"level_{s_edge:.6g}".replace(".", "p")
    saved_surface, saved_meta = make_xyz_surface(
        level_dir / "boozer_surface.npz",
        nfp=nfp,
        order=int(config["boozer"]["surface_order"]),
        stellsym=stellsym,
        nphi=53,
        ntheta=55,
    )
    saved_fixed = residual_for_iota_G(
        saved_surface,
        field,
        iota=float(saved_meta["iota"]),
        G=float(saved_meta["G"]),
    )
    saved_optimized = optimize_iota_G_on_fixed_surface(saved_surface, field)

    plot_residual_sweep(rows, args.output_dir / "boozer_residual_vs_rho.png")
    output = {
        "case": args.case_file.name,
        "s_edge": s_edge,
        "alpha_fit": args.alpha_fit,
        "alpha_iota_coeffs": alpha_fit.iota_coeffs.tolist(),
        "theoretical_G": theoretical_G,
        "surface_optimization_performed": False,
        "spectral_projection_only": True,
        "rows": rows,
        "saved_boozer_surface": {
            "metadata": saved_meta,
            "fixed_metadata_iota_G": saved_fixed,
            "optimized_iota_G_on_fixed_surface": saved_optimized,
        },
        "total_time_s": float(time.perf_counter() - started),
    }
    write_json(args.output_dir / "summary.json", output)


if __name__ == "__main__":
    main()
