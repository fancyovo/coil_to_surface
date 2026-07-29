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
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simsopt.geo import SurfaceXYZTensorFourier

from stellarator_eval.alpha_clebsch import load_alpha_fit
from stellarator_eval.config import BoozerConfig, SurfaceScanConfig
from stellarator_eval.field import build_field
from stellarator_eval.serialization import write_json
from stellarator_eval.surface import surface_points_from_level_gpu
from stellarator_eval.toroidal_correction import (
    ToroidalCorrectionFit,
    evaluate_toroidal_correction,
    fit_toroidal_correction,
)
from scripts.desc_psi_volume_initial_guess_experiment import (
    dataclass_from_dict,
    load_field_input,
    load_json,
    load_psi_model,
)
from scripts.diagnose_alpha_boozer_residual import (
    fit_tensor_surface,
    parse_floats,
    parse_ints,
    reparameterize_surface,
    residual_for_iota_G,
)


TWOPI = 2.0 * np.pi


def sampled_surface(
    surface,
    *,
    nphi: int,
    ntheta: int,
    phi_shift: float = 0.0,
    theta_shift: float = 0.0,
):
    sampled = SurfaceXYZTensorFourier(
        mpol=surface.mpol,
        ntor=surface.ntor,
        stellsym=surface.stellsym,
        nfp=surface.nfp,
        quadpoints_phi=(np.arange(nphi) + phi_shift) / (surface.nfp * nphi),
        quadpoints_theta=(np.arange(ntheta) + theta_shift) / ntheta,
    )
    sampled.set_dofs(surface.get_dofs())
    return sampled


def surface_field_data(surface, field, iota: float) -> dict[str, np.ndarray]:
    xyz = surface.gamma()
    xphi = surface.gammadash1()
    xtheta = surface.gammadash2()
    field.set_points(xyz.reshape((-1, 3)))
    B = field.B().reshape(xyz.shape)
    B2 = np.sum(B * B, axis=2)
    tangent = xphi + iota * xtheta
    local_G = np.sum(B * tangent, axis=2)
    return {
        "xyz": xyz,
        "B": B,
        "B2": B2,
        "tangent": tangent,
        "local_G": local_G,
    }


def scalar_residual_metrics(
    data: dict[str, np.ndarray],
    *,
    G: float,
    scale: np.ndarray,
) -> dict[str, float]:
    B = data["B"]
    B2 = data["B2"]
    tangent = data["tangent"] / scale[..., None]
    Bnorm = np.sqrt(B2)
    residual = G * B - B2[..., None] * tangent
    weighted = residual / Bnorm[..., None]
    local_G = np.sum(B * tangent, axis=2)
    tangent_parallel = (
        np.sum(tangent * B, axis=2) / np.maximum(B2, 1e-30)
    )[..., None] * B
    direction_sine = np.linalg.norm(tangent - tangent_parallel, axis=2) / np.maximum(
        np.linalg.norm(tangent, axis=2), 1e-30
    )
    return {
        "relative_l2": float(
            np.linalg.norm(weighted) / max(abs(G) * np.sqrt(Bnorm.size), 1e-30)
        ),
        "direction_angle_p95_deg": float(
            np.degrees(np.arcsin(np.clip(np.percentile(direction_sine, 95), 0.0, 1.0)))
        ),
        "local_G_mean": float(np.mean(local_G)),
        "local_G_relative_std": float(
            np.std(local_G) / max(abs(np.mean(local_G)), 1e-30)
        ),
        "point_relative_p95": float(
            np.percentile(
                np.linalg.norm(residual, axis=2)
                / np.maximum(abs(G) * Bnorm, 1e-30),
                95,
            )
        ),
    }


def periodic_surface_interpolator(surface, *, nphi: int, ntheta: int):
    sampled = sampled_surface(surface, nphi=nphi, ntheta=ntheta)
    xyz = sampled.gamma()
    phi = np.arange(nphi)[:, None] / (surface.nfp * nphi)
    radius = np.sqrt(xyz[:, :, 0] ** 2 + xyz[:, :, 1] ** 2)
    angle_offset = np.angle(
        np.exp(1j * (np.arctan2(xyz[:, :, 1], xyz[:, :, 0]) - TWOPI * phi))
    )
    z = xyz[:, :, 2]

    def interpolate(old_phi, old_theta):
        coordinates = np.vstack(
            [
                np.mod(old_phi * surface.nfp, 1.0).ravel() * nphi,
                np.mod(old_theta, 1.0).ravel() * ntheta,
            ]
        )
        shape = np.shape(old_phi)
        evaluated_radius = map_coordinates(
            radius, coordinates, order=5, mode="grid-wrap"
        ).reshape(shape)
        evaluated_offset = map_coordinates(
            angle_offset, coordinates, order=5, mode="grid-wrap"
        ).reshape(shape)
        evaluated_z = map_coordinates(z, coordinates, order=5, mode="grid-wrap").reshape(
            shape
        )
        geometric_angle = TWOPI * old_phi + evaluated_offset
        return np.stack(
            [
                evaluated_radius * np.cos(geometric_angle),
                evaluated_radius * np.sin(geometric_angle),
                evaluated_z,
            ],
            axis=2,
        )

    return interpolate


def corrected_surface(
    source,
    correction: ToroidalCorrectionFit,
    *,
    output_order: int,
    nphi: int,
    ntheta: int,
    interpolation_size: int,
) -> tuple[SurfaceXYZTensorFourier, dict[str, float]]:
    phi_b = np.arange(nphi)[:, None] / (source.nfp * nphi)
    theta_b = np.arange(ntheta)[None, :] / ntheta
    phi_b, theta_b = np.broadcast_arrays(phi_b, theta_b)
    alpha = theta_b - correction.iota * phi_b
    old_phi = phi_b.copy()
    iteration_count = 0
    for iteration_count in range(1, 21):
        old_theta = alpha + correction.iota * old_phi
        nu, _, _, along_field = evaluate_toroidal_correction(
            correction, old_theta, old_phi
        )
        residual = old_phi + nu - phi_b
        jacobian = 1.0 + along_field
        if np.min(jacobian) <= 0.0:
            raise RuntimeError(
                f"non-invertible toroidal correction: min Jacobian={np.min(jacobian):.6g}"
            )
        old_phi -= residual / jacobian
        if np.max(np.abs(residual)) < 1e-13:
            break
    old_theta = alpha + correction.iota * old_phi
    nu, _, _, along_field = evaluate_toroidal_correction(
        correction, old_theta, old_phi
    )
    inversion_residual = old_phi + nu - phi_b
    interpolate = periodic_surface_interpolator(
        source, nphi=interpolation_size, ntheta=interpolation_size
    )
    xyz = interpolate(old_phi, old_theta)
    fitted, spectral_fit = fit_tensor_surface(
        xyz, source.nfp, output_order, source.stellsym
    )
    return fitted, {
        "newton_iterations": int(iteration_count),
        "inversion_residual_max_turns": float(np.max(np.abs(inversion_residual))),
        "mapping_jacobian_min": float(np.min(1.0 + along_field)),
        "mapping_jacobian_max": float(np.max(1.0 + along_field)),
        **spectral_fit,
    }


def plot_radial(rows: list[dict], selected_order: int, output: Path) -> None:
    selected = sorted(
        [row for row in rows if row["nu_order"] == selected_order],
        key=lambda row: row["rho"],
    )
    rho = [row["rho"] for row in selected]
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))
    axes[0].semilogy(
        rho,
        [row["before"]["relative_l2"] for row in selected],
        "o--",
        color="#777777",
        label="alpha only",
    )
    axes[0].semilogy(
        rho,
        [row["analytic_corrected"]["relative_l2"] for row in selected],
        "o-",
        color="#d55e00",
        label="analytic nu correction",
    )
    axes[0].semilogy(
        rho,
        [row["simsopt_corrected"]["relative_l2"] for row in selected],
        "o-",
        color="#0072b2",
        label="reprojected Simsopt surface",
    )
    axes[1].semilogy(
        rho,
        [row["before"]["direction_angle_p95_deg"] for row in selected],
        "o--",
        color="#777777",
        label="alpha only",
    )
    axes[1].semilogy(
        rho,
        [row["simsopt_corrected"]["direction_angle_p95_deg"] for row in selected],
        "o-",
        color="#0072b2",
        label="alpha + nu",
    )
    axes[2].semilogy(
        rho,
        [row["before"]["local_G_relative_std"] for row in selected],
        "o--",
        color="#777777",
        label="alpha only",
    )
    axes[2].semilogy(
        rho,
        [row["simsopt_corrected"]["local_G_relative_std"] for row in selected],
        "o-",
        color="#0072b2",
        label="alpha + nu",
    )
    axes[0].set_ylabel("normalized Simsopt Boozer residual")
    axes[1].set_ylabel("direction mismatch p95 [deg]")
    axes[2].set_ylabel(r"std$(G_{local})/|mean(G_{local})|$")
    for axis in axes:
        axis.set_xlabel(r"$\rho$")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].legend(frameon=False, fontsize=9)
    axes[2].legend(frameon=False, fontsize=9)
    figure.suptitle(
        f"Fixed-surface toroidal correction (nu Fourier order {selected_order})"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=190)
    plt.close(figure)


def plot_order_scan(rows: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
    available_rho = sorted({row["rho"] for row in rows})
    requested = [0.2, 0.5, 0.8, 1.0]
    colors = ["#009e73", "#0072b2", "#e69f00", "#d55e00"]
    for target, color in zip(requested, colors):
        rho = min(available_rho, key=lambda value: abs(value - target))
        selected = sorted(
            [row for row in rows if row["rho"] == rho],
            key=lambda row: row["nu_order"],
        )
        orders = [row["nu_order"] for row in selected]
        axes[0].semilogy(
            orders,
            [row["simsopt_corrected"]["relative_l2"] for row in selected],
            "o-",
            color=color,
            label=fr"$\rho={rho:.1f}$",
        )
        axes[1].semilogy(
            orders,
            [row["nu_fit"]["relative_l2"] for row in selected],
            "o-",
            color=color,
        )
    axes[0].set_ylabel("corrected Simsopt residual")
    axes[1].set_ylabel(r"MDE fit $\|D\nu-h\|/\|h\|$")
    for axis in axes:
        axis.set_xlabel(r"$\nu$ Fourier order")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    figure.suptitle("Toroidal-correction spectral convergence")
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
    parser.add_argument("--s-edge", type=float, default=None)
    parser.add_argument(
        "--rho-values", default="0.12,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
    )
    parser.add_argument("--nu-orders", default="4,8,12")
    parser.add_argument("--surface-order", type=int, default=12)
    parser.add_argument("--extract-order", type=int, default=24)
    parser.add_argument("--fit-nphi", type=int, default=65)
    parser.add_argument("--fit-ntheta", type=int, default=67)
    parser.add_argument("--validation-nphi", type=int, default=57)
    parser.add_argument("--validation-ntheta", type=int, default=59)
    parser.add_argument("--interpolation-size", type=int, default=97)
    parser.add_argument("--regularization", type=float, default=0.0)
    parser.add_argument("--save-surfaces", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    summary = load_json(args.run_dir / "summary.json")
    config = summary["config"]
    nfp = int(summary["nfp"])
    s_edge = (
        float(args.s_edge)
        if args.s_edge is not None
        else float(summary["best_surface"]["psi_level"])
    )
    model = load_psi_model(args.run_dir / "psi_model.npz")
    alpha_fit = load_alpha_fit(args.alpha_dir / args.alpha_fit)
    field_input = load_field_input(args.case_file, "raw")
    field = build_field(
        field_input, current_unit=config.get("current_unit") or "A"
    ).field
    scan_cfg = dataclass_from_dict(SurfaceScanConfig, config.get("scan"))
    boozer_cfg = dataclass_from_dict(BoozerConfig, config.get("boozer"))

    rows = []
    saved_surfaces = []
    rho_values = parse_floats(args.rho_values)
    nu_orders = parse_ints(args.nu_orders)
    selected_order = max(nu_orders)
    surface_dir = args.output_dir / "surfaces"
    if args.save_surfaces:
        surface_dir.mkdir()
    for rho in rho_values:
        xyz, radii, extraction = surface_points_from_level_gpu(
            model,
            s_edge * rho * rho,
            args.extract_order,
            scan_cfg,
            boozer_cfg,
        )
        parameterized, alpha_transform = reparameterize_surface(
            xyz, float(rho), nfp, alpha_fit, "alpha_clockwise"
        )
        alpha_surface, alpha_projection = fit_tensor_surface(
            parameterized,
            nfp,
            args.surface_order,
            bool(boozer_cfg.stellsym),
        )
        iota = float(alpha_fit.iota(np.asarray([rho]))[0])
        training = sampled_surface(
            alpha_surface, nphi=args.fit_nphi, ntheta=args.fit_ntheta
        )
        training_data = surface_field_data(training, field, iota)
        G = float(np.mean(training_data["local_G"]))
        surface_stem = f"rho_{rho:.6g}".replace(".", "p")
        if args.save_surfaces:
            alpha_path = surface_dir / f"{surface_stem}_alpha.npz"
            np.savez(
                alpha_path,
                dofs=alpha_surface.get_dofs(),
                iota=iota,
                G=G,
                nfp=nfp,
                order=args.surface_order,
                stellsym=bool(boozer_cfg.stellsym),
                rho=float(rho),
                s_edge=s_edge,
                s_level=float(s_edge * rho * rho),
                radius_mean_m=float(np.mean(radii)),
                spectral_fit_rms_m=alpha_projection["spectral_fit_rms_m"],
                kind="alpha",
            )
        target = training_data["local_G"] / G - 1.0
        train_phi = np.arange(args.fit_nphi)[:, None] / (nfp * args.fit_nphi)
        train_theta = np.arange(args.fit_ntheta)[None, :] / args.fit_ntheta
        validation = sampled_surface(
            alpha_surface,
            nphi=args.validation_nphi,
            ntheta=args.validation_ntheta,
            phi_shift=0.371,
            theta_shift=0.413,
        )
        validation_data = surface_field_data(validation, field, iota)
        validation_phi = (
            np.arange(args.validation_nphi)[:, None] + 0.371
        ) / (nfp * args.validation_nphi)
        validation_theta = (
            np.arange(args.validation_ntheta)[None, :] + 0.413
        ) / args.validation_ntheta
        before = residual_for_iota_G(validation, field, iota=iota, G=G)

        for nu_order in nu_orders:
            correction = fit_toroidal_correction(
                train_theta,
                train_phi,
                target,
                iota=iota,
                nfp=nfp,
                mpol=nu_order,
                ntor=nu_order,
                regularization=args.regularization,
            )
            nu_validation, _, _, along_field = evaluate_toroidal_correction(
                correction, validation_theta, validation_phi
            )
            analytic = scalar_residual_metrics(
                validation_data, G=G, scale=1.0 + along_field
            )
            corrected, mapping = corrected_surface(
                alpha_surface,
                correction,
                output_order=args.surface_order,
                nphi=args.fit_nphi,
                ntheta=args.fit_ntheta,
                interpolation_size=args.interpolation_size,
            )
            corrected_validation = sampled_surface(
                corrected,
                nphi=args.validation_nphi,
                ntheta=args.validation_ntheta,
                phi_shift=0.371,
                theta_shift=0.413,
            )
            simsopt_corrected = residual_for_iota_G(
                corrected_validation, field, iota=iota, G=G
            )
            if args.save_surfaces and nu_order == selected_order:
                corrected_path = surface_dir / f"{surface_stem}_alpha_nu.npz"
                np.savez(
                    corrected_path,
                    dofs=corrected.get_dofs(),
                    iota=iota,
                    G=G,
                    nfp=nfp,
                    order=args.surface_order,
                    stellsym=bool(boozer_cfg.stellsym),
                    rho=float(rho),
                    s_edge=s_edge,
                    s_level=float(s_edge * rho * rho),
                    radius_mean_m=float(np.mean(radii)),
                    spectral_fit_rms_m=mapping["spectral_fit_rms_m"],
                    kind="alpha_nu",
                )
                saved_surfaces.append(
                    {
                        "rho": float(rho),
                        "alpha": str(alpha_path),
                        "alpha_nu": str(corrected_path),
                    }
                )
            rows.append(
                {
                    "rho": float(rho),
                    "s_level": float(s_edge * rho * rho),
                    "nu_order": int(nu_order),
                    "surface_order": int(args.surface_order),
                    "iota": iota,
                    "G": G,
                    "radius_mean_m": float(np.mean(radii)),
                    "extraction": {
                        key: float(value) for key, value in extraction.items()
                    },
                    "alpha_transform": alpha_transform,
                    "alpha_projection": alpha_projection,
                    "before": before,
                    "nu_fit": correction.diagnostics,
                    "nu_validation": {
                        "nu_rms_turns": float(
                            np.sqrt(np.mean(nu_validation * nu_validation))
                        ),
                        "nu_max_abs_turns": float(np.max(np.abs(nu_validation))),
                        "mapping_jacobian_min": float(np.min(1.0 + along_field)),
                        "mapping_jacobian_max": float(np.max(1.0 + along_field)),
                    },
                    "analytic_corrected": analytic,
                    "mapping_and_projection": mapping,
                    "simsopt_corrected": simsopt_corrected,
                }
            )

    plot_radial(
        rows,
        selected_order,
        args.output_dir / "toroidal_correction_vs_rho.png",
    )
    plot_order_scan(rows, args.output_dir / "toroidal_correction_order_scan.png")
    output = {
        "case": args.case_file.name,
        "s_edge": s_edge,
        "alpha_dir": str(args.alpha_dir),
        "alpha_fit": args.alpha_fit,
        "alpha_iota_coeffs": alpha_fit.iota_coeffs.tolist(),
        "surface_optimization_performed": False,
        "coordinate_correction_only": True,
        "surface_order": int(args.surface_order),
        "nu_orders": nu_orders,
        "regularization": float(args.regularization),
        "selected_nu_order": int(selected_order),
        "saved_surfaces": saved_surfaces,
        "rows": rows,
        "total_time_s": float(time.perf_counter() - started),
    }
    write_json(args.output_dir / "summary.json", output)


if __name__ == "__main__":
    main()
