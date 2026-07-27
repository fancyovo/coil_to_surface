from __future__ import annotations

import argparse
import json
import os
import sys
import time
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

from simsopt.geo import SurfaceXYZTensorFourier, ToroidalFlux

from stellarator_eval.axis import b_components
from stellarator_eval.config import BoozerConfig, SurfaceScanConfig
from stellarator_eval.desc_joint_ls import (
    JointSpectralFitConfig,
    WeightedSampleSet,
    apply_joint_fit,
    fit_joint_rzl_data,
)
from stellarator_eval.field import build_field
from stellarator_eval.psi import psi_and_gradient, psi_ray_value_and_derivative
from stellarator_eval.serialization import write_json
from stellarator_eval.surface import level_curve_phi0, surface_points_from_level

from scripts.desc_external_rzl_data_ls_experiment import axis_sample_points
from scripts.desc_joint_rzl_initial_guess_experiment import boundary_mismatch, rk4_period_samples
from scripts.desc_psi_volume_initial_guess_experiment import (
    boundary_layer_points,
    build_equilibrium,
    dataclass_from_dict,
    fit_axis_curve,
    load_field_input,
    load_json,
    load_psi_model,
    make_xyz_surface,
    psi_layer_points,
    write_vmec_input_from_surface,
)

TWOPI = 2.0 * np.pi


def _parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _dataset(nodes, values, weight: float, label: str) -> WeightedSampleSet:
    return WeightedSampleSet(
        nodes=np.asarray(nodes, dtype=float),
        values=np.asarray(values, dtype=float),
        weight=float(weight),
        label=label,
    )


def _surface_for_sampling(surface, *, nfp: int, nphi: int, ntheta: int):
    sampled = SurfaceXYZTensorFourier(
        mpol=surface.mpol,
        ntor=surface.ntor,
        stellsym=surface.stellsym,
        nfp=nfp,
        quadpoints_phi=np.linspace(0.0, 1.0 / nfp, nphi, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, ntheta, endpoint=False),
    )
    sampled.set_dofs(surface.get_dofs())
    return sampled


def geometric_boundary_points(
    surface,
    model,
    *,
    nfp: int,
    nphi: int,
    ntheta: int,
    dense_theta: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    sampled = _surface_for_sampling(surface, nfp=nfp, nphi=nphi, ntheta=dense_theta)
    xyz = sampled.gamma()
    phi = TWOPI * sampled.quadpoints_phi
    theta_target = np.linspace(0.0, TWOPI, ntheta, endpoint=False)
    R_out = np.empty((nphi, ntheta), dtype=float)
    Z_out = np.empty((nphi, ntheta), dtype=float)
    monotone_failures = 0
    for i, pp in enumerate(phi):
        R = np.sqrt(xyz[i, :, 0] ** 2 + xyz[i, :, 1] ** 2)
        Z = xyz[i, :, 2]
        ra, za, _, _ = model.axis_at(np.asarray([pp]))
        angle = np.mod(np.arctan2(Z - za[0], R - ra[0]), TWOPI)
        order = np.argsort(angle)
        angle = angle[order]
        R = R[order]
        Z = Z[order]
        unique, idx = np.unique(angle, return_index=True)
        if len(unique) < dense_theta // 2:
            monotone_failures += 1
        R = R[idx]
        Z = Z[idx]
        angle_ext = np.concatenate([unique[-1:] - TWOPI, unique, unique[:1] + TWOPI])
        R_ext = np.concatenate([R[-1:], R, R[:1]])
        Z_ext = np.concatenate([Z[-1:], Z, Z[:1]])
        R_out[i] = np.interp(theta_target, angle_ext, R_ext)
        Z_out[i] = np.interp(theta_target, angle_ext, Z_ext)
    pg, tg = np.meshgrid(phi, theta_target, indexing="ij")
    nodes = np.column_stack([np.ones(pg.size), tg.ravel(), pg.ravel()])
    return (
        nodes,
        R_out.ravel(),
        Z_out.ravel(),
        {"monotone_failure_sections": int(monotone_failures)},
    )


def psi_surface_diagnostics(field, model, psi_edge: float, scan_cfg: SurfaceScanConfig, *, order: int, rho_values):
    surfaces = []
    radii = []
    layers = []
    nphi = 2 * order + 1
    ntheta = 2 * order + 1
    phi = np.linspace(0.0, TWOPI / model.nfp, nphi, endpoint=False)
    theta = np.linspace(0.0, TWOPI, ntheta, endpoint=False)
    pg, tg = np.meshgrid(phi, theta, indexing="ij")
    for rho in rho_values:
        level = float(psi_edge * rho * rho)
        xyz, radius, _ = surface_points_from_level(model, level, order, scan_cfg)
        R = np.sqrt(xyz[..., 0] ** 2 + xyz[..., 1] ** 2)
        Z = xyz[..., 2]
        psi, gr, gz, gp = psi_and_gradient(model, R.ravel(), Z.ravel(), pg.ravel())
        eps_phi = 1e-7
        psi_phi_plus, _, _, _ = psi_and_gradient(
            model, R.ravel(), Z.ravel(), pg.ravel() + eps_phi
        )
        gp_fd = (psi_phi_plus - psi) / eps_phi
        _, dpsi = psi_ray_value_and_derivative(
            model, radius.ravel(), tg.ravel(), pg.ravel()
        )
        br, bphi, bz = b_components(field, R.ravel(), Z.ravel(), pg.ravel())
        bdot = br * gr + bz * gz + (bphi / R.ravel()) * gp
        bnorm = np.sqrt(br**2 + bphi**2 + bz**2)
        gradnorm = np.sqrt(gr**2 + gz**2 + (gp / R.ravel()) ** 2)
        angle = np.abs(bdot) / np.maximum(bnorm * gradnorm, 1e-14)
        layers.append(
            {
                "rho": float(rho),
                "psi_level": level,
                "root_rms": float(np.sqrt(np.mean((psi - level) ** 2))),
                "root_max_abs": float(np.max(np.abs(psi - level))),
                "grad_phi_fd_error_max": float(np.max(np.abs(gp_fd - gp))),
                "grad_phi_fd_error_rms": float(np.sqrt(np.mean((gp_fd - gp) ** 2))),
                "dpsi_dr_min": float(np.min(dpsi)),
                "dpsi_dr_nonpositive_fraction": float(np.mean(dpsi <= 0.0)),
                "bdotgrad_angle_mean": float(np.mean(angle)),
                "bdotgrad_angle_p95": float(np.percentile(angle, 95)),
                "bdotgrad_angle_max": float(np.max(angle)),
                "radius_min": float(np.min(radius)),
                "radius_max": float(np.max(radius)),
            }
        )
        surfaces.append(xyz)
        radii.append(radius)
    radii = np.asarray(radii)
    gaps = np.diff(radii, axis=0)
    return {
        "rho_values": np.asarray(rho_values, dtype=float),
        "phi": phi,
        "theta": theta,
        "surfaces": np.asarray(surfaces),
        "radii": radii,
        "layers": layers,
        "adjacent_radius_gap_min": float(np.min(gaps)),
        "adjacent_radius_gap_nonpositive_fraction": float(np.mean(gaps <= 0.0)),
    }


def poincare_return_diagnostics(
    field,
    model,
    psi_edge: float,
    scan_cfg: SurfaceScanConfig,
    *,
    rho_values,
    n_alpha: int,
    periods: int,
    steps_per_period: int,
):
    layers = []
    hits = []
    for rho in rho_values:
        level = float(psi_edge * rho * rho)
        _, R, Z, _ = level_curve_phi0(model, level, n_alpha, scan_cfg)
        layer_hits = []
        for _ in range(periods):
            _, _, _, R, Z = rk4_period_samples(
                field,
                R,
                Z,
                model.nfp,
                n_zeta=1,
                steps=steps_per_period,
            )
            layer_hits.append(np.column_stack([R.copy(), Z.copy()]))
        layer_hits = np.asarray(layer_hits)
        psi, _, _, _ = psi_and_gradient(
            model,
            layer_hits[..., 0].ravel(),
            layer_hits[..., 1].ravel(),
            np.zeros(layer_hits.shape[0] * layer_hits.shape[1]),
        )
        drift = psi - level
        drift_by_period = drift.reshape(layer_hits.shape[:2])
        _, gr, gz, gp = psi_and_gradient(
            model,
            layer_hits[..., 0].ravel(),
            layer_hits[..., 1].ravel(),
            np.zeros(layer_hits.shape[0] * layer_hits.shape[1]),
        )
        gradnorm = np.sqrt(gr**2 + gz**2 + (gp / layer_hits[..., 0].ravel()) ** 2)
        distance = np.abs(drift) / np.maximum(gradnorm, 1e-14)
        layers.append(
            {
                "rho": float(rho),
                "psi_level": level,
                "psi_drift_rms": float(np.sqrt(np.mean(drift**2))),
                "psi_drift_p95_abs": float(np.percentile(np.abs(drift), 95)),
                "psi_drift_max_abs": float(np.max(np.abs(drift))),
                "distance_drift_p95": float(np.percentile(distance, 95)),
                "distance_drift_max": float(np.max(distance)),
                "psi_drift_rms_by_period": [
                    float(np.sqrt(np.mean(row**2))) for row in drift_by_period
                ],
            }
        )
        hits.append(layer_hits)
    return {"rho_values": np.asarray(rho_values), "layers": layers, "hits": hits}


def jacobian_stats(eq, *, radial_count: int = 25, angular_factor: int = 2):
    from desc.grid import LinearGrid

    rho = np.linspace(0.02, 1.0, radial_count)
    grid = LinearGrid(
        rho=rho,
        M=max(angular_factor * eq.M, 12),
        N=max(angular_factor * eq.N, 12),
        NFP=eq.NFP,
        sym=False,
        axis=False,
    )
    data = eq.compute(["sqrt(g)", "sqrt(g)_PEST", "R", "Z"], grid=grid)
    sqrtg = np.asarray(data["sqrt(g)"], dtype=float)
    sqrtg_pest = np.asarray(data["sqrt(g)_PEST"], dtype=float)

    def summarize(values):
        finite = values[np.isfinite(values)]
        positive = float(np.mean(finite > 0.0))
        negative = float(np.mean(finite < 0.0))
        return {
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "positive_fraction": positive,
            "negative_fraction": negative,
            "minority_sign_fraction": float(min(positive, negative)),
        }

    per_rho = []
    node_rho = np.asarray(grid.nodes[:, 0])
    majority_sign = -1.0 if np.mean(sqrtg < 0.0) >= np.mean(sqrtg > 0.0) else 1.0
    for rr in rho:
        mask = np.isclose(node_rho, rr)
        per_rho.append(
            {
                "rho": float(rr),
                "wrong_sign_fraction": float(np.mean(np.sign(sqrtg[mask]) != majority_sign)),
                "sqrtg_min": float(np.min(sqrtg[mask])),
                "sqrtg_max": float(np.max(sqrtg[mask])),
            }
        )
    return {
        "sqrtg": summarize(sqrtg),
        "sqrtg_pest": summarize(sqrtg_pest),
        "is_nested": bool(eq.is_nested(grid=grid)),
        "per_rho": per_rho,
        "grid_nodes": np.asarray(grid.nodes),
        "sqrtg_values": sqrtg,
    }


def fit_rz_variant(
    *,
    name: str,
    input_path: Path,
    toroidal_flux: float,
    axis_curve,
    nfp: int,
    model,
    resolution: int,
    interior_nodes,
    interior_R,
    interior_Z,
    boundary_nodes,
    boundary_R,
    boundary_Z,
    fit_cfg: JointSpectralFitConfig,
    interior_weight: float,
    boundary_weight: float,
    axis_weight: float,
    axis_zeta_count: int,
):
    from desc.geometry import FourierRZToroidalSurface

    desc_surface = FourierRZToroidalSurface.from_input_file(str(input_path))
    eq = build_equilibrium(
        desc_surface,
        toroidal_flux,
        resolution,
        resolution,
        resolution,
        axis_curve=axis_curve,
        constructor_ensure_nested=False,
    )
    axis_nodes, axis_R, axis_Z = axis_sample_points(
        model, nfp=nfp, nzeta=axis_zeta_count
    )
    R_sets = [
        _dataset(interior_nodes, interior_R, interior_weight, "interior_R"),
        _dataset(boundary_nodes, boundary_R, boundary_weight, "boundary_R"),
        _dataset(axis_nodes, axis_R, axis_weight, "axis_R"),
    ]
    Z_sets = [
        _dataset(interior_nodes, interior_Z, interior_weight, "interior_Z"),
        _dataset(boundary_nodes, boundary_Z, boundary_weight, "boundary_Z"),
        _dataset(axis_nodes, axis_Z, axis_weight, "axis_Z"),
    ]
    t0 = time.perf_counter()
    fit = fit_joint_rzl_data(
        eq,
        R_datasets=R_sets,
        Z_datasets=Z_sets,
        config=fit_cfg,
    )
    fit_time = time.perf_counter() - t0
    apply_joint_fit(eq, fit)
    jac = jacobian_stats(eq)
    boundary_R_fit = np.asarray(eq.R_basis.evaluate(boundary_nodes)) @ fit.R_lmn
    boundary_Z_fit = np.asarray(eq.Z_basis.evaluate(boundary_nodes)) @ fit.Z_lmn
    axis_R_fit = np.asarray(eq.R_basis.evaluate(axis_nodes)) @ fit.R_lmn
    axis_Z_fit = np.asarray(eq.Z_basis.evaluate(axis_nodes)) @ fit.Z_lmn

    def vector_rms(Ra, Za, Rb, Zb):
        return float(np.sqrt(np.mean((Ra - Rb) ** 2 + (Za - Zb) ** 2)))

    result = {
        "name": name,
        "resolution": int(resolution),
        "fit_time_s": float(fit_time),
        "matrix_shape": list(fit.matrix_shape),
        "R_fit_rms": fit.diagnostics["R_fit_rms"],
        "Z_fit_rms": fit.diagnostics["Z_fit_rms"],
        "boundary_parametric_rms": vector_rms(
            boundary_R_fit, boundary_Z_fit, boundary_R, boundary_Z
        ),
        "axis_parametric_rms": vector_rms(axis_R_fit, axis_Z_fit, axis_R, axis_Z),
        "jacobian": {k: v for k, v in jac.items() if k not in {"grid_nodes", "sqrtg_values"}},
    }
    return result, eq, jac


def plot_psi_sections(psi_diag, output: Path):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    surfaces = psi_diag["surfaces"]
    phi = psi_diag["phi"]
    rho_values = psi_diag["rho_values"]
    indices = np.linspace(0, len(phi) - 1, 4).round().astype(int)
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(rho_values)))
    for ax, idx in zip(axes.ravel(), indices):
        for color, rho, xyz in zip(colors, rho_values, surfaces):
            R = np.sqrt(xyz[idx, :, 0] ** 2 + xyz[idx, :, 1] ** 2)
            Z = xyz[idx, :, 2]
            ax.plot(np.r_[R, R[0]], np.r_[Z, Z[0]], color=color, lw=1.3, label=f"{rho:.2f}")
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        ax.set_title(f"phi={phi[idx] / np.pi:.3f} pi")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[::2], labels[::2], title="rho", loc="center right")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_poincare(returns, model, psi_edge: float, scan_cfg, output: Path):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rho_values = returns["rho_values"]
    colors = plt.cm.plasma(np.linspace(0.08, 0.92, len(rho_values)))
    fig, ax = plt.subplots(figsize=(7.5, 7.0), constrained_layout=True)
    for color, rho, hits in zip(colors, rho_values, returns["hits"]):
        _, R, Z, _ = level_curve_phi0(
            model, float(psi_edge * rho * rho), 512, scan_cfg
        )
        ax.plot(np.r_[R, R[0]], np.r_[Z, Z[0]], color=color, lw=1.2)
        ax.scatter(hits[..., 0].ravel(), hits[..., 1].ravel(), color=color, s=3, alpha=0.55)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("Poincare returns at phi=0 over fitted psi contours")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_sweep(results, output: Path):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        groups.setdefault(item["case"], []).append(item)
    for label, values in groups.items():
        values = sorted(values, key=lambda x: x["resolution"])
        x = [v["resolution"] for v in values]
        y = [v["jacobian"]["sqrtg"]["minority_sign_fraction"] for v in values]
        z = [v["R_fit_rms"] + v["Z_fit_rms"] for v in values]
        axes[0].plot(x, y, marker="o", label=label)
        axes[1].plot(x, z, marker="o", label=label)
    axes[0].set_ylabel("sqrt(g) minority-sign fraction")
    axes[1].set_ylabel("R RMS + Z RMS [m]")
    for ax in axes:
        ax.set_xlabel("DESC L=M=N")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_jacobian_map(snapshot, output: Path):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    nodes = snapshot["jac"]["grid_nodes"]
    values = snapshot["jac"]["sqrtg_values"]
    rho = np.unique(nodes[:, 0])
    theta = np.unique(nodes[:, 1])
    zeta = np.unique(nodes[:, 2])
    cube = values.reshape((len(theta), len(rho), len(zeta)), order="F")
    majority = -1.0 if np.mean(values < 0.0) >= np.mean(values > 0.0) else 1.0
    wrong = np.mean(np.sign(cube) != majority, axis=2).T
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    mesh = ax.pcolormesh(theta, rho, wrong, shading="auto", cmap="magma", vmin=0.0, vmax=1.0)
    fig.colorbar(mesh, ax=ax, label="wrong-sign fraction over zeta")
    ax.set_xlabel("theta")
    ax.set_ylabel("rho")
    ax.set_title(snapshot["case"])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_rz_fit_sections(snapshot, output: Path):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    nodes = snapshot["target_nodes"]
    target_R = snapshot["target_R"]
    target_Z = snapshot["target_Z"]
    eq = snapshot["eq"]
    fit_R = np.asarray(eq.R_basis.evaluate(nodes)) @ np.asarray(eq.R_lmn)
    fit_Z = np.asarray(eq.Z_basis.evaluate(nodes)) @ np.asarray(eq.Z_lmn)
    zeta_values = np.unique(nodes[:, 2])
    selected = zeta_values[np.linspace(0, len(zeta_values) - 1, 4).round().astype(int)]
    rho_values = np.unique(nodes[:, 0])
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(rho_values)))
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    for ax, zeta in zip(axes.ravel(), selected):
        for color, rho in zip(colors, rho_values):
            mask = np.isclose(nodes[:, 2], zeta) & np.isclose(nodes[:, 0], rho)
            if np.count_nonzero(mask) < 3:
                continue
            theta_order = np.argsort(nodes[mask, 1])
            rt = target_R[mask][theta_order]
            zt = target_Z[mask][theta_order]
            rf = fit_R[mask][theta_order]
            zf = fit_Z[mask][theta_order]
            ax.plot(np.r_[rt, rt[0]], np.r_[zt, zt[0]], color=color, lw=1.2)
            ax.plot(np.r_[rf, rf[0]], np.r_[zf, zf[0]], color=color, lw=0.9, ls="--")
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        ax.set_title(f"zeta={zeta / np.pi:.3f} pi")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
    fig.suptitle(f"{snapshot['case']}: psi targets (solid), DESC fit (dashed)")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Diagnose psi nestedness and DESC R/Z volume fitting.")
    parser.add_argument("--case-file", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--key", default="raw")
    parser.add_argument("--gpu-lib-path", default="gpu_backend/build_mixed/libstellarator_gpu.so")
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--psi-backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--psi-order", type=int, default=10)
    parser.add_argument("--rho-min", type=float, default=0.10)
    parser.add_argument("--rho-layers", type=int, default=9)
    parser.add_argument("--edge-scales", default="1.0,0.75,0.5,0.35")
    parser.add_argument("--resolutions", default="4,6,8")
    parser.add_argument("--interior-weight", type=float, default=1.0)
    parser.add_argument("--boundary-weight", type=float, default=40.0)
    parser.add_argument("--axis-weight", type=float, default=20.0)
    parser.add_argument("--axis-zeta-count", type=int, default=32)
    parser.add_argument("--rz-ridge", type=float, default=1e-8)
    parser.add_argument("--poincare-periods", type=int, default=32)
    parser.add_argument("--poincare-alpha", type=int, default=8)
    parser.add_argument("--poincare-steps", type=int, default=360)
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_device))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_file = Path(args.case_file)
    run_dir = Path(args.run_dir)
    summary = load_json(run_dir / "summary.json")
    config_dict = summary.get("config", {})
    nfp = int(summary["nfp"])
    current_unit = config_dict.get("current_unit") or "A"
    scan_cfg = dataclass_from_dict(SurfaceScanConfig, config_dict.get("scan"))
    boozer_cfg = dataclass_from_dict(BoozerConfig, config_dict.get("boozer"))
    scan_cfg = replace(scan_cfg, gpu_lib_path=args.gpu_lib_path, gpu_device=args.gpu_device)
    boozer_cfg = replace(boozer_cfg, gpu_lib_path=args.gpu_lib_path, gpu_device=args.gpu_device)

    best = summary["best_surface"]
    psi_edge = float(best["psi_level"])
    level_dir = run_dir / f"level_{psi_edge:.6g}".replace(".", "p")
    surface_npz = level_dir / "boozer_surface.npz"
    surface_order = int(config_dict.get("boozer", {}).get("surface_order", 6))
    stellsym = bool(config_dict.get("boozer", {}).get("stellsym", True))

    field_input = load_field_input(case_file, args.key)
    built = build_field(field_input, current_unit=current_unit)
    model = load_psi_model(run_dir / "psi_model.npz")
    surface, surface_meta = make_xyz_surface(
        surface_npz,
        nfp=nfp,
        order=surface_order,
        stellsym=stellsym,
        nphi=96,
        ntheta=144,
    )
    toroidal_flux = float(ToroidalFlux(surface, built.field).J())
    input_path = output_dir / "boundary_input.check"
    write_vmec_input_from_surface(surface, toroidal_flux, input_path)
    axis_curve, axis_info = fit_axis_curve(run_dir / "axis_data.npz", nfp=nfp, axis_order=8)

    rho_diag = np.linspace(max(args.rho_min, 0.05), 1.0, 13)
    psi_diag = psi_surface_diagnostics(
        built.field,
        model,
        psi_edge,
        scan_cfg,
        order=max(args.psi_order, 12),
        rho_values=rho_diag,
    )
    poincare_rho = np.asarray([0.2, 0.4, 0.6, 0.8, 1.0])
    returns = poincare_return_diagnostics(
        built.field,
        model,
        psi_edge,
        scan_cfg,
        rho_values=poincare_rho,
        n_alpha=args.poincare_alpha,
        periods=args.poincare_periods,
        steps_per_period=args.poincare_steps,
    )

    plot_psi_sections(psi_diag, output_dir / "psi_nested_sections.png")
    plot_poincare(returns, model, psi_edge, scan_cfg, output_dir / "psi_poincare_phi0.png")

    edge_scales = _parse_float_list(args.edge_scales)
    resolutions = _parse_int_list(args.resolutions)
    sweep_results = []
    snapshots = []
    fit_cfg = JointSpectralFitConfig(rz_ridge=args.rz_ridge, l_ridge=args.rz_ridge)
    for edge_scale in edge_scales:
        edge_level = psi_edge * edge_scale * edge_scale
        rho_values = np.linspace(args.rho_min, 0.9, args.rho_layers)
        interior_nodes, interior_R, interior_Z, _ = psi_layer_points(
            model,
            edge_level,
            rho_values,
            args.psi_order,
            scan_cfg,
            boozer_cfg,
            backend=args.psi_backend,
        )
        for resolution in resolutions:
            psi_boundary_nodes, psi_boundary_R, psi_boundary_Z, _ = psi_layer_points(
                model,
                edge_level,
                np.asarray([1.0]),
                max(resolution, args.psi_order),
                scan_cfg,
                boozer_cfg,
                backend=args.psi_backend,
            )
            boundary_variants = {
                "psi_ray": (psi_boundary_nodes, psi_boundary_R, psi_boundary_Z, {}),
            }
            if abs(edge_scale - 1.0) < 1e-12:
                native = boundary_layer_points(
                    surface, 1.0, nfp, 2 * resolution + 1, 2 * resolution + 1
                )
                geometric = geometric_boundary_points(
                    surface,
                    model,
                    nfp=nfp,
                    nphi=2 * resolution + 1,
                    ntheta=2 * resolution + 1,
                )
                boundary_variants["boozer_native"] = (*native, {})
                boundary_variants["boozer_geometric"] = geometric
            for boundary_name, boundary_data in boundary_variants.items():
                boundary_nodes, boundary_R, boundary_Z, boundary_info = boundary_data
                case_name = f"edge_{edge_scale:g}_{boundary_name}"
                result, eq, jac = fit_rz_variant(
                    name=case_name,
                    input_path=input_path,
                    toroidal_flux=toroidal_flux,
                    axis_curve=axis_curve,
                    nfp=nfp,
                    model=model,
                    resolution=resolution,
                    interior_nodes=interior_nodes,
                    interior_R=interior_R,
                    interior_Z=interior_Z,
                    boundary_nodes=boundary_nodes,
                    boundary_R=boundary_R,
                    boundary_Z=boundary_Z,
                    fit_cfg=fit_cfg,
                    interior_weight=args.interior_weight,
                    boundary_weight=args.boundary_weight,
                    axis_weight=args.axis_weight,
                    axis_zeta_count=args.axis_zeta_count,
                )
                result["case"] = case_name
                result["edge_scale"] = float(edge_scale)
                result["edge_psi_level"] = float(edge_level)
                result["boundary_parameterization"] = boundary_name
                result["boundary_info"] = boundary_info
                sweep_results.append(result)
                if resolution == 6 and edge_scale == 1.0:
                    snapshots.append(
                        {
                            "case": case_name,
                            "eq": eq,
                            "jac": jac,
                            "target_nodes": np.vstack([interior_nodes, boundary_nodes]),
                            "target_R": np.concatenate([interior_R, boundary_R]),
                            "target_Z": np.concatenate([interior_Z, boundary_Z]),
                        }
                    )

    plot_sweep(sweep_results, output_dir / "rz_resolution_sweep.png")
    if snapshots:
        worst = max(
            snapshots,
            key=lambda item: item["jac"]["sqrtg"]["minority_sign_fraction"],
        )
        best_snapshot = min(
            snapshots,
            key=lambda item: item["jac"]["sqrtg"]["minority_sign_fraction"],
        )
        plot_jacobian_map(worst, output_dir / "rz_jacobian_worst.png")
        plot_jacobian_map(best_snapshot, output_dir / "rz_jacobian_best.png")
        psi_snapshot = next(
            (item for item in snapshots if item["case"].endswith("psi_ray")),
            best_snapshot,
        )
        plot_rz_fit_sections(psi_snapshot, output_dir / "rz_fit_sections.png")

    result = {
        "case_file": str(case_file),
        "run_dir": str(run_dir),
        "psi_edge": psi_edge,
        "surface_meta": surface_meta,
        "axis_fit": axis_info,
        "psi_fit_info": model.fit_info,
        "psi_nestedness": {
            "layers": psi_diag["layers"],
            "adjacent_radius_gap_min": psi_diag["adjacent_radius_gap_min"],
            "adjacent_radius_gap_nonpositive_fraction": psi_diag[
                "adjacent_radius_gap_nonpositive_fraction"
            ],
        },
        "poincare_returns": returns["layers"],
        "rz_sweep": sweep_results,
        "artifacts": {
            "psi_nested_sections": str(output_dir / "psi_nested_sections.png"),
            "psi_poincare_phi0": str(output_dir / "psi_poincare_phi0.png"),
            "rz_resolution_sweep": str(output_dir / "rz_resolution_sweep.png"),
            "rz_jacobian_worst": str(output_dir / "rz_jacobian_worst.png"),
            "rz_jacobian_best": str(output_dir / "rz_jacobian_best.png"),
            "rz_fit_sections": str(output_dir / "rz_fit_sections.png"),
        },
    }
    write_json(output_dir / "summary.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
