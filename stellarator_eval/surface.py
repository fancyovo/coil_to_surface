from __future__ import annotations

import time
from dataclasses import dataclass
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from scipy.spatial import cKDTree
from simsopt.geo import BoozerSurface, SurfaceXYZTensorFourier, Volume, boozer_surface_residual

from .axis import b_components, rk4_one_period
from .config import BoozerConfig, SurfaceScanConfig
from .psi import PsiModel, psi_and_gradient, psi_ray_value_and_derivative

TWOPI = 2.0 * np.pi
MU0 = 4.0e-7 * np.pi


@dataclass
class LevelScreen:
    psi_level: float
    ok: bool
    reason: str
    radius_min: float
    radius_mean: float
    radius_max: float
    end_distance_p95: float
    rel_end_distance_p95: float
    trace_time_s: float


def _quadratic_radius(model: PsiModel, psi_level, theta, phi, max_radius):
    X = np.cos(theta)
    Z = np.sin(theta)
    q = X**2
    for c, mode in zip(model.coeffs, model.modes):
        if mode.a + mode.b != 2:
            continue
        arg = mode.m * model.nfp * phi
        trig = np.cos(arg) if mode.kind == "cos" else np.sin(arg)
        q += c * (X**mode.a) * (Z**mode.b) * trig
    fallback = model.a * np.sqrt(max(psi_level, 1e-16))
    rho = np.where(q > 1e-10, model.a * np.sqrt(np.maximum(psi_level, 0.0) / q), fallback)
    return np.clip(rho, 1e-12 * model.a, max_radius)


def level_curve_phi0(model: PsiModel, psi_level: float, n_alpha: int, cfg: SurfaceScanConfig):
    theta = np.linspace(0.0, TWOPI, n_alpha, endpoint=False)
    phi = 0.0
    max_radius = cfg.max_radius_scale * model.a
    rho = _quadratic_radius(model, psi_level, theta, phi, max_radius)
    for _ in range(cfg.curve_newton_maxiter):
        psi, dpsi = psi_ray_value_and_derivative(model, rho, theta, phi)
        f = psi - psi_level
        if np.max(np.abs(f)) <= cfg.curve_newton_tol:
            break
        denom = np.where(np.abs(dpsi) > 1e-14, dpsi, np.where(dpsi >= 0.0, 1e-14, -1e-14))
        step = np.clip(f / denom, -0.45 * np.maximum(np.abs(rho), 1e-8 * model.a), 0.45 * np.maximum(np.abs(rho), 1e-8 * model.a))
        trial = np.clip(rho - step, 1e-12 * model.a, max_radius)
        psi_trial, _ = psi_ray_value_and_derivative(model, trial, theta, phi)
        take = np.abs(psi_trial - psi_level) <= np.abs(f)
        rho[take] = trial[take]
        rho[~take] = 0.5 * (rho[~take] + trial[~take])
    ra = model.R_axis[0]
    za = model.Z_axis[0]
    return theta, ra + rho * np.cos(theta), za + rho * np.sin(theta), rho


def screen_level(field, model: PsiModel, psi_level: float, cfg: SurfaceScanConfig) -> LevelScreen:
    theta, R, Z, rho = level_curve_phi0(model, psi_level, cfg.n_alpha, cfg)
    t0 = time.perf_counter()
    Re, Ze = rk4_one_period(field, R, Z, model.nfp, cfg.trace_steps)
    trace_time = time.perf_counter() - t0
    phi_end = np.full_like(Re, TWOPI / model.nfp)
    psi_end, gr, gz, gp = psi_and_gradient(model, Re, Ze, phi_end)
    grad_norm = np.sqrt(gr**2 + gz**2 + (gp / Re) ** 2)
    distance = np.abs(psi_end - psi_level) / np.maximum(grad_norm, 1e-14)
    p95 = float(np.percentile(distance, 95))
    radius_mean = float(np.mean(rho))
    rel = p95 / max(radius_mean, 1e-14)
    ok = bool((p95 <= cfg.drift_abs_tol) and (rel <= cfg.drift_rel_tol) and np.max(rho) < cfg.max_radius_scale * model.a * 0.999)
    reason = "ok" if ok else "drift_or_radius_failed"
    return LevelScreen(
        psi_level=float(psi_level),
        ok=ok,
        reason=reason,
        radius_min=float(np.min(rho)),
        radius_mean=radius_mean,
        radius_max=float(np.max(rho)),
        end_distance_p95=p95,
        rel_end_distance_p95=float(rel),
        trace_time_s=trace_time,
    )


def surface_points_from_level(model: PsiModel, psi_level: float, order: int, cfg: SurfaceScanConfig):
    nphi = 2 * order + 1
    ntheta = 2 * order + 1
    phis = np.linspace(0.0, TWOPI / model.nfp, nphi, endpoint=False)
    thetas = np.linspace(0.0, TWOPI, ntheta, endpoint=False)
    xyz = np.empty((nphi, ntheta, 3))
    radii = np.empty((nphi, ntheta))
    max_radius = cfg.max_radius_scale * model.a
    for i, phi in enumerate(phis):
        theta = thetas.copy()
        rho = _quadratic_radius(model, psi_level, theta, phi, max_radius)
        for _ in range(cfg.curve_newton_maxiter):
            psi, dpsi = psi_ray_value_and_derivative(model, rho, theta, phi)
            f = psi - psi_level
            if np.max(np.abs(f)) <= cfg.curve_newton_tol:
                break
            denom = np.where(np.abs(dpsi) > 1e-14, dpsi, np.where(dpsi >= 0.0, 1e-14, -1e-14))
            step = np.clip(f / denom, -0.45 * np.maximum(np.abs(rho), 1e-8 * model.a), 0.45 * np.maximum(np.abs(rho), 1e-8 * model.a))
            rho = np.clip(rho - step, 1e-12 * model.a, max_radius)
        ra, za, _, _ = model.axis_at(np.full_like(theta, phi))
        R = ra + rho * np.cos(theta)
        Z = za + rho * np.sin(theta)
        cp = np.cos(phi)
        sp = np.sin(phi)
        xyz[i, :, 0] = R * cp
        xyz[i, :, 1] = R * sp
        xyz[i, :, 2] = Z
        radii[i] = rho
    return xyz, radii


def fit_xyz_tensor_surface(xyz, nfp: int, order: int, stellsym: bool):
    nphi, ntheta, _ = xyz.shape
    surf = SurfaceXYZTensorFourier(
        mpol=order,
        ntor=order,
        stellsym=stellsym,
        nfp=nfp,
        quadpoints_phi=np.linspace(0.0, 1.0 / nfp, nphi, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, ntheta, endpoint=False),
    )
    surf.least_squares_fit(xyz)
    rms = float(np.sqrt(np.mean((surf.gamma() - xyz) ** 2)))
    return surf, rms


def clone_surface(surf):
    other = SurfaceXYZTensorFourier(
        mpol=surf.mpol,
        ntor=surf.ntor,
        stellsym=surf.stellsym,
        nfp=surf.nfp,
        quadpoints_phi=surf.quadpoints_phi,
        quadpoints_theta=surf.quadpoints_theta,
    )
    other.set_dofs(surf.get_dofs().copy())
    return other


def helical_qs_metric(boozer_surface, biotsavart, helicity_m: int, helicity_n: int, sdim: int = 16, n_alpha: int = 64):
    in_surface = boozer_surface.surface
    phis = np.linspace(0.0, 1.0 / in_surface.nfp, 2 * sdim, endpoint=False)
    thetas = np.linspace(0.0, 1.0, 2 * sdim, endpoint=False)
    surface = SurfaceXYZTensorFourier(
        mpol=in_surface.mpol,
        ntor=in_surface.ntor,
        stellsym=in_surface.stellsym,
        nfp=in_surface.nfp,
        quadpoints_phi=phis,
        quadpoints_theta=thetas,
    )
    surface.set_dofs(in_surface.get_dofs())
    biotsavart.set_points(surface.gamma().reshape((-1, 3)))
    b = biotsavart.B().reshape((len(phis), len(thetas), 3))
    modb = np.linalg.norm(b, axis=2)
    normal = surface.normal()
    ds = np.linalg.norm(normal, axis=2)
    theta = thetas[None, :] * TWOPI
    zeta = phis[:, None] * surface.nfp * TWOPI
    alpha = (helicity_m * theta - helicity_n * zeta) % TWOPI
    bins = np.linspace(0.0, TWOPI, n_alpha + 1)
    b_qs = np.zeros_like(modb)
    for k in range(n_alpha):
        mask = (alpha >= bins[k]) & (alpha < bins[k + 1])
        if np.any(mask):
            b_qs[mask] = np.mean(modb[mask] * ds[mask]) / np.mean(ds[mask])
    return float(np.mean(ds * (modb - b_qs) ** 2) / np.mean(ds * b_qs**2))


def evaluate_boozer_surface(field, model: PsiModel, psi_level: float, scan_cfg: SurfaceScanConfig, boozer_cfg: BoozerConfig, out_npz=None):
    result: dict[str, object] = {"psi_level": float(psi_level)}
    t0 = time.perf_counter()
    xyz, radii = surface_points_from_level(model, psi_level, boozer_cfg.surface_order, scan_cfg)
    result["extract_surface_time_s"] = time.perf_counter() - t0
    result["radius_min"] = float(np.min(radii))
    result["radius_mean"] = float(np.mean(radii))
    result["radius_max"] = float(np.max(radii))

    t0 = time.perf_counter()
    surf, fit_rms = fit_xyz_tensor_surface(xyz, model.nfp, boozer_cfg.surface_order, boozer_cfg.stellsym)
    result["surface_fit_time_s"] = time.perf_counter() - t0
    result["surface_fit_rms"] = float(fit_rms)
    result["initial_volume"] = float(Volume(surf).J())

    current_sum = float(sum(abs(c.current.get_value()) for c in field.coils))
    g0 = MU0 * current_sum
    result["G0"] = float(g0)
    try:
        r0 = boozer_surface_residual(surf, boozer_cfg.initial_iota, g0, field, derivatives=0)[0]
        result["initial_boozer_residual_norm"] = float(np.linalg.norm(r0))
    except Exception as exc:
        result["initial_boozer_residual_error"] = repr(exc)

    volume = Volume(surf)
    target_volume = volume.J()
    boozer = BoozerSurface(field, surf, volume, target_volume)
    try:
        t0 = time.perf_counter()
        ls = boozer.minimize_boozer_penalty_constraints_ls(
            tol=boozer_cfg.ls_tol,
            maxiter=boozer_cfg.ls_maxiter,
            iota=boozer_cfg.initial_iota,
            G=g0,
            constraint_weight=boozer_cfg.constraint_weight,
            weight_inv_modB=True,
        )
        result["ls_time_s"] = time.perf_counter() - t0
        result["ls_success"] = bool(ls.get("success", False))
        result["ls_iota"] = float(ls["iota"])
        result["ls_G"] = float(ls["G"])
        result["ls_residual_norm"] = float(np.linalg.norm(boozer_surface_residual(surf, ls["iota"], ls["G"], field, derivatives=0)[0]))
    except Exception as exc:
        result["ls_error"] = repr(exc)
        result["total_time_s"] = sum(v for k, v in result.items() if k.endswith("_time_s") and isinstance(v, float))
        return result

    surf2 = clone_surface(surf)
    volume2 = Volume(surf2)
    boozer2 = BoozerSurface(field, surf2, volume2, target_volume)
    try:
        t0 = time.perf_counter()
        newton = boozer2.solve_residual_equation_exactly_newton(
            tol=boozer_cfg.newton_tol,
            maxiter=boozer_cfg.newton_maxiter,
            iota=float(result["ls_iota"]),
            G=float(result["ls_G"]),
            verbose=False,
        )
        result["newton_time_s"] = time.perf_counter() - t0
        result["newton_success"] = bool(newton.get("success", False))
        result["newton_iter"] = int(newton.get("iter", -1))
        result["iota"] = float(newton["iota"])
        result["G"] = float(newton["G"])
        result["volume"] = float(volume2.J())
        result["newton_residual_norm"] = float(np.linalg.norm(boozer_surface_residual(surf2, newton["iota"], newton["G"], field, derivatives=0)[0]))
        if out_npz is not None:
            np.savez(out_npz, dofs=surf2.get_dofs(), iota=result["iota"], G=result["G"], psi_level=psi_level)
    except Exception as exc:
        result["newton_error"] = repr(exc)
        result["total_time_s"] = sum(v for k, v in result.items() if k.endswith("_time_s") and isinstance(v, float))
        return result

    if result.get("newton_success"):
        t0 = time.perf_counter()
        result["qs_error_QA_1_0"] = helical_qs_metric(boozer2, field, 1, 0, boozer_cfg.qs_sdim)
        result["qs_error_QH_1_1"] = helical_qs_metric(boozer2, field, 1, 1, boozer_cfg.qs_sdim)
        result["qs_error_QP_0_1"] = helical_qs_metric(boozer2, field, 0, 1, boozer_cfg.qs_sdim)
        result["qs_time_s"] = time.perf_counter() - t0

    result["total_time_s"] = sum(v for k, v in result.items() if k.endswith("_time_s") and isinstance(v, float))
    return result
