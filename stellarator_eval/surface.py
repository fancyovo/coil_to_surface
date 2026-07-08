from __future__ import annotations

import time
from dataclasses import dataclass
import os
from pathlib import Path
import inspect

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from scipy.spatial import cKDTree

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
    curve_newton_time_s: float
    end_distance_p95: float
    rel_end_distance_p95: float
    trace_time_s: float
    iota_estimate: float | None = None
    iota_estimate_std: float | None = None


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


def _radial_poly_coeffs_phi0(model: PsiModel, theta):
    cth = np.cos(theta)
    sth = np.sin(theta)
    degree = max(mode.a + mode.b for mode in model.modes)
    coeffs = np.zeros((degree + 1, len(theta)))
    coeffs[2] += cth**2
    for c, mode in zip(model.coeffs, model.modes):
        if mode.kind == "sin":
            continue
        deg = mode.a + mode.b
        coeffs[deg] += c * (cth**mode.a) * (sth**mode.b)
    return coeffs


def _eval_radial_poly(coeffs, rho, a_scale):
    u = rho / a_scale
    psi = np.zeros_like(u)
    dpsi = np.zeros_like(u)
    for deg in range(2, coeffs.shape[0]):
        c = coeffs[deg]
        if not np.any(c):
            continue
        psi += c * (u**deg)
        dpsi += c * deg * (u ** (deg - 1)) / a_scale
    return psi, dpsi


def level_curve_phi0(model: PsiModel, psi_level: float, n_alpha: int, cfg: SurfaceScanConfig):
    theta = np.linspace(0.0, TWOPI, n_alpha, endpoint=False)
    max_radius = cfg.max_radius_scale * model.a
    poly = _radial_poly_coeffs_phi0(model, theta)
    q = poly[2]
    fallback = model.a * np.sqrt(max(psi_level, 1e-16))
    rho = np.where(q > 1e-10, model.a * np.sqrt(max(psi_level, 0.0) / q), fallback)
    rho = np.clip(rho, 1e-12 * model.a, max_radius)
    for _ in range(cfg.curve_newton_maxiter):
        psi, dpsi = _eval_radial_poly(poly, rho, model.a)
        f = psi - psi_level
        if np.max(np.abs(f)) <= cfg.curve_newton_tol:
            break
        denom = np.where(np.abs(dpsi) > 1e-14, dpsi, np.where(dpsi >= 0.0, 1e-14, -1e-14))
        step = np.clip(f / denom, -0.45 * np.maximum(np.abs(rho), 1e-8 * model.a), 0.45 * np.maximum(np.abs(rho), 1e-8 * model.a))
        trial = np.clip(rho - step, 1e-12 * model.a, max_radius)
        psi_trial, _ = _eval_radial_poly(poly, trial, model.a)
        take = np.abs(psi_trial - psi_level) <= np.abs(f)
        rho[take] = trial[take]
        rho[~take] = 0.5 * (rho[~take] + trial[~take])
    ra = model.R_axis[0]
    za = model.Z_axis[0]
    return theta, ra + rho * np.cos(theta), za + rho * np.sin(theta), rho


def estimate_iota_from_endpoint(model: PsiModel, theta0, Re, Ze) -> tuple[float, float]:
    phi_end = np.full_like(Re, TWOPI / model.nfp)
    ra, za, _, _ = model.axis_at(phi_end)
    theta1 = np.arctan2(Ze - za, Re - ra)
    delta = np.angle(np.exp(1j * (theta1 - theta0)))
    mean_phase = np.angle(np.mean(np.exp(1j * delta)))
    residual = np.angle(np.exp(1j * (delta - mean_phase)))
    scale = TWOPI / model.nfp
    # The local geometric angle used here has the opposite orientation from the
    # Boozer theta convention used by Simsopt's BoozerSurface.
    return float(-mean_phase / scale), float(np.std(residual) / scale)


def screen_level(field, model: PsiModel, psi_level: float, cfg: SurfaceScanConfig) -> LevelScreen:
    t_curve = time.perf_counter()
    theta, R, Z, rho = level_curve_phi0(model, psi_level, cfg.n_alpha, cfg)
    curve_time = time.perf_counter() - t_curve
    t0 = time.perf_counter()
    Re, Ze = rk4_one_period(field, R, Z, model.nfp, cfg.trace_steps)
    iota_estimate, iota_estimate_std = estimate_iota_from_endpoint(model, theta, Re, Ze)
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
        curve_newton_time_s=curve_time,
        end_distance_p95=p95,
        rel_end_distance_p95=float(rel),
        trace_time_s=trace_time,
        iota_estimate=iota_estimate,
        iota_estimate_std=iota_estimate_std,
    )


def _level_screen_from_endpoint(
    model: PsiModel,
    psi_level: float,
    rho,
    Re,
    Ze,
    cfg: SurfaceScanConfig,
    trace_time: float,
    curve_time: float,
    reason_ok="ok",
    theta0=None,
):
    phi_end = np.full_like(Re, TWOPI / model.nfp)
    psi_end, gr, gz, gp = psi_and_gradient(model, Re, Ze, phi_end)
    grad_norm = np.sqrt(gr**2 + gz**2 + (gp / Re) ** 2)
    distance = np.abs(psi_end - psi_level) / np.maximum(grad_norm, 1e-14)
    p95 = float(np.percentile(distance, 95))
    radius_mean = float(np.mean(rho))
    rel = p95 / max(radius_mean, 1e-14)
    ok = bool((p95 <= cfg.drift_abs_tol) and (rel <= cfg.drift_rel_tol) and np.max(rho) < cfg.max_radius_scale * model.a * 0.999)
    iota_estimate = None
    iota_estimate_std = None
    if theta0 is not None:
        iota_estimate, iota_estimate_std = estimate_iota_from_endpoint(model, theta0, Re, Ze)
    return LevelScreen(
        psi_level=float(psi_level),
        ok=ok,
        reason=reason_ok if ok else "drift_or_radius_failed",
        radius_min=float(np.min(rho)),
        radius_mean=radius_mean,
        radius_max=float(np.max(rho)),
        curve_newton_time_s=curve_time,
        end_distance_p95=p95,
        rel_end_distance_p95=float(rel),
        trace_time_s=trace_time,
        iota_estimate=iota_estimate,
        iota_estimate_std=iota_estimate_std,
    )


def screen_levels_gpu(field_input, model: PsiModel, levels, cfg: SurfaceScanConfig, current_unit: str = "MA") -> list[dict]:
    try:
        import sys

        gpu_python = Path(__file__).resolve().parents[1] / "gpu_backend" / "python"
        if str(gpu_python) not in sys.path:
            sys.path.insert(0, str(gpu_python))
        from stellarator_gpu import CoilFieldGpu
    except Exception as exc:
        raise RuntimeError(f"GPU backend import failed: {exc!r}") from exc

    curves = []
    for level in levels:
        t_curve = time.perf_counter()
        theta, R, Z, rho = level_curve_phi0(model, float(level), cfg.n_alpha, cfg)
        curves.append(
            {
                "psi_level": float(level),
                "theta": theta,
                "R": R,
                "Z": Z,
                "rho": rho,
                "curve_time": time.perf_counter() - t_curve,
            }
        )
    if not curves:
        return []

    R0 = np.concatenate([c["R"] for c in curves])
    Z0 = np.concatenate([c["Z"] for c in curves])
    offsets = np.cumsum([0] + [len(c["R"]) for c in curves])
    unit = current_unit.lower()
    if unit in {"ma", "megaamp", "megaamps"}:
        currents = np.asarray(field_input.currents, dtype=float) * 1e6
    elif unit in {"a", "amp", "amps"}:
        currents = np.asarray(field_input.currents, dtype=float)
    else:
        raise ValueError(f"unknown current_unit={current_unit!r}; use 'MA' or 'A'")
    lib_path = Path(cfg.gpu_lib_path)
    if not lib_path.is_absolute():
        lib_path = Path.cwd() / lib_path
    t_create = time.perf_counter()
    gpu_field = CoilFieldGpu(
        lib_path,
        field_input.coeffs_x,
        field_input.coeffs_y,
        field_input.coeffs_z,
        currents,
        nfp=model.nfp,
        segments_per_coil=cfg.gpu_segments_per_coil,
        device_id=cfg.gpu_device,
    )
    gpu_create_time = time.perf_counter() - t_create
    try:
        t_trace = time.perf_counter()
        Re, Ze = gpu_field.trace_period_blockline_precision(
            R0,
            Z0,
            steps=cfg.trace_steps,
            precision=cfg.gpu_trace_precision,
            threads_per_line=cfg.gpu_threads_per_line,
            nfp=model.nfp,
        )
        trace_total = time.perf_counter() - t_trace
        per_line_time = trace_total / max(len(R0), 1)
        results: list[dict] = []
        for i, curve in enumerate(curves):
            sl = slice(offsets[i], offsets[i + 1])
            screen = _level_screen_from_endpoint(
                model,
                curve["psi_level"],
                curve["rho"],
                Re[sl],
                Ze[sl],
                cfg,
                trace_time=per_line_time * (offsets[i + 1] - offsets[i]),
                curve_time=curve["curve_time"],
                theta0=curve["theta"],
            ).__dict__
            screen["trace_backend"] = "gpu"
            screen["trace_precision"] = cfg.gpu_trace_precision
            screen["gpu_batch_trace_time_s"] = trace_total
            screen["gpu_create_time_s"] = gpu_create_time
            results.append(screen)

        verify_candidates = sorted([r for r in results if r.get("ok")], key=lambda r: r["psi_level"], reverse=True)[
            : cfg.gpu_verify_candidates
        ]
        if verify_candidates and cfg.gpu_verify_precision:
            verify_levels = {float(r["psi_level"]) for r in verify_candidates}
            verify_indices = [i for i, c in enumerate(curves) if c["psi_level"] in verify_levels]
            Rv = np.concatenate([curves[i]["R"] for i in verify_indices])
            Zv = np.concatenate([curves[i]["Z"] for i in verify_indices])
            voffsets = np.cumsum([0] + [len(curves[i]["R"]) for i in verify_indices])
            t_verify = time.perf_counter()
            Rev, Zev = gpu_field.trace_period_blockline_precision(
                Rv,
                Zv,
                steps=cfg.trace_steps,
                precision=cfg.gpu_verify_precision,
                threads_per_line=cfg.gpu_threads_per_line,
                nfp=model.nfp,
            )
            verify_time = time.perf_counter() - t_verify
            for j, i in enumerate(verify_indices):
                sl = slice(voffsets[j], voffsets[j + 1])
                verified = _level_screen_from_endpoint(
                    model,
                    curves[i]["psi_level"],
                    curves[i]["rho"],
                    Rev[sl],
                    Zev[sl],
                    cfg,
                    trace_time=verify_time / max(len(verify_indices), 1),
                    curve_time=curves[i]["curve_time"],
                    reason_ok="ok_verified",
                    theta0=curves[i]["theta"],
                ).__dict__
                results[i]["verify_precision"] = cfg.gpu_verify_precision
                results[i]["verify_trace_time_s"] = verify_time / max(len(verify_indices), 1)
                results[i]["verify_ok"] = verified["ok"]
                results[i]["verify_end_distance_p95"] = verified["end_distance_p95"]
                results[i]["verify_rel_end_distance_p95"] = verified["rel_end_distance_p95"]
                if not verified["ok"]:
                    results[i]["ok"] = False
                    results[i]["reason"] = "gpu_verify_failed"
        return results
    finally:
        gpu_field.close()


def surface_points_from_level(model: PsiModel, psi_level: float, order: int, cfg: SurfaceScanConfig):
    t_newton = time.perf_counter()
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
    return xyz, radii, time.perf_counter() - t_newton


def surface_points_from_level_gpu(model: PsiModel, psi_level: float, order: int, cfg: SurfaceScanConfig, boozer_cfg: BoozerConfig):
    import sys

    gpu_python = Path(__file__).resolve().parents[1] / "gpu_backend" / "python"
    if str(gpu_python) not in sys.path:
        sys.path.insert(0, str(gpu_python))
    from stellarator_gpu import surface_points_from_level_gpu as gpu_surface_points_from_level

    poly_degree = int(model.fit_info.get("poly_degree", max(m.a + m.b for m in model.modes)))
    m_tor = int(model.fit_info.get("m_tor", max(m.m for m in model.modes)))
    mode_kind = np.array([0 if m.kind == "cos" else 1 for m in model.modes], dtype=np.int32)
    lib_path = Path(boozer_cfg.gpu_lib_path)
    if not lib_path.is_absolute():
        lib_path = Path.cwd() / lib_path
    xyz, radii, stats = gpu_surface_points_from_level(
        lib_path,
        model.coeffs,
        np.array([m.a for m in model.modes], dtype=np.int32),
        np.array([m.b for m in model.modes], dtype=np.int32),
        np.array([m.m for m in model.modes], dtype=np.int32),
        mode_kind,
        nfp=model.nfp,
        a=model.a,
        poly_degree=poly_degree,
        m_tor=m_tor,
        axis_R=model.R_axis,
        axis_Z=model.Z_axis,
        order=order,
        psi_level=float(psi_level),
        maxiter=cfg.curve_newton_maxiter,
        tol=cfg.curve_newton_tol,
        max_radius_scale=cfg.max_radius_scale,
        device_id=boozer_cfg.gpu_device,
    )
    return xyz, radii, stats


def fit_xyz_tensor_surface(xyz, nfp: int, order: int, stellsym: bool):
    from simsopt.geo import SurfaceXYZTensorFourier

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
    from simsopt.geo import SurfaceXYZTensorFourier

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
    from simsopt.geo import SurfaceXYZTensorFourier

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
    from simsopt.geo import BoozerSurface, Volume, boozer_surface_residual

    def total_surface_time() -> float:
        return float(
            result.get("extract_surface_time_s", 0.0)
            + result.get("surface_fit_time_s", 0.0)
            + result.get("ls_time_s", 0.0)
            + result.get("newton_time_s", 0.0)
            + result.get("qs_time_s", 0.0)
        )

    result: dict[str, object] = {"psi_level": float(psi_level)}
    t0 = time.perf_counter()
    if boozer_cfg.surface_extract_backend.lower() == "gpu":
        try:
            xyz, radii, extract_stats = surface_points_from_level_gpu(model, psi_level, boozer_cfg.surface_order, scan_cfg, boozer_cfg)
            newton_time = float(extract_stats["newton_s"])
            result["extract_surface_backend"] = "gpu"
            result["level_surface_coeff_build_time_s"] = float(extract_stats["coeff_build_s"])
            result["level_surface_copy_in_time_s"] = float(extract_stats["copy_in_s"])
            result["level_surface_copy_out_time_s"] = float(extract_stats["copy_out_s"])
            result["level_surface_gpu_total_time_s"] = float(extract_stats["total_s"])
        except Exception as exc:
            xyz, radii, newton_time = surface_points_from_level(model, psi_level, boozer_cfg.surface_order, scan_cfg)
            result["extract_surface_backend"] = "gpu_fallback_cpu"
            result["extract_surface_gpu_error"] = repr(exc)
    else:
        xyz, radii, newton_time = surface_points_from_level(model, psi_level, boozer_cfg.surface_order, scan_cfg)
        result["extract_surface_backend"] = "cpu"
    result["extract_surface_time_s"] = time.perf_counter() - t0
    result["level_surface_1d_newton_time_s"] = float(newton_time)
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
        ls_kwargs = {
            "tol": boozer_cfg.ls_tol,
            "maxiter": boozer_cfg.ls_maxiter,
            "iota": boozer_cfg.initial_iota,
            "G": g0,
            "constraint_weight": boozer_cfg.constraint_weight,
        }
        if "weight_inv_modB" in inspect.signature(boozer.minimize_boozer_penalty_constraints_ls).parameters:
            ls_kwargs["weight_inv_modB"] = True
        ls = boozer.minimize_boozer_penalty_constraints_ls(**ls_kwargs)
        result["ls_time_s"] = time.perf_counter() - t0
        result["ls_success"] = bool(ls.get("success", False))
        result["ls_iota"] = float(ls["iota"])
        result["ls_G"] = float(ls["G"])
        result["ls_residual_norm"] = float(np.linalg.norm(boozer_surface_residual(surf, ls["iota"], ls["G"], field, derivatives=0)[0]))
    except Exception as exc:
        result["ls_error"] = repr(exc)
        result["total_time_s"] = total_surface_time()
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
        result["total_time_s"] = total_surface_time()
        return result

    if result.get("newton_success"):
        t0 = time.perf_counter()
        result["qs_error_QA_1_0"] = helical_qs_metric(boozer2, field, 1, 0, boozer_cfg.qs_sdim)
        result["qs_error_QH_1_1"] = helical_qs_metric(boozer2, field, 1, 1, boozer_cfg.qs_sdim)
        result["qs_error_QP_0_1"] = helical_qs_metric(boozer2, field, 0, 1, boozer_cfg.qs_sdim)
        result["qs_time_s"] = time.perf_counter() - t0

    result["total_time_s"] = total_surface_time()
    return result
