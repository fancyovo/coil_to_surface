from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from stellarator_gpu import CoilFieldGpu, load_case

TWOPI = 2.0 * np.pi


@dataclass
class PsiData:
    coeffs: np.ndarray
    mode_a: np.ndarray
    mode_b: np.ndarray
    mode_m: np.ndarray
    mode_kind: np.ndarray
    nfp: int
    a: float
    phi_axis: np.ndarray
    R_axis: np.ndarray
    Z_axis: np.ndarray
    R_axis_phi: np.ndarray
    Z_axis_phi: np.ndarray


def load_psi(path: str | Path) -> PsiData:
    d = np.load(path, allow_pickle=True)
    return PsiData(
        coeffs=np.asarray(d["coeffs"], dtype=np.float64),
        mode_a=np.asarray(d["mode_a"], dtype=np.int32),
        mode_b=np.asarray(d["mode_b"], dtype=np.int32),
        mode_m=np.asarray(d["mode_m"], dtype=np.int32),
        mode_kind=np.asarray(d["mode_kind"]).astype(str),
        nfp=int(d["nfp"]),
        a=float(d["a"]),
        phi_axis=np.asarray(d["phi_axis"], dtype=np.float64),
        R_axis=np.asarray(d["R_axis"], dtype=np.float64),
        Z_axis=np.asarray(d["Z_axis"], dtype=np.float64),
        R_axis_phi=np.asarray(d["R_axis_phi"], dtype=np.float64),
        Z_axis_phi=np.asarray(d["Z_axis_phi"], dtype=np.float64),
    )


def interp_periodic(phi, phi_axis, values, nfp: int):
    period = TWOPI / nfp
    p = np.mod(np.asarray(phi, dtype=np.float64), period)
    x = np.r_[phi_axis, period]
    y = np.r_[values, values[0]]
    return np.interp(p, x, y)


def axis_at(model: PsiData, phi):
    return (
        interp_periodic(phi, model.phi_axis, model.R_axis, model.nfp),
        interp_periodic(phi, model.phi_axis, model.Z_axis, model.nfp),
        interp_periodic(phi, model.phi_axis, model.R_axis_phi, model.nfp),
        interp_periodic(phi, model.phi_axis, model.Z_axis_phi, model.nfp),
    )


def mono_and_derivatives(ax, bz, X, Z):
    val = (X**ax) * (Z**bz)
    dx = np.zeros_like(X)
    dz = np.zeros_like(X)
    if ax:
        dx = ax * (X ** (ax - 1)) * (Z**bz)
    if bz:
        dz = bz * (X**ax) * (Z ** (bz - 1))
    return val, dx, dz


def trig(kind, m, phi, nfp):
    arg = m * nfp * phi
    if kind == "cos":
        return np.cos(arg), -m * nfp * np.sin(arg)
    return np.sin(arg), m * nfp * np.cos(arg)


def psi_and_gradient(model: PsiData, R, Z, phi):
    R = np.asarray(R, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    ra, za, rap, zap = axis_at(model, phi)
    X = (R - ra) / model.a
    Zc = (Z - za) / model.a
    psi = X**2
    grad_R = 2.0 * X / model.a
    grad_Z = np.zeros_like(X)
    grad_phi = -rap * grad_R
    for c, ax, bz, m, kind in zip(model.coeffs, model.mode_a, model.mode_b, model.mode_m, model.mode_kind):
        mono, mono_x, mono_z = mono_and_derivatives(int(ax), int(bz), X, Zc)
        tr, tr_phi = trig(kind, int(m), phi, model.nfp)
        dR = mono_x * tr / model.a
        dZ = mono_z * tr / model.a
        dPhi = mono * tr_phi - rap * dR - zap * dZ
        psi += c * mono * tr
        grad_R += c * dR
        grad_Z += c * dZ
        grad_phi += c * dPhi
    return psi, grad_R, grad_Z, grad_phi


def psi_ray_value_and_derivative(model: PsiData, rho, theta, phi):
    rho = np.asarray(rho, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    X = rho * np.cos(theta) / model.a
    Z = rho * np.sin(theta) / model.a
    dX = np.cos(theta) / model.a
    dZ = np.sin(theta) / model.a
    psi = X**2
    dpsi = 2.0 * X * dX
    for c, ax, bz, m, kind in zip(model.coeffs, model.mode_a, model.mode_b, model.mode_m, model.mode_kind):
        tr, _ = trig(kind, int(m), phi, model.nfp)
        mono, mono_x, mono_z = mono_and_derivatives(int(ax), int(bz), X, Z)
        psi += c * mono * tr
        dpsi += c * (mono_x * dX + mono_z * dZ) * tr
    return psi, dpsi


def quadratic_radius(model: PsiData, psi_level, theta, phi, max_radius):
    X = np.cos(theta)
    Z = np.sin(theta)
    q = X**2
    for c, ax, bz, m, kind in zip(model.coeffs, model.mode_a, model.mode_b, model.mode_m, model.mode_kind):
        if int(ax) + int(bz) != 2:
            continue
        tr, _ = trig(kind, int(m), phi, model.nfp)
        q += c * (X ** int(ax)) * (Z ** int(bz)) * tr
    fallback = model.a * np.sqrt(np.maximum(psi_level, 1e-16))
    rho = np.where(q > 1e-10, model.a * np.sqrt(np.maximum(psi_level, 0.0) / q), fallback)
    return np.clip(rho, 1e-12 * model.a, max_radius)


def level_curve_phi0(model: PsiData, psi_level: float, n_alpha: int, max_radius_scale: float, tol: float, maxiter: int):
    theta = np.linspace(0.0, TWOPI, n_alpha, endpoint=False)
    phi = 0.0
    max_radius = max_radius_scale * model.a
    rho = quadratic_radius(model, psi_level, theta, phi, max_radius)
    iters = 0
    max_abs_f = np.inf
    for iters in range(1, maxiter + 1):
        psi, dpsi = psi_ray_value_and_derivative(model, rho, theta, phi)
        f = psi - psi_level
        max_abs_f = float(np.max(np.abs(f)))
        if max_abs_f <= tol:
            break
        denom = np.where(np.abs(dpsi) > 1e-14, dpsi, np.where(dpsi >= 0.0, 1e-14, -1e-14))
        step = np.clip(
            f / denom,
            -0.45 * np.maximum(np.abs(rho), 1e-8 * model.a),
            0.45 * np.maximum(np.abs(rho), 1e-8 * model.a),
        )
        trial = np.clip(rho - step, 1e-12 * model.a, max_radius)
        psi_trial, _ = psi_ray_value_and_derivative(model, trial, theta, phi)
        take = np.abs(psi_trial - psi_level) <= np.abs(f)
        rho[take] = trial[take]
        rho[~take] = 0.5 * (rho[~take] + trial[~take])
    ra = model.R_axis[0]
    za = model.Z_axis[0]
    return theta, ra + rho * np.cos(theta), za + rho * np.sin(theta), rho, iters, max_abs_f


def level_curves_phi0_batched(
    model: PsiData,
    psi_levels,
    n_alpha: int,
    max_radius_scale: float,
    tol: float,
    maxiter: int,
):
    levels = np.asarray(psi_levels, dtype=np.float64)[:, None]
    theta = np.linspace(0.0, TWOPI, n_alpha, endpoint=False)[None, :]
    phi = 0.0
    max_radius = max_radius_scale * model.a
    rho = quadratic_radius(model, levels, theta, phi, max_radius)
    iters = np.zeros(len(psi_levels), dtype=np.int32)
    max_abs_f_level = np.full(len(psi_levels), np.inf, dtype=np.float64)
    active = np.ones(len(psi_levels), dtype=bool)
    for it in range(1, maxiter + 1):
        psi, dpsi = psi_ray_value_and_derivative(model, rho, theta, phi)
        f = psi - levels
        max_abs = np.max(np.abs(f), axis=1)
        newly_done = active & (max_abs <= tol)
        iters[newly_done] = it
        max_abs_f_level[newly_done] = max_abs[newly_done]
        active = active & (max_abs > tol)
        if not np.any(active):
            break
        denom = np.where(np.abs(dpsi) > 1e-14, dpsi, np.where(dpsi >= 0.0, 1e-14, -1e-14))
        step = np.clip(
            f / denom,
            -0.45 * np.maximum(np.abs(rho), 1e-8 * model.a),
            0.45 * np.maximum(np.abs(rho), 1e-8 * model.a),
        )
        trial = np.clip(rho - step, 1e-12 * model.a, max_radius)
        psi_trial, _ = psi_ray_value_and_derivative(model, trial, theta, phi)
        take = np.abs(psi_trial - levels) <= np.abs(f)
        update_mask = active[:, None]
        rho = np.where(update_mask & take, trial, rho)
        rho = np.where(update_mask & ~take, 0.5 * (rho + trial), rho)
    if np.any(active):
        psi, _ = psi_ray_value_and_derivative(model, rho, theta, phi)
        f = psi - levels
        max_abs = np.max(np.abs(f), axis=1)
        iters[active] = maxiter
        max_abs_f_level[active] = max_abs[active]
    theta_1d = theta.ravel()
    ra = model.R_axis[0]
    za = model.Z_axis[0]
    curves = []
    for i, level in enumerate(psi_levels):
        rho_i = rho[i]
        curves.append(
            {
                "psi_level": float(level),
                "theta": theta_1d,
                "R": ra + rho_i * np.cos(theta_1d),
                "Z": za + rho_i * np.sin(theta_1d),
                "rho": rho_i,
                "newton_time_s": None,
                "newton_iters": int(iters[i]),
                "newton_max_abs_f": float(max_abs_f_level[i]),
                "radius_min": float(np.min(rho_i)),
                "radius_mean": float(np.mean(rho_i)),
                "radius_max": float(np.max(rho_i)),
            }
        )
    return curves


def radial_poly_coeffs_phi0(model: PsiData, theta):
    theta = np.asarray(theta, dtype=np.float64)
    cth = np.cos(theta)
    sth = np.sin(theta)
    degree = int(np.max(model.mode_a + model.mode_b))
    p = np.zeros((degree + 1, theta.size), dtype=np.float64)
    p[2] += cth**2
    for c, ax, bz, kind in zip(model.coeffs, model.mode_a, model.mode_b, model.mode_kind):
        if kind == "sin":
            continue
        deg = int(ax) + int(bz)
        p[deg] += c * (cth ** int(ax)) * (sth ** int(bz))
    return p


def eval_radial_poly(p, rho, a_scale):
    u = rho / a_scale
    psi = np.zeros_like(u)
    dpsi = np.zeros_like(u)
    for deg in range(2, p.shape[0]):
        coeff = p[deg][None, :]
        if not np.any(coeff):
            continue
        u_pow = u**deg
        psi += coeff * u_pow
        dpsi += coeff * deg * (u ** (deg - 1)) / a_scale
    return psi, dpsi


def level_curves_phi0_poly(
    model: PsiData,
    psi_levels,
    n_alpha: int,
    max_radius_scale: float,
    tol: float,
    maxiter: int,
):
    levels = np.asarray(psi_levels, dtype=np.float64)[:, None]
    theta = np.linspace(0.0, TWOPI, n_alpha, endpoint=False)
    max_radius = max_radius_scale * model.a
    p = radial_poly_coeffs_phi0(model, theta)
    q = p[2][None, :]
    fallback = model.a * np.sqrt(np.maximum(levels, 1e-16))
    rho = np.where(q > 1e-10, model.a * np.sqrt(np.maximum(levels, 0.0) / q), fallback)
    rho = np.clip(rho, 1e-12 * model.a, max_radius)
    iters = np.zeros(len(psi_levels), dtype=np.int32)
    max_abs_f_level = np.full(len(psi_levels), np.inf, dtype=np.float64)
    active = np.ones(len(psi_levels), dtype=bool)
    for it in range(1, maxiter + 1):
        psi, dpsi = eval_radial_poly(p, rho, model.a)
        f = psi - levels
        max_abs = np.max(np.abs(f), axis=1)
        newly_done = active & (max_abs <= tol)
        iters[newly_done] = it
        max_abs_f_level[newly_done] = max_abs[newly_done]
        active = active & (max_abs > tol)
        if not np.any(active):
            break
        denom = np.where(np.abs(dpsi) > 1e-14, dpsi, np.where(dpsi >= 0.0, 1e-14, -1e-14))
        step = np.clip(
            f / denom,
            -0.45 * np.maximum(np.abs(rho), 1e-8 * model.a),
            0.45 * np.maximum(np.abs(rho), 1e-8 * model.a),
        )
        trial = np.clip(rho - step, 1e-12 * model.a, max_radius)
        psi_trial, _ = eval_radial_poly(p, trial, model.a)
        take = np.abs(psi_trial - levels) <= np.abs(f)
        update_mask = active[:, None]
        rho = np.where(update_mask & take, trial, rho)
        rho = np.where(update_mask & ~take, 0.5 * (rho + trial), rho)
    if np.any(active):
        psi, _ = eval_radial_poly(p, rho, model.a)
        f = psi - levels
        max_abs = np.max(np.abs(f), axis=1)
        iters[active] = maxiter
        max_abs_f_level[active] = max_abs[active]
    ra = model.R_axis[0]
    za = model.Z_axis[0]
    curves = []
    for i, level in enumerate(psi_levels):
        rho_i = rho[i]
        curves.append(
            {
                "psi_level": float(level),
                "theta": theta,
                "R": ra + rho_i * np.cos(theta),
                "Z": za + rho_i * np.sin(theta),
                "rho": rho_i,
                "newton_time_s": None,
                "newton_iters": int(iters[i]),
                "newton_max_abs_f": float(max_abs_f_level[i]),
                "radius_min": float(np.min(rho_i)),
                "radius_mean": float(np.mean(rho_i)),
                "radius_max": float(np.max(rho_i)),
            }
        )
    return curves


def parse_levels(text: str):
    return [float(x) for x in text.replace(",", " ").split()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lib", default="build_mixed/libstellarator_gpu.so")
    p.add_argument("--case-file", default="../examples/01.json")
    p.add_argument("--key", default="raw")
    p.add_argument("--psi-model", required=True)
    p.add_argument("--segments", type=int, default=256)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--levels", default="0.001,0.002,0.004,0.008,0.012,0.02,0.04,0.08,0.12,0.16")
    p.add_argument("--n-alpha", type=int, default=256)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--max-radius-scale", type=float, default=1.0)
    p.add_argument("--drift-abs-tol", type=float, default=5e-4)
    p.add_argument("--drift-rel-tol", type=float, default=0.30)
    p.add_argument("--newton-tol", type=float, default=1e-12)
    p.add_argument("--newton-maxiter", type=int, default=20)
    p.add_argument("--newton-mode", choices=["serial", "batched", "poly"], default="serial")
    p.add_argument("--precisions", default="fp32,mixed64,fp64")
    p.add_argument("--threads-per-line", type=int, default=256)
    p.add_argument("--output", default="psi0_screen_gpu_bench.json")
    args = p.parse_args()

    model = load_psi(args.psi_model)
    cx, cy, cz, currents, nfp = load_case(args.case_file, args.key)
    if nfp != model.nfp:
        raise ValueError(f"case nfp={nfp} does not match psi model nfp={model.nfp}")
    levels = parse_levels(args.levels)
    precisions = [x.strip() for x in args.precisions.replace(",", " ").split()]

    t0 = time.perf_counter()
    if args.newton_mode == "batched":
        curves = level_curves_phi0_batched(
            model, levels, args.n_alpha, args.max_radius_scale, args.newton_tol, args.newton_maxiter
        )
    elif args.newton_mode == "poly":
        curves = level_curves_phi0_poly(
            model, levels, args.n_alpha, args.max_radius_scale, args.newton_tol, args.newton_maxiter
        )
    else:
        curves = []
        for level in levels:
            t_level = time.perf_counter()
            theta, R, Z, rho, iters, max_abs_f = level_curve_phi0(
                model, level, args.n_alpha, args.max_radius_scale, args.newton_tol, args.newton_maxiter
            )
            curves.append(
                {
                    "psi_level": float(level),
                    "theta": theta,
                    "R": R,
                    "Z": Z,
                    "rho": rho,
                    "newton_time_s": time.perf_counter() - t_level,
                    "newton_iters": int(iters),
                    "newton_max_abs_f": float(max_abs_f),
                    "radius_min": float(np.min(rho)),
                    "radius_mean": float(np.mean(rho)),
                    "radius_max": float(np.max(rho)),
                }
            )
    curve_total = time.perf_counter() - t0
    if args.newton_mode in {"batched", "poly"}:
        for curve in curves:
            curve["newton_time_s"] = curve_total / max(len(curves), 1)

    R0 = np.concatenate([c["R"] for c in curves])
    Z0 = np.concatenate([c["Z"] for c in curves])
    level_index = np.concatenate([np.full(args.n_alpha, i, dtype=np.int32) for i in range(len(curves))])

    t0 = time.perf_counter()
    field = CoilFieldGpu(args.lib, cx, cy, cz, currents, nfp=nfp, segments_per_coil=args.segments, device_id=args.device)
    field_create_time = time.perf_counter() - t0

    trace_results = {}
    end_points = {}
    for precision in precisions:
        # Warm up once; then time the same full batch.
        field.trace_period_blockline_precision(
            R0, Z0, steps=args.steps, precision=precision, threads_per_line=args.threads_per_line, nfp=nfp
        )
        t_trace = time.perf_counter()
        Re, Ze = field.trace_period_blockline_precision(
            R0, Z0, steps=args.steps, precision=precision, threads_per_line=args.threads_per_line, nfp=nfp
        )
        trace_time = time.perf_counter() - t_trace
        end_points[precision] = (Re, Ze)
        rows = []
        phi_end = np.full(args.n_alpha, TWOPI / nfp)
        for i, curve in enumerate(curves):
            mask = level_index == i
            rr = Re[mask]
            zz = Ze[mask]
            psi_end, gr, gz, gp = psi_and_gradient(model, rr, zz, phi_end)
            grad_norm = np.sqrt(gr**2 + gz**2 + (gp / rr) ** 2)
            distance = np.abs(psi_end - curve["psi_level"]) / np.maximum(grad_norm, 1e-14)
            p95 = float(np.percentile(distance, 95))
            rel = p95 / max(curve["radius_mean"], 1e-14)
            ok = bool(
                (p95 <= args.drift_abs_tol)
                and (rel <= args.drift_rel_tol)
                and (curve["radius_max"] < args.max_radius_scale * model.a * 0.999)
            )
            rows.append(
                {
                    "psi_level": curve["psi_level"],
                    "ok": ok,
                    "end_distance_p95": p95,
                    "rel_end_distance_p95": float(rel),
                    "radius_mean": curve["radius_mean"],
                    "radius_max": curve["radius_max"],
                }
            )
        trace_results[precision] = {
            "trace_time_s": trace_time,
            "points": int(len(R0)),
            "level_results": rows,
        }

    diffs = {}
    if "fp64" in end_points:
        r64, z64 = end_points["fp64"]
        for precision, (rp, zp) in end_points.items():
            if precision == "fp64":
                continue
            err = np.sqrt((rp - r64) ** 2 + (zp - z64) ** 2)
            diffs[precision + "_vs_fp64"] = {
                "p50": float(np.percentile(err, 50)),
                "p95": float(np.percentile(err, 95)),
                "max": float(np.max(err)),
            }

    result = {
        "case_file": args.case_file,
        "key": args.key,
        "psi_model": args.psi_model,
        "nfp": nfp,
        "levels": levels,
        "n_alpha": args.n_alpha,
        "newton_mode": args.newton_mode,
        "total_points": int(len(R0)),
        "curve_newton_total_s": curve_total,
        "curve_newton_per_level_s": [float(c["newton_time_s"]) for c in curves],
        "curve_newton_iters": [int(c["newton_iters"]) for c in curves],
        "curve_newton_max_abs_f": [float(c["newton_max_abs_f"]) for c in curves],
        "field_create_time_s": field_create_time,
        "trace": trace_results,
        "end_point_diffs": diffs,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    field.close()


if __name__ == "__main__":
    main()
