from __future__ import annotations

import time
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from .config import AxisGAConfig
from .timing import record_b_call

TWOPI = 2.0 * np.pi


@dataclass
class AxisResult:
    has_axis: bool
    best_R: float
    best_Z: float
    best_residual: float
    generation: int
    history: list[dict]
    phi: np.ndarray
    R: np.ndarray
    Z: np.ndarray
    R_phi: np.ndarray
    Z_phi: np.ndarray
    time_s: float
    search_time_s: float = 0.0
    trace_time_s: float = 0.0
    backend: str = "cpu"
    trace_error: str = ""
    failure_reason: str = ""
    search_best_residual: float = float("nan")
    topology_class: str = ""
    topology_trace: float = float("nan")
    topology_det: float = float("nan")
    topology_stability_margin: float = float("nan")
    topology_robust: bool = False
    topology_ellipse_aspect: float = float("nan")
    topology_time_s: float = 0.0


def b_components(field, r, z, phi):
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    cp = np.cos(phi)
    sp = np.sin(phi)
    xyz = np.ascontiguousarray(np.column_stack([r * cp, r * sp, z]))
    t0 = time.perf_counter()
    field.set_points(xyz)
    b = field.B()
    record_b_call(time.perf_counter() - t0, len(xyz))
    br = b[:, 0] * cp + b[:, 1] * sp
    bphi = -b[:, 0] * sp + b[:, 1] * cp
    bz = b[:, 2]
    return br, bphi, bz


def rk4_one_period(field, r0, z0, nfp: int, steps: int):
    period = TWOPI / nfp
    h = period / steps
    r = np.asarray(r0, dtype=float).copy()
    z = np.asarray(z0, dtype=float).copy()

    def rhs(phi, rr, zz):
        br, bphi, bz = b_components(field, rr, zz, phi)
        tiny = 1e-14
        denom = np.where(np.abs(bphi) < tiny, np.where(bphi >= 0.0, tiny, -tiny), bphi)
        return rr * br / denom, rr * bz / denom

    for s in range(steps):
        phi = s * h
        k1r, k1z = rhs(phi, r, z)
        k2r, k2z = rhs(phi + 0.5 * h, r + 0.5 * h * k1r, z + 0.5 * h * k1z)
        k3r, k3z = rhs(phi + 0.5 * h, r + 0.5 * h * k2r, z + 0.5 * h * k2z)
        k4r, k4z = rhs(phi + h, r + h * k3r, z + h * k3z)
        r += (h / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r)
        z += (h / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)
    return r, z


def rk4_period_samples(field, r0, z0, nfp: int, *, n_zeta: int, steps: int):
    """Trace one field period and retain uniformly spaced intermediate states."""
    period = TWOPI / nfp
    steps = int(max(steps, n_zeta))
    h = period / steps
    r = np.asarray(r0, dtype=float).copy()
    z = np.asarray(z0, dtype=float).copy()
    store_steps = np.floor(np.linspace(0, steps, n_zeta, endpoint=False)).astype(int)
    store_lookup = {int(step): idx for idx, step in enumerate(store_steps)}
    r_hist = np.empty((len(r), n_zeta))
    z_hist = np.empty((len(r), n_zeta))
    phi_hist = np.empty(n_zeta)

    def rhs(phi, rr, zz):
        br, bphi, bz = b_components(field, rr, zz, phi)
        tiny = 1e-14
        denom = np.where(
            np.abs(bphi) < tiny,
            np.where(bphi >= 0.0, tiny, -tiny),
            bphi,
        )
        return rr * br / denom, rr * bz / denom

    for step in range(steps):
        if step in store_lookup:
            idx = store_lookup[step]
            r_hist[:, idx] = r
            z_hist[:, idx] = z
            phi_hist[idx] = step * h
        phi = step * h
        k1r, k1z = rhs(phi, r, z)
        k2r, k2z = rhs(phi + 0.5 * h, r + 0.5 * h * k1r, z + 0.5 * h * k1z)
        k3r, k3z = rhs(phi + 0.5 * h, r + 0.5 * h * k2r, z + 0.5 * h * k2z)
        k4r, k4z = rhs(phi + h, r + h * k3r, z + h * k3z)
        r += (h / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r)
        z += (h / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)
    return phi_hist, r_hist, z_hist, r, z


def _initial_grid(rc, zc, span, grid):
    rs = np.linspace(rc - span, rc + span, grid)
    zs = np.linspace(zc - span, zc + span, grid)
    rg, zg = np.meshgrid(rs, zs, indexing="xy")
    return rg.ravel(), zg.ravel()


def _pairwise_midpoints(r, z):
    return (0.5 * (r[:, None] + r[None, :])).ravel(), (0.5 * (z[:, None] + z[None, :])).ravel()


def _unique_points(r, z, decimals=14):
    pts = np.column_stack([r, z])
    _, idx = np.unique(np.round(pts, decimals=decimals), axis=0, return_index=True)
    idx = np.sort(idx)
    return r[idx], z[idx]


def search_axis_ga(field, nfp: int, r_center: float, cfg: AxisGAConfig) -> tuple[dict, list[dict]]:
    r, z = _initial_grid(r_center, cfg.z_center, cfg.span, cfg.grid)
    history: list[dict] = []
    best = {}
    for gen in range(cfg.max_generations + 1):
        t0 = time.perf_counter()
        re, ze = rk4_one_period(field, r, z, nfp, cfg.rk4_steps)
        residual = np.sqrt((re - r) ** 2 + (ze - z) ** 2)
        order = np.argsort(residual)
        top = order[: cfg.keep]
        dt = time.perf_counter() - t0
        row = {
            "generation": gen,
            "population": int(len(r)),
            "best_R": float(r[order[0]]),
            "best_Z": float(z[order[0]]),
            "best_residual": float(residual[order[0]]),
            "top_median_residual": float(np.median(residual[top])),
            "top_max_residual": float(np.max(residual[top])),
            "top_R_span": float(np.max(r[top]) - np.min(r[top])),
            "top_Z_span": float(np.max(z[top]) - np.min(z[top])),
            "time_s": dt,
        }
        history.append(row)
        best = row
        if row["best_residual"] <= cfg.tol or gen == cfg.max_generations:
            break
        r, z = _pairwise_midpoints(r[top], z[top])
        r, z = _unique_points(r, z)
        target = cfg.grid * cfg.grid
        if len(r) < target:
            local_span = max(row["top_R_span"], row["top_Z_span"], cfg.span * 2.0 ** (-(gen + 4)), 1e-10)
            rg, zg = _initial_grid(row["best_R"], row["best_Z"], local_span, cfg.grid)
            r, z = _unique_points(np.r_[r, rg], np.r_[z, zg])
        if len(r) > target:
            r = r[:target]
            z = z[:target]
    return best, history


def _trace_gpu_population(gpu_field, r, z, nfp: int, cfg: AxisGAConfig, precision: str):
    return gpu_field.trace_period_blockline_precision(
        r,
        z,
        steps=cfg.rk4_steps,
        precision=precision,
        threads_per_line=cfg.gpu_threads_per_line,
        nfp=nfp,
    )


def _rebuild_population_from_top(r_ordered, z_ordered, best_r, best_z, grid, keep, target, local_span):
    seed_r = r_ordered[:keep]
    seed_z = z_ordered[:keep]
    r, z = _pairwise_midpoints(seed_r, seed_z)
    r, z = _unique_points(r, z)
    if len(r) < target:
        rg, zg = _initial_grid(best_r, best_z, local_span, grid)
        r, z = _unique_points(np.r_[r, rg], np.r_[z, zg])
    if len(r) > target:
        r = r[:target]
        z = z[:target]
    return r, z


def search_axis_ga_gpu(gpu_field, nfp: int, r_center: float, cfg: AxisGAConfig) -> tuple[dict, list[dict]]:
    r, z = _initial_grid(r_center, cfg.z_center, cfg.span, cfg.grid)
    history: list[dict] = []
    best: dict = {}
    stage = "coarse" if cfg.staged else "single"
    stage_generation = 0
    total_generations = cfg.max_generations + cfg.fine_max_generations if cfg.staged else cfg.max_generations
    for gen in range(total_generations + 1):
        precision = cfg.coarse_precision if stage == "coarse" else cfg.fine_precision
        if not cfg.staged:
            precision = cfg.gpu_trace_precision
        t0 = time.perf_counter()
        re, ze = _trace_gpu_population(gpu_field, r, z, nfp, cfg, precision)
        trace_time = time.perf_counter() - t0
        residual = np.sqrt((re - r) ** 2 + (ze - z) ** 2)
        order = np.argsort(residual)
        keep = cfg.fine_keep if stage == "fine" else cfg.keep
        grid = cfg.fine_grid if stage == "fine" else cfg.grid
        target = grid * grid
        top = order[:keep]
        row = {
            "generation": gen,
            "stage": stage,
            "stage_generation": stage_generation,
            "precision": precision,
            "population": int(len(r)),
            "best_R": float(r[order[0]]),
            "best_Z": float(z[order[0]]),
            "best_residual": float(residual[order[0]]),
            "top_median_residual": float(np.median(residual[top])),
            "top_max_residual": float(np.max(residual[top])),
            "top_R_span": float(np.max(r[top]) - np.min(r[top])),
            "top_Z_span": float(np.max(z[top]) - np.min(z[top])),
            "time_s": trace_time,
        }
        history.append(row)
        best = row
        if cfg.staged and stage == "coarse" and row["best_residual"] <= cfg.switch_tol:
            stage = "fine"
            stage_generation = 0
            local_span = max(row["top_R_span"], row["top_Z_span"], cfg.fine_span_min, cfg.span * 2.0 ** (-(gen + 4)))
            r, z = _rebuild_population_from_top(
                r[order], z[order], row["best_R"], row["best_Z"], cfg.fine_grid, cfg.fine_keep, cfg.fine_grid * cfg.fine_grid, local_span
            )
            continue
        if row["best_residual"] <= cfg.tol:
            break
        if (not cfg.staged and gen == cfg.max_generations) or (cfg.staged and stage == "coarse" and gen == cfg.max_generations):
            break
        if cfg.staged and stage == "fine" and stage_generation >= cfg.fine_max_generations:
            break
        r, z = _pairwise_midpoints(r[top], z[top])
        r, z = _unique_points(r, z)
        if len(r) < target:
            local_span = max(row["top_R_span"], row["top_Z_span"], cfg.span * 2.0 ** (-(gen + 4)), 1e-10)
            rg, zg = _initial_grid(row["best_R"], row["best_Z"], local_span, grid)
            r, z = _unique_points(np.r_[r, rg], np.r_[z, zg])
        if len(r) > target:
            r = r[:target]
            z = z[:target]
        stage_generation += 1
    if cfg.gpu_verify_precision:
        t0 = time.perf_counter()
        rv = np.array([best["best_R"]], dtype=float)
        zv = np.array([best["best_Z"]], dtype=float)
        re, ze = _trace_gpu_population(gpu_field, rv, zv, nfp, cfg, cfg.gpu_verify_precision)
        verify_time = time.perf_counter() - t0
        best["search_residual"] = float(best["best_residual"])
        best["best_residual"] = float(np.hypot(re[0] - rv[0], ze[0] - zv[0]))
        best["verify_precision"] = cfg.gpu_verify_precision
        best["verify_time_s"] = verify_time
        best["verify_end_R"] = float(re[0])
        best["verify_end_Z"] = float(ze[0])
    return best, history


def _curve_values_from_coeff(coeff: np.ndarray, t: np.ndarray) -> np.ndarray:
    coeff = np.asarray(coeff, dtype=float)
    order = (coeff.size - 1) // 2
    out = np.full_like(t, coeff[0], dtype=float)
    for k in range(1, order + 1):
        out += coeff[k] * np.cos(k * t) + coeff[order + k] * np.sin(k * t)
    return out


def _fixed_point_domain(field_input, r_center: float, cfg: AxisGAConfig) -> dict:
    t = np.linspace(0.0, TWOPI, 160, endpoint=False)
    r_parts = []
    z_parts = []
    for xcoef, ycoef, zcoef in zip(field_input.coeffs_x, field_input.coeffs_y, field_input.coeffs_z):
        x = _curve_values_from_coeff(xcoef, t)
        y = _curve_values_from_coeff(ycoef, t)
        z = _curve_values_from_coeff(zcoef, t)
        r_parts.append(np.sqrt(x * x + y * y))
        z_parts.append(z)
    if r_parts:
        coil_r = np.concatenate(r_parts)
        coil_z = np.concatenate(z_parts)
        rq0, rq1 = np.quantile(coil_r, [0.05, 0.95])
        z_abs = float(np.quantile(np.abs(coil_z), 0.95))
        r_span = max(cfg.span, 0.45 * float(rq1 - rq0))
        z_span = max(cfg.span, 1.15 * z_abs)
    else:
        r_span = cfg.span
        z_span = cfg.span
    r_min = max(float(cfg.fixed_point_r_floor), float(min(r_center - cfg.span, r_center - r_span)))
    r_max = float(max(r_center + cfg.span, r_center + r_span))
    if r_max <= r_min:
        r_max = r_min + max(cfg.span, 0.1)
    return {"r_min": r_min, "r_max": r_max, "z_min": -z_span, "z_max": z_span}


def _angle_delta(a: float, b: float) -> float:
    d = b - a
    while d <= -math.pi:
        d += 2.0 * math.pi
    while d > math.pi:
        d -= 2.0 * math.pi
    return d


def _fixed_point_candidates(rs, zs, dr_grid, dz_grid, residual_grid, cfg: AxisGAConfig, *, max_candidates: int) -> list[dict]:
    grid = len(rs)
    angles = np.arctan2(dz_grid, dr_grid)
    candidates: list[dict] = []
    for j in range(grid - 1):
        for i in range(grid - 1):
            vals = [angles[j, i], angles[j, i + 1], angles[j + 1, i + 1], angles[j + 1, i]]
            if not np.all(np.isfinite(vals)):
                continue
            winding = (
                _angle_delta(vals[0], vals[1])
                + _angle_delta(vals[1], vals[2])
                + _angle_delta(vals[2], vals[3])
                + _angle_delta(vals[3], vals[0])
            )
            if abs(winding) > math.pi:
                candidates.append(
                    {
                        "R": 0.5 * float(rs[i] + rs[i + 1]),
                        "Z": 0.5 * float(zs[j] + zs[j + 1]),
                        "kind": "winding",
                        "cell_residual_min": float(np.min(residual_grid[j : j + 2, i : i + 2])),
                    }
                )
    local_mins: list[dict] = []
    for j in range(1, grid - 1):
        for i in range(1, grid - 1):
            val = residual_grid[j, i]
            if np.isfinite(val) and val <= np.min(residual_grid[j - 1 : j + 2, i - 1 : i + 2]):
                local_mins.append({"R": float(rs[i]), "Z": float(zs[j]), "kind": "local_min", "cell_residual_min": float(val)})
    local_mins.sort(key=lambda x: x["cell_residual_min"])
    candidates.extend(local_mins[: cfg.fixed_point_local_min_candidates])
    step = max(float(rs[1] - rs[0]) if grid > 1 else 0.0, float(zs[1] - zs[0]) if grid > 1 else 0.0)
    min_dist = 0.75 * step
    candidates.sort(key=lambda x: (0 if x["kind"] == "winding" else 1, x["cell_residual_min"]))
    out: list[dict] = []
    for cand in candidates:
        if cand["R"] < cfg.fixed_point_r_floor:
            continue
        if all(np.hypot(cand["R"] - prev["R"], cand["Z"] - prev["Z"]) >= min_dist for prev in out):
            out.append(cand)
        if len(out) >= max_candidates:
            break
    return out


def _fixed_point_eval(gpu_field, r, z, nfp: int, cfg: AxisGAConfig, precision: str):
    re, ze = _trace_gpu_population(gpu_field, r, z, nfp, cfg, precision)
    dr = re - r
    dz = ze - z
    return dr, dz, np.hypot(dr, dz)


def _topology_stability_margin(trace: float, det: float) -> float:
    if not np.isfinite(trace) or not np.isfinite(det) or det <= 0.0:
        return float("-inf")
    return float(2.0 - abs(trace) / math.sqrt(det))


def _classify_map_topology(trace: float, det: float) -> str:
    if not np.isfinite(trace) or not np.isfinite(det):
        return "invalid"
    if det <= 0.0:
        return "hyperbolic"
    normalized_trace = abs(trace) / math.sqrt(det)
    if normalized_trace < 2.0:
        return "elliptic"
    if normalized_trace > 2.0:
        return "hyperbolic"
    return "parabolic"


def _ellipse_aspect_from_jacobian(a: float, b: float, c: float, d: float, det: float) -> tuple[float, list[float]]:
    if not np.isfinite(det) or det <= 0.0:
        return float("inf"), [float("nan"), float("nan")]
    scale = math.sqrt(det)
    a, b, c, d = a / scale, b / scale, c / scale, d / scale
    q = np.array([[c, 0.5 * (d - a)], [0.5 * (d - a), -b]], dtype=float)
    eig = np.linalg.eigvalsh(q)
    if eig[0] < 0.0 and eig[1] < 0.0:
        eig = -eig[::-1]
    if eig[0] <= 0.0 or not np.all(np.isfinite(eig)):
        return float("inf"), [float(eig[0]), float(eig[1])]
    return float(math.sqrt(eig[1] / eig[0])), [float(eig[0]), float(eig[1])]


def _fixed_point_add_topology(gpu_field, items: list[dict], nfp: int, cfg: AxisGAConfig, domain: dict) -> float:
    if not items or not cfg.fixed_point_topology_filter:
        return 0.0
    span = max(domain["r_max"] - domain["r_min"], domain["z_max"] - domain["z_min"])
    h = max(cfg.fixed_point_topology_fd_abs, cfg.fixed_point_topology_fd_rel * span)
    r = np.array([item["best_R"] for item in items], dtype=float)
    z = np.array([item["best_Z"] for item in items], dtype=float)
    rp = np.maximum(cfg.fixed_point_r_floor, r + h)
    rm = np.maximum(cfg.fixed_point_r_floor, r - h)
    zp = z + h
    zm = z - h
    eval_r = np.concatenate([rp, rm, r, r])
    eval_z = np.concatenate([z, z, zp, zm])
    t0 = time.perf_counter()
    fdr, fdz, _ = _fixed_point_eval(gpu_field, eval_r, eval_z, nfp, cfg, cfg.gpu_verify_precision or cfg.gpu_trace_precision)
    dt = time.perf_counter() - t0
    n = len(items)
    pr = eval_r + fdr
    pz = eval_z + fdz
    denom_r = np.maximum(rp - rm, 1e-300)
    dpr_dr = (pr[:n] - pr[n : 2 * n]) / denom_r
    dpz_dr = (pz[:n] - pz[n : 2 * n]) / denom_r
    dpr_dz = (pr[2 * n : 3 * n] - pr[3 * n :]) / (2.0 * h)
    dpz_dz = (pz[2 * n : 3 * n] - pz[3 * n :]) / (2.0 * h)
    trace = dpr_dr + dpz_dz
    det = dpr_dr * dpz_dz - dpr_dz * dpz_dr
    for i, item in enumerate(items):
        aspect, q_eig = _ellipse_aspect_from_jacobian(
            float(dpr_dr[i]),
            float(dpr_dz[i]),
            float(dpz_dr[i]),
            float(dpz_dz[i]),
            float(det[i]),
        )
        normalized_trace = float(trace[i] / math.sqrt(det[i])) if np.isfinite(det[i]) and det[i] > 0.0 else float("nan")
        stability_margin = _topology_stability_margin(float(trace[i]), float(det[i]))
        item["topology_class"] = _classify_map_topology(float(trace[i]), float(det[i]))
        item["topology_trace"] = float(trace[i])
        item["topology_det"] = float(det[i])
        item["topology_normalized_trace"] = normalized_trace
        item["topology_stability_margin"] = stability_margin
        item["topology_robust"] = bool(
            item["topology_class"] == "elliptic"
            and stability_margin >= cfg.fixed_point_topology_margin
        )
        item["topology_fd_h"] = float(h)
        item["topology_ellipse_aspect"] = aspect
        item["topology_invariant_q_eig"] = q_eig
    return dt


def _choose_fixed_point_axis(top: list[dict], cfg: AxisGAConfig) -> dict:
    if not top:
        raise ValueError("no fixed-point candidates to choose from")
    if not cfg.fixed_point_topology_filter:
        return dict(min(top, key=lambda x: x["best_residual"]))
    eligible = [item for item in top if item["best_residual"] <= cfg.tol and item.get("topology_class") == "elliptic"]
    if eligible:
        robust = [item for item in eligible if item.get("topology_robust", False)]
        preferred = robust or eligible
        if cfg.fixed_point_prefer_round_elliptic:
            chosen = dict(min(preferred, key=lambda x: (x.get("topology_ellipse_aspect", float("inf")), x["best_residual"])))
        else:
            chosen = dict(min(preferred, key=lambda x: x["best_residual"]))
        chosen["topology_accepted"] = True
        return chosen
    best = dict(min(top, key=lambda x: x["best_residual"]))
    best["topology_accepted"] = False
    return best


def _fixed_point_axis_prefer(candidate: dict, incumbent: dict) -> bool:
    cand_robust = candidate.get("topology_robust", False)
    inc_robust = incumbent.get("topology_robust", False)
    if cand_robust and not inc_robust:
        return True
    if inc_robust and not cand_robust:
        return False
    cand_ok = candidate.get("topology_accepted", False)
    inc_ok = incumbent.get("topology_accepted", False)
    if cand_ok and not inc_ok:
        return True
    if inc_ok and not cand_ok:
        return False
    return candidate["best_residual"] < incumbent["best_residual"]


def _fixed_point_refine(gpu_field, candidates: list[dict], nfp: int, cfg: AxisGAConfig, domain: dict, *, newton_iters: int):
    if not candidates:
        return [], {"newton_time_s": 0.0, "newton_iterations": 0, "newton_evaluated_points": 0}
    r = np.array([c["R"] for c in candidates], dtype=float)
    z = np.array([c["Z"] for c in candidates], dtype=float)
    kind = [c["kind"] for c in candidates]
    dr, dz, residual = _fixed_point_eval(gpu_field, r, z, nfp, cfg, cfg.gpu_trace_precision)
    span = max(domain["r_max"] - domain["r_min"], domain["z_max"] - domain["z_min"])
    h = max(cfg.fixed_point_fd_abs, cfg.fixed_point_fd_rel * span)
    evaluated = len(r)
    actual_iters = 0
    t0 = time.perf_counter()
    for it in range(newton_iters):
        active = np.where(residual > cfg.tol)[0]
        if active.size == 0:
            break
        actual_iters = it + 1
        ra = r[active]
        za = z[active]
        rp = np.maximum(cfg.fixed_point_r_floor, ra + h)
        rm = np.maximum(cfg.fixed_point_r_floor, ra - h)
        eval_r = np.concatenate([rp, rm, ra, ra])
        eval_z = np.concatenate([za, za, za + h, za - h])
        fdr, fdz, _ = _fixed_point_eval(gpu_field, eval_r, eval_z, nfp, cfg, cfg.gpu_trace_precision)
        evaluated += eval_r.size
        n = active.size
        denom_r = np.maximum(rp - rm, 1e-300)
        j11 = (fdr[:n] - fdr[n : 2 * n]) / denom_r
        j21 = (fdz[:n] - fdz[n : 2 * n]) / denom_r
        j12 = (fdr[2 * n : 3 * n] - fdr[3 * n :]) / (2.0 * h)
        j22 = (fdz[2 * n : 3 * n] - fdz[3 * n :]) / (2.0 * h)
        det = j11 * j22 - j12 * j21
        good = np.abs(det) > 1e-14
        step_r = np.zeros(n, dtype=float)
        step_z = np.zeros(n, dtype=float)
        step_r[good] = (-dr[active][good] * j22[good] + j12[good] * dz[active][good]) / det[good]
        step_z[good] = (j21[good] * dr[active][good] - j11[good] * dz[active][good]) / det[good]
        step_norm = np.hypot(step_r, step_z)
        scale = np.minimum(1.0, 0.25 * span / np.maximum(step_norm, 1e-300))
        step_r *= scale
        step_z *= scale
        accepted = np.zeros(n, dtype=bool)
        for alpha in (1.0, 0.5, 0.25, 0.125):
            trial_r = np.clip(ra + alpha * step_r, cfg.fixed_point_r_floor, domain["r_max"])
            trial_z = np.clip(za + alpha * step_z, domain["z_min"], domain["z_max"])
            tdr, tdz, tres = _fixed_point_eval(gpu_field, trial_r, trial_z, nfp, cfg, cfg.gpu_trace_precision)
            evaluated += trial_r.size
            improve = (~accepted) & np.isfinite(tres) & (tres < residual[active])
            if not np.any(improve):
                continue
            idx = active[improve]
            r[idx] = trial_r[improve]
            z[idx] = trial_z[improve]
            dr[idx] = tdr[improve]
            dz[idx] = tdz[improve]
            residual[idx] = tres[improve]
            accepted[improve] = True
    dt = time.perf_counter() - t0
    refined = []
    for idx in np.argsort(residual):
        refined.append(
            {
                "best_R": float(r[idx]),
                "best_Z": float(z[idx]),
                "best_residual": float(residual[idx]),
                "search_residual": float(residual[idx]),
                "candidate_kind": kind[idx],
            }
        )
    return refined, {"newton_time_s": dt, "newton_iterations": int(actual_iters), "newton_evaluated_points": int(evaluated)}


def search_axis_fixed_point_gpu(
    gpu_field,
    field_input,
    nfp: int,
    r_center: float,
    cfg: AxisGAConfig,
    *,
    grid: int,
    max_candidates: int,
    newton_iters: int,
    stage: str,
) -> tuple[dict, list[dict]]:
    domain = _fixed_point_domain(field_input, r_center, cfg)
    rs = np.linspace(domain["r_min"], domain["r_max"], grid)
    zs = np.linspace(domain["z_min"], domain["z_max"], grid)
    rg, zg = np.meshgrid(rs, zs, indexing="xy")
    r0 = np.ascontiguousarray(rg.ravel(), dtype=float)
    z0 = np.ascontiguousarray(zg.ravel(), dtype=float)
    t_grid = time.perf_counter()
    dr, dz, residual = _fixed_point_eval(gpu_field, r0, z0, nfp, cfg, cfg.gpu_trace_precision)
    grid_time = time.perf_counter() - t_grid
    residual_grid = residual.reshape(grid, grid)
    candidates = _fixed_point_candidates(rs, zs, dr.reshape(grid, grid), dz.reshape(grid, grid), residual_grid, cfg, max_candidates=max_candidates)
    refined, stats = _fixed_point_refine(gpu_field, candidates, nfp, cfg, domain, newton_iters=newton_iters)
    if refined:
        top = refined[: max(1, cfg.fixed_point_verify_top)]
        rv = np.array([item["best_R"] for item in top], dtype=float)
        zv = np.array([item["best_Z"] for item in top], dtype=float)
        t_verify = time.perf_counter()
        vdr, vdz, vres = _fixed_point_eval(gpu_field, rv, zv, nfp, cfg, cfg.gpu_verify_precision or cfg.gpu_trace_precision)
        verify_time = time.perf_counter() - t_verify
        for i, item in enumerate(top):
            item["search_residual"] = item["best_residual"]
            item["best_residual"] = float(vres[i])
            item["verify_end_R"] = float(rv[i] + vdr[i])
            item["verify_end_Z"] = float(zv[i] + vdz[i])
            item["verify_precision"] = cfg.gpu_verify_precision
            item["verify_time_s"] = verify_time
        topology_time = _fixed_point_add_topology(gpu_field, top, nfp, cfg, domain)
        top.sort(key=lambda x: x["best_residual"])
        best = _choose_fixed_point_axis(top, cfg)
    else:
        idx = int(np.argmin(residual))
        verify_time = 0.0
        topology_time = 0.0
        best = {
            "best_R": float(r0[idx]),
            "best_Z": float(z0[idx]),
            "best_residual": float(residual[idx]),
            "search_residual": float(residual[idx]),
            "candidate_kind": "grid_best_no_candidate",
            "topology_class": "",
            "topology_accepted": False,
            "topology_robust": False,
        }
    best["generation"] = 0
    row = {
        "generation": 0,
        "stage": stage,
        "method": "fixed_point",
        "grid": int(grid),
        "population": int(r0.size),
        "best_R": float(best["best_R"]),
        "best_Z": float(best["best_Z"]),
        "best_residual": float(best["best_residual"]),
        "grid_best_residual": float(np.min(residual)),
        "candidate_count": int(len(candidates)),
        "time_s": float(grid_time + stats["newton_time_s"] + verify_time + topology_time),
        "grid_time_s": float(grid_time),
        "verify_time_s": float(verify_time),
        "topology_time_s": float(topology_time),
        "topology_filter_enabled": bool(cfg.fixed_point_topology_filter),
        "topology_require_elliptic": bool(cfg.fixed_point_require_elliptic),
        "verified_candidates": [
            {
                "rank_by_residual": int(i),
                "best_R": float(item["best_R"]),
                "best_Z": float(item["best_Z"]),
                "best_residual": float(item["best_residual"]),
                "candidate_kind": item.get("candidate_kind", ""),
                "topology_class": item.get("topology_class", ""),
                "topology_trace": float(item.get("topology_trace", float("nan"))),
                "topology_det": float(item.get("topology_det", float("nan"))),
                "topology_normalized_trace": float(item.get("topology_normalized_trace", float("nan"))),
                "topology_stability_margin": float(item.get("topology_stability_margin", float("nan"))),
                "topology_robust": bool(item.get("topology_robust", False)),
                "topology_ellipse_aspect": float(item.get("topology_ellipse_aspect", float("inf"))),
            }
            for i, item in enumerate(top if refined else [])
        ],
        **stats,
        **domain,
    }
    return best, [row]


def trace_axis(field, r0: float, z0: float, nfp: int, steps: int):
    period = TWOPI / nfp

    def rhs(phi, rz):
        r, z = rz
        br, bphi, bz = b_components(field, np.array([r]), np.array([z]), phi)
        tiny = 1e-14
        denom = bphi[0] if abs(bphi[0]) > tiny else np.copysign(tiny, bphi[0] if bphi[0] != 0 else 1.0)
        return [r * br[0] / denom, r * bz[0] / denom]

    phi = np.linspace(0.0, period, steps, endpoint=False)
    sol = solve_ivp(rhs, (0.0, period), [r0, z0], t_eval=phi, method="DOP853", rtol=1e-10, atol=1e-12)
    if not sol.success:
        raise RuntimeError(f"axis trace failed: {sol.message}")
    R = sol.y[0]
    Z = sol.y[1]
    br, bphi, bz = b_components(field, R, Z, phi)
    tiny = 1e-14
    denom = np.where(
        np.abs(bphi) > tiny,
        bphi,
        np.copysign(tiny, np.where(bphi != 0.0, bphi, 1.0)),
    )
    R_phi = R * br / denom
    Z_phi = R * bz / denom
    return phi, R, Z, R_phi, Z_phi


def find_axis(field, nfp: int, coil_r0: float, cfg: AxisGAConfig) -> AxisResult:
    t0 = time.perf_counter()
    t_search = time.perf_counter()
    best, history = search_axis_ga(field, nfp, coil_r0, cfg)
    search_time = time.perf_counter() - t_search
    has_axis = bool(best["best_residual"] <= cfg.tol)
    failure_reason = "" if has_axis else "residual_above_tol"
    phi = np.empty(0, dtype=float)
    R = np.empty(0, dtype=float)
    Z = np.empty(0, dtype=float)
    R_phi = np.empty(0, dtype=float)
    Z_phi = np.empty(0, dtype=float)
    trace_time = 0.0
    trace_error = ""
    if has_axis:
        t_trace = time.perf_counter()
        try:
            phi, R, Z, R_phi, Z_phi = trace_axis(field, best["best_R"], best["best_Z"], nfp, cfg.axis_trace_steps)
        except Exception as exc:
            has_axis = False
            failure_reason = "trace_failed"
            trace_error = repr(exc)
        trace_time = time.perf_counter() - t_trace
    return AxisResult(
        has_axis=has_axis,
        best_R=float(best["best_R"]),
        best_Z=float(best["best_Z"]),
        best_residual=float(best["best_residual"]),
        search_best_residual=float(best.get("search_residual", best["best_residual"])),
        generation=int(best["generation"]),
        history=history,
        phi=phi,
        R=R,
        Z=Z,
        R_phi=R_phi,
        Z_phi=Z_phi,
        time_s=time.perf_counter() - t0,
        search_time_s=search_time,
        trace_time_s=trace_time,
        backend="cpu",
        trace_error=trace_error,
        failure_reason=failure_reason,
        topology_class=str(best.get("topology_class", "")),
        topology_trace=float(best.get("topology_trace", float("nan"))),
        topology_det=float(best.get("topology_det", float("nan"))),
        topology_stability_margin=float(best.get("topology_stability_margin", float("nan"))),
        topology_robust=bool(best.get("topology_robust", False)),
        topology_ellipse_aspect=float(best.get("topology_ellipse_aspect", float("nan"))),
        topology_time_s=float(sum(float(row.get("topology_time_s", 0.0)) for row in history)),
    )


def find_axis_gpu(field_input, trace_field, nfp: int, coil_r0: float, cfg: AxisGAConfig, current_unit: str = "MA") -> AxisResult:
    import sys

    gpu_python = Path(__file__).resolve().parents[1] / "gpu_backend" / "python"
    if str(gpu_python) not in sys.path:
        sys.path.insert(0, str(gpu_python))
    from stellarator_gpu import CoilFieldGpu

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

    t0 = time.perf_counter()
    t_create = time.perf_counter()
    gpu_field = CoilFieldGpu(
        lib_path,
        field_input.coeffs_x,
        field_input.coeffs_y,
        field_input.coeffs_z,
        currents,
        nfp=nfp,
        segments_per_coil=cfg.gpu_segments_per_coil,
        device_id=cfg.gpu_device,
    )
    gpu_create_time = time.perf_counter() - t_create
    try:
        t_search = time.perf_counter()
        if cfg.method == "ga":
            best, history = search_axis_ga_gpu(gpu_field, nfp, coil_r0, cfg)
        elif cfg.method == "fixed_point":
            best, history = search_axis_fixed_point_gpu(
                gpu_field,
                field_input,
                nfp,
                coil_r0,
                cfg,
                grid=cfg.fixed_point_grid,
                max_candidates=cfg.fixed_point_max_candidates,
                newton_iters=cfg.fixed_point_newton_iters,
                stage="fast",
            )
            needs_topology_fallback = (
                cfg.fixed_point_topology_filter
                and cfg.fixed_point_require_elliptic
                and best["best_residual"] <= cfg.tol
                and not best.get("topology_robust", False)
            )
            if best["best_residual"] > cfg.tol or needs_topology_fallback:
                fb_best, fb_history = search_axis_fixed_point_gpu(
                    gpu_field,
                    field_input,
                    nfp,
                    coil_r0,
                    cfg,
                    grid=cfg.fixed_point_fallback_grid,
                    max_candidates=cfg.fixed_point_fallback_max_candidates,
                    newton_iters=cfg.fixed_point_fallback_newton_iters,
                    stage="fallback",
                )
                history.extend(fb_history)
                if _fixed_point_axis_prefer(fb_best, best):
                    best = fb_best
        else:
            raise ValueError(f"unknown axis search method {cfg.method!r}")
        search_time = time.perf_counter() - t_search
    finally:
        gpu_field.close()
    has_axis = bool(best["best_residual"] <= cfg.tol)
    failure_reason = "" if has_axis else "residual_above_tol"
    if has_axis and cfg.method == "fixed_point" and cfg.fixed_point_topology_filter and cfg.fixed_point_require_elliptic:
        if best.get("topology_class") != "elliptic" or not best.get("topology_accepted", False):
            has_axis = False
            failure_reason = "no_elliptic_axis_candidate"
    phi = np.empty(0, dtype=float)
    R = np.empty(0, dtype=float)
    Z = np.empty(0, dtype=float)
    R_phi = np.empty(0, dtype=float)
    Z_phi = np.empty(0, dtype=float)
    trace_time = 0.0
    trace_error = ""
    if has_axis:
        t_trace = time.perf_counter()
        try:
            phi, R, Z, R_phi, Z_phi = trace_axis(trace_field, best["best_R"], best["best_Z"], nfp, cfg.axis_trace_steps)
        except Exception as exc:
            has_axis = False
            failure_reason = "trace_failed"
            trace_error = repr(exc)
        trace_time = time.perf_counter() - t_trace
    if history:
        history[0]["gpu_create_time_s"] = gpu_create_time
    return AxisResult(
        has_axis=has_axis,
        best_R=float(best["best_R"]),
        best_Z=float(best["best_Z"]),
        best_residual=float(best["best_residual"]),
        search_best_residual=float(best.get("search_residual", best["best_residual"])),
        generation=int(best["generation"]),
        history=history,
        phi=phi,
        R=R,
        Z=Z,
        R_phi=R_phi,
        Z_phi=Z_phi,
        time_s=time.perf_counter() - t0,
        search_time_s=search_time,
        trace_time_s=trace_time,
        backend="gpu",
        trace_error=trace_error,
        failure_reason=failure_reason,
        topology_class=str(best.get("topology_class", "")),
        topology_trace=float(best.get("topology_trace", float("nan"))),
        topology_det=float(best.get("topology_det", float("nan"))),
        topology_stability_margin=float(best.get("topology_stability_margin", float("nan"))),
        topology_robust=bool(best.get("topology_robust", False)),
        topology_ellipse_aspect=float(best.get("topology_ellipse_aspect", float("nan"))),
        topology_time_s=float(sum(float(row.get("topology_time_s", 0.0)) for row in history)),
    )


def interp_periodic(phi, phi_axis, values, nfp: int):
    period = TWOPI / nfp
    p = np.mod(np.asarray(phi, dtype=float), period)
    x = np.r_[phi_axis, period]
    y = np.r_[values, values[0]]
    return np.interp(p, x, y)


def interp_periodic_hermite(phi, phi_axis, values, derivatives, nfp: int):
    """Interpolate a periodic value and its derivative consistently."""
    phi_axis = np.asarray(phi_axis, dtype=float)
    values = np.asarray(values, dtype=float)
    derivatives = np.asarray(derivatives, dtype=float)
    if not (phi_axis.ndim == values.ndim == derivatives.ndim == 1):
        raise ValueError("periodic Hermite inputs must be one-dimensional")
    if not (len(phi_axis) == len(values) == len(derivatives)) or len(phi_axis) < 2:
        raise ValueError("periodic Hermite inputs must have matching lengths >= 2")

    period = TWOPI / nfp
    p = np.mod(np.asarray(phi, dtype=float), period)
    idx = np.searchsorted(phi_axis, p, side="right") - 1
    idx = np.where(idx < 0, len(phi_axis) - 1, idx)
    next_idx = (idx + 1) % len(phi_axis)
    x0 = phi_axis[idx]
    x1 = np.where(next_idx == 0, period + phi_axis[0], phi_axis[next_idx])
    p_local = np.where(p < x0, p + period, p)
    h = x1 - x0
    t = (p_local - x0) / h

    y0 = values[idx]
    y1 = values[next_idx]
    d0 = derivatives[idx]
    d1 = derivatives[next_idx]
    t2 = t * t
    t3 = t2 * t
    value = (
        (2.0 * t3 - 3.0 * t2 + 1.0) * y0
        + (t3 - 2.0 * t2 + t) * h * d0
        + (-2.0 * t3 + 3.0 * t2) * y1
        + (t3 - t2) * h * d1
    )
    derivative = (
        (6.0 * t2 - 6.0 * t) * y0 / h
        + (3.0 * t2 - 4.0 * t + 1.0) * d0
        + (-6.0 * t2 + 6.0 * t) * y1 / h
        + (3.0 * t2 - 2.0 * t) * d1
    )
    return value, derivative
