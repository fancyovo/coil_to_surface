from __future__ import annotations

import time
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
    dphi = phi[1] - phi[0]
    R_phi = np.gradient(R, dphi, edge_order=2)
    Z_phi = np.gradient(Z, dphi, edge_order=2)
    return phi, R, Z, R_phi, Z_phi


def find_axis(field, nfp: int, coil_r0: float, cfg: AxisGAConfig) -> AxisResult:
    t0 = time.perf_counter()
    t_search = time.perf_counter()
    best, history = search_axis_ga(field, nfp, coil_r0, cfg)
    search_time = time.perf_counter() - t_search
    t_trace = time.perf_counter()
    phi, R, Z, R_phi, Z_phi = trace_axis(field, best["best_R"], best["best_Z"], nfp, cfg.axis_trace_steps)
    trace_time = time.perf_counter() - t_trace
    return AxisResult(
        has_axis=bool(best["best_residual"] <= cfg.tol),
        best_R=float(best["best_R"]),
        best_Z=float(best["best_Z"]),
        best_residual=float(best["best_residual"]),
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
        best, history = search_axis_ga_gpu(gpu_field, nfp, coil_r0, cfg)
        search_time = time.perf_counter() - t_search
    finally:
        gpu_field.close()
    t_trace = time.perf_counter()
    phi, R, Z, R_phi, Z_phi = trace_axis(trace_field, best["best_R"], best["best_Z"], nfp, cfg.axis_trace_steps)
    trace_time = time.perf_counter() - t_trace
    if history:
        history[0]["gpu_create_time_s"] = gpu_create_time
    return AxisResult(
        has_axis=bool(best["best_residual"] <= cfg.tol),
        best_R=float(best["best_R"]),
        best_Z=float(best["best_Z"]),
        best_residual=float(best["best_residual"]),
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
    )


def interp_periodic(phi, phi_axis, values, nfp: int):
    period = TWOPI / nfp
    p = np.mod(np.asarray(phi, dtype=float), period)
    x = np.r_[phi_axis, period]
    y = np.r_[values, values[0]]
    return np.interp(p, x, y)
