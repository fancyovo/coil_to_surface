from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .axis import b_components, interp_periodic_hermite
from .config import PsiFitConfig

TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class PolyMode:
    a: int
    b: int
    m: int
    kind: str


@dataclass
class PsiModel:
    coeffs: np.ndarray
    modes: list[PolyMode]
    nfp: int
    a: float
    phi_axis: np.ndarray
    R_axis: np.ndarray
    Z_axis: np.ndarray
    R_axis_phi: np.ndarray
    Z_axis_phi: np.ndarray
    fit_info: dict

    def axis_at(self, phi):
        ra, rap = interp_periodic_hermite(
            phi, self.phi_axis, self.R_axis, self.R_axis_phi, self.nfp
        )
        za, zap = interp_periodic_hermite(
            phi, self.phi_axis, self.Z_axis, self.Z_axis_phi, self.nfp
        )
        return ra, za, rap, zap


def build_modes(poly_degree: int, m_tor: int) -> list[PolyMode]:
    modes: list[PolyMode] = []
    for deg in range(2, poly_degree + 1):
        for ax in range(deg, -1, -1):
            bz = deg - ax
            for m in range(m_tor + 1):
                if ax == 2 and bz == 0 and m == 0:
                    continue
                if m == 0:
                    modes.append(PolyMode(ax, bz, m, "cos"))
                else:
                    modes.append(PolyMode(ax, bz, m, "cos"))
                    modes.append(PolyMode(ax, bz, m, "sin"))
    return modes


def _trig(mode: PolyMode, phi, nfp: int):
    arg = mode.m * nfp * phi
    if mode.kind == "cos":
        return np.cos(arg), -mode.m * nfp * np.sin(arg)
    return np.sin(arg), mode.m * nfp * np.cos(arg)


def _mono(mode: PolyMode, X, Z):
    val = (X**mode.a) * (Z**mode.b)
    dx = np.zeros_like(X)
    dz = np.zeros_like(X)
    if mode.a:
        dx = mode.a * (X ** (mode.a - 1)) * (Z**mode.b)
    if mode.b:
        dz = mode.b * (X**mode.a) * (Z ** (mode.b - 1))
    return val, dx, dz


def _basis_components(mode: PolyMode, X, Z, phi, R_axis_phi, Z_axis_phi, nfp: int, a_scale: float):
    mono, mono_x, mono_z = _mono(mode, X, Z)
    trig, trig_phi = _trig(mode, phi, nfp)
    dR = mono_x * trig / a_scale
    dZ = mono_z * trig / a_scale
    dPhi = mono * trig_phi - R_axis_phi * dR - Z_axis_phi * dZ
    return mono * trig, dR, dZ, dPhi


def _fixed_components(X, R_axis_phi, a_scale: float):
    val = X**2
    dR = 2.0 * X / a_scale
    dZ = np.zeros_like(X)
    dPhi = -R_axis_phi * dR
    return val, dR, dZ, dPhi


def _trig_tables(phi, nfp: int, m_tor: int):
    n = len(phi)
    cols = 1 + 2 * m_tor
    trig = np.empty((n, cols))
    trig_phi = np.empty((n, cols))
    trig[:, 0] = 1.0
    trig_phi[:, 0] = 0.0
    col = 1
    for m in range(1, m_tor + 1):
        arg = m * nfp * phi
        s = np.sin(arg)
        c = np.cos(arg)
        fac = m * nfp
        trig[:, col] = c
        trig_phi[:, col] = -fac * s
        col += 1
        trig[:, col] = s
        trig_phi[:, col] = fac * c
        col += 1
    return trig, trig_phi


def _power_table(x, degree: int):
    powers = [np.ones_like(x)]
    if degree >= 1:
        powers.append(x)
    for k in range(2, degree + 1):
        powers.append(powers[-1] * x)
    return powers


def _fill_design_matrix(mat, X, Zc, phi, rap, zap, br, bphi, bz, rr, nfp: int, cfg: PsiFitConfig):
    trig, trig_phi = _trig_tables(phi, nfp, cfg.m_tor)
    xpow = _power_table(X, cfg.poly_degree)
    zpow = _power_table(Zc, cfg.poly_degree)
    cphi = bphi / rr
    cR = br - cphi * rap
    cZ = bz - cphi * zap
    col = 0
    for deg in range(2, cfg.poly_degree + 1):
        for ax in range(deg, -1, -1):
            bz_exp = deg - ax
            mono = xpow[ax] * zpow[bz_exp]
            if ax:
                mono_x = ax * xpow[ax - 1] * zpow[bz_exp]
            else:
                mono_x = 0.0
            if bz_exp:
                mono_z = bz_exp * xpow[ax] * zpow[bz_exp - 1]
            else:
                mono_z = 0.0
            spatial = (cR * mono_x + cZ * mono_z) / cfg.a
            toroidal = cphi * mono
            block = spatial[:, None] * trig + toroidal[:, None] * trig_phi
            if ax == 2 and bz_exp == 0:
                block = block[:, 1:]
            width = block.shape[1]
            mat[:, col : col + width] = block
            col += width
    if col != mat.shape[1]:
        raise RuntimeError(f"internal basis size mismatch: filled {col}, expected {mat.shape[1]}")


def _mode_orders(model: PsiModel):
    poly_degree = int(model.fit_info.get("poly_degree", max(m.a + m.b for m in model.modes)))
    m_tor = int(model.fit_info.get("m_tor", max(m.m for m in model.modes)))
    return poly_degree, m_tor


def _coeff_block(coeffs, idx: int, ax: int, bz_exp: int, m_tor: int):
    cols = 1 + 2 * m_tor
    block = np.zeros(cols)
    if ax == 2 and bz_exp == 0:
        block[0] = 1.0
        width = cols - 1
        block[1:] = coeffs[idx : idx + width]
        return block, idx + width
    width = cols
    block[:] = coeffs[idx : idx + width]
    return block, idx + width


def _mode_arrays(modes: list[PolyMode]):
    kind = np.array([0 if m.kind == "cos" else 1 for m in modes], dtype=np.int32)
    return (
        np.array([m.a for m in modes], dtype=np.int32),
        np.array([m.b for m in modes], dtype=np.int32),
        np.array([m.m for m in modes], dtype=np.int32),
        kind,
    )


def psi_and_gradient(model: PsiModel, R, Z, phi):
    ra, za, rap, zap = model.axis_at(phi)
    X = (R - ra) / model.a
    Zc = (Z - za) / model.a
    poly_degree, m_tor = _mode_orders(model)
    trig, trig_phi = _trig_tables(phi, model.nfp, m_tor)
    xpow = _power_table(X, poly_degree)
    zpow = _power_table(Zc, poly_degree)
    psi = np.zeros_like(X)
    grad_R = np.zeros_like(X)
    grad_Z = np.zeros_like(X)
    grad_phi = np.zeros_like(X)
    idx = 0
    for deg in range(2, poly_degree + 1):
        for ax in range(deg, -1, -1):
            bz_exp = deg - ax
            block, idx = _coeff_block(model.coeffs, idx, ax, bz_exp, m_tor)
            amp = trig @ block
            amp_phi = trig_phi @ block
            mono = xpow[ax] * zpow[bz_exp]
            if ax:
                mono_x = ax * xpow[ax - 1] * zpow[bz_exp]
            else:
                mono_x = 0.0
            if bz_exp:
                mono_z = bz_exp * xpow[ax] * zpow[bz_exp - 1]
            else:
                mono_z = 0.0
            dR = mono_x * amp / model.a
            dZ = mono_z * amp / model.a
            psi += mono * amp
            grad_R += dR
            grad_Z += dZ
            grad_phi += mono * amp_phi - rap * dR - zap * dZ
    if idx != len(model.coeffs):
        raise RuntimeError(f"internal coefficient size mismatch: used {idx}, expected {len(model.coeffs)}")
    return psi, grad_R, grad_Z, grad_phi


def _make_training_points(axis, nfp: int, cfg: PsiFitConfig):
    dr_vals = np.linspace(-cfg.a, cfg.a, cfg.n_r)
    dz_vals = np.linspace(-cfg.a, cfg.a, cfg.n_z)
    phi_vals = np.linspace(0.0, TWOPI / nfp, cfg.n_phi, endpoint=False)
    drg, dzg, pg = np.meshgrid(dr_vals, dz_vals, phi_vals, indexing="ij")
    dr = drg.ravel()
    dz = dzg.ravel()
    phi = pg.ravel()
    rho = np.sqrt(dr**2 + dz**2)
    keep = (rho >= cfg.rho_min) & (rho <= cfg.a)
    ra, _ = interp_periodic_hermite(
        phi[keep], axis.phi, axis.R, axis.R_phi, nfp
    )
    za, _ = interp_periodic_hermite(
        phi[keep], axis.phi, axis.Z, axis.Z_phi, nfp
    )
    return ra + dr[keep], za + dz[keep], phi[keep]


def _b_components_gpu(gpu_field, r, z, phi, *, precision: str = "fp64"):
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    phi = np.asarray(phi, dtype=float)
    cp = np.cos(phi)
    sp = np.sin(phi)
    xyz = np.ascontiguousarray(np.column_stack([r * cp, r * sp, z]))
    b = gpu_field.eval_B(xyz, precision=precision)
    br = b[:, 0] * cp + b[:, 1] * sp
    bphi = -b[:, 0] * sp + b[:, 1] * cp
    bz = b[:, 2]
    return br, bphi, bz


def _make_gpu_field(field_input, nfp: int, cfg: PsiFitConfig, current_unit: str):
    import sys

    gpu_python = Path(__file__).resolve().parents[1] / "gpu_backend" / "python"
    if str(gpu_python) not in sys.path:
        sys.path.insert(0, str(gpu_python))
    from stellarator_gpu import CoilFieldGpu

    lib_path = Path(cfg.gpu_lib_path)
    if not lib_path.is_absolute():
        lib_path = Path.cwd() / lib_path
    unit = current_unit.lower()
    if unit in {"ma", "megaamp", "megaamps"}:
        currents = np.asarray(field_input.currents, dtype=float) * 1e6
    elif unit in {"a", "amp", "amps"}:
        currents = np.asarray(field_input.currents, dtype=float)
    else:
        raise ValueError(f"unknown current_unit={current_unit!r}; use 'MA' or 'A'")
    return CoilFieldGpu(
        lib_path,
        field_input.coeffs_x,
        field_input.coeffs_y,
        field_input.coeffs_z,
        currents,
        nfp=nfp,
        segments_per_coil=cfg.gpu_segments_per_coil,
        device_id=cfg.gpu_device,
    )


def _assemble(field, modes, R, Z, phi, axis, nfp: int, cfg: PsiFitConfig, b_sampler=None, normal_eq_sampler=None):
    n_modes = len(modes)
    ata = np.zeros((n_modes, n_modes))
    atb = np.zeros(n_modes)
    rhs_norm2 = 0.0
    timings = {
        "assemble_interp_s": 0.0,
        "assemble_b_sample_s": 0.0,
        "assemble_basis_s": 0.0,
        "assemble_normal_eq_s": 0.0,
        "assemble_normal_eq_gpu_s": 0.0,
        "assemble_normal_eq_cpu_s": 0.0,
    }
    if b_sampler is None:
        b_sampler = lambda rr, zz, pp: b_components(field, rr, zz, pp)
    for start in range(0, len(R), cfg.batch_size):
        stop = min(start + cfg.batch_size, len(R))
        rr = R[start:stop]
        zz = Z[start:stop]
        pp = phi[start:stop]
        t = time.perf_counter()
        ra, rap = interp_periodic_hermite(pp, axis.phi, axis.R, axis.R_phi, nfp)
        za, zap = interp_periodic_hermite(pp, axis.phi, axis.Z, axis.Z_phi, nfp)
        X = (rr - ra) / cfg.a
        Zc = (zz - za) / cfg.a
        timings["assemble_interp_s"] += time.perf_counter() - t
        t = time.perf_counter()
        br, bphi, bz = b_sampler(rr, zz, pp)
        timings["assemble_b_sample_s"] += time.perf_counter() - t
        t = time.perf_counter()
        _, dR0, dZ0, dPhi0 = _fixed_components(X, rap, cfg.a)
        rhs = -(br * dR0 + bz * dZ0 + (bphi / rr) * dPhi0)
        mat = np.empty((len(rr), n_modes))
        _fill_design_matrix(mat, X, Zc, pp, rap, zap, br, bphi, bz, rr, nfp, cfg)
        timings["assemble_basis_s"] += time.perf_counter() - t
        t = time.perf_counter()
        if normal_eq_sampler is None:
            ata += mat.T @ mat
            atb += mat.T @ rhs
            timings["assemble_normal_eq_cpu_s"] += time.perf_counter() - t
        else:
            batch_ata, batch_atb = normal_eq_sampler(mat, rhs)
            ata += batch_ata
            atb += batch_atb
            timings["assemble_normal_eq_gpu_s"] += time.perf_counter() - t
        rhs_norm2 += float(rhs @ rhs)
        timings["assemble_normal_eq_s"] += time.perf_counter() - t
    return ata, atb, rhs_norm2, timings


def _solve(ata, atb, ridge):
    scale = np.sqrt(np.maximum(np.diag(ata), 1e-30))
    scaled = ata / np.outer(scale, scale)
    scaled.flat[:: scaled.shape[0] + 1] += ridge
    coeff_scaled = np.linalg.solve(scaled, atb / scale)
    return coeff_scaled / scale, float(np.linalg.cond(scaled))


def validate_model(field, model: PsiModel, cfg: PsiFitConfig, b_sampler=None):
    rng = np.random.default_rng(20260704)
    phi = rng.uniform(0.0, TWOPI / model.nfp, cfg.validation_points)
    theta = rng.uniform(0.0, TWOPI, cfg.validation_points)
    u_min = (cfg.rho_min / cfg.a) ** 2
    rho = cfg.a * np.sqrt(rng.uniform(u_min, 1.0, cfg.validation_points))
    ra, za, _, _ = model.axis_at(phi)
    R = ra + rho * np.cos(theta)
    Z = za + rho * np.sin(theta)
    if b_sampler is None:
        b_sampler = lambda rr, zz, pp: b_components(field, rr, zz, pp)
    t = time.perf_counter()
    br, bphi, bz = b_sampler(R, Z, phi)
    b_time = time.perf_counter() - t
    psi, gr, gz, gp = psi_and_gradient(model, R, Z, phi)
    vals = br * gr + bz * gz + (bphi / R) * gp
    b_norm = np.sqrt(br**2 + bphi**2 + bz**2)
    grad_norm = np.sqrt(gr**2 + gz**2 + (gp / R) ** 2)
    angle = np.abs(vals) / np.maximum(b_norm * grad_norm, 1e-14)
    return {
        "validation_rms": float(np.sqrt(np.mean(vals**2))),
        "validation_angle_mean": float(np.mean(angle)),
        "validation_angle_p95": float(np.percentile(angle, 95)),
        "validation_angle_max": float(np.max(angle)),
        "validation_angle_l2": float(np.sqrt(np.sum(vals**2) / np.sum((b_norm * grad_norm) ** 2))),
        "psi_min": float(np.min(psi)),
        "psi_max": float(np.max(psi)),
        "validation_b_sample_s": float(b_time),
    }


def fit_psi(field, axis, nfp: int, cfg: PsiFitConfig, field_input=None, current_unit: str = "MA") -> PsiModel:
    t0 = time.perf_counter()
    backend = cfg.backend.lower()
    if backend not in {"cpu", "gpu", "fullgpu"}:
        raise ValueError(f"unknown psi backend {cfg.backend!r}; use 'cpu', 'gpu', or 'fullgpu'")
    if backend in {"gpu", "fullgpu"} and field_input is None:
        raise ValueError("GPU psi backends require field_input")
    gpu_field = None
    b_sampler = None
    normal_eq_sampler = None
    gpu_create_time = 0.0
    if backend in {"gpu", "fullgpu"}:
        tg = time.perf_counter()
        gpu_field = _make_gpu_field(field_input, nfp, cfg, current_unit)
        gpu_create_time = time.perf_counter() - tg
        b_sampler = lambda rr, zz, pp: _b_components_gpu(gpu_field, rr, zz, pp)
    normal_eq_backend = cfg.normal_eq_backend.lower()
    normal_eq_precision = cfg.normal_eq_precision.lower()
    linear_solver = cfg.linear_solver.lower()
    if normal_eq_precision not in {"fp64", "fp32"}:
        raise ValueError("normal_eq_precision must be 'fp64' or 'fp32'")
    if linear_solver not in {"normal_eq", "qr"}:
        raise ValueError("linear_solver must be 'normal_eq' or 'qr'")
    if backend == "fullgpu":
        if linear_solver == "normal_eq":
            normal_eq_backend = "gpu_full"
        else:
            normal_eq_backend = "gpu_full_qr"
    else:
        if linear_solver != "normal_eq":
            raise ValueError("linear_solver='qr' is currently only supported with psi backend 'fullgpu'")
        if normal_eq_backend == "auto":
            normal_eq_backend = "gpu" if backend == "gpu" else "cpu"
        if normal_eq_backend not in {"cpu", "gpu"}:
            raise ValueError(f"unknown normal_eq_backend {cfg.normal_eq_backend!r}; use 'auto', 'cpu', or 'gpu'")
    if normal_eq_backend == "gpu":
        if gpu_field is None:
            raise ValueError("GPU normal equation requires psi backend 'gpu'")
        normal_eq_sampler = lambda mat, rhs: gpu_field.normal_eq(mat, rhs, precision=normal_eq_precision)
    else:
        if normal_eq_backend == "cpu":
            normal_eq_precision = "fp64"
    t_modes = time.perf_counter()
    modes = build_modes(cfg.poly_degree, cfg.m_tor)
    modes_time = time.perf_counter() - t_modes
    t_points = time.perf_counter()
    R, Z, phi = _make_training_points(axis, nfp, cfg)
    points_time = time.perf_counter() - t_points
    if backend == "fullgpu":
        mode_a, mode_b, mode_m, mode_kind = _mode_arrays(modes)
        coeffs, train_rms, gpu_fit_stats = gpu_field.fit_psi_fullgpu(
            R,
            Z,
            phi,
            axis.R,
            axis.Z,
            axis.R_phi,
            axis.Z_phi,
            mode_a,
            mode_b,
            mode_m,
            mode_kind,
            a=cfg.a,
            poly_degree=cfg.poly_degree,
            m_tor=cfg.m_tor,
            ridge=cfg.ridge,
            precision=cfg.normal_eq_precision,
            solver=linear_solver,
        )
        cond = -1.0
        assemble_time = float(gpu_fit_stats["assemble_s"] + gpu_fit_stats["linear_prep_s"])
        solve_time = float(gpu_fit_stats["solve_s"])
        assemble_timings = {
            "assemble_interp_s": 0.0,
            "assemble_b_sample_s": 0.0,
            "assemble_basis_s": float(gpu_fit_stats["assemble_s"]),
            "assemble_normal_eq_s": float(gpu_fit_stats["linear_prep_s"]) if linear_solver == "normal_eq" else 0.0,
            "assemble_normal_eq_gpu_s": float(gpu_fit_stats["linear_prep_s"]) if linear_solver == "normal_eq" else 0.0,
            "assemble_normal_eq_cpu_s": 0.0,
            "qr_prep_s": float(gpu_fit_stats["linear_prep_s"]) if linear_solver == "qr" else 0.0,
            "qr_transpose_s": float(gpu_fit_stats["qr_transpose_s"]),
            "qr_scale_s": float(gpu_fit_stats["qr_scale_s"]),
            "qr_factor_s": float(gpu_fit_stats["qr_factor_s"]),
            "qr_apply_qtb_s": float(gpu_fit_stats["qr_apply_qtb_s"]),
            "qr_tri_s": float(gpu_fit_stats["qr_tri_s"]),
            "fullgpu_copy_in_s": float(gpu_fit_stats["copy_in_s"]),
            "fullgpu_residual_s": float(gpu_fit_stats["residual_s"]),
            "fullgpu_copy_out_s": float(gpu_fit_stats["copy_out_s"]),
            "fullgpu_total_kernel_s": float(gpu_fit_stats["total_s"]),
        }
    else:
        t_assemble = time.perf_counter()
        try:
            ata, atb, rhs_norm2, assemble_timings = _assemble(
                field, modes, R, Z, phi, axis, nfp, cfg, b_sampler=b_sampler, normal_eq_sampler=normal_eq_sampler
            )
        finally:
            pass
        assemble_time = time.perf_counter() - t_assemble
        t_solve = time.perf_counter()
        coeffs, cond = _solve(ata, atb, cfg.ridge)
        solve_time = time.perf_counter() - t_solve
        train_resid = float(np.sqrt(abs(coeffs @ (ata @ coeffs) - 2.0 * coeffs @ atb + rhs_norm2)))
        train_rms = train_resid / np.sqrt(len(R))
    model = PsiModel(
        coeffs=coeffs,
        modes=modes,
        nfp=nfp,
        a=cfg.a,
        phi_axis=axis.phi,
        R_axis=axis.R,
        Z_axis=axis.Z,
        R_axis_phi=axis.R_phi,
        Z_axis_phi=axis.Z_phi,
        fit_info={
            "basis": "cartesian_poly_toroidal_fourier",
            "backend": backend,
            "linear_solver": linear_solver,
            "normal_eq_backend": normal_eq_backend,
            "normal_eq_precision": normal_eq_precision,
            "fixed_term": "x^2 coefficient = 1",
            "poly_degree": cfg.poly_degree,
            "m_tor": cfg.m_tor,
            "n_modes": len(modes),
            "training_points": int(len(R)),
            "condition_number": cond,
            "train_rms": float(train_rms),
            "gpu_create_s": float(gpu_create_time),
            "mode_build_s": float(modes_time),
            "training_point_s": float(points_time),
            "assemble_s": float(assemble_time),
            **{k: float(v) for k, v in assemble_timings.items()},
            "solve_s": float(solve_time),
            "time_s": time.perf_counter() - t0,
        },
    )
    t_validation = time.perf_counter()
    try:
        model.fit_info.update(validate_model(field, model, cfg, b_sampler=b_sampler))
        model.fit_info["validation_s"] = float(time.perf_counter() - t_validation)
        model.fit_info["time_s"] = time.perf_counter() - t0
        return model
    finally:
        if gpu_field is not None:
            gpu_field.close()


def psi_ray_value_and_derivative(model: PsiModel, rho, theta, phi):
    rho = np.asarray(rho, dtype=float)
    theta = np.asarray(theta, dtype=float)
    X = rho * np.cos(theta) / model.a
    Z = rho * np.sin(theta) / model.a
    dX = np.cos(theta) / model.a
    dZ = np.sin(theta) / model.a
    psi = X**2
    dpsi = 2.0 * X * dX
    for c, mode in zip(model.coeffs, model.modes):
        trig, _ = _trig(mode, phi, model.nfp)
        mono, mono_x, mono_z = _mono(mode, X, Z)
        psi += c * mono * trig
        dpsi += c * (mono_x * dX + mono_z * dZ) * trig
    return psi, dpsi


def model_to_npz_dict(model: PsiModel) -> dict:
    return {
        "coeffs": model.coeffs,
        "mode_a": np.array([m.a for m in model.modes], dtype=int),
        "mode_b": np.array([m.b for m in model.modes], dtype=int),
        "mode_m": np.array([m.m for m in model.modes], dtype=int),
        "mode_kind": np.array([m.kind for m in model.modes]),
        "nfp": model.nfp,
        "a": model.a,
        "phi_axis": model.phi_axis,
        "R_axis": model.R_axis,
        "Z_axis": model.Z_axis,
        "R_axis_phi": model.R_axis_phi,
        "Z_axis_phi": model.Z_axis_phi,
        **{f"info_{k}": v for k, v in model.fit_info.items() if np.isscalar(v) or isinstance(v, str)},
    }
