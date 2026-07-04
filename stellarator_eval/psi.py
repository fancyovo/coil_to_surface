from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .axis import b_components, interp_periodic
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
        return (
            interp_periodic(phi, self.phi_axis, self.R_axis, self.nfp),
            interp_periodic(phi, self.phi_axis, self.Z_axis, self.nfp),
            interp_periodic(phi, self.phi_axis, self.R_axis_phi, self.nfp),
            interp_periodic(phi, self.phi_axis, self.Z_axis_phi, self.nfp),
        )


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


def psi_and_gradient(model: PsiModel, R, Z, phi):
    ra, za, rap, zap = model.axis_at(phi)
    X = (R - ra) / model.a
    Zc = (Z - za) / model.a
    psi, grad_R, grad_Z, grad_phi = _fixed_components(X, rap, model.a)
    for c, mode in zip(model.coeffs, model.modes):
        val, dR, dZ, dPhi = _basis_components(mode, X, Zc, phi, rap, zap, model.nfp, model.a)
        psi += c * val
        grad_R += c * dR
        grad_Z += c * dZ
        grad_phi += c * dPhi
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
    ra = interp_periodic(phi[keep], axis.phi, axis.R, nfp)
    za = interp_periodic(phi[keep], axis.phi, axis.Z, nfp)
    return ra + dr[keep], za + dz[keep], phi[keep]


def _assemble(field, modes, R, Z, phi, axis, nfp: int, cfg: PsiFitConfig):
    n_modes = len(modes)
    ata = np.zeros((n_modes, n_modes))
    atb = np.zeros(n_modes)
    rhs_norm2 = 0.0
    for start in range(0, len(R), cfg.batch_size):
        stop = min(start + cfg.batch_size, len(R))
        rr = R[start:stop]
        zz = Z[start:stop]
        pp = phi[start:stop]
        ra = interp_periodic(pp, axis.phi, axis.R, nfp)
        za = interp_periodic(pp, axis.phi, axis.Z, nfp)
        rap = interp_periodic(pp, axis.phi, axis.R_phi, nfp)
        zap = interp_periodic(pp, axis.phi, axis.Z_phi, nfp)
        X = (rr - ra) / cfg.a
        Zc = (zz - za) / cfg.a
        br, bphi, bz = b_components(field, rr, zz, pp)
        _, dR0, dZ0, dPhi0 = _fixed_components(X, rap, cfg.a)
        rhs = -(br * dR0 + bz * dZ0 + (bphi / rr) * dPhi0)
        mat = np.empty((len(rr), n_modes))
        for j, mode in enumerate(modes):
            _, dR, dZ, dPhi = _basis_components(mode, X, Zc, pp, rap, zap, nfp, cfg.a)
            mat[:, j] = br * dR + bz * dZ + (bphi / rr) * dPhi
        ata += mat.T @ mat
        atb += mat.T @ rhs
        rhs_norm2 += float(rhs @ rhs)
    return ata, atb, rhs_norm2


def _solve(ata, atb, ridge):
    scale = np.sqrt(np.maximum(np.diag(ata), 1e-30))
    scaled = ata / np.outer(scale, scale)
    scaled.flat[:: scaled.shape[0] + 1] += ridge
    coeff_scaled = np.linalg.solve(scaled, atb / scale)
    return coeff_scaled / scale, float(np.linalg.cond(scaled))


def validate_model(field, model: PsiModel, cfg: PsiFitConfig):
    rng = np.random.default_rng(20260704)
    phi = rng.uniform(0.0, TWOPI / model.nfp, cfg.validation_points)
    theta = rng.uniform(0.0, TWOPI, cfg.validation_points)
    u_min = (cfg.rho_min / cfg.a) ** 2
    rho = cfg.a * np.sqrt(rng.uniform(u_min, 1.0, cfg.validation_points))
    ra, za, _, _ = model.axis_at(phi)
    R = ra + rho * np.cos(theta)
    Z = za + rho * np.sin(theta)
    br, bphi, bz = b_components(field, R, Z, phi)
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
    }


def fit_psi(field, axis, nfp: int, cfg: PsiFitConfig) -> PsiModel:
    t0 = time.perf_counter()
    modes = build_modes(cfg.poly_degree, cfg.m_tor)
    R, Z, phi = _make_training_points(axis, nfp, cfg)
    ata, atb, rhs_norm2 = _assemble(field, modes, R, Z, phi, axis, nfp, cfg)
    coeffs, cond = _solve(ata, atb, cfg.ridge)
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
            "fixed_term": "x^2 coefficient = 1",
            "poly_degree": cfg.poly_degree,
            "m_tor": cfg.m_tor,
            "n_modes": len(modes),
            "training_points": int(len(R)),
            "condition_number": cond,
            "train_rms": float(train_rms),
            "time_s": time.perf_counter() - t0,
        },
    )
    model.fit_info.update(validate_model(field, model, cfg))
    return model


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
