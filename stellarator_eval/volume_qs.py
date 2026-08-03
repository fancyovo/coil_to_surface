from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import time
from pathlib import Path

import numpy as np

from .alpha_clebsch import (
    ClebschMode,
    FluxCalibration,
    build_clebsch_modes,
    zernike_radial,
)
from .config import VolumeQSConfig
from .psi import PolyMode, PsiModel


TWOPI = 2.0 * np.pi
MU0 = 4.0e-7 * np.pi


@dataclass
class StraightFieldFit:
    modes: list[ClebschMode]
    lambda_coeffs: np.ndarray
    iota_coeffs: np.ndarray
    diagnostics: dict

    def iota(self, rho):
        u = np.asarray(rho, dtype=float) ** 2
        return sum(coefficient * u**power for power, coefficient in enumerate(self.iota_coeffs))


@dataclass
class FluxScaleFit:
    coefficients: np.ndarray
    s_edge: float
    diagnostics: dict

    def derivative(self, s):
        u = np.asarray(s, dtype=float) / self.s_edge
        return sum(coefficient * u**power for power, coefficient in enumerate(self.coefficients))

    def evaluate(self, s):
        u = np.asarray(s, dtype=float) / self.s_edge
        return self.s_edge * sum(
            coefficient * u ** (power + 1) / (power + 1)
            for power, coefficient in enumerate(self.coefficients)
        )

    @property
    def psi_edge(self) -> float:
        return float(self.s_edge * sum(c / (k + 1) for k, c in enumerate(self.coefficients)))


def load_psi_model(path: str | Path) -> PsiModel:
    data = np.load(path)
    modes = [
        PolyMode(int(a), int(b), int(m), str(kind))
        for a, b, m, kind in zip(data["mode_a"], data["mode_b"], data["mode_m"], data["mode_kind"])
    ]
    fit_info = {
        key.removeprefix("info_"): data[key].item()
        for key in data.files
        if key.startswith("info_")
    }
    return PsiModel(
        coeffs=np.asarray(data["coeffs"], dtype=float),
        modes=modes,
        nfp=int(data["nfp"]),
        a=float(data["a"]),
        phi_axis=np.asarray(data["phi_axis"], dtype=float),
        R_axis=np.asarray(data["R_axis"], dtype=float),
        Z_axis=np.asarray(data["Z_axis"], dtype=float),
        R_axis_phi=np.asarray(data["R_axis_phi"], dtype=float),
        Z_axis_phi=np.asarray(data["Z_axis_phi"], dtype=float),
        fit_info=fit_info,
    )


def _psi_tensor(model: PsiModel):
    degree = max(max(mode.a + mode.b for mode in model.modes), 2)
    m_tor = max(max(mode.m for mode in model.modes), 0)
    monomials = [(a, degree_now - a) for degree_now in range(2, degree + 1) for a in range(degree_now, -1, -1)]
    monomial_index = {mode: index for index, mode in enumerate(monomials)}
    coefficients = np.zeros((len(monomials), 1 + 2 * m_tor), dtype=float)
    coefficients[monomial_index[(2, 0)], 0] = 1.0
    for coefficient, mode in zip(model.coeffs, model.modes):
        trig_index = 0 if mode.m == 0 else 2 * mode.m - (1 if mode.kind == "cos" else 0)
        coefficients[monomial_index[(mode.a, mode.b)], trig_index] = coefficient
    return monomials, coefficients, degree, m_tor


def evaluate_psi_tensor_numpy(model: PsiModel, R, Z, phi):
    R, Z, phi = np.broadcast_arrays(np.asarray(R, float), np.asarray(Z, float), np.asarray(phi, float))
    shape = R.shape
    R = R.ravel()
    Z = Z.ravel()
    phi = phi.ravel()
    ra, za, rap, zap = model.axis_at(phi)
    x = (R - ra) / model.a
    z = (Z - za) / model.a
    monomials, coefficients, degree, m_tor = _psi_tensor(model)
    xp = np.column_stack([x**power for power in range(degree + 1)])
    zp = np.column_stack([z**power for power in range(degree + 1)])
    mono = np.column_stack([xp[:, a] * zp[:, b] for a, b in monomials])
    mono_x = np.column_stack([
        np.zeros_like(x) if a == 0 else a * xp[:, a - 1] * zp[:, b] for a, b in monomials
    ])
    mono_z = np.column_stack([
        np.zeros_like(z) if b == 0 else b * xp[:, a] * zp[:, b - 1] for a, b in monomials
    ])
    trig = [np.ones_like(phi)]
    trig_phi = [np.zeros_like(phi)]
    for m in range(1, m_tor + 1):
        argument = m * model.nfp * phi
        trig.extend([np.cos(argument), np.sin(argument)])
        trig_phi.extend([-m * model.nfp * np.sin(argument), m * model.nfp * np.cos(argument)])
    trig = np.column_stack(trig)
    trig_phi = np.column_stack(trig_phi)
    value = np.einsum("ni,ij,nj->n", mono, coefficients, trig, optimize=True)
    derivative_x = np.einsum("ni,ij,nj->n", mono_x, coefficients, trig, optimize=True)
    derivative_z = np.einsum("ni,ij,nj->n", mono_z, coefficients, trig, optimize=True)
    derivative_phi_fixed = np.einsum("ni,ij,nj->n", mono, coefficients, trig_phi, optimize=True)
    grad_R = derivative_x / model.a
    grad_Z = derivative_z / model.a
    grad_phi = derivative_phi_fixed - rap * grad_R - zap * grad_Z
    return tuple(array.reshape(shape) for array in (value, grad_R, grad_Z, grad_phi))


def evaluate_psi_tensor_torch(model: PsiModel, R, Z, phi, *, device: str = "cuda", precision: str = "fp32"):
    import torch

    dtype = torch.float32 if precision == "fp32" else torch.float64
    R = np.asarray(R, dtype=float)
    shape = R.shape
    R_flat = R.ravel()
    Z_flat = np.asarray(Z, dtype=float).ravel()
    phi_flat = np.asarray(phi, dtype=float).ravel()
    ra, za, rap, zap = model.axis_at(phi_flat)
    x = torch.as_tensor((R_flat - ra) / model.a, dtype=dtype, device=device)
    z = torch.as_tensor((Z_flat - za) / model.a, dtype=dtype, device=device)
    phi_tensor = torch.as_tensor(phi_flat, dtype=dtype, device=device)
    monomials, coefficients, degree, m_tor = _psi_tensor(model)
    xp = torch.stack([x**power for power in range(degree + 1)], dim=1)
    zp = torch.stack([z**power for power in range(degree + 1)], dim=1)
    mono = torch.stack([xp[:, a] * zp[:, b] for a, b in monomials], dim=1)
    mono_x = torch.stack([
        torch.zeros_like(x) if a == 0 else a * xp[:, a - 1] * zp[:, b] for a, b in monomials
    ], dim=1)
    mono_z = torch.stack([
        torch.zeros_like(z) if b == 0 else b * xp[:, a] * zp[:, b - 1] for a, b in monomials
    ], dim=1)
    trig = [torch.ones_like(phi_tensor)]
    trig_phi = [torch.zeros_like(phi_tensor)]
    for m in range(1, m_tor + 1):
        argument = m * model.nfp * phi_tensor
        trig.extend([torch.cos(argument), torch.sin(argument)])
        trig_phi.extend([-m * model.nfp * torch.sin(argument), m * model.nfp * torch.cos(argument)])
    trig = torch.stack(trig, dim=1)
    trig_phi = torch.stack(trig_phi, dim=1)
    coefficient_tensor = torch.as_tensor(coefficients, dtype=dtype, device=device)
    value = torch.einsum("ni,ij,nj->n", mono, coefficient_tensor, trig)
    derivative_x = torch.einsum("ni,ij,nj->n", mono_x, coefficient_tensor, trig)
    derivative_z = torch.einsum("ni,ij,nj->n", mono_z, coefficient_tensor, trig)
    derivative_phi_fixed = torch.einsum("ni,ij,nj->n", mono, coefficient_tensor, trig_phi)
    grad_R = derivative_x / model.a
    grad_Z = derivative_z / model.a
    rap_tensor = torch.as_tensor(rap, dtype=dtype, device=device)
    zap_tensor = torch.as_tensor(zap, dtype=dtype, device=device)
    grad_phi = derivative_phi_fixed - rap_tensor * grad_R - zap_tensor * grad_Z
    output = tuple(array.detach().cpu().numpy().reshape(shape) for array in (value, grad_R, grad_Z, grad_phi))
    del xp, zp, mono, mono_x, mono_z, trig, trig_phi, coefficient_tensor
    return output


def _cartesian_gradient(radial, toroidal, vertical, phi, R):
    cosine = np.cos(phi)
    sine = np.sin(phi)
    physical_toroidal = toroidal / R
    return np.column_stack([
        radial * cosine - physical_toroidal * sine,
        radial * sine + physical_toroidal * cosine,
        vertical,
    ])


def _surface_radius_on_rays(
    model: PsiModel,
    level: float,
    theta,
    phi,
    *,
    max_radius: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the fitted polynomial surface on independent (theta, phi) rays."""
    theta, phi = np.broadcast_arrays(np.asarray(theta, float), np.asarray(phi, float))
    theta = theta.ravel()
    phi = phi.ravel()
    cosine = np.cos(theta)
    sine = np.sin(theta)
    degree = max(mode.a + mode.b for mode in model.modes)
    coefficients = np.zeros((degree + 1, len(theta)), dtype=float)
    coefficients[2] = cosine * cosine
    for coefficient, mode in zip(model.coeffs, model.modes):
        argument = mode.m * model.nfp * phi
        trig = np.cos(argument) if mode.kind == "cos" else np.sin(argument)
        coefficients[mode.a + mode.b] += (
            coefficient * cosine**mode.a * sine**mode.b * trig
        )
    limit = model.a if max_radius is None else min(float(max_radius), model.a)
    quadratic = coefficients[2]
    safe_quadratic = np.where(quadratic > 1e-10, quadratic, 1.0)
    radius = np.where(
        quadratic > 1e-10,
        model.a * np.sqrt(max(float(level), 0.0) / safe_quadratic),
        model.a * np.sqrt(max(float(level), 1e-16)),
    )
    radius = np.clip(radius, 1e-12 * model.a, limit)
    residual = np.full_like(radius, np.inf)
    for _ in range(30):
        normalized = radius / model.a
        value = coefficients[-1].copy()
        derivative = np.zeros_like(radius)
        for power in range(degree - 1, -1, -1):
            derivative = derivative * normalized + value
            value = value * normalized + coefficients[power]
        derivative /= model.a
        residual = value - level
        if float(np.max(np.abs(residual))) <= 1e-12:
            break
        denominator = np.where(
            np.abs(derivative) > 1e-14,
            derivative,
            np.where(derivative >= 0.0, 1e-14, -1e-14),
        )
        step = np.clip(
            residual / denominator,
            -0.4 * np.maximum(radius, 1e-10),
            0.4 * np.maximum(radius, 1e-10),
        )
        radius = np.clip(radius - step, 1e-12 * model.a, limit)
    return radius, np.abs(residual)


def sample_volume_points(
    model: PsiModel,
    config: VolumeQSConfig,
    *,
    device: str = "cuda",
    extent: float | None = None,
) -> dict[str, np.ndarray]:
    extent = model.a if extent is None else min(float(extent), model.a)
    lower = config.s_edge * config.rho_min**2
    n_phi, n_theta, n_radial = _volume_candidate_lattice_shape(
        config.point_count,
        config.grid_phi,
        config.ray_candidate_oversampling,
    )
    p_index, t_index, r_index = np.meshgrid(
        np.arange(n_phi),
        np.arange(n_theta),
        np.arange(n_radial),
        indexing="ij",
    )
    phi_flat = (
        (p_index.ravel() + 0.417) * TWOPI / (model.nfp * n_phi)
    )
    geometric_theta = TWOPI * (
        (
            t_index.ravel()
            + 0.613
            + 0.38196601125 * r_index.ravel()
            + 0.217 * p_index.ravel()
        )
        % n_theta
    ) / n_theta
    boundary_radius, boundary_residual = _surface_radius_on_rays(
        model,
        config.s_edge,
        geometric_theta,
        phi_flat,
        max_radius=extent,
    )
    radial_fraction = np.sqrt(
        config.rho_min**2
        + (1.0 - config.rho_min**2)
        * (r_index.ravel() + 0.371)
        / n_radial
    )
    radius = boundary_radius * radial_fraction
    dr_flat = radius * np.cos(geometric_theta)
    dz_flat = radius * np.sin(geometric_theta)
    ra, za, _, _ = model.axis_at(phi_flat)
    R = ra + dr_flat
    Z = za + dz_flat
    s, grad_R, grad_Z, grad_phi_coordinate = evaluate_psi_tensor_torch(
        model, R, Z, phi_flat, device=device, precision=config.precision
    )
    keep = (
        np.isfinite(s)
        & (s >= lower)
        & (s <= config.s_edge)
        & (R > 0.0)
        & (boundary_residual <= config.flux_boundary_tolerance)
    )
    available = int(np.sum(keep))
    minimum_count = _minimum_volume_sample_count(
        point_count=config.point_count,
        alpha_fit_point_count=config.alpha_fit_point_count,
        candidate_count=len(phi_flat),
        minimum_candidate_valid_fraction=config.minimum_candidate_valid_fraction,
    )
    if available < minimum_count:
        raise RuntimeError(
            f"surface-volume sampler produced {available} valid points, fewer than the "
            f"fixed-budget minimum {minimum_count}"
        )
    indices = np.flatnonzero(keep)
    if available > config.point_count:
        selection = np.floor(np.linspace(0, available, config.point_count, endpoint=False)).astype(int)
        indices = indices[selection]
    selected_boundary_radius = boundary_radius[indices]
    R = R[indices]
    Z = Z[indices]
    phi = phi_flat[indices]
    s = s[indices]
    grad_R = grad_R[indices]
    grad_Z = grad_Z[indices]
    grad_phi_coordinate = grad_phi_coordinate[indices]
    ra, za, rap, zap = model.axis_at(phi)
    x = R - ra
    z = Z - za
    radius2 = np.maximum(x * x + z * z, 1e-30)
    theta = -np.arctan2(z, x)
    theta_R = z / radius2
    theta_Z = -x / radius2
    theta_phi = (x * zap - z * rap) / radius2
    grad_s = _cartesian_gradient(grad_R, grad_phi_coordinate, grad_Z, phi, R)
    grad_theta = _cartesian_gradient(theta_R, theta_phi, theta_Z, phi, R)
    grad_phi = _cartesian_gradient(np.zeros_like(R), np.ones_like(R), np.zeros_like(R), phi, R)
    xyz = np.column_stack([R * np.cos(phi), R * np.sin(phi), Z])
    return {
        "xyz": xyz,
        "R": R,
        "Z": Z,
        "phi": phi,
        "s": s,
        "rho": np.sqrt(np.clip(s / config.s_edge, 0.0, None)),
        "theta": theta,
        "grad_s": grad_s,
        "grad_theta": grad_theta,
        "grad_phi": grad_phi,
        "volume_weight": _physical_volume_weights(R, selected_boundary_radius),
        "nfp": np.asarray(model.nfp),
        "candidate_count": np.asarray([len(phi_flat)]),
        "available_count": np.asarray([available]),
        "minimum_count": np.asarray([minimum_count]),
        "candidate_valid_fraction": np.asarray([available / len(phi_flat)]),
    }


def _volume_candidate_lattice_shape(
    point_count: int,
    grid_phi: int,
    oversampling: float,
) -> tuple[int, int, int]:
    if point_count < 1:
        raise ValueError("point_count must be positive")
    if not np.isfinite(oversampling) or oversampling < 1.0:
        raise ValueError("ray candidate oversampling must be at least 1")
    candidate_target = int(np.ceil(point_count * oversampling))
    n_phi = max(8, int(grid_phi))
    per_phi = int(np.ceil(candidate_target / n_phi))
    n_theta = max(16, int(np.ceil(np.sqrt(per_phi))))
    n_radial = int(np.ceil(per_phi / n_theta))
    return n_phi, n_theta, n_radial


def _physical_volume_weights(R, boundary_radius):
    return np.asarray(R) * np.asarray(boundary_radius) ** 2


def _minimum_volume_sample_count(
    point_count,
    alpha_fit_point_count,
    candidate_count,
    minimum_candidate_valid_fraction=0.95,
):
    fraction = float(minimum_candidate_valid_fraction)
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("minimum_candidate_valid_fraction must be between 0 and 1")
    return max(
        int(alpha_fit_point_count),
        int(point_count),
        int(np.ceil(fraction * candidate_count)),
    )


def _budget_flux_levels(candidate_levels, maximum_attempts: int = 5) -> list[float]:
    levels = sorted(
        {float(value) for value in candidate_levels if float(value) > 0.0},
        reverse=True,
    )
    if len(levels) <= maximum_attempts:
        return levels
    selected = levels[:2]
    for level in levels[2:]:
        if level <= 0.65 * selected[-1]:
            selected.append(level)
        if len(selected) >= maximum_attempts:
            break
    if len(selected) < maximum_attempts and levels[-1] not in selected:
        selected.append(levels[-1])
    return selected[:maximum_attempts]


def _balanced_radial_weights(rho, bins: int):
    index = np.minimum((np.asarray(rho) * bins).astype(int), bins - 1)
    counts = np.bincount(index, minlength=bins)
    weights = 1.0 / np.sqrt(np.maximum(counts[index], 1))
    return weights / np.sqrt(np.mean(weights * weights))


def _basis_derivatives_numpy(modes, rho, theta, phi, nfp, coefficients):
    derivative_theta = np.zeros_like(rho, dtype=float)
    derivative_phi = np.zeros_like(rho, dtype=float)
    radial_cache = {}
    phase_cache = {}
    for coefficient, mode in zip(coefficients, modes):
        radial = radial_cache.setdefault((mode.l, mode.m), zernike_radial(rho, mode.l, mode.m))
        cosine, sine = phase_cache.setdefault(
            (mode.m, mode.n),
            (np.cos(mode.m * theta - mode.n * nfp * phi), np.sin(mode.m * theta - mode.n * nfp * phi)),
        )
        if mode.kind == "cos":
            derivative_theta += coefficient * (-mode.m * radial * sine)
            derivative_phi += coefficient * (mode.n * nfp * radial * sine)
        else:
            derivative_theta += coefficient * (mode.m * radial * cosine)
            derivative_phi += coefficient * (-mode.n * nfp * radial * cosine)
    return derivative_theta, derivative_phi


def fit_straight_field_alpha(points: dict, B, config: VolumeQSConfig, *, device: str = "cuda") -> StraightFieldFit:
    import torch

    start = time.perf_counter()
    dtype = torch.float32 if config.precision == "fp32" else torch.float64
    B = np.asarray(B, dtype=float)
    grad_s = np.asarray(points["grad_s"], dtype=float)
    normal_coefficient = np.sum(B * grad_s, axis=1) / np.maximum(np.sum(grad_s * grad_s, axis=1), 1e-30)
    tangent_B = B - normal_coefficient[:, None] * grad_s
    along_theta = np.sum(tangent_B * points["grad_theta"], axis=1)
    along_phi = np.sum(tangent_B * points["grad_phi"], axis=1)
    modes = build_clebsch_modes(
        config.alpha_radial_order,
        config.alpha_poloidal_order,
        config.alpha_toroidal_order,
    )
    count = len(B)
    columns = len(modes) + config.iota_degree + 1
    rho = torch.as_tensor(points["rho"], dtype=dtype, device=device)
    theta = torch.as_tensor(points["theta"], dtype=dtype, device=device)
    phi = torch.as_tensor(points["phi"], dtype=dtype, device=device)
    bt_theta = torch.as_tensor(along_theta, dtype=dtype, device=device)
    bt_phi = torch.as_tensor(along_phi, dtype=dtype, device=device)
    weights_np = _balanced_radial_weights(points["rho"], config.radial_bins) / np.maximum(np.linalg.norm(B, axis=1), 1e-30)
    weights_np /= np.sqrt(np.mean(weights_np * weights_np))
    weights = torch.as_tensor(weights_np, dtype=dtype, device=device)
    design = torch.empty((count, columns), dtype=dtype, device=device)
    radial_cache = {}
    phase_cache = {}
    for column, mode in enumerate(modes):
        key = (mode.l, mode.m)
        if key not in radial_cache:
            radial_cache[key] = torch.as_tensor(
                zernike_radial(points["rho"], mode.l, mode.m), dtype=dtype, device=device
            )
        phase_key = (mode.m, mode.n)
        if phase_key not in phase_cache:
            argument = mode.m * theta - mode.n * int(points["nfp"]) * phi
            phase_cache[phase_key] = (torch.cos(argument), torch.sin(argument))
        cosine, sine = phase_cache[phase_key]
        radial = radial_cache[key]
        if mode.kind == "cos":
            derivative_theta = -mode.m * radial * sine
            derivative_phi = mode.n * points["nfp"] * radial * sine
        else:
            derivative_theta = mode.m * radial * cosine
            derivative_phi = -mode.n * points["nfp"] * radial * cosine
        design[:, column] = weights * (derivative_theta * bt_theta + derivative_phi * bt_phi)
    u = rho * rho
    for power in range(config.iota_degree + 1):
        design[:, len(modes) + power] = -weights * u**power * bt_phi
    rhs = -weights * bt_theta
    column_scale = torch.linalg.vector_norm(design, dim=0).clamp_min(1e-20)
    design /= column_scale[None, :]
    if config.alpha_ridge > 0.0:
        ridge = torch.sqrt(torch.as_tensor(config.alpha_ridge, dtype=dtype, device=device))
        design = torch.cat([design, ridge * torch.eye(columns, dtype=dtype, device=device)], dim=0)
        rhs = torch.cat([rhs, torch.zeros(columns, dtype=dtype, device=device)])
    torch.cuda.synchronize() if str(device).startswith("cuda") else None
    assemble_s = time.perf_counter() - start
    solve_start = time.perf_counter()
    solution_scaled = torch.linalg.lstsq(design, rhs[:, None], driver="gels").solution[:, 0]
    torch.cuda.synchronize() if str(device).startswith("cuda") else None
    solve_s = time.perf_counter() - solve_start
    weighted_residual = design[:count] @ solution_scaled - rhs[:count]
    weighted_relative_l2 = torch.linalg.vector_norm(weighted_residual) / torch.linalg.vector_norm(
        rhs[:count]
    ).clamp_min(1e-20)
    coefficients = (solution_scaled / column_scale).detach().cpu().numpy()
    fit = StraightFieldFit(
        modes=modes,
        lambda_coeffs=coefficients[: len(modes)],
        iota_coeffs=coefficients[len(modes) :],
        diagnostics={},
    )
    iota = fit.iota(points["rho"])
    fit.diagnostics = {
        "point_count": int(count),
        "column_count": int(columns),
        "precision": config.precision,
        "assemble_s": float(assemble_s),
        "solve_s": float(solve_s),
        "total_s": float(time.perf_counter() - start),
        "relative_l2": float(weighted_relative_l2.item()),
        "residual_weighting": "radial_balance/|B|",
        "normal_B_relative_l2": float(np.linalg.norm(normal_coefficient[:, None] * grad_s) / np.linalg.norm(B)),
        "iota_min": float(np.min(iota)),
        "iota_max": float(np.max(iota)),
    }
    return fit


def calibrate_toroidal_flux_gpu(
    model: PsiModel,
    gpu_field,
    config: VolumeQSConfig,
    *,
    levels=None,
) -> FluxCalibration:
    """Calibrate fitted s to signed toroidal flux in two vectorized batches."""
    start = time.perf_counter()
    if levels is None:
        level_fraction = (
            np.arange(1, config.flux_level_count + 1) / config.flux_level_count
        ) ** 2
        levels = config.s_edge * level_fraction
    else:
        levels = np.asarray(levels, dtype=float)
        if np.any(levels <= 0.0) or np.any(np.diff(levels) <= 0.0):
            raise ValueError("flux calibration levels must be positive and increasing")
        if levels[-1] > config.s_edge * (1.0 + 1e-12):
            raise ValueError("flux calibration levels must not exceed config.s_edge")
    phis = np.linspace(0.0, TWOPI / model.nfp, config.flux_phi_count, endpoint=False)
    theta = (np.arange(config.flux_theta_count) + 0.5) * TWOPI / config.flux_theta_count
    phi_grid, theta_grid = np.meshgrid(phis, theta, indexing="ij")
    phi_flat = phi_grid.ravel()
    theta_flat = theta_grid.ravel()
    cosine = np.cos(theta_flat)
    sine = np.sin(theta_flat)
    polynomial_degree_s = max(mode.a + mode.b for mode in model.modes)
    radial_coefficients = np.zeros((polynomial_degree_s + 1, len(theta_flat)), dtype=float)
    radial_coefficients[2] = cosine * cosine
    for coefficient, mode in zip(model.coeffs, model.modes):
        argument = mode.m * model.nfp * phi_flat
        trig = np.cos(argument) if mode.kind == "cos" else np.sin(argument)
        radial_coefficients[mode.a + mode.b] += (
            coefficient * cosine**mode.a * sine**mode.b * trig
        )
    radial_coefficients = radial_coefficients.reshape(
        polynomial_degree_s + 1, config.flux_phi_count, config.flux_theta_count
    )
    q = radial_coefficients[2]
    radius = np.where(
        q[:, None, :] > 1e-10,
        model.a * np.sqrt(np.maximum(levels[None, :, None], 0.0) / q[:, None, :]),
        model.a * np.sqrt(np.maximum(levels[None, :, None], 1e-16)),
    )
    radius = np.clip(radius, 1e-12 * model.a, model.a)
    for _ in range(30):
        normalized_radius = radius / model.a
        value = np.zeros_like(radius)
        derivative = np.zeros_like(radius)
        for degree in range(2, polynomial_degree_s + 1):
            coefficient = radial_coefficients[degree][:, None, :]
            value += coefficient * normalized_radius**degree
            derivative += (
                degree
                * coefficient
                * normalized_radius ** (degree - 1)
                / model.a
            )
        residual = value - levels[None, :, None]
        if float(np.max(np.abs(residual))) <= 1e-12:
            break
        floor = np.where(derivative >= 0.0, 1e-14, -1e-14)
        denominator = np.where(np.abs(derivative) > 1e-14, derivative, floor)
        step = np.clip(
            residual / denominator,
            -0.4 * np.maximum(radius, 1e-10),
            0.4 * np.maximum(radius, 1e-10),
        )
        radius = np.clip(radius - step, 1e-12 * model.a, model.a)
    normalized_radius = radius / model.a
    boundary_value = sum(
        radial_coefficients[degree][:, None, :] * normalized_radius**degree
        for degree in range(2, polynomial_degree_s + 1)
    )
    boundary_residual = np.abs(boundary_value - levels[None, :, None])
    boundary_radius = radius
    geometry_s = time.perf_counter() - start

    gauss_x, gauss_w = np.polynomial.legendre.leggauss(config.flux_radial_quadrature)
    radial = 0.5 * boundary_radius[..., None] * (gauss_x + 1.0)
    radial_weight = 0.5 * boundary_radius[..., None] * gauss_w
    theta_full = theta[None, None, :, None]
    phi_full = phis[:, None, None, None]
    axis_R, axis_Z, _, _ = model.axis_at(phis)
    R = axis_R[:, None, None, None] + radial * np.cos(theta_full)
    Z = axis_Z[:, None, None, None] + radial * np.sin(theta_full)
    phi_values = np.broadcast_to(phi_full, R.shape)
    xyz = np.column_stack(
        [R.ravel() * np.cos(phi_values.ravel()), R.ravel() * np.sin(phi_values.ravel()), Z.ravel()]
    )
    field_start = time.perf_counter()
    field = gpu_field.eval_B(xyz, precision=config.precision)
    field_s = time.perf_counter() - field_start
    B_phi = (
        -field[:, 0] * np.sin(phi_values.ravel())
        + field[:, 1] * np.cos(phi_values.ravel())
    ).reshape(R.shape)
    integrand = B_phi * radial
    psi_sections = (TWOPI / config.flux_theta_count) * np.sum(
        integrand * radial_weight, axis=(2, 3)
    ) / TWOPI

    psi_mean = np.mean(psi_sections, axis=0)
    polynomial_degree = config.flux_degree + 1
    design = np.column_stack([levels**power for power in range(1, polynomial_degree + 1)])
    coefficients, *_ = np.linalg.lstsq(design, psi_mean, rcond=None)
    fitted = design @ coefficients
    derivative_grid = np.column_stack(
        [power * levels ** (power - 1) for power in range(1, polynomial_degree + 1)]
    ) @ coefficients
    section_std = np.std(psi_sections, axis=0)
    section_relative_std = section_std / np.maximum(np.abs(psi_mean), 1e-30)
    boundary_residual_by_level = np.max(boundary_residual, axis=(0, 2))
    edge_radius = boundary_radius[:, -1, :]
    edge_axis_R = axis_R[:, None]
    edge_integrand = (
        0.5 * edge_axis_R * edge_radius**2
        + np.cos(theta)[None, :] * edge_radius**3 / 3.0
    )
    edge_cross_section_area = np.pi * np.mean(edge_radius**2, axis=1)
    edge_volume = TWOPI * TWOPI * float(np.mean(edge_integrand))
    axis_major_radius = float(np.mean(axis_R))
    effective_minor_radius = np.sqrt(
        max(edge_volume, 0.0) / max(2.0 * np.pi**2 * axis_major_radius, 1e-30)
    )
    diagnostics = {
        "time_s": float(time.perf_counter() - start),
        "geometry_s": float(geometry_s),
        "field_s": float(field_s),
        "edge_s": float(levels[-1]),
        "edge_psi_toroidal_per_radian": float(fitted[-1]),
        "edge_flux_weber": float(TWOPI * fitted[-1]),
        "fit_relative_rms": float(
            np.sqrt(np.mean((fitted - psi_mean) ** 2)) / max(abs(psi_mean[-1]), 1e-30)
        ),
        "section_relative_std_max": float(np.max(section_relative_std)),
        "section_relative_std_edge": float(section_relative_std[-1]),
        "section_relative_std_by_level": section_relative_std.tolist(),
        "boundary_residual_max": float(np.max(boundary_residual)),
        "boundary_residual_edge": float(boundary_residual_by_level[-1]),
        "boundary_residual_by_level": boundary_residual_by_level.tolist(),
        "boundary_radius_max": float(np.max(boundary_radius)),
        "boundary_radius_edge_max": float(np.max(boundary_radius[:, -1, :])),
        "boundary_cross_section_area_edge_mean": float(np.mean(edge_cross_section_area)),
        "boundary_volume_edge": float(edge_volume),
        "boundary_axis_major_radius_mean": axis_major_radius,
        "boundary_effective_minor_radius_edge": float(effective_minor_radius),
        "boundary_effective_inverse_aspect_ratio": float(
            effective_minor_radius / max(axis_major_radius, 1e-30)
        ),
        "derivative_min": float(np.min(derivative_grid)),
        "derivative_max": float(np.max(derivative_grid)),
        "monotone": bool(np.all(derivative_grid * np.sign(fitted[-1]) > 0.0)),
    }
    diagnostics["quality_ok"] = bool(
        diagnostics["boundary_residual_max"] <= config.flux_boundary_tolerance
        and diagnostics["section_relative_std_edge"]
        <= config.flux_section_relative_std_tolerance
        and diagnostics["monotone"]
    )
    return FluxCalibration(
        s_knots=levels,
        psi_knots=psi_mean,
        psi_by_section=psi_sections,
        phi_sections=phis,
        polynomial_coeffs=np.asarray(coefficients),
        polynomial_degree=polynomial_degree,
        diagnostics=diagnostics,
    )


def apply_flux_coordinates(points: dict, calibration: FluxCalibration) -> None:
    """Attach physical toroidal-flux coordinates and gradients in place."""
    psi = calibration.evaluate(points["s"])
    derivative = calibration.derivative(points["s"])
    points["psi"] = psi
    points["grad_psi"] = derivative[:, None] * points["grad_s"]
    points["rho"] = np.sqrt(np.clip(psi / calibration.psi_edge, 0.0, None))


def subset_volume_points(points: dict, indices) -> dict:
    indices = np.asarray(indices, dtype=int)
    count = len(points["s"])
    return {
        key: value[indices] if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == count else value
        for key, value in points.items()
    }


def fit_alpha_vector_gpu_qr(
    points: dict,
    B,
    config: VolumeQSConfig,
    *,
    device: str = "cuda",
) -> StraightFieldFit:
    """Fit lambda and iota from B = grad(psi) x grad(alpha) with dense GPU QR."""
    import torch

    start = time.perf_counter()
    dtype = torch.float32 if config.precision == "fp32" else torch.float64
    B = np.asarray(B, dtype=float)
    cross_theta_np = np.cross(points["grad_psi"], points["grad_theta"])
    cross_phi_np = np.cross(points["grad_psi"], points["grad_phi"])
    modes = build_clebsch_modes(
        config.alpha_radial_order,
        config.alpha_poloidal_order,
        config.alpha_toroidal_order,
    )
    count = len(B)
    columns = len(modes) + config.iota_degree + 1
    rho = torch.as_tensor(points["rho"], dtype=dtype, device=device)
    theta = torch.as_tensor(points["theta"], dtype=dtype, device=device)
    phi = torch.as_tensor(points["phi"], dtype=dtype, device=device)
    cross_theta = torch.as_tensor(cross_theta_np, dtype=dtype, device=device)
    cross_phi = torch.as_tensor(cross_phi_np, dtype=dtype, device=device)
    bfield = torch.as_tensor(B, dtype=dtype, device=device)
    # The physical vector equation is already consistently scaled in tesla.
    # Extra 1/|B| or radial weights bias iota when the fitted s coordinate is imperfect.
    weights_np = np.ones(count, dtype=float)
    weights = torch.as_tensor(weights_np, dtype=dtype, device=device)
    rhs = ((bfield - cross_theta) * weights[:, None]).reshape(-1)
    design = torch.empty((3 * count, columns), dtype=dtype, device=device)

    radial_cache = {}
    phase_cache = {}
    for column, mode in enumerate(modes):
        key = (mode.l, mode.m)
        if key not in radial_cache:
            radial_cache[key] = torch.as_tensor(
                zernike_radial(points["rho"], mode.l, mode.m), dtype=dtype, device=device
            )
        phase_key = (mode.m, mode.n)
        if phase_key not in phase_cache:
            argument = mode.m * theta - mode.n * int(points["nfp"]) * phi
            phase_cache[phase_key] = (torch.cos(argument), torch.sin(argument))
        cosine, sine = phase_cache[phase_key]
        radial = radial_cache[key]
        if mode.kind == "cos":
            derivative_theta = -mode.m * radial * sine
            derivative_phi = mode.n * int(points["nfp"]) * radial * sine
        else:
            derivative_theta = mode.m * radial * cosine
            derivative_phi = -mode.n * int(points["nfp"]) * radial * cosine
        vector = derivative_theta[:, None] * cross_theta + derivative_phi[:, None] * cross_phi
        design[:, column] = (vector * weights[:, None]).reshape(-1)
    u = rho * rho
    for power in range(config.iota_degree + 1):
        vector = -(u**power)[:, None] * cross_phi
        design[:, len(modes) + power] = (vector * weights[:, None]).reshape(-1)

    column_scale = torch.linalg.vector_norm(design, dim=0).clamp_min(1e-20)
    design /= column_scale[None, :]
    if config.alpha_ridge > 0.0:
        ridge = torch.sqrt(torch.as_tensor(config.alpha_ridge, dtype=dtype, device=device))
        design = torch.cat([design, ridge * torch.eye(columns, dtype=dtype, device=device)], dim=0)
        rhs = torch.cat([rhs, torch.zeros(columns, dtype=dtype, device=device)])
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    assemble_s = time.perf_counter() - start
    solve_start = time.perf_counter()
    solution_scaled = torch.linalg.lstsq(design, rhs[:, None], driver="gels").solution[:, 0]
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    solve_s = time.perf_counter() - solve_start
    coefficients = (solution_scaled / column_scale).detach().cpu().numpy()
    fit = StraightFieldFit(
        modes=modes,
        lambda_coeffs=coefficients[: len(modes)],
        iota_coeffs=coefficients[len(modes) :],
        diagnostics={},
    )
    lambda_theta, lambda_phi = _basis_derivatives_numpy(
        modes, points["rho"], points["theta"], points["phi"], int(points["nfp"]), fit.lambda_coeffs
    )
    iota = fit.iota(points["rho"])
    reconstructed = (
        (1.0 + lambda_theta)[:, None] * cross_theta_np
        + (lambda_phi - iota)[:, None] * cross_phi_np
    )
    residual = reconstructed - B
    fit.diagnostics = {
        "point_count": int(count),
        "row_count": int(3 * count),
        "column_count": int(columns),
        "precision": config.precision,
        "assemble_s": float(assemble_s),
        "solve_s": float(solve_s),
        "total_s": float(time.perf_counter() - start),
        "relative_l2": float(np.linalg.norm(residual) / np.linalg.norm(B)),
        "relative_point_p95": float(
            np.percentile(np.linalg.norm(residual, axis=1) / np.maximum(np.linalg.norm(B, axis=1), 1e-30), 95)
        ),
        "one_plus_lambda_theta_min": float(np.min(1.0 + lambda_theta)),
        "noninvertible_fraction": float(np.mean(1.0 + lambda_theta <= 0.0)),
        "iota_min": float(np.min(iota)),
        "iota_max": float(np.max(iota)),
    }
    return fit


def evaluate_alpha_derivatives(fit: StraightFieldFit, points: dict):
    lambda_theta, lambda_phi = _basis_derivatives_numpy(
        fit.modes,
        points["rho"],
        points["theta"],
        points["phi"],
        int(points["nfp"]),
        fit.lambda_coeffs,
    )
    return lambda_theta, lambda_phi, fit.iota(points["rho"])


def evaluate_straight_field_alpha(points: dict, B, fit: StraightFieldFit) -> dict:
    B = np.asarray(B, dtype=float)
    grad_s = np.asarray(points["grad_s"], dtype=float)
    normal_coefficient = np.sum(B * grad_s, axis=1) / np.maximum(
        np.sum(grad_s * grad_s, axis=1), 1e-30
    )
    tangent_B = B - normal_coefficient[:, None] * grad_s
    along_theta = np.sum(tangent_B * points["grad_theta"], axis=1)
    along_phi = np.sum(tangent_B * points["grad_phi"], axis=1)
    lambda_theta, lambda_phi, iota = evaluate_alpha_derivatives(fit, points)
    residual = (1.0 + lambda_theta) * along_theta + (lambda_phi - iota) * along_phi
    return {
        "point_count": int(len(B)),
        "relative_l2": float(np.linalg.norm(residual) / max(np.linalg.norm(along_theta), 1e-30)),
        "normal_B_relative_l2": float(
            np.linalg.norm(normal_coefficient[:, None] * grad_s) / np.linalg.norm(B)
        ),
        "iota_min": float(np.min(iota)),
        "iota_max": float(np.max(iota)),
    }


def fit_flux_scale(points: dict, B, alpha_fit: StraightFieldFit, config: VolumeQSConfig) -> FluxScaleFit:
    start = time.perf_counter()
    B = np.asarray(B, dtype=float)
    grad_s = np.asarray(points["grad_s"], dtype=float)
    normal_coefficient = np.sum(B * grad_s, axis=1) / np.maximum(np.sum(grad_s * grad_s, axis=1), 1e-30)
    tangent_B = B - normal_coefficient[:, None] * grad_s
    lambda_theta, lambda_phi, iota = evaluate_alpha_derivatives(alpha_fit, points)
    cross_theta = np.cross(grad_s, points["grad_theta"])
    cross_phi = np.cross(grad_s, points["grad_phi"])
    direction = (1.0 + lambda_theta)[:, None] * cross_theta + (lambda_phi - iota)[:, None] * cross_phi
    u = np.asarray(points["s"]) / config.s_edge
    weights = _balanced_radial_weights(points["rho"], config.radial_bins) / np.maximum(np.linalg.norm(B, axis=1), 1e-30)
    matrix = np.column_stack([
        (weights[:, None] * direction * u[:, None] ** power).reshape(-1)
        for power in range(config.flux_degree + 1)
    ])
    rhs = (weights[:, None] * tangent_B).reshape(-1)
    coefficients, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    reconstructed = sum(
        coefficient * u**power for power, coefficient in enumerate(coefficients)
    )[:, None] * direction
    residual = reconstructed - tangent_B
    fit = FluxScaleFit(np.asarray(coefficients), config.s_edge, diagnostics={})
    derivative_grid = fit.derivative(np.linspace(0.0, config.s_edge, 201))
    fit.diagnostics = {
        "degree": int(config.flux_degree),
        "total_s": float(time.perf_counter() - start),
        "relative_l2": float(np.linalg.norm(residual) / np.linalg.norm(tangent_B)),
        "derivative_min": float(np.min(derivative_grid)),
        "derivative_max": float(np.max(derivative_grid)),
        "psi_edge": fit.psi_edge,
        "monotone": bool(np.all(derivative_grid > 0.0) or np.all(derivative_grid < 0.0)),
    }
    return fit


def vacuum_G(currents_a, nfp: int, toroidal_flux: float) -> float:
    """Return the vacuum Boozer G for radian angular coordinates."""
    toroidal_flux = float(toroidal_flux)
    if not np.isfinite(toroidal_flux) or toroidal_flux == 0.0:
        raise ValueError("vacuum G requires nonzero signed toroidal flux")
    linked_current = 2 * int(nfp) * np.sum(np.abs(np.asarray(currents_a, dtype=float)))
    magnitude = MU0 * linked_current / TWOPI
    return float(np.copysign(magnitude, toroidal_flux))


def compute_f_c(points: dict, B, grad_B, alpha_fit: StraightFieldFit, flux_fit: FluxScaleFit, *, M: int, N: int, G: float, I: float = 0.0):
    B = np.asarray(B, dtype=float)
    grad_B = np.asarray(grad_B, dtype=float)
    magnitude = np.linalg.norm(B, axis=1)
    grad_magnitude = np.einsum("nij,ni->nj", grad_B, B) / np.maximum(magnitude[:, None], 1e-30)
    grad_psi = points.get("grad_psi")
    if grad_psi is None:
        F = flux_fit.derivative(points["s"])
        grad_psi = F[:, None] * points["grad_s"]
    iota = alpha_fit.iota(points["rho"])
    A = np.sum(np.cross(B, grad_psi) * grad_magnitude, axis=1)
    C = np.sum(B * grad_magnitude, axis=1)
    first = (M * iota - N) * A
    second = (M * G + N * I) * C
    f_c = first - second
    normalized = f_c / np.maximum(magnitude**3, 1e-30)
    return {
        "B_magnitude": magnitude,
        "grad_B_magnitude": grad_magnitude,
        "grad_psi": grad_psi,
        "iota": iota,
        "A": A,
        "C": C,
        "first_term": first,
        "second_term": second,
        "f_C": f_c,
        "f_C_over_B3": normalized,
    }


def summarize_volume_qs(points: dict, fields: dict, radial_bins: int) -> dict:
    rho = np.asarray(points["rho"])
    weight = np.asarray(points["volume_weight"])
    normalized = np.asarray(fields["f_C_over_B3"])
    raw = np.asarray(fields["f_C"])

    def effective_fraction(selected):
        selected_weight = weight[selected]
        return float(
            np.sum(selected_weight) ** 2
            / max(len(selected_weight) * np.sum(selected_weight**2), 1e-300)
        )

    def weighted_rms(value, selected):
        selected_weight = weight[selected]
        return float(np.sqrt(np.sum(selected_weight * value[selected] ** 2) / np.sum(selected_weight)))

    bins = []
    edges = np.linspace(float(np.min(rho)), float(np.max(rho)), radial_bins + 1)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (rho >= lower) & (rho < upper if index + 1 < radial_bins else rho <= upper)
        if not np.any(selected):
            continue
        bins.append({
            "rho_min": float(lower),
            "rho_max": float(upper),
            "count": int(np.sum(selected)),
            "f_C_rms_T3": weighted_rms(raw, selected),
            "f_C_over_B3_rms": weighted_rms(normalized, selected),
        })
    selected = np.ones(len(rho), dtype=bool)
    cancellation = np.abs(fields["f_C"]) / np.maximum(
        np.abs(fields["first_term"]) + np.abs(fields["second_term"]), 1e-30
    )
    return {
        "point_count": int(len(rho)),
        "f_C_rms_T3": weighted_rms(raw, selected),
        "f_C_over_B3_rms": weighted_rms(normalized, selected),
        "f_C_over_B3_abs_p95": float(np.percentile(np.abs(normalized), 95)),
        "volume_weight_effective_fraction": effective_fraction(selected),
        "cancellation_ratio_median": float(np.median(cancellation)),
        "radial_bins": bins,
    }


def evaluate_volume_qs_model(
    field_input,
    model: PsiModel,
    candidate_levels,
    config: VolumeQSConfig,
    *,
    current_unit: str = "A",
    target_helicity: tuple[int, int] | None = None,
) -> dict:
    """Evaluate volume f_C from a stable fitted invariant and screened levels."""
    import sys

    from .field import normalize_currents

    gpu_python = Path(__file__).resolve().parents[1] / "gpu_backend" / "python"
    if str(gpu_python) not in sys.path:
        sys.path.insert(0, str(gpu_python))
    from stellarator_gpu import CoilFieldGpu

    started = time.perf_counter()
    timings = {}
    currents_a = normalize_currents(field_input.currents, current_unit)
    gpu_lib = Path(config.gpu_lib_path)
    if not gpu_lib.is_absolute():
        gpu_lib = Path.cwd() / gpu_lib
    create_start = time.perf_counter()
    gpu = CoilFieldGpu(
        gpu_lib,
        field_input.coeffs_x,
        field_input.coeffs_y,
        field_input.coeffs_z,
        currents_a,
        field_input.nfp,
        segments_per_coil=config.gpu_segments_per_coil,
        device_id=config.gpu_device,
    )
    timings["gpu_field_create_s"] = float(time.perf_counter() - create_start)
    try:
        flux_attempts = []
        calibration = None
        selected_config = None
        for level in _budget_flux_levels(candidate_levels):
            trial_config = replace(config, s_edge=level)
            flux_start = time.perf_counter()
            trial = calibrate_toroidal_flux_gpu(model, gpu, trial_config)
            attempt = {
                "s_edge": level,
                "time_s": float(time.perf_counter() - flux_start),
                **trial.diagnostics,
            }
            flux_attempts.append(attempt)
            if trial.diagnostics["quality_ok"]:
                calibration = trial
                selected_config = trial_config
                break
        if calibration is None or selected_config is None:
            return {
                "status": "failed",
                "reason": "no screened level passed flux calibration quality gates",
                "flux_attempts": flux_attempts,
                "timing": {**timings, "downstream_total_s": float(time.perf_counter() - started)},
            }

        extent = min(
            model.a,
            1.02 * float(calibration.diagnostics["boundary_radius_edge_max"]),
        )
        points_start = time.perf_counter()
        sampling_attempts = []
        while True:
            try:
                points = sample_volume_points(
                    model,
                    selected_config,
                    device=f"cuda:{selected_config.gpu_device}",
                    extent=extent,
                )
                sampling_attempts.append(
                    {"grid_xy": int(selected_config.grid_xy), "status": "ok"}
                )
                break
            except RuntimeError as exc:
                sampling_attempts.append(
                    {
                        "grid_xy": int(selected_config.grid_xy),
                        "status": "insufficient_points",
                        "reason": str(exc),
                    }
                )
                next_grid = int(np.ceil(selected_config.grid_xy * 1.25))
                if next_grid > 256:
                    raise
                selected_config = replace(selected_config, grid_xy=next_grid)
        apply_flux_coordinates(points, calibration)
        timings["volume_points_s"] = float(time.perf_counter() - points_start)
        field_start = time.perf_counter()
        B, grad_B = gpu.eval_B_grad(points["xyz"], precision=selected_config.precision)
        timings["B_grad_B_s"] = float(time.perf_counter() - field_start)
    finally:
        gpu.close()

    fit_count = min(selected_config.alpha_fit_point_count, len(B))
    fit_indices = np.floor(np.linspace(0, len(B), fit_count, endpoint=False)).astype(int)
    alpha = fit_straight_field_alpha(
        subset_volume_points(points, fit_indices),
        B[fit_indices],
        selected_config,
        device=f"cuda:{selected_config.gpu_device}",
    )
    timings["alpha_total_s"] = float(alpha.diagnostics["total_s"])
    timings["alpha_qr_s"] = float(alpha.diagnostics["solve_s"])

    target_helicity = target_helicity or (1, 0)
    specs = {
        "target": target_helicity,
        "QA": (1, 0),
        "QH_plus": (1, int(field_input.nfp)),
        "QH_minus": (1, -int(field_input.nfp)),
        "QP": (0, int(field_input.nfp)),
    }
    G = vacuum_G(currents_a, field_input.nfp, calibration.psi_edge)
    metric_start = time.perf_counter()
    metrics = {}
    for name, (M, N) in specs.items():
        fields = compute_f_c(points, B, grad_B, alpha, calibration, M=M, N=N, G=G)
        metrics[name] = summarize_volume_qs(points, fields, selected_config.radial_bins)
    timings["qs_metrics_s"] = float(time.perf_counter() - metric_start)
    timings["downstream_total_s"] = float(time.perf_counter() - started)
    return {
        "status": "ok",
        "target_helicity": list(target_helicity),
        "s_edge": float(selected_config.s_edge),
        "sampling_extent": float(extent),
        "sampling": {
            "point_count": int(len(points["s"])),
            "alpha_fit_point_count": int(fit_count),
            "candidate_count": int(points["candidate_count"][0]),
            "available_count": int(points["available_count"][0]),
            "attempts": sampling_attempts,
        },
        "flux_attempts": flux_attempts,
        "flux": {
            "coefficients": np.asarray(calibration.polynomial_coeffs).tolist(),
            "diagnostics": calibration.diagnostics,
        },
        "alpha": {
            "iota_coefficients": alpha.iota_coeffs.tolist(),
            "diagnostics": alpha.diagnostics,
        },
        "G": float(G),
        "metrics": metrics,
        "timing": timings,
    }
