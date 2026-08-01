from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Iterable

import numpy as np
from scipy.interpolate import PchipInterpolator

from .psi import PsiModel, psi_and_gradient, psi_ray_value_and_derivative

TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class ClebschMode:
    l: int
    m: int
    n: int
    kind: str


@dataclass
class FluxCalibration:
    s_knots: np.ndarray
    psi_knots: np.ndarray
    psi_by_section: np.ndarray
    phi_sections: np.ndarray
    polynomial_coeffs: np.ndarray
    polynomial_degree: int
    diagnostics: dict

    def evaluate(self, s):
        s = np.asarray(s, dtype=float)
        out = np.zeros_like(s)
        for power, coeff in enumerate(self.polynomial_coeffs, start=1):
            out += coeff * s**power
        return out

    def derivative(self, s):
        s = np.asarray(s, dtype=float)
        out = np.zeros_like(s)
        for power, coeff in enumerate(self.polynomial_coeffs, start=1):
            out += power * coeff * s ** (power - 1)
        return out

    @property
    def psi_edge(self) -> float:
        return float(self.evaluate(self.s_knots[-1]))


@dataclass
class AlphaFitResult:
    modes: list[ClebschMode]
    lambda_coeffs: np.ndarray
    iota_coeffs: np.ndarray
    radial_order: int
    poloidal_order: int
    toroidal_order: int
    iota_degree: int
    diagnostics: dict

    def iota(self, rho):
        u = np.asarray(rho, dtype=float) ** 2
        return sum(c * u**k for k, c in enumerate(self.iota_coeffs))


def load_alpha_fit(path) -> AlphaFitResult:
    data = np.load(path)
    modes = [
        ClebschMode(int(l), int(m), int(n), str(kind))
        for l, m, n, kind in zip(
            data["mode_l"], data["mode_m"], data["mode_n"], data["mode_kind"]
        )
    ]
    return AlphaFitResult(
        modes=modes,
        lambda_coeffs=np.asarray(data["lambda_coeffs"], dtype=float),
        iota_coeffs=np.asarray(data["iota_coeffs"], dtype=float),
        radial_order=int(data["radial_order"]),
        poloidal_order=int(data["poloidal_order"]),
        toroidal_order=int(data["toroidal_order"]),
        iota_degree=int(data["iota_degree"]),
        diagnostics={},
    )


def load_flux_calibration(path) -> FluxCalibration:
    data = np.load(path)
    coeffs = np.asarray(data["polynomial_coeffs"], dtype=float)
    return FluxCalibration(
        s_knots=np.asarray(data["s_knots"], dtype=float),
        psi_knots=np.asarray(data["psi_knots"], dtype=float),
        psi_by_section=np.asarray(data["psi_by_section"], dtype=float),
        phi_sections=np.asarray(data["phi_sections"], dtype=float),
        polynomial_coeffs=coeffs,
        polynomial_degree=len(coeffs),
        diagnostics={},
    )


def build_clebsch_modes(
    radial_order: int,
    poloidal_order: int,
    toroidal_order: int,
) -> list[ClebschMode]:
    """Build a complete real Fourier-Zernike basis with flux functions removed."""
    modes: list[ClebschMode] = []
    m_max = min(int(radial_order), int(poloidal_order))
    for m in range(m_max + 1):
        radial_degrees = range(m, int(radial_order) + 1, 2)
        n_values: Iterable[int]
        if m == 0:
            n_values = range(1, int(toroidal_order) + 1)
        else:
            n_values = range(-int(toroidal_order), int(toroidal_order) + 1)
        for n in n_values:
            for l in radial_degrees:
                modes.append(ClebschMode(l, m, n, "cos"))
                modes.append(ClebschMode(l, m, n, "sin"))
    return modes


def zernike_radial(rho, l: int, m: int):
    """Evaluate the standard radial Zernike polynomial R_l^m on [0, 1]."""
    rho = np.asarray(rho, dtype=float)
    if l < m or (l - m) % 2:
        return np.zeros_like(rho)
    result = np.zeros_like(rho)
    half_plus = (l + m) // 2
    half_minus = (l - m) // 2
    for k in range(half_minus + 1):
        coeff = ((-1) ** k) * math.factorial(l - k)
        coeff /= (
            math.factorial(k)
            * math.factorial(half_plus - k)
            * math.factorial(half_minus - k)
        )
        result += coeff * rho ** (l - 2 * k)
    return result


def _level_radii(
    model: PsiModel,
    level: float,
    theta: np.ndarray,
    phi: float,
    *,
    max_radius_scale: float,
    maxiter: int = 30,
    tol: float = 1e-12,
) -> np.ndarray:
    from .surface import _quadratic_radius

    max_radius = float(max_radius_scale) * model.a
    phi_values = np.full_like(theta, float(phi))
    radius = _quadratic_radius(model, level, theta, phi_values, max_radius)
    for _ in range(maxiter):
        value, derivative = psi_ray_value_and_derivative(model, radius, theta, phi_values)
        residual = value - level
        if float(np.max(np.abs(residual))) <= tol:
            break
        floor = np.where(derivative >= 0.0, 1e-14, -1e-14)
        denom = np.where(np.abs(derivative) > 1e-14, derivative, floor)
        step = np.clip(
            residual / denom,
            -0.4 * np.maximum(radius, 1e-10),
            0.4 * np.maximum(radius, 1e-10),
        )
        radius = np.clip(radius - step, 1e-12 * model.a, max_radius)
    return radius


def calibrate_toroidal_flux(
    model: PsiModel,
    levels,
    b_sampler: Callable,
    *,
    phi_count: int = 8,
    theta_count: int = 256,
    radial_quadrature: int = 24,
    polynomial_degree: int = 4,
    max_radius_scale: float = 1.0,
) -> FluxCalibration:
    """Calibrate the fitted invariant against signed toroidal flux / (2*pi)."""
    t0 = time.perf_counter()
    levels = np.asarray(levels, dtype=float)
    if np.any(levels <= 0.0) or np.any(np.diff(levels) <= 0.0):
        raise ValueError("flux calibration levels must be positive and increasing")
    phis = np.linspace(0.0, TWOPI / model.nfp, int(phi_count), endpoint=False)
    theta = (np.arange(int(theta_count)) + 0.5) * TWOPI / int(theta_count)
    gauss_x, gauss_w = np.polynomial.legendre.leggauss(int(radial_quadrature))
    psi_sections = np.empty((len(phis), len(levels)), dtype=float)

    for iphi, phi in enumerate(phis):
        ra, za, _, _ = model.axis_at(np.asarray([phi]))
        for ilevel, level in enumerate(levels):
            boundary_radius = _level_radii(
                model,
                float(level),
                theta,
                float(phi),
                max_radius_scale=max_radius_scale,
            )
            radius = 0.5 * boundary_radius[:, None] * (gauss_x[None, :] + 1.0)
            radial_weight = 0.5 * boundary_radius[:, None] * gauss_w[None, :]
            rr = float(ra[0]) + radius * np.cos(theta[:, None])
            zz = float(za[0]) + radius * np.sin(theta[:, None])
            pp = np.full(rr.size, float(phi))
            _, bphi, _ = b_sampler(rr.ravel(), zz.ravel(), pp)
            integrand = np.asarray(bphi).reshape(rr.shape) * radius
            flux = (TWOPI / len(theta)) * np.sum(integrand * radial_weight)
            psi_sections[iphi, ilevel] = flux / TWOPI

    psi_mean = np.mean(psi_sections, axis=0)
    design = np.column_stack([levels**k for k in range(1, int(polynomial_degree) + 1)])
    coeffs, *_ = np.linalg.lstsq(design, psi_mean, rcond=None)
    fitted = design @ coeffs
    derivative_grid = np.column_stack(
        [k * levels ** (k - 1) for k in range(1, int(polynomial_degree) + 1)]
    ) @ coeffs
    section_std = np.std(psi_sections, axis=0)
    diagnostics = {
        "time_s": float(time.perf_counter() - t0),
        "edge_s": float(levels[-1]),
        "edge_psi_toroidal_per_radian": float(fitted[-1]),
        "edge_flux_weber": float(TWOPI * fitted[-1]),
        "fit_relative_rms": float(
            np.sqrt(np.mean((fitted - psi_mean) ** 2)) / max(abs(psi_mean[-1]), 1e-30)
        ),
        "section_relative_std_max": float(
            np.max(section_std / np.maximum(np.abs(psi_mean), 1e-30))
        ),
        "derivative_min": float(np.min(derivative_grid)),
        "derivative_max": float(np.max(derivative_grid)),
        "monotone": bool(np.all(derivative_grid * np.sign(fitted[-1]) > 0.0)),
    }
    return FluxCalibration(
        s_knots=levels,
        psi_knots=psi_mean,
        psi_by_section=psi_sections,
        phi_sections=phis,
        polynomial_coeffs=np.asarray(coeffs),
        polynomial_degree=int(polynomial_degree),
        diagnostics=diagnostics,
    )


def sample_uniform_volume(
    model: PsiModel,
    s_edge: float,
    *,
    count: int,
    grid_xy: int,
    grid_phi: int,
    rho_min: float = 0.06,
    offset_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select a deterministic, shifted Cartesian lattice inside an s level."""
    spacing = 2.0 * model.a / int(grid_xy)
    shift_x = ((0.5 + 0.271828 * offset_seed) % 1.0) * spacing
    shift_z = ((0.5 + 0.618034 * offset_seed) % 1.0) * spacing
    x = -model.a + shift_x + np.arange(int(grid_xy)) * spacing
    z = -model.a + shift_z + np.arange(int(grid_xy)) * spacing
    xx, zz = np.meshgrid(x, z, indexing="ij")
    xx = xx.ravel()
    zz = zz.ravel()
    phi_offset = ((0.5 + 0.414214 * offset_seed) % 1.0) / int(grid_phi)
    phis = (np.arange(int(grid_phi)) / int(grid_phi) + phi_offset) * TWOPI / model.nfp
    kept: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    lower = float(s_edge) * float(rho_min) ** 2
    for phi in phis:
        p = np.full_like(xx, phi)
        ra, za, _, _ = model.axis_at(p)
        rr = ra + xx
        zphys = za + zz
        s, _, _, _ = psi_and_gradient(model, rr, zphys, p)
        mask = (s >= lower) & (s <= float(s_edge))
        if np.any(mask):
            kept.append((rr[mask], zphys[mask], p[mask], s[mask]))
    if not kept:
        raise RuntimeError("uniform volume grid contains no points inside the requested s level")
    arrays = [np.concatenate([item[i] for item in kept]) for i in range(4)]
    available = len(arrays[0])
    if available < int(count):
        raise RuntimeError(f"uniform volume grid produced {available} points, fewer than requested {count}")
    if available > int(count):
        # Evenly span the accepted lattice instead of concentrating a prefix in phi.
        indices = np.floor(np.linspace(0, available, int(count), endpoint=False)).astype(int)
        arrays = [array[indices] for array in arrays]
    return tuple(np.asarray(array) for array in arrays)  # type: ignore[return-value]


def disjoint_train_validation_indices(
    point_count: int, train_count: int, validation_count: int
) -> tuple[np.ndarray, np.ndarray]:
    requested = int(train_count) + int(validation_count)
    if requested > int(point_count):
        raise ValueError("train and validation counts exceed sampled point count")
    validation = np.floor(
        np.linspace(0, point_count, validation_count, endpoint=False)
    ).astype(int)
    keep = np.ones(point_count, dtype=bool)
    keep[validation] = False
    training_pool = np.flatnonzero(keep)
    if len(training_pool) > train_count:
        choose = np.floor(
            np.linspace(0, len(training_pool), train_count, endpoint=False)
        ).astype(int)
        training_pool = training_pool[choose]
    return training_pool, validation


def alpha_coordinates_from_volume_points(points: dict) -> dict[str, np.ndarray]:
    grad_psi = np.asarray(points["grad_psi"])
    return {
        "rho": np.asarray(points["rho"]),
        "theta": np.asarray(points["theta"]),
        "phi": np.asarray(points["phi"]),
        "grad_psi": grad_psi,
        "cross_theta": np.cross(grad_psi, points["grad_theta"]),
        "cross_phi": np.cross(grad_psi, points["grad_phi"]),
    }


def physical_coordinate_data(
    model: PsiModel,
    calibration: FluxCalibration,
    R,
    Z,
    phi,
    s=None,
) -> dict[str, np.ndarray]:
    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=float)
    phi = np.asarray(phi, dtype=float)
    if s is None:
        s, grad_R, grad_Z, grad_phi = psi_and_gradient(model, R, Z, phi)
    else:
        s = np.asarray(s, dtype=float)
        _, grad_R, grad_Z, grad_phi = psi_and_gradient(model, R, Z, phi)
    dpsi_ds = calibration.derivative(s)
    grad_psi = np.column_stack(
        [dpsi_ds * grad_R, dpsi_ds * grad_phi / R, dpsi_ds * grad_Z]
    )
    psi = calibration.evaluate(s)
    rho = np.sqrt(np.clip(psi / calibration.psi_edge, 0.0, None))

    ra, za, rap, zap = model.axis_at(phi)
    x = R - ra
    z = Z - za
    radius2 = np.maximum(x * x + z * z, 1e-30)
    theta = -np.arctan2(z, x)
    # theta is clockwise so that grad(psi_toroidal) x grad(theta) has the
    # physical toroidal-field orientation for psi_toroidal = flux / (2*pi).
    theta_R = z / radius2
    theta_Z = -x / radius2
    theta_phi = (x * zap - z * rap) / radius2
    grad_theta = np.column_stack([theta_R, theta_phi / R, theta_Z])
    grad_phi_coord = np.column_stack([np.zeros_like(R), 1.0 / R, np.zeros_like(R)])
    cross_theta = np.cross(grad_psi, grad_theta)
    cross_phi = np.cross(grad_psi, grad_phi_coord)
    return {
        "s": s,
        "psi": psi,
        "rho": rho,
        "theta": theta,
        "phi": phi,
        "grad_psi": grad_psi,
        "grad_theta": grad_theta,
        "grad_phi": grad_phi_coord,
        "cross_theta": cross_theta,
        "cross_phi": cross_phi,
    }


def _basis_fields_numpy(
    modes: list[ClebschMode],
    rho: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    nfp: int,
    coeffs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = np.zeros_like(rho)
    derivative_theta = np.zeros_like(rho)
    derivative_phi = np.zeros_like(rho)
    if coeffs is None:
        coeffs = np.ones(len(modes))
    radial_cache: dict[tuple[int, int], np.ndarray] = {}
    phase_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for coeff, mode in zip(coeffs, modes):
        if coeff == 0.0:
            continue
        radial = radial_cache.setdefault((mode.l, mode.m), zernike_radial(rho, mode.l, mode.m))
        cosine, sine = phase_cache.setdefault(
            (mode.m, mode.n),
            (
                np.cos(mode.m * theta - mode.n * nfp * phi),
                np.sin(mode.m * theta - mode.n * nfp * phi),
            ),
        )
        if mode.kind == "cos":
            trig = cosine
            dtheta = -mode.m * sine
            dphi = mode.n * nfp * sine
        else:
            trig = sine
            dtheta = mode.m * cosine
            dphi = -mode.n * nfp * cosine
        value += coeff * radial * trig
        derivative_theta += coeff * radial * dtheta
        derivative_phi += coeff * radial * dphi
    return value, derivative_theta, derivative_phi


def evaluate_lambda(
    result: AlphaFitResult,
    rho,
    theta,
    phi,
    nfp: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _basis_fields_numpy(
        result.modes,
        np.asarray(rho, dtype=float),
        np.asarray(theta, dtype=float),
        np.asarray(phi, dtype=float),
        nfp,
        result.lambda_coeffs,
    )


def evaluate_alpha_fit(
    result: AlphaFitResult,
    coordinates: dict[str, np.ndarray],
    B: np.ndarray,
    nfp: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    rho = coordinates["rho"]
    lam, lambda_theta, lambda_phi = _basis_fields_numpy(
        result.modes,
        rho,
        coordinates["theta"],
        coordinates["phi"],
        nfp,
        result.lambda_coeffs,
    )
    iota = result.iota(rho)
    reconstructed = (
        (1.0 + lambda_theta)[:, None] * coordinates["cross_theta"]
        + (lambda_phi - iota)[:, None] * coordinates["cross_phi"]
    )
    B = np.asarray(B, dtype=float)
    residual = reconstructed - B
    bnorm = np.linalg.norm(B, axis=1)
    residual_norm = np.linalg.norm(residual, axis=1)
    grad_psi = coordinates["grad_psi"]
    normal = np.sum(B * grad_psi, axis=1) / np.maximum(
        np.sum(grad_psi * grad_psi, axis=1), 1e-30
    )
    normal_B = normal[:, None] * grad_psi
    tangent_B = B - normal_B
    component_names = ("R", "phi", "Z")
    component_metrics = {}
    for component, name in enumerate(component_names):
        component_metrics[name] = {
            "rms": float(np.sqrt(np.mean(residual[:, component] ** 2))),
            "relative_l2": float(
                np.linalg.norm(residual[:, component])
                / max(np.linalg.norm(B[:, component]), 1e-30)
            ),
        }
    radial_bins = []
    for lower, upper in zip(np.linspace(0.0, 1.0, 6)[:-1], np.linspace(0.0, 1.0, 6)[1:]):
        mask = (rho >= lower) & (rho < upper if upper < 1.0 else rho <= upper)
        if np.any(mask):
            radial_bins.append(
                {
                    "rho_min": float(lower),
                    "rho_max": float(upper),
                    "count": int(np.sum(mask)),
                    "relative_l2": float(
                        np.linalg.norm(residual[mask]) / np.linalg.norm(B[mask])
                    ),
                }
            )
    metrics = {
        "point_count": int(len(B)),
        "relative_l2": float(np.linalg.norm(residual) / np.linalg.norm(B)),
        "relative_point_mean": float(np.mean(residual_norm / np.maximum(bnorm, 1e-30))),
        "relative_point_p95": float(np.percentile(residual_norm / np.maximum(bnorm, 1e-30), 95)),
        "relative_point_max": float(np.max(residual_norm / np.maximum(bnorm, 1e-30))),
        "normal_floor_relative_l2": float(np.linalg.norm(normal_B) / np.linalg.norm(B)),
        "tangent_relative_l2": float(np.linalg.norm(residual + normal_B) / np.linalg.norm(tangent_B)),
        "lambda_rms": float(np.sqrt(np.mean(lam * lam))),
        "lambda_max_abs": float(np.max(np.abs(lam))),
        "lambda_theta_min": float(np.min(lambda_theta)),
        "one_plus_lambda_theta_min": float(np.min(1.0 + lambda_theta)),
        "one_plus_lambda_theta_p01": float(np.percentile(1.0 + lambda_theta, 1)),
        "one_plus_lambda_theta_p05": float(np.percentile(1.0 + lambda_theta, 5)),
        "noninvertible_fraction": float(np.mean(1.0 + lambda_theta <= 0.0)),
        "iota_min": float(np.min(iota)),
        "iota_max": float(np.max(iota)),
        "components": component_metrics,
        "radial_bins": radial_bins,
    }
    fields = {
        "lambda": lam,
        "lambda_theta": lambda_theta,
        "lambda_phi": lambda_phi,
        "iota": iota,
        "B_reconstructed": reconstructed,
        "B_residual": residual,
    }
    return metrics, fields


def fit_alpha_gpu_qr(
    coordinates: dict[str, np.ndarray],
    B: np.ndarray,
    *,
    nfp: int,
    radial_order: int,
    poloidal_order: int,
    toroidal_order: int,
    iota_degree: int = 3,
    relative_weighting: bool = False,
    device: str = "cuda",
    precision: str = "fp32",
) -> AlphaFitResult:
    """Solve the unconstrained dense Clebsch fit with a GPU QR factorization."""
    import torch

    t0 = time.perf_counter()
    if precision not in {"fp32", "fp64"}:
        raise ValueError("precision must be 'fp32' or 'fp64'")
    modes = build_clebsch_modes(radial_order, poloidal_order, toroidal_order)
    B = np.asarray(B, dtype=float)
    count = len(B)
    dtype = torch.float32 if precision == "fp32" else torch.float64
    rho = torch.as_tensor(coordinates["rho"], dtype=dtype, device=device)
    theta = torch.as_tensor(coordinates["theta"], dtype=dtype, device=device)
    phi = torch.as_tensor(coordinates["phi"], dtype=dtype, device=device)
    cross_theta = torch.as_tensor(coordinates["cross_theta"], dtype=dtype, device=device)
    cross_phi = torch.as_tensor(coordinates["cross_phi"], dtype=dtype, device=device)
    bfield = torch.as_tensor(B, dtype=dtype, device=device)
    rhs_matrix = bfield - cross_theta
    if relative_weighting:
        weights = torch.rsqrt(torch.sum(bfield * bfield, dim=1).clamp_min(1e-30))
    else:
        weights = torch.ones(count, dtype=dtype, device=device)
    rhs = (rhs_matrix * weights[:, None]).reshape(-1, 1)
    columns = len(modes) + int(iota_degree) + 1
    design = torch.empty((3 * count, columns), dtype=dtype, device=device)

    radial_cache: dict[tuple[int, int], torch.Tensor] = {}
    phase_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    for column, mode in enumerate(modes):
        key = (mode.l, mode.m)
        if key not in radial_cache:
            radial_cache[key] = torch.as_tensor(
                zernike_radial(coordinates["rho"], mode.l, mode.m),
                dtype=dtype,
                device=device,
            )
        phase_key = (mode.m, mode.n)
        if phase_key not in phase_cache:
            argument = mode.m * theta - mode.n * nfp * phi
            phase_cache[phase_key] = (torch.cos(argument), torch.sin(argument))
        cosine, sine = phase_cache[phase_key]
        radial = radial_cache[key]
        if mode.kind == "cos":
            derivative_theta = -mode.m * radial * sine
            derivative_phi = mode.n * nfp * radial * sine
        else:
            derivative_theta = mode.m * radial * cosine
            derivative_phi = -mode.n * nfp * radial * cosine
        vector = derivative_theta[:, None] * cross_theta + derivative_phi[:, None] * cross_phi
        design[:, column] = (vector * weights[:, None]).reshape(-1)

    u = rho * rho
    for power in range(int(iota_degree) + 1):
        vector = -(u**power)[:, None] * cross_phi
        design[:, len(modes) + power] = (vector * weights[:, None]).reshape(-1)

    assemble_s = time.perf_counter() - t0
    column_scale = torch.linalg.vector_norm(design, dim=0).clamp_min(1e-30)
    design /= column_scale[None, :]
    torch.cuda.synchronize()
    solve_start = time.perf_counter()
    solution_scaled = torch.linalg.lstsq(design, rhs, driver="gels").solution[:, 0]
    torch.cuda.synchronize()
    solve_s = time.perf_counter() - solve_start
    weighted_residual = design @ solution_scaled - rhs[:, 0]
    coefficients = (solution_scaled / column_scale).detach().cpu().numpy()
    diagnostics = {
        "point_count": int(count),
        "row_count": int(3 * count),
        "column_count": int(columns),
        "assemble_s": float(assemble_s),
        "solve_s": float(solve_s),
        "total_s": float(time.perf_counter() - t0),
        "solver": "torch.linalg.lstsq(gels)",
        "precision": precision,
        "relative_weighting": bool(relative_weighting),
        "weighted_residual_rms": float(torch.sqrt(torch.mean(weighted_residual**2)).item()),
        "column_scale_min": float(torch.min(column_scale).item()),
        "column_scale_max": float(torch.max(column_scale).item()),
    }
    del design, rhs, weighted_residual, solution_scaled
    torch.cuda.empty_cache()
    return AlphaFitResult(
        modes=modes,
        lambda_coeffs=coefficients[: len(modes)],
        iota_coeffs=coefficients[len(modes) :],
        radial_order=int(radial_order),
        poloidal_order=int(poloidal_order),
        toroidal_order=int(toroidal_order),
        iota_degree=int(iota_degree),
        diagnostics=diagnostics,
    )


def pchip_flux_diagnostic(calibration: FluxCalibration, points: int = 400) -> dict:
    """Compare the polynomial calibration to a shape-preserving interpolation."""
    interpolator = PchipInterpolator(
        np.r_[0.0, calibration.s_knots],
        np.r_[0.0, calibration.psi_knots],
    )
    s = np.linspace(0.0, calibration.s_knots[-1], int(points))
    pchip = interpolator(s)
    polynomial = calibration.evaluate(s)
    return {
        "relative_l2": float(np.linalg.norm(polynomial - pchip) / np.linalg.norm(pchip)),
        "max_abs": float(np.max(np.abs(polynomial - pchip))),
    }
