from __future__ import annotations

from typing import Callable

import numpy as np


TWOPI = 2.0 * np.pi


def magnetic_axis_quadrature(model, sample_count: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return one-period axis points and d(x, y, z)/dphi tangents."""

    count = int(sample_count or len(model.phi_axis))
    if count < 16:
        raise ValueError("magnetic-axis circulation requires at least 16 samples")
    phi = np.linspace(0.0, TWOPI / int(model.nfp), count, endpoint=False)
    radius, vertical, radius_phi, vertical_phi = model.axis_at(phi)
    cosine = np.cos(phi)
    sine = np.sin(phi)
    points = np.column_stack((radius * cosine, radius * sine, vertical))
    tangents = np.column_stack(
        (
            radius_phi * cosine - radius * sine,
            radius_phi * sine + radius * cosine,
            vertical_phi,
        )
    )
    return points, tangents


def magnetic_axis_circulation(
    model,
    evaluate_B: Callable[[np.ndarray], np.ndarray],
    *,
    sample_count: int | None = None,
) -> float:
    """Compute the full-device Ampere circulation along the periodic axis."""

    points, tangents = magnetic_axis_quadrature(model, sample_count)
    field = np.asarray(evaluate_B(points), dtype=float)
    if field.shape != points.shape:
        raise ValueError(f"B must have shape {points.shape}, got {field.shape}")
    integrand = np.einsum("ij,ij->i", field, tangents)
    circulation = TWOPI * float(np.mean(integrand))
    if not np.isfinite(circulation):
        raise ValueError("magnetic-axis circulation is not finite")
    return circulation


def simsopt_axis_circulation(field, model, *, sample_count: int | None = None) -> float:
    def evaluate(points: np.ndarray) -> np.ndarray:
        field.set_points(points)
        return field.B()

    return magnetic_axis_circulation(model, evaluate, sample_count=sample_count)


def vacuum_G_from_circulation(circulation: float, toroidal_flux: float) -> float:
    """Return radian-coordinate vacuum G using the measured Ampere circulation."""

    circulation = float(circulation)
    toroidal_flux = float(toroidal_flux)
    if not np.isfinite(circulation):
        raise ValueError("vacuum G requires finite magnetic-axis circulation")
    if not np.isfinite(toroidal_flux) or toroidal_flux == 0.0:
        raise ValueError("vacuum G requires nonzero signed toroidal flux")
    return float(np.copysign(abs(circulation) / TWOPI, toroidal_flux))
