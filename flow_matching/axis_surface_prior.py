from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


CURVE_ORDER = 16
COEFF_COUNT = 2 * CURVE_ORDER + 1
TOKEN_DIM = 3 * COEFF_COUNT + 1
MU0 = 4.0e-7 * math.pi


@dataclass(frozen=True)
class PriorSample:
    tokens: np.ndarray
    reference_axis: np.ndarray
    metadata: dict[str, Any]


_FAMILY_PARAMETERS = {
    "near_circular": {
        "axis_scale": 0.38,
        "surface_corrugation": 0.035,
        "ellipticity": 0.055,
        "warp_fraction": 0.055,
    },
    "balanced": {
        "axis_scale": 0.70,
        "surface_corrugation": 0.075,
        "ellipticity": 0.105,
        "warp_fraction": 0.125,
    },
    "helical": {
        "axis_scale": 1.00,
        "surface_corrugation": 0.115,
        "ellipticity": 0.145,
        "warp_fraction": 0.205,
    },
}


def supported_conditions() -> tuple[tuple[int, int], ...]:
    """Conditions used by the prior study; nc=5 is deliberately absent."""
    return tuple([(nfp, 1) for nfp in range(4, 9)] + [(nfp, nc) for nc in range(2, 5) for nfp in range(2, 9)])


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-12:
        raise ValueError("cannot normalize a nearly zero vector")
    return vector / norm


def _rotate(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _unit(axis)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        vector * cosine
        + np.cross(axis, vector) * sine
        + axis * float(np.dot(axis, vector)) * (1.0 - cosine)
    )


def _signed_angle(first: np.ndarray, second: np.ndarray, axis: np.ndarray) -> float:
    first = _unit(first - axis * float(np.dot(first, axis)))
    second = _unit(second - axis * float(np.dot(second, axis)))
    return math.atan2(float(np.dot(axis, np.cross(first, second))), float(np.dot(first, second)))


def _axis_and_frame(
    *,
    nfp: int,
    radial_coefficients: np.ndarray,
    vertical_coefficients: np.ndarray,
    samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phi = 2.0 * math.pi * np.arange(samples, dtype=float) / samples
    radius = np.ones(samples, dtype=float)
    height = np.zeros(samples, dtype=float)
    for mode, (radial, vertical) in enumerate(
        zip(radial_coefficients, vertical_coefficients, strict=True), start=1
    ):
        argument = mode * nfp * phi
        radius += radial * np.cos(argument)
        height += vertical * np.sin(argument)
    axis = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), height))
    tangent = np.roll(axis, -1, axis=0) - np.roll(axis, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)

    normal = np.empty_like(axis)
    radial_direction = np.asarray([math.cos(phi[0]), math.sin(phi[0]), 0.0])
    normal[0] = _unit(radial_direction - tangent[0] * float(np.dot(radial_direction, tangent[0])))
    for index in range(samples - 1):
        cross = np.cross(tangent[index], tangent[index + 1])
        sine = float(np.linalg.norm(cross))
        cosine = float(np.clip(np.dot(tangent[index], tangent[index + 1]), -1.0, 1.0))
        if sine < 1.0e-12:
            transported = normal[index]
        else:
            transported = _rotate(normal[index], cross / sine, math.atan2(sine, cosine))
        normal[index + 1] = _unit(
            transported - tangent[index + 1] * float(np.dot(transported, tangent[index + 1]))
        )

    cross = np.cross(tangent[-1], tangent[0])
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(tangent[-1], tangent[0]), -1.0, 1.0))
    closure = normal[-1] if sine < 1.0e-12 else _rotate(normal[-1], cross / sine, math.atan2(sine, cosine))
    mismatch = _signed_angle(closure, normal[0], tangent[0])
    for index in range(samples):
        normal[index] = _rotate(normal[index], tangent[index], mismatch * index / samples)
        normal[index] = _unit(normal[index] - tangent[index] * float(np.dot(normal[index], tangent[index])))
    binormal = np.cross(tangent, normal)
    binormal /= np.linalg.norm(binormal, axis=1, keepdims=True)
    return phi, axis, normal, binormal


def _periodic_interp(values: np.ndarray, phi: np.ndarray) -> np.ndarray:
    samples = len(values)
    coordinate = np.mod(phi, 2.0 * math.pi) * samples / (2.0 * math.pi)
    lower = np.floor(coordinate).astype(int)
    alpha = (coordinate - lower)[:, None]
    return values[lower % samples] * (1.0 - alpha) + values[(lower + 1) % samples] * alpha


def _fit_fourier(points: np.ndarray, order: int = CURVE_ORDER) -> tuple[np.ndarray, float, float]:
    count = len(points)
    theta = 2.0 * math.pi * np.arange(count, dtype=float) / count
    design = [np.ones(count, dtype=float)]
    for mode in range(1, order + 1):
        design.extend((np.sin(mode * theta), np.cos(mode * theta)))
    matrix = np.column_stack(design)
    coefficients = np.linalg.lstsq(matrix, points, rcond=None)[0].T
    fitted = matrix @ coefficients.T
    residual = np.linalg.norm(fitted - points, axis=1)
    return coefficients, float(np.sqrt(np.mean(residual**2))), float(np.max(residual))


def evaluate_fourier(tokens: np.ndarray, samples: int = 256) -> np.ndarray:
    values = np.atleast_2d(np.asarray(tokens, dtype=float))
    if values.shape[1] != TOKEN_DIM:
        raise ValueError(f"tokens must have {TOKEN_DIM} columns")
    theta = 2.0 * math.pi * np.arange(samples, dtype=float) / samples
    design = [np.ones(samples, dtype=float)]
    for mode in range(1, CURVE_ORDER + 1):
        design.extend((np.sin(mode * theta), np.cos(mode * theta)))
    matrix = np.column_stack(design)
    curves = []
    for token in values:
        coefficients = token[: 3 * COEFF_COUNT].reshape(3, COEFF_COUNT)
        curves.append(matrix @ coefficients.T)
    return np.asarray(curves)


def gauss_linking_number(first: np.ndarray, second: np.ndarray) -> float:
    """Midpoint quadrature of the Gauss linking integral for two closed polygons."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first_next = np.roll(first, -1, axis=0)
    second_next = np.roll(second, -1, axis=0)
    first_delta = first_next - first
    second_delta = second_next - second
    separation = (first + first_next)[:, None, :] * 0.5 - (second + second_next)[None, :, :] * 0.5
    denominator = np.linalg.norm(separation, axis=2) ** 3
    numerator = np.einsum(
        "ijk,ijk->ij",
        np.cross(first_delta[:, None, :], second_delta[None, :, :]),
        separation,
    )
    return float(np.sum(numerator / np.maximum(denominator, 1.0e-20)) / (4.0 * math.pi))


def _solve_contour(
    *,
    theta: np.ndarray,
    level: float,
    nfp: int,
    terms: list[tuple[int, int, float, float]],
) -> np.ndarray:
    phi = np.full_like(theta, level)
    for _ in range(10):
        value = phi - level
        derivative = np.ones_like(phi)
        for poloidal_mode, field_mode, amplitude, phase in terms:
            argument = poloidal_mode * theta - field_mode * nfp * phi + phase
            value += amplitude * np.sin(argument)
            derivative -= amplitude * field_mode * nfp * np.cos(argument)
        update = value / derivative
        phi -= update
        if float(np.max(np.abs(update))) < 1.0e-13:
            break
    return phi


def _axis_coefficients(rng: np.random.Generator, nfp: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    modes = np.arange(1, 4, dtype=float)
    envelope = scale * 0.42 / (nfp * modes) ** 2
    radial = rng.normal(0.0, envelope)
    vertical = rng.normal(0.0, envelope)
    # A coherent first harmonic is retained so the prior is not a collection of circular axes.
    coherent = scale * rng.uniform(0.12, 0.30) / nfp**2
    radial[0] += rng.choice((-1.0, 1.0)) * coherent
    vertical[0] += rng.choice((-1.0, 1.0)) * coherent * rng.uniform(0.65, 1.0)
    return radial, vertical


def sample_axis_surface_prior(
    *,
    seed: int,
    nfp: int,
    n_base_coils: int,
    family: str = "balanced",
    curve_samples: int = 512,
    frame_samples: int = 2048,
    target_field_t: float = 1.0,
) -> PriorSample:
    """Draw simple coils as contours on a random tube around a random reference axis.

    The distribution is analytic and does not load or estimate any quantity from QUASR.
    The returned reference axis is a construction aid, not a claimed magnetic axis.
    """
    if (int(nfp), int(n_base_coils)) not in supported_conditions():
        raise ValueError("unsupported (nfp, n_base_coils); this prior requires nc<=4")
    if family not in _FAMILY_PARAMETERS:
        raise ValueError(f"unknown family {family!r}")
    if curve_samples < 64 or frame_samples < 4 * curve_samples:
        raise ValueError("curve_samples must be >=64 and frame_samples >=4*curve_samples")
    parameters = _FAMILY_PARAMETERS[family]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(nfp), int(n_base_coils)]))
    radial, vertical = _axis_coefficients(rng, nfp, parameters["axis_scale"])
    frame_phi, axis, normal, binormal = _axis_and_frame(
        nfp=nfp,
        radial_coefficients=radial,
        vertical_coefficients=vertical,
        samples=frame_samples,
    )

    physical_coil_count = 2 * nfp * n_base_coils
    angular_spacing = 2.0 * math.pi / physical_coil_count
    spacing_limited_radius = 1.0 - 0.092 / max(2.0 * math.sin(0.5 * angular_spacing), 1.0e-6)
    minor_radius_high = min(0.40, max(0.18, spacing_limited_radius))
    minor_radius_low = min(0.25, max(0.145, 0.72 * minor_radius_high))
    minor_radius = rng.uniform(minor_radius_low, minor_radius_high)
    ellipticity = rng.uniform(-parameters["ellipticity"], parameters["ellipticity"])
    corrugation = rng.uniform(0.25, 1.0) * parameters["surface_corrugation"]
    corrugation_phase = rng.uniform(0.0, 2.0 * math.pi)

    half_period = math.pi / nfp
    level_spacing = half_period / n_base_coils
    global_shift = rng.uniform(-0.12, 0.12) * level_spacing
    levels = (np.arange(n_base_coils, dtype=float) + 0.5) * level_spacing + global_shift
    raw_terms: list[tuple[int, int, float, float]] = []
    for poloidal_mode, field_mode in ((1, 1), (2, 1), (1, 2), (3, 1)):
        weight = 1.0 / (poloidal_mode**1.5 * field_mode)
        raw_terms.append((poloidal_mode, field_mode, rng.normal() * weight, rng.uniform(0.0, 2.0 * math.pi)))
    raw_l1 = sum(abs(term[2]) for term in raw_terms)
    warp_budget = parameters["warp_fraction"] * level_spacing
    derivative_budget = 0.48 / max(nfp, 1)
    scale = min(warp_budget / max(raw_l1, 1.0e-12), derivative_budget / max(sum(abs(t[2]) * t[1] for t in raw_terms), 1.0e-12))
    terms = [(m, n, amplitude * scale, phase) for m, n, amplitude, phase in raw_terms]

    theta = 2.0 * math.pi * np.arange(curve_samples, dtype=float) / curve_samples
    coefficient_blocks = []
    fit_rms = []
    fit_max = []
    contour_ranges = []
    raw_contours = []
    for level in levels:
        phi = _solve_contour(theta=theta, level=float(level), nfp=nfp, terms=terms)
        axis_at_phi = _periodic_interp(axis, phi)
        normal_at_phi = _periodic_interp(normal, phi)
        binormal_at_phi = _periodic_interp(binormal, phi)
        phase = theta - nfp * phi + corrugation_phase
        radius_modulation = 1.0 + corrugation * np.cos(phase) + 0.35 * corrugation * np.cos(2.0 * theta + corrugation_phase)
        normal_radius = minor_radius * (1.0 + ellipticity) * radius_modulation
        binormal_radius = minor_radius * (1.0 - ellipticity) * radius_modulation
        points = (
            axis_at_phi
            + normal_radius[:, None] * np.cos(theta)[:, None] * normal_at_phi
            + binormal_radius[:, None] * np.sin(theta)[:, None] * binormal_at_phi
        )
        coefficients, rms, maximum = _fit_fourier(points)
        coefficient_blocks.append(coefficients)
        fit_rms.append(rms)
        fit_max.append(maximum)
        contour_ranges.append([float(np.min(phi)), float(np.max(phi))])
        raw_contours.append(points)

    axis_for_link = axis[:: max(1, frame_samples // 128)][:128]
    link = gauss_linking_number(np.asarray(raw_contours[0])[:: max(1, curve_samples // 128)][:128], axis_for_link)
    if abs(link) < 0.75:
        raise RuntimeError(f"generated contour lost its reference-axis linking: {link:.6g}")
    total_linked_current_a = 2.0 * math.pi * 1.0 * float(target_field_t) / MU0
    current_a = math.copysign(total_linked_current_a / physical_coil_count, link)
    tokens = np.empty((n_base_coils, TOKEN_DIM), dtype=np.float64)
    for index, coefficients in enumerate(coefficient_blocks):
        tokens[index, : 3 * COEFF_COUNT] = coefficients.reshape(-1)
        tokens[index, -1] = current_a

    metadata: dict[str, Any] = {
        "format": "axis_surface_contour_prior_v1",
        "seed": int(seed),
        "nfp": int(nfp),
        "n_base_coils": int(n_base_coils),
        "family": family,
        "independent_of_quasr": True,
        "reference_axis_role": "construction_reference_only",
        "axis_radial_coefficients": radial.tolist(),
        "axis_vertical_coefficients": vertical.tolist(),
        "minor_radius": float(minor_radius),
        "ellipticity": float(ellipticity),
        "surface_corrugation": float(corrugation),
        "contour_levels": levels.tolist(),
        "contour_phi_ranges": contour_ranges,
        "scalar_terms": [
            {"poloidal_mode": m, "field_period_mode": n, "amplitude_rad": amplitude, "phase_rad": phase}
            for m, n, amplitude, phase in terms
        ],
        "scalar_monotonicity_lower_bound": float(1.0 - sum(abs(amplitude) * n * nfp for _, n, amplitude, _ in terms)),
        "curve_fit_rms_max_m": float(max(fit_rms)),
        "curve_fit_abs_max_m": float(max(fit_max)),
        "reference_linking_number": float(link),
        "target_field_t": float(target_field_t),
        "total_linked_current_a": float(total_linked_current_a),
        "base_current_a": float(current_a),
    }
    return PriorSample(tokens=tokens, reference_axis=axis, metadata=metadata)
