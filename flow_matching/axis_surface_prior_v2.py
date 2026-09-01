from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from flow_matching.axis_surface_prior import (
    COEFF_COUNT,
    CURVE_ORDER,
    MU0,
    TOKEN_DIM,
    _axis_and_frame,
    _fit_fourier,
    _periodic_interp,
    _solve_contour,
    gauss_linking_number,
    supported_conditions,
)


@dataclass(frozen=True)
class ShapedPriorPrototype:
    tokens: np.ndarray
    reference_axis: np.ndarray
    inner_reference_surface: np.ndarray
    winding_surface: np.ndarray
    metadata: dict[str, Any]


_PRESETS = {
    "balanced_stellarator": {
        "axis_radial": 0.145,
        "axis_vertical": 0.130,
        "axis_second": 0.026,
        "minor_radius": 0.330,
        "size_variation": 0.14,
        "elongation_variation": 0.28,
        "cross_section_rotation": 0.38,
        "triangularity": 0.085,
        "helical_ripple": 0.075,
        "contour_warp": 0.24,
    },
    "axis_dominant": {
        "axis_radial": 0.205,
        "axis_vertical": 0.180,
        "axis_second": 0.042,
        "minor_radius": 0.315,
        "size_variation": 0.12,
        "elongation_variation": 0.24,
        "cross_section_rotation": 0.34,
        "triangularity": 0.070,
        "helical_ripple": 0.060,
        "contour_warp": 0.22,
    },
    "surface_dominant": {
        "axis_radial": 0.155,
        "axis_vertical": 0.140,
        "axis_second": 0.030,
        "minor_radius": 0.345,
        "size_variation": 0.22,
        "elongation_variation": 0.40,
        "cross_section_rotation": 0.58,
        "triangularity": 0.135,
        "helical_ripple": 0.120,
        "contour_warp": 0.31,
    },
}


def prototype_presets() -> tuple[str, ...]:
    return tuple(_PRESETS)


def _coherent_axis_coefficients(
    rng: np.random.Generator,
    *,
    nfp: int,
    radial_amplitude: float,
    vertical_amplitude: float,
    second_amplitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    # The mild nfp scaling controls curvature without erasing the stellarator bend.
    nfp_scale = float(np.clip((4.0 / nfp) ** 0.72, 0.64, 1.35))
    radial = np.zeros(3, dtype=float)
    vertical = np.zeros(3, dtype=float)
    radial[0] = radial_amplitude * nfp_scale * rng.uniform(0.90, 1.10)
    vertical[0] = vertical_amplitude * nfp_scale * rng.uniform(0.90, 1.10)
    radial[1] = second_amplitude * nfp_scale * rng.uniform(-1.0, 1.0)
    vertical[1] = second_amplitude * nfp_scale * rng.uniform(-1.0, 1.0)
    radial[2] = 0.25 * second_amplitude * nfp_scale * rng.uniform(-1.0, 1.0)
    vertical[2] = 0.25 * second_amplitude * nfp_scale * rng.uniform(-1.0, 1.0)
    return radial, vertical


def _surface_points(
    *,
    axis: np.ndarray,
    normal: np.ndarray,
    binormal: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    nfp: int,
    parameters: dict[str, float],
    scale: float,
) -> np.ndarray:
    shape = np.broadcast_shapes(np.shape(phi), np.shape(theta))
    phi_flat = np.broadcast_to(phi, shape).reshape(-1)
    theta_flat = np.broadcast_to(theta, shape).reshape(-1)
    axis_at_phi = _periodic_interp(axis, phi_flat)
    normal_at_phi = _periodic_interp(normal, phi_flat)
    binormal_at_phi = _periodic_interp(binormal, phi_flat)

    field_angle = nfp * phi_flat
    size = 1.0 + parameters["size_variation"] * np.cos(field_angle)
    elongation = np.exp(parameters["elongation_variation"] * np.sin(field_angle))
    normal_radius = scale * parameters["minor_radius"] * size / np.sqrt(elongation)
    binormal_radius = scale * parameters["minor_radius"] * size * np.sqrt(elongation)
    rotation = parameters["cross_section_rotation"] * np.sin(field_angle)
    poloidal = theta_flat + rotation
    triangle_phase = 2.0 * theta_flat - field_angle
    ripple = 1.0 + parameters["helical_ripple"] * np.cos(theta_flat - field_angle)

    normal_offset = ripple * normal_radius * np.cos(poloidal)
    binormal_offset = ripple * binormal_radius * np.sin(poloidal)
    triangle_scale = scale * parameters["minor_radius"] * parameters["triangularity"]
    normal_offset += triangle_scale * np.cos(triangle_phase)
    binormal_offset += 0.55 * triangle_scale * np.sin(triangle_phase)
    points = (
        axis_at_phi
        + normal_offset[:, None] * normal_at_phi
        + binormal_offset[:, None] * binormal_at_phi
    )
    return points.reshape(*shape, 3)


def sample_shaped_prior_prototype(
    *,
    seed: int,
    nfp: int,
    n_base_coils: int,
    preset: str,
    curve_samples: int = 512,
    frame_samples: int = 2048,
    surface_phi_samples: int = 192,
    surface_theta_samples: int = 96,
    target_field_t: float = 1.0,
) -> ShapedPriorPrototype:
    """Construct a visibly three-dimensional stellarator prior for visual review.

    The inner surface is a geometric reference only. It is not a field-derived
    magnetic surface, and the construction axis is not a verified magnetic axis.
    """
    if (int(nfp), int(n_base_coils)) not in supported_conditions():
        raise ValueError("unsupported (nfp, n_base_coils); prototypes require nc<=4")
    if preset not in _PRESETS:
        raise ValueError(f"unknown preset {preset!r}")
    parameters = dict(_PRESETS[preset])
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(nfp), int(n_base_coils), 2]))
    radial, vertical = _coherent_axis_coefficients(
        rng,
        nfp=nfp,
        radial_amplitude=parameters["axis_radial"],
        vertical_amplitude=parameters["axis_vertical"],
        second_amplitude=parameters["axis_second"],
    )
    _, axis, normal, binormal = _axis_and_frame(
        nfp=nfp,
        radial_coefficients=radial,
        vertical_coefficients=vertical,
        samples=frame_samples,
    )

    surface_phi = 2.0 * math.pi * np.arange(surface_phi_samples, dtype=float) / surface_phi_samples
    surface_theta = 2.0 * math.pi * np.arange(surface_theta_samples, dtype=float) / surface_theta_samples
    phi_grid, theta_grid = np.meshgrid(surface_phi, surface_theta, indexing="ij")
    inner_surface = _surface_points(
        axis=axis,
        normal=normal,
        binormal=binormal,
        phi=phi_grid,
        theta=theta_grid,
        nfp=nfp,
        parameters=parameters,
        scale=0.52,
    )
    winding_surface = _surface_points(
        axis=axis,
        normal=normal,
        binormal=binormal,
        phi=phi_grid,
        theta=theta_grid,
        nfp=nfp,
        parameters=parameters,
        scale=1.0,
    )

    half_period = math.pi / nfp
    level_spacing = half_period / n_base_coils
    levels = (np.arange(n_base_coils, dtype=float) + 0.5) * level_spacing
    phase = rng.uniform(-0.35, 0.35)
    amplitude = parameters["contour_warp"] / nfp
    terms = [
        (1, 1, amplitude, phase),
        (2, 1, 0.30 * amplitude, -0.7 * phase),
    ]
    theta = 2.0 * math.pi * np.arange(curve_samples, dtype=float) / curve_samples
    blocks = []
    fit_rms = []
    fit_max = []
    raw_contours = []
    for level in levels:
        phi = _solve_contour(theta=theta, level=float(level), nfp=nfp, terms=terms)
        points = _surface_points(
            axis=axis,
            normal=normal,
            binormal=binormal,
            phi=phi,
            theta=theta,
            nfp=nfp,
            parameters=parameters,
            scale=1.0,
        )
        coefficients, rms, maximum = _fit_fourier(points)
        blocks.append(coefficients)
        fit_rms.append(rms)
        fit_max.append(maximum)
        raw_contours.append(points)

    axis_for_link = axis[:: max(1, frame_samples // 128)][:128]
    link = gauss_linking_number(raw_contours[0][:: max(1, curve_samples // 128)][:128], axis_for_link)
    if abs(link) < 0.75:
        raise RuntimeError(f"prototype contour lost its reference-axis linking: {link:.6g}")
    physical_count = 2 * nfp * n_base_coils
    total_linked_current_a = 2.0 * math.pi * float(target_field_t) / MU0
    current_a = math.copysign(total_linked_current_a / physical_count, link)
    tokens = np.empty((n_base_coils, TOKEN_DIM), dtype=np.float64)
    for index, coefficients in enumerate(blocks):
        tokens[index, : 3 * COEFF_COUNT] = coefficients.reshape(-1)
        tokens[index, -1] = current_a

    cylindrical_radius = np.linalg.norm(axis[:, :2], axis=1)
    surface_axis = _periodic_interp(axis, surface_phi)
    surface_radius = np.linalg.norm(winding_surface - surface_axis[:, None, :], axis=2).mean(axis=1)
    metadata: dict[str, Any] = {
        "format": "axis_surface_contour_prior_visual_v2",
        "status": "geometry_only_awaiting_user_acceptance",
        "preset": preset,
        "seed": int(seed),
        "nfp": int(nfp),
        "n_base_coils": int(n_base_coils),
        "reference_axis_role": "construction_reference_only",
        "inner_surface_role": "geometric_plasma_like_reference_only",
        "independent_of_quasr_statistics": True,
        "axis_radial_coefficients": radial.tolist(),
        "axis_vertical_coefficients": vertical.tolist(),
        "axis_R_peak_to_peak_m": float(np.ptp(cylindrical_radius)),
        "axis_Z_peak_to_peak_m": float(np.ptp(axis[:, 2])),
        "winding_section_mean_radius_peak_to_peak_m": float(np.ptp(surface_radius)),
        "curve_fit_rms_max_m": float(max(fit_rms)),
        "curve_fit_abs_max_m": float(max(fit_max)),
        "reference_linking_number": float(link),
        "scalar_monotonicity_lower_bound": float(1.0 - sum(abs(term[2]) * term[1] * nfp for term in terms)),
        "parameters": parameters,
    }
    return ShapedPriorPrototype(
        tokens=tokens,
        reference_axis=axis,
        inner_reference_surface=inner_surface,
        winding_surface=winding_surface,
        metadata=metadata,
    )
