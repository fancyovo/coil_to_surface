from __future__ import annotations

import math
from typing import Any

import numpy as np


CURVE_ORDER = 16
COEFF_COUNT = 2 * CURVE_ORDER + 1
TOKEN_DIM = 3 * COEFF_COUNT + 1


def q_down(value: float, scale: float, power: float, fallback: float = 0.0) -> float:
    if not math.isfinite(value) or scale <= 0.0:
        return fallback
    x = max(value, 0.0) / scale
    return 1.0 / (1.0 + x**power)


def q_up(value: float, scale: float, power: float, fallback: float = 0.0) -> float:
    if not math.isfinite(value) or value <= 0.0 or scale <= 0.0:
        return fallback
    return 1.0 / (1.0 + (scale / value) ** power)


def coil_score_decomposition(native: dict[str, Any]) -> dict[str, float]:
    diagnostics = native["diagnostics"]
    quality = {
        "length": q_down(float(diagnostics["coil_length_mean"]), 7.0, 1.4, 0.6),
        "curvature_p95": q_down(
            float(diagnostics["coil_curvature_p95"]), 10.0, 1.3, 0.5
        ),
        "curvature_max": q_down(
            float(diagnostics["coil_curvature_max"]), 35.0, 1.2, 0.5
        ),
        "spacing": q_up(
            float(diagnostics["coil_min_intercoil_distance"]), 0.08, 1.1, 0.45
        ),
        "axis_distance": q_up(
            float(diagnostics["coil_min_axis_distance"]), 0.20, 1.2, 0.45
        ),
        "high_mode": q_down(
            float(diagnostics["coil_high_mode_energy_fraction"]), 0.05, 1.0, 0.7
        ),
        "current": q_down(
            float(diagnostics["coil_current_abs_max_a"]), 2.0e6, 1.0, 0.7
        ),
    }
    weights = {
        "length": 0.16,
        "curvature_p95": 0.20,
        "curvature_max": 0.12,
        "spacing": 0.20,
        "axis_distance": 0.12,
        "high_mode": 0.13,
        "current": 0.07,
    }
    result = {
        f"quality_{name}": value for name, value in quality.items()
    }
    result.update(
        {
            f"points_{name}": 100.0 * weights[name] * value
            for name, value in quality.items()
        }
    )
    result["points_curvature"] = (
        result["points_curvature_p95"] + result["points_curvature_max"]
    )
    result["points_distance"] = (
        result["points_spacing"] + result["points_axis_distance"]
    )
    result["points_other"] = (
        result["points_length"]
        + result["points_high_mode"]
        + result["points_current"]
    )
    result["points_total"] = sum(
        result[f"points_{name}"] for name in weights
    )
    recorded = float(native["components"]["coil"])
    if abs(result["points_total"] - recorded) > 2.0e-8:
        raise ValueError(
            "reconstructed coil component differs from native score: "
            f"{result['points_total']:.12g} vs {recorded:.12g}"
        )
    return result


def tokens_from_raw(raw: dict[str, Any]) -> np.ndarray:
    x = np.asarray(raw["x"], dtype=np.float64)
    y = np.asarray(raw["y"], dtype=np.float64)
    z = np.asarray(raw["z"], dtype=np.float64)
    current = np.asarray(raw["current"], dtype=np.float64)[:, None]
    tokens = np.concatenate((x, y, z, current), axis=1)
    if tokens.ndim != 2 or tokens.shape[1] != TOKEN_DIM:
        raise ValueError(f"expected tokens with shape (nc,{TOKEN_DIM}), got {tokens.shape}")
    return tokens


def fourier_design(samples: int, order: int = CURVE_ORDER) -> np.ndarray:
    theta = 2.0 * math.pi * np.arange(samples, dtype=np.float64) / samples
    columns = [np.ones(samples, dtype=np.float64)]
    for mode in range(1, order + 1):
        columns.extend((np.sin(mode * theta), np.cos(mode * theta)))
    return np.column_stack(columns)


def evaluate_curves(tokens: np.ndarray, samples: int = 1024) -> np.ndarray:
    values = np.atleast_2d(np.asarray(tokens, dtype=np.float64))
    if values.shape[1] != TOKEN_DIM:
        raise ValueError(f"expected token dimension {TOKEN_DIM}")
    design = fourier_design(samples)
    return np.asarray(
        [design @ row[: 3 * COEFF_COUNT].reshape(3, COEFF_COUNT).T for row in values]
    )


def fit_curves(points: np.ndarray, currents: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    points = np.asarray(points, dtype=np.float64)
    currents = np.asarray(currents, dtype=np.float64).reshape(-1)
    if points.ndim != 3 or points.shape[0] != len(currents) or points.shape[2] != 3:
        raise ValueError("points must have shape (nc,samples,3)")
    design = fourier_design(points.shape[1])
    rows = []
    residuals = []
    for curve, current in zip(points, currents, strict=True):
        coefficients = np.linalg.lstsq(design, curve, rcond=None)[0].T
        fitted = design @ coefficients.T
        residual = np.linalg.norm(fitted - curve, axis=1)
        rows.append(np.concatenate((coefficients.reshape(-1), [current])))
        residuals.append(residual)
    all_residuals = np.concatenate(residuals)
    return np.asarray(rows), {
        "fit_rms_m": float(np.sqrt(np.mean(all_residuals**2))),
        "fit_p95_m": float(np.quantile(all_residuals, 0.95)),
        "fit_max_m": float(np.max(all_residuals)),
    }


def load_full_axis_points(axis_data: Any) -> np.ndarray:
    phi = np.asarray(axis_data["phi"], dtype=np.float64)
    radius = np.asarray(axis_data["R"], dtype=np.float64)
    z = np.asarray(axis_data["Z"], dtype=np.float64)
    nfp = int(np.asarray(axis_data["nfp"]).item())
    periods = []
    for period in range(nfp):
        angle = phi + 2.0 * math.pi * period / nfp
        periods.append(
            np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))
        )
    return np.concatenate(periods, axis=0)


def nearest_reference(points: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    flat = points.reshape(-1, 3)
    nearest = np.empty(len(flat), dtype=np.int64)
    distance = np.empty(len(flat), dtype=np.float64)
    for begin in range(0, len(flat), 256):
        block = flat[begin : begin + 256]
        squared = np.sum((block[:, None, :] - reference[None, :, :]) ** 2, axis=2)
        local = np.argmin(squared, axis=1)
        nearest[begin : begin + len(block)] = local
        distance[begin : begin + len(block)] = np.sqrt(
            squared[np.arange(len(block)), local]
        )
    return nearest.reshape(points.shape[:-1]), distance.reshape(points.shape[:-1])


def scale_about_axis(
    tokens: np.ndarray,
    axis_points: np.ndarray,
    scale: float,
    *,
    samples: int = 1024,
) -> tuple[np.ndarray, dict[str, float]]:
    if not 0.0 < scale <= 1.0:
        raise ValueError("scale must lie in (0,1]")
    source_points = evaluate_curves(tokens, samples=samples)
    nearest, source_distance = nearest_reference(source_points, axis_points)
    anchors = axis_points[nearest]
    scaled_points = anchors + scale * (source_points - anchors)
    fitted_tokens, fit = fit_curves(scaled_points, tokens[:, -1])
    fit.update(
        {
            "requested_scale": float(scale),
            "source_axis_distance_mean_m": float(np.mean(source_distance)),
            "target_axis_distance_mean_m": float(scale * np.mean(source_distance)),
        }
    )
    return fitted_tokens, fit


def rotate_z(points: np.ndarray, angle: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    result = points.copy()
    result[..., 0] = cosine * points[..., 0] - sine * points[..., 1]
    result[..., 1] = sine * points[..., 0] + cosine * points[..., 1]
    return result


def full_coil_curves(tokens: np.ndarray, nfp: int, *, samples: int = 512) -> list[np.ndarray]:
    base = evaluate_curves(tokens, samples=samples)
    result = []
    for curve in base:
        for reflected in (False, True):
            symmetric = curve.copy()
            if reflected:
                symmetric[:, 1:] *= -1.0
            for period in range(nfp):
                result.append(rotate_z(symmetric, 2.0 * math.pi * period / nfp))
    return result


def curve_lengths(tokens: np.ndarray, *, samples: int = 4096) -> np.ndarray:
    curves = evaluate_curves(tokens, samples=samples)
    return np.asarray(
        [np.linalg.norm(np.roll(curve, -1, axis=0) - curve, axis=1).sum() for curve in curves]
    )


def effective_radius_m(tokens: np.ndarray) -> float:
    return float(np.mean(curve_lengths(tokens)) / (2.0 * math.pi))

