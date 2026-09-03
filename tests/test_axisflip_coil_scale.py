from __future__ import annotations

import numpy as np
import pytest

from scripts.axisflip_coil_scale import (
    COEFF_COUNT,
    TOKEN_DIM,
    coil_score_decomposition,
    effective_radius_m,
    evaluate_curves,
    scale_about_axis,
    scale_about_coil_axis_anchor,
)


def circle_token(radius: float = 2.0) -> np.ndarray:
    token = np.zeros((1, TOKEN_DIM), dtype=np.float64)
    token[0, 2] = radius
    token[0, COEFF_COUNT + 1] = radius
    token[0, -1] = 1.0e5
    return token


def test_effective_radius_and_axis_scaling() -> None:
    source = circle_token(2.0)
    assert effective_radius_m(source) == pytest.approx(2.0, rel=2.0e-6)
    scaled, diagnostics = scale_about_axis(
        source, np.zeros((1, 3), dtype=np.float64), 0.4, samples=256
    )
    assert diagnostics["fit_rms_m"] < 1.0e-12
    assert effective_radius_m(scaled) == pytest.approx(0.8, rel=2.0e-6)
    radii = np.linalg.norm(evaluate_curves(scaled, samples=128)[0, :, :2], axis=1)
    assert np.max(np.abs(radii - 0.8)) < 1.0e-12
    assert scaled[0, -1] == source[0, -1]


def test_per_coil_axis_anchor_is_an_exact_similarity() -> None:
    source = np.concatenate((circle_token(2.0), circle_token(1.0)), axis=0)
    source[0, 0] = 3.0
    source[1, 0] = -4.0
    axis = np.asarray(((3.0, 0.0, 0.0), (-4.0, 0.0, 0.0)))
    scaled, diagnostics = scale_about_coil_axis_anchor(
        source, axis, 0.25, samples=256
    )
    assert diagnostics["fit_max_m"] < 1.0e-12
    lengths = np.asarray(
        [
            np.linalg.norm(np.roll(curve, -1, axis=0) - curve, axis=1).sum()
            for curve in evaluate_curves(scaled, samples=4096)
        ]
    )
    assert lengths / (2.0 * np.pi) == pytest.approx((0.5, 0.25), rel=2.0e-6)
    assert scaled[:, -1] == pytest.approx(source[:, -1])


def test_coil_score_decomposition_matches_native_weighting() -> None:
    diagnostics = {
        "coil_length_mean": 3.0,
        "coil_curvature_p95": 8.0,
        "coil_curvature_max": 20.0,
        "coil_min_intercoil_distance": 0.04,
        "coil_min_axis_distance": 0.45,
        "coil_high_mode_energy_fraction": 0.01,
        "coil_current_abs_max_a": 1.2e5,
    }
    provisional = {
        "diagnostics": diagnostics,
        "components": {"coil": 0.0},
    }
    provisional["components"]["coil"] = sum(
        (
            16.0 / (1.0 + (3.0 / 7.0) ** 1.4),
            20.0 / (1.0 + (8.0 / 10.0) ** 1.3),
            12.0 / (1.0 + (20.0 / 35.0) ** 1.2),
            20.0 / (1.0 + (0.08 / 0.04) ** 1.1),
            12.0 / (1.0 + (0.20 / 0.45) ** 1.2),
            13.0 / (1.0 + (0.01 / 0.05)),
            7.0 / (1.0 + (1.2e5 / 2.0e6)),
        )
    )
    result = coil_score_decomposition(provisional)
    assert result["points_total"] == pytest.approx(provisional["components"]["coil"])
    assert result["points_curvature"] == pytest.approx(
        result["points_curvature_p95"] + result["points_curvature_max"]
    )
    assert result["points_distance"] == pytest.approx(
        result["points_spacing"] + result["points_axis_distance"]
    )
