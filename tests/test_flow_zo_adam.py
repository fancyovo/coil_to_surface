from __future__ import annotations

import json

import numpy as np

from scripts.optimize_flow_prior_zo_adam import (
    gradient_from_pairs,
    load_initial_noise,
    orthogonal_directions,
    prior_penalty_and_gradient,
)
from scripts.optimize_flow_prior_standard_adam import robust_direction_deltas


def test_load_initial_noise_accepts_generic_start(tmp_path):
    path = tmp_path / "start.json"
    noise = np.arange(300, dtype=np.float32).reshape(3, 100)
    path.write_text(
        json.dumps({"flow_prior_start": {"noise": noise.tolist()}}),
        encoding="utf-8",
    )

    loaded, payload = load_initial_noise(path)

    np.testing.assert_array_equal(loaded, noise)
    assert "flow_prior_start" in payload


def test_load_initial_noise_accepts_standard_adam_and_subspace_bfgs(tmp_path):
    noise = np.arange(300, dtype=np.float32).reshape(3, 100)
    for key in ("flow_prior_standard_adam", "flow_prior_subspace_bfgs"):
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps({key: {"noise": noise.tolist()}}), encoding="utf-8")
        loaded, payload = load_initial_noise(path)
        np.testing.assert_array_equal(loaded, noise)
        assert key in payload


def test_orthogonal_directions_have_unit_rms():
    directions = orthogonal_directions(np.random.default_rng(73), (2, 3), 4)
    flat = directions.reshape(4, -1).astype(np.float64)
    np.testing.assert_allclose(
        np.sqrt(np.mean(flat * flat, axis=1)), np.ones(4), atol=2.0e-7
    )
    gram = flat @ flat.T
    np.testing.assert_allclose(
        gram, np.eye(4) * flat.shape[1], atol=2.0e-6
    )


def test_full_orthogonal_gradient_recovers_linear_objective():
    rng = np.random.default_rng(91)
    shape = (2, 3)
    center = rng.normal(size=shape)
    coefficients = rng.normal(size=shape)
    directions = orthogonal_directions(rng, shape, int(np.prod(shape)))
    perturbation = 0.03

    def objective(value):
        return float(np.sum(coefficients * value))

    plus = np.asarray(
        [objective(center + perturbation * direction) for direction in directions]
    )
    minus = np.asarray(
        [objective(center - perturbation * direction) for direction in directions]
    )
    gradient, raw_delta = gradient_from_pairs(
        plus,
        minus,
        directions,
        perturbation,
        delta_clip=None,
    )
    assert np.all(np.abs(raw_delta) > 0.0)
    np.testing.assert_allclose(gradient, coefficients, rtol=2.0e-6, atol=2.0e-6)


def test_direction_delta_clipping_limits_gradient_contribution():
    directions = np.asarray([[[1.0, -1.0]]], dtype=np.float32)
    gradient, raw_delta = gradient_from_pairs(
        np.asarray([100.0]),
        np.asarray([0.0]),
        directions,
        0.5,
        delta_clip=4.0,
    )
    np.testing.assert_allclose(raw_delta, [100.0])
    np.testing.assert_allclose(gradient, [[4.0, -4.0]])


def test_robust_direction_filter_drops_invalid_and_rescales_valid_directions():
    used, invalid, outlier, limit = robust_direction_deltas(
        np.asarray([-58.0, -0.5, -57.0, -0.2]),
        ["no_axis", "ok", "no_axis", "ok", "ok", "ok", "ok", "ok"],
        outlier_ratio=8.0,
        mad_factor=8.0,
    )

    np.testing.assert_array_equal(invalid, [True, False, True, False])
    np.testing.assert_array_equal(outlier, [False, False, False, False])
    np.testing.assert_allclose(used, [0.0, -1.0, 0.0, -0.4])
    assert limit is None


def test_robust_direction_filter_is_scale_invariant_for_valid_outlier():
    statuses = ["ok"] * 8
    delta = np.asarray([0.4, 0.5, 0.6, 100.0])
    used, invalid, outlier, limit = robust_direction_deltas(
        delta, statuses, outlier_ratio=8.0, mad_factor=8.0
    )
    scaled, scaled_invalid, scaled_outlier, scaled_limit = robust_direction_deltas(
        100.0 * delta, statuses, outlier_ratio=8.0, mad_factor=8.0
    )

    np.testing.assert_array_equal(invalid, np.zeros(4, dtype=bool))
    np.testing.assert_array_equal(outlier, [False, False, False, True])
    np.testing.assert_array_equal(scaled_invalid, invalid)
    np.testing.assert_array_equal(scaled_outlier, outlier)
    np.testing.assert_allclose(scaled, 100.0 * used)
    assert np.isclose(scaled_limit, 100.0 * limit)


def test_prior_penalty_is_inactive_inside_soft_region():
    noise = np.asarray([[0.5, -1.0, 1.5]], dtype=float)
    penalty, gradient = prior_penalty_and_gradient(
        noise,
        rms_soft=2.0,
        coordinate_soft=4.0,
        rms_weight=5.0,
        coordinate_weight=5.0,
    )
    assert penalty == 0.0
    np.testing.assert_array_equal(gradient, np.zeros_like(noise))


def test_prior_gradient_points_outward_beyond_soft_region():
    noise = np.asarray([[5.0, -5.0, 3.0]], dtype=float)
    penalty, gradient = prior_penalty_and_gradient(
        noise,
        rms_soft=2.0,
        coordinate_soft=4.0,
        rms_weight=5.0,
        coordinate_weight=5.0,
    )
    assert penalty > 0.0
    assert np.all(gradient * noise > 0.0)
