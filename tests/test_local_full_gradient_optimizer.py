from __future__ import annotations

import numpy as np

from scripts.optimize_flow_prior_local_full_gradient_adam import (
    coordinate_gradient,
    damped_inverse_bfgs,
    endpoint_latents,
    gradient_probe,
    random_direction_endpoints,
    random_direction_gradient,
    recorded_native_result,
)


def linear_scores(states: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    return np.einsum("nij,ij->n", states, gradient)


def test_coordinate_gradient_recovers_linear_objective() -> None:
    center = np.arange(6, dtype=np.float32).reshape(2, 3) / 10.0
    gradient = np.arange(1, 7, dtype=np.float64).reshape(2, 3)
    perturbation = 0.005

    endpoints = endpoint_latents(center, perturbation)
    estimate = coordinate_gradient(
        linear_scores(endpoints, gradient), perturbation
    ).reshape(center.shape)

    np.testing.assert_allclose(estimate, gradient, rtol=2.0e-5)


def test_complete_orthogonal_bank_recovers_linear_objective() -> None:
    center = np.arange(6, dtype=np.float32).reshape(2, 3) / 10.0
    gradient = np.arange(1, 7, dtype=np.float64).reshape(2, 3)
    directions = np.eye(6, dtype=np.float32).reshape(6, 2, 3) * np.sqrt(6.0)
    perturbation = 0.005

    endpoints = random_direction_endpoints(center, perturbation, directions)
    estimate = random_direction_gradient(
        linear_scores(endpoints, gradient), perturbation, directions
    )

    np.testing.assert_allclose(estimate, gradient, rtol=2.0e-5)


def test_random_gradient_probe_is_reproducible_and_centered() -> None:
    center = np.arange(6, dtype=np.float32).reshape(2, 3) / 10.0
    first_rng = np.random.default_rng(1234)
    second_rng = np.random.default_rng(1234)

    directions, endpoints = gradient_probe(
        center,
        mode="random-orthogonal",
        perturbation=0.005,
        random_direction_count=4,
        rng=first_rng,
    )
    repeat_directions, repeat_endpoints = gradient_probe(
        center,
        mode="random-orthogonal",
        perturbation=0.005,
        random_direction_count=4,
        rng=second_rng,
    )

    np.testing.assert_array_equal(directions, repeat_directions)
    np.testing.assert_array_equal(endpoints, repeat_endpoints)
    np.testing.assert_allclose(
        0.5 * (endpoints[0::2] + endpoints[1::2]),
        np.broadcast_to(center, (len(directions),) + center.shape),
    )


def test_coordinate_gradient_probe_does_not_advance_rng() -> None:
    center = np.zeros((2, 3), dtype=np.float32)
    rng = np.random.default_rng(42)
    before = rng.bit_generator.state
    directions, endpoints = gradient_probe(
        center,
        mode="coordinate",
        perturbation=0.005,
        random_direction_count=4,
        rng=rng,
    )

    assert directions is None
    assert rng.bit_generator.state == before
    assert endpoints.shape == (12, 2, 3)


def test_inverse_bfgs_matches_diagonal_quadratic_curvature() -> None:
    inverse, details = damped_inverse_bfgs(
        np.eye(2),
        np.asarray([1.0, 0.0]),
        np.asarray([2.0, 0.0]),
    )

    assert details["updated"]
    np.testing.assert_allclose(inverse, np.diag([0.5, 1.0]))


def test_inverse_bfgs_damps_bad_curvature_without_losing_spd() -> None:
    inverse, details = damped_inverse_bfgs(
        np.eye(2),
        np.asarray([1.0, 0.0]),
        np.asarray([-1.0, 0.0]),
    )

    assert details["updated"]
    assert details["damped"]
    assert np.all(np.linalg.eigvalsh(inverse) > 0.0)


def test_recorded_native_result_preserves_anchor_branch() -> None:
    result = {"status": "ok", "diagnostics": {"axis_R": 1.0, "axis_Z": 0.0}}
    payload = {"flow_prior_local_full_gradient_adam": {"native_score": result}}

    assert recorded_native_result(payload) is result
    assert recorded_native_result({"flow_prior_start": {"noise": []}}) is None


def test_recorded_native_result_accepts_screened_start() -> None:
    result = {"status": "ok", "diagnostics": {"axis_R": 1.0, "axis_Z": 0.0}}
    payload = {"flow_prior_screening": {"native_score": result}}

    assert recorded_native_result(payload) is result
