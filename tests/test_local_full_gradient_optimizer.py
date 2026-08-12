from __future__ import annotations

import numpy as np

from scripts.optimize_flow_prior_local_full_gradient_adam import (
    coordinate_gradient,
    damped_inverse_bfgs,
    endpoint_latents,
    random_direction_endpoints,
    random_direction_gradient,
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
