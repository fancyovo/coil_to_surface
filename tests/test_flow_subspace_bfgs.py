from __future__ import annotations

import numpy as np

from scripts.optimize_flow_prior_subspace_bfgs import (
    choose_line_search_candidate,
    damped_inverse_bfgs,
    gradient_cosine,
    initial_inverse_hessian,
    projected_central_gradient,
    rms_orthonormal_basis,
)


def test_rms_basis_is_orthonormal_and_uses_candidates() -> None:
    shape = (2, 3)
    first = np.arange(1, 7, dtype=float).reshape(shape)
    basis = rms_orthonormal_basis(
        [first], shape=shape, rank=4, rng=np.random.default_rng(19)
    )
    flat = basis.reshape(4, -1).astype(np.float64)
    np.testing.assert_allclose(flat @ flat.T, np.eye(4) * 6, atol=2.0e-6)
    expected = first / np.sqrt(np.mean(first * first))
    np.testing.assert_allclose(basis[0], expected, atol=2.0e-7)


def test_projected_gradient_recovers_quadratic_derivative() -> None:
    center = np.asarray([0.4, -0.2, 0.7])
    perturbation = 0.003

    def objective(value: np.ndarray) -> float:
        return float(2.0 * value[0] - 3.0 * value[1] + value[2] ** 2)

    plus = np.asarray(
        [objective(center + perturbation * np.eye(3)[index]) for index in range(3)]
    )
    minus = np.asarray(
        [objective(center - perturbation * np.eye(3)[index]) for index in range(3)]
    )
    gradient = projected_central_gradient(plus, minus, perturbation)
    np.testing.assert_allclose(gradient, [2.0, -3.0, 1.4], atol=1.0e-12)


def test_damped_bfgs_stays_symmetric_positive_definite() -> None:
    inverse = np.eye(3)
    step = np.asarray([0.2, -0.1, 0.05])
    objective_gradient_delta = np.asarray([0.5, -0.05, 0.02])
    updated, info = damped_inverse_bfgs(inverse, step, objective_gradient_delta)
    assert info["updated"] is True
    np.testing.assert_allclose(updated, updated.T, atol=1.0e-14)
    assert np.all(np.linalg.eigvalsh(updated) > 0.0)


def test_initial_inverse_hessian_uses_maximum_curvature() -> None:
    inverse, scale = initial_inverse_hessian(np.asarray([-4.0, -9.0, 2.0]))
    assert scale == 6.5
    np.testing.assert_allclose(inverse, np.eye(3) / 6.5)


def test_line_search_chooses_best_valid_improvement() -> None:
    selected = choose_line_search_candidate(
        np.asarray([10.2, 10.5, 12.0, 10.4]),
        np.asarray([True, True, False, True]),
        current_score=10.0,
        min_improvement=0.01,
    )
    assert selected == 1
    assert (
        choose_line_search_candidate(
            np.asarray([9.0, 10.0]),
            np.asarray([True, True]),
            current_score=10.0,
            min_improvement=0.01,
        )
        is None
    )


def test_gradient_cosine_handles_regular_and_zero_vectors() -> None:
    assert gradient_cosine(np.asarray([1.0, 0.0]), np.asarray([1.0, 1.0])) > 0.7
    assert gradient_cosine(np.zeros(2), np.zeros(2)) == 1.0
