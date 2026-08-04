import numpy as np

from scripts.optimize_qh_g3_informed_subspace_adam import (
    exact_subspace_gradient,
    informed_orthogonal_directions,
)


def score_result(score: float, *, surface_level: float = 0.25) -> dict:
    return {
        "status": "ok",
        "score": score,
        "diagnostics": {
            "surface_level": surface_level,
            "stable_surface_count": 4,
            "surface_long_trace_rejected_count": 0,
            "flux_attempt_count": 1,
            "volume_candidate_count": 100,
            "volume_available_count": 100,
            "volume_point_count": 100,
            "alpha_column_count": 20,
        },
    }


def test_informed_basis_is_rms_normalized_and_orthogonal() -> None:
    rng = np.random.default_rng(7)
    gradient = np.arange(1.0, 13.0).reshape(3, 4)
    directions = informed_orthogonal_directions(rng, gradient, random_count=5)
    flattened = directions.reshape(len(directions), -1).astype(np.float64)

    gram = flattened @ flattened.T / gradient.size
    assert np.allclose(gram, np.eye(len(directions)), atol=2.0e-7)
    assert np.dot(flattened[0], gradient.ravel()) > 0.0


def test_full_informed_basis_recovers_linear_gradient() -> None:
    rng = np.random.default_rng(11)
    gradient = np.asarray([[2.0, -1.0, 0.5], [0.25, 3.0, -2.0]])
    directions = informed_orthogonal_directions(
        rng, gradient, random_count=gradient.size - 1
    )
    perturbation = 0.005
    slopes = directions.reshape(len(directions), -1) @ gradient.ravel()
    plus = [score_result(50.0 + perturbation * slope) for slope in slopes]
    minus = [score_result(50.0 - perturbation * slope) for slope in slopes]

    recovered, rows, predicted_gain = exact_subspace_gradient(
        score_result(50.0), plus, minus, directions, perturbation
    )

    assert recovered is not None
    assert all(row["valid"] for row in rows)
    assert predicted_gain > 0.0
    assert np.allclose(recovered, gradient, rtol=2.0e-6, atol=2.0e-6)


def test_branch_change_is_not_used_as_a_derivative() -> None:
    rng = np.random.default_rng(13)
    gradient = np.ones((2, 3))
    directions = informed_orthogonal_directions(rng, gradient, random_count=1)
    plus = [score_result(51.0), score_result(51.0, surface_level=0.2)]
    minus = [score_result(49.0), score_result(49.0)]

    recovered, rows, _ = exact_subspace_gradient(
        score_result(50.0), plus, minus, directions, 0.005
    )

    assert recovered is not None
    assert rows[0]["valid"]
    assert not rows[1]["valid"]
