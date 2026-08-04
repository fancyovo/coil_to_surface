import numpy as np

from scripts.optimize_qh_g3_informed_subspace_adam import (
    best_improving_branch_endpoint,
    best_improving_endpoint,
    exact_subspace_gradient,
    informed_orthogonal_directions,
    projected_trust_update,
    score_improves_by,
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


def test_projected_trust_update_preserves_direction_and_rms() -> None:
    gradient = np.asarray([[2.0, -1.0, 0.5], [0.25, 3.0, -2.0]])
    update = projected_trust_update(gradient, step_rms=0.0025)

    assert update is not None
    assert np.allclose(update / gradient, update.flat[0] / gradient.flat[0])
    assert np.isclose(np.sqrt(np.mean(np.square(update))), 0.0025)
    assert np.vdot(update, gradient) > 0.0


def test_projected_trust_update_rejects_zero_gradient() -> None:
    assert projected_trust_update(np.zeros((2, 3)), step_rms=0.0025) is None


def test_one_sided_same_branch_endpoint_is_used_as_a_derivative() -> None:
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
    assert rows[0]["difference_scheme"] == "centered"
    assert rows[1]["valid"]
    assert rows[1]["difference_scheme"] == "backward"


def test_direction_without_a_center_branch_endpoint_is_rejected() -> None:
    direction = np.ones((1, 2, 3))
    recovered, rows, predicted_gain = exact_subspace_gradient(
        score_result(50.0),
        [score_result(51.0, surface_level=0.2)],
        [score_result(49.0, surface_level=0.36)],
        direction,
        0.005,
    )

    assert recovered is None
    assert not rows[0]["valid"]
    assert rows[0]["difference_scheme"] == "invalid"
    assert predicted_gain == 0.0


def test_improving_branch_endpoint_is_selected_separately() -> None:
    center = score_result(50.0)
    endpoints = [
        score_result(60.0),
        score_result(51.0, surface_level=0.36),
        score_result(52.0, surface_level=0.36),
        score_result(49.0, surface_level=0.49),
    ]

    selected = best_improving_branch_endpoint(center, endpoints, minimum_gain=0.1)

    assert selected == 2


def test_best_probe_endpoint_can_stay_on_the_center_branch() -> None:
    center = score_result(50.0)
    endpoints = [
        score_result(50.5),
        score_result(52.0, surface_level=0.36),
        score_result(51.0),
    ]

    assert best_improving_endpoint(
        center,
        endpoints,
        same_branch_minimum_gain=0.1,
        branch_minimum_gain=0.1,
    ) == 1
    assert best_improving_endpoint(
        center,
        endpoints[:1],
        same_branch_minimum_gain=0.1,
        branch_minimum_gain=0.1,
    ) == 0


def test_probe_and_branch_endpoint_thresholds_are_independent() -> None:
    center = score_result(50.0)
    endpoints = [
        score_result(50.005),
        score_result(50.009, surface_level=0.36),
    ]

    assert best_improving_endpoint(
        center,
        endpoints,
        same_branch_minimum_gain=0.001,
        branch_minimum_gain=0.01,
    ) == 0


def test_branch_endpoint_must_beat_smooth_candidate() -> None:
    smooth = score_result(52.0)

    assert not score_improves_by(score_result(52.005), smooth, minimum_gain=0.01)
    assert score_improves_by(score_result(52.02), smooth, minimum_gain=0.01)
    assert not score_improves_by(None, smooth, minimum_gain=0.01)
