import numpy as np

from scripts.qh_score_noise_sensitivity import curve_positions, perturbation_metrics


def test_curve_positions_constant_curve() -> None:
    tokens = np.zeros((1, 100), dtype=np.float64)
    tokens[0, 0] = 1.25
    tokens[0, 33] = -0.5
    tokens[0, 66] = 0.75
    positions = curve_positions(tokens, samples=16)
    assert positions.shape == (1, 16, 3)
    expected = np.broadcast_to([1.25, -0.5, 0.75], positions.shape)
    np.testing.assert_allclose(positions, expected)


def test_zero_perturbation_metrics() -> None:
    tokens = np.zeros((2, 100), dtype=np.float64)
    tokens[:, -1] = (1.0, -1.0)
    metrics = perturbation_metrics(tokens, tokens)
    assert all(value == 0.0 for value in metrics.values())


def test_current_change_does_not_move_curve() -> None:
    reference = np.zeros((2, 100), dtype=np.float64)
    changed = reference.copy()
    changed[:, -1] = (2.0, -3.0)
    metrics = perturbation_metrics(changed, reference)
    assert metrics["position_delta_rms_m"] == 0.0
    assert metrics["coefficient_relative_l2"] == 0.0
    assert metrics["current_relative_l2"] > 0.0
