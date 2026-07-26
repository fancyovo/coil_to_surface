import sys
from pathlib import Path

import numpy as np


GPU_PYTHON = Path(__file__).resolve().parents[1] / "gpu_backend" / "python"
if str(GPU_PYTHON) not in sys.path:
    sys.path.insert(0, str(GPU_PYTHON))

from stellarator_gpu import eval_B_grad_segments_cpu, eval_B_segments_cpu


def test_segment_field_gradient_matches_centered_finite_difference():
    rng = np.random.default_rng(20260726)
    segment_position = rng.normal(size=(17, 3))
    segment_weight = rng.normal(size=(17, 3)) * 2.5e5
    points = rng.normal(size=(8, 3)) + np.asarray([2.0, -1.5, 0.7])

    field, gradient = eval_B_grad_segments_cpu(points, segment_position, segment_weight)
    np.testing.assert_allclose(
        field,
        eval_B_segments_cpu(points, segment_position, segment_weight),
        rtol=1e-14,
        atol=1e-14,
    )

    step = 2e-6
    finite_difference = np.empty_like(gradient)
    for coordinate in range(3):
        offset = np.zeros(3)
        offset[coordinate] = step
        plus = eval_B_segments_cpu(points + offset, segment_position, segment_weight)
        minus = eval_B_segments_cpu(points - offset, segment_position, segment_weight)
        finite_difference[:, :, coordinate] = (plus - minus) / (2.0 * step)

    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-8, atol=2e-10)


def test_segment_field_gradient_is_divergence_free():
    rng = np.random.default_rng(1024)
    segment_position = rng.normal(size=(23, 3))
    segment_weight = rng.normal(size=(23, 3)) * 1e5
    points = rng.normal(size=(13, 3)) + np.asarray([3.0, 2.0, -2.0])
    _, gradient = eval_B_grad_segments_cpu(points, segment_position, segment_weight)
    divergence = np.trace(gradient, axis1=1, axis2=2)
    np.testing.assert_allclose(divergence, 0.0, atol=1e-13)
