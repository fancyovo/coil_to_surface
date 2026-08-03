from __future__ import annotations

import torch

from flow_matching.geometry import curve_metrics, geometry_eligible, reference_bounds


def test_curve_metrics_and_bounds_are_per_sample():
    tokens = torch.zeros(4, 3, 100)
    tokens[..., 0] = 1.0
    tokens[..., 2] = 0.2
    tokens[..., 33] = 0.1
    tokens[..., 35] = 0.2
    tokens[..., 66] = 0.05
    tokens[..., 68] = 0.1
    tokens[..., -1] = 1.0e6 / 3.0
    metrics = curve_metrics(tokens, samples=32)
    assert all(values.shape == (4,) for values in metrics.values())
    bounds = reference_bounds(metrics)
    assert geometry_eligible(metrics, bounds).shape == (4,)
