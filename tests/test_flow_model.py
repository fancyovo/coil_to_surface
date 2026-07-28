from __future__ import annotations

import torch

from flow_matching.flow import flow_matching_batch, flow_matching_loss, sample_heun
from flow_matching.model import CoilFlowTransformer


def small_model() -> CoilFlowTransformer:
    return CoilFlowTransformer(width=64, layers=2, heads=4, hidden=128)


def test_model_is_permutation_equivariant_without_positions():
    torch.manual_seed(3)
    model = small_model().eval()
    tokens = torch.randn(2, 4, 100)
    time = torch.tensor([0.2, 0.7])
    nfp = torch.tensor([3, 5])
    permutation = torch.tensor([2, 0, 3, 1])
    expected = model(tokens, time, nfp)[:, permutation]
    actual = model(tokens[:, permutation], time, nfp)
    torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)


def test_flow_loss_and_heun_shapes_are_finite():
    torch.manual_seed(5)
    model = small_model().eval()
    data = torch.randn(3, 2, 100)
    mixed, time, target = flow_matching_batch(data)
    prediction = model(mixed, time, torch.full((3,), 4, dtype=torch.long))
    loss = flow_matching_loss(prediction, target)
    assert loss.ndim == 0 and torch.isfinite(loss)
    sampled = sample_heun(
        model,
        torch.randn_like(data),
        torch.full((3,), 4, dtype=torch.long),
        steps=3,
    )
    assert sampled.shape == data.shape
    assert torch.isfinite(sampled).all()
