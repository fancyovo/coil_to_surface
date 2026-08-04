from __future__ import annotations

import numpy as np
import torch

from flow_matching.data import CoilNormalizer
from flow_matching.vjp import (
    canonicalize_currents_torch,
    integrate_flow_differentiable,
    inverse_normalizer_torch,
)


class LinearVelocity(torch.nn.Module):
    def forward(self, state, time, nfp, mask=None):
        del time, nfp
        value = 0.1 * state
        return value if mask is None else value * mask[..., None]


def test_differentiable_rk4_has_correct_vjp() -> None:
    state = torch.randn(1, 2, 100, dtype=torch.float64, requires_grad=True)
    nfp = torch.tensor([4])
    result = integrate_flow_differentiable(
        LinearVelocity(), state, nfp, steps=16, checkpoint_steps=4
    )
    (gradient,) = torch.autograd.grad(result.sum(), state)
    np.testing.assert_allclose(
        gradient.detach().numpy(),
        np.full_like(gradient.detach().numpy(), np.exp(0.1)),
        rtol=2.0e-10,
        atol=2.0e-10,
    )


def test_torch_current_gauge_matches_numpy_normalizer() -> None:
    normalizer = CoilNormalizer(
        mean=np.zeros(100, dtype=np.float32),
        std=np.ones(100, dtype=np.float32),
        current_l1_a={"4:2": 12.0},
    )
    normalized = torch.zeros((1, 2, 100), dtype=torch.float32)
    normalized[0, :, -1] = torch.tensor([-2.0, 1.0])
    physical, dominant, sign = inverse_normalizer_torch(normalized, normalizer, (4, 2))
    expected = normalizer.inverse(normalized.numpy(), (4, 2))
    np.testing.assert_allclose(physical.detach().numpy(), expected, rtol=0.0, atol=0.0)
    assert dominant.tolist() == [0]
    assert sign.tolist() == [-1.0]


def test_current_gauge_vjp_matches_centered_difference() -> None:
    tokens = torch.zeros((1, 2, 100), dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        tokens[0, :, -1] = torch.tensor([3.0, -1.0])
    cotangent = torch.zeros_like(tokens)
    cotangent[0, :, -1] = torch.tensor([0.25, -0.7])
    physical, _, _ = canonicalize_currents_torch(tokens, 8.0)
    (gradient,) = torch.autograd.grad(torch.sum(physical * cotangent), tokens)
    direction = torch.zeros_like(tokens)
    direction[0, :, -1] = torch.tensor([0.4, -0.2])
    step = 1.0e-6
    plus, _, _ = canonicalize_currents_torch(tokens.detach() + step * direction, 8.0)
    minus, _, _ = canonicalize_currents_torch(tokens.detach() - step * direction, 8.0)
    finite = torch.sum((plus - minus) * cotangent) / (2.0 * step)
    predicted = torch.sum(gradient * direction)
    np.testing.assert_allclose(predicted.item(), finite.item(), rtol=2.0e-9, atol=2.0e-9)
