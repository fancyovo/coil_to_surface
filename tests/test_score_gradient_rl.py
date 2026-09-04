from __future__ import annotations

import numpy as np
import torch
from torch import nn

from flow_matching.data import CoilNormalizer
from flow_matching.model import CoilFlowTransformer
from flow_matching.score_gradient_rl import (
    N_BASE_COILS,
    TOKEN_DIM,
    coordinate_gradient_directional_check,
    cfm_loss_terms,
    map_score_gradient_to_flow,
    valid_flow_terms_with_transport,
)


def _normalizer(scale: float, mean: float = 0.0) -> CoilNormalizer:
    return CoilNormalizer(
        mean=np.full(TOKEN_DIM, mean, dtype=np.float32),
        std=np.full(TOKEN_DIM, scale, dtype=np.float32),
        current_l1_a={"8:3": 3.0},
        clip=float("inf"),
    )


def test_score_gradient_mapping_matches_coordinate_directional_identity() -> None:
    flow_normalizer = _normalizer(2.0)
    optimizer_normalizer = _normalizer(4.0)
    flow = np.zeros((N_BASE_COILS, TOKEN_DIM), dtype=np.float32)
    flow[:, -1] = np.asarray([0.7, 0.2, 0.1], dtype=np.float32)
    gradient = np.linspace(-1.0, 1.0, flow.size, dtype=np.float64).reshape(flow.shape)
    mapped = map_score_gradient_to_flow(
        flow,
        gradient,
        flow_normalizer,
        optimizer_normalizer,
        current_l1_a=3.0,
        nfp=8,
    )
    direction = np.random.default_rng(7).normal(size=flow.shape)
    direction /= np.linalg.norm(direction)
    error = coordinate_gradient_directional_check(
        flow,
        gradient,
        mapped,
        flow_normalizer,
        optimizer_normalizer,
        current_l1_a=3.0,
        nfp=8,
        direction=direction,
        step=1.0e-3,
    )
    assert error < 2.0e-3
    # Geometry is an affine normalizer map: dx_optimizer/dx_flow = 2/4.
    np.testing.assert_allclose(mapped[:, :TOKEN_DIM - 1], gradient[:, :TOKEN_DIM - 1] * 0.5, rtol=1e-6)


class _TinyVelocity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, tokens: torch.Tensor, time: torch.Tensor, nfp: torch.Tensor) -> torch.Tensor:
        del time, nfp
        return self.scale * tokens


def test_transport_terms_build_parameter_gradient_graph() -> None:
    torch.manual_seed(3)
    model = _TinyVelocity()
    data = torch.randn(3, N_BASE_COILS, TOKEN_DIM)
    score_gradient = torch.randn_like(data)
    weights = torch.ones(TOKEN_DIM)
    generator = torch.Generator().manual_seed(11)
    ordinary, transport = valid_flow_terms_with_transport(
        model,
        data,
        score_gradient,
        feature_weights=weights,
        monte_carlo_samples=4,
        generator=generator,
    )
    assert ordinary.shape == (3,)
    assert transport.shape == (3,)
    objective = (ordinary + 0.03 * transport).mean()
    objective.backward()
    assert model.scale.grad is not None
    assert torch.isfinite(model.scale.grad)


def test_cfm_loss_accepts_independent_draws() -> None:
    model = _TinyVelocity()
    data = torch.zeros(5, N_BASE_COILS, TOKEN_DIM)
    noise = torch.ones_like(data)
    time = torch.linspace(0.1, 0.9, len(data))
    loss = cfm_loss_terms(
        model,
        data,
        feature_weights=torch.ones(TOKEN_DIM),
        noise=noise,
        time_value=time,
    )
    assert loss.shape == (5,)
    assert torch.all(torch.isfinite(loss))


def test_transformer_transport_term_supports_second_order_parameter_backward() -> None:
    torch.manual_seed(5)
    model = CoilFlowTransformer(width=8, layers=1, heads=2, hidden=16)
    data = torch.randn(2, N_BASE_COILS, TOKEN_DIM)
    score_gradient = torch.randn_like(data)
    ordinary, transport = valid_flow_terms_with_transport(
        model,
        data,
        score_gradient,
        feature_weights=torch.ones(TOKEN_DIM),
        monte_carlo_samples=2,
        generator=torch.Generator().manual_seed(13),
    )
    (ordinary.mean() + 0.01 * transport.mean()).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
