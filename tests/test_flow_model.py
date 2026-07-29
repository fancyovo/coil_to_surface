from __future__ import annotations

import torch

from flow_matching.flow import (
    flow_matching_batch,
    flow_matching_loss,
    physical_flow_feature_weights,
    physical_flow_loss_components,
    sample_heun,
)
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


def test_physical_feature_weights_preserve_geometry_block_mass():
    std = torch.ones(100)
    feature, physical = physical_flow_feature_weights(
        std, relative_geometry_weight=0.05, current_feature_weight=1.0
    )
    torch.testing.assert_close(feature[:99].sum(), torch.tensor(99.0))
    torch.testing.assert_close(physical.sum(), torch.tensor(99.0))
    torch.testing.assert_close(feature[-1], torch.tensor(1.0))
    torch.testing.assert_close(physical[0] / physical[1], torch.tensor(2.0))
    torch.testing.assert_close(physical[33] / physical[34], torch.tensor(2.0))


def test_weighted_loss_matches_manual_reduction():
    prediction = torch.tensor([[[1.0, 2.0]]])
    target = torch.zeros_like(prediction)
    weights = torch.tensor([1.0, 3.0])
    loss = flow_matching_loss(prediction, target, feature_weights=weights)
    torch.testing.assert_close(loss, torch.tensor((1.0 + 3.0 * 4.0) / 4.0))


def test_physical_loss_components_share_one_definition():
    torch.manual_seed(17)
    prediction = torch.randn(2, 3, 100)
    target = torch.randn_like(prediction)
    feature, physical = physical_flow_feature_weights(torch.linspace(0.1, 1.0, 100))
    total, geometry_physical, geometry_relative, current = physical_flow_loss_components(
        prediction,
        target,
        feature_weights=feature,
        physical_geometry_weights=physical,
    )
    torch.testing.assert_close(
        total, flow_matching_loss(prediction, target, feature_weights=feature)
    )
    torch.testing.assert_close(
        geometry_physical,
        flow_matching_loss(
            prediction[..., :99], target[..., :99], feature_weights=physical
        ),
    )
    torch.testing.assert_close(
        geometry_relative, flow_matching_loss(prediction[..., :99], target[..., :99])
    )
    torch.testing.assert_close(
        current, flow_matching_loss(prediction[..., -1:], target[..., -1:])
    )
