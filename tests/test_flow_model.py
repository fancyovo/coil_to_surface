from __future__ import annotations

import torch

from flow_matching.flow import (
    flow_matching_batch,
    flow_matching_loss,
    integrate_flow,
    physical_flow_feature_weights,
    physical_flow_loss_components,
    sample_heun,
    sample_rk4,
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


class TimeDependentVelocity(torch.nn.Module):
    def forward(self, state, time, nfp, mask=None):
        del nfp
        velocity = (1.0 + time[:, None, None]) * torch.ones_like(state)
        return velocity if mask is None else velocity * mask[..., None]


class ValidatedVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.validation_calls = 0
        self.velocity_calls = 0

    def validate_nfp(self, nfp):
        self.validation_calls += 1
        if torch.any(nfp < 1):
            raise ValueError("invalid nfp")

    def forward(self, state, time, nfp, mask=None):
        raise AssertionError("integrators must use the once-validated path")

    def forward_unchecked(self, state, time, nfp, mask=None):
        del time, nfp
        self.velocity_calls += 1
        return state if mask is None else state * mask[..., None]


def test_integrator_validates_nfp_once_before_rk4_loop():
    model = ValidatedVelocity()
    initial = torch.ones(1, 2, 100)
    result = integrate_flow(model, initial, torch.tensor([4]), steps=3, method="rk4")
    assert model.validation_calls == 1
    assert model.velocity_calls == 12
    torch.testing.assert_close(result, initial * torch.exp(torch.tensor(1.0)), rtol=2e-4, atol=2e-4)


def test_rk4_integrates_in_both_directions_and_closes():
    model = TimeDependentVelocity()
    initial = torch.randn(2, 3, 100)
    nfp = torch.tensor([3, 4])
    final = sample_rk4(model, initial, nfp, steps=4)
    torch.testing.assert_close(final, initial + 1.5, rtol=0.0, atol=2.0e-6)
    recovered = integrate_flow(
        model, final, nfp, start_time=1.0, end_time=0.0, steps=4, method="rk4"
    )
    torch.testing.assert_close(recovered, initial, rtol=0.0, atol=2.0e-6)


def test_bidirectional_integrator_respects_mask():
    model = TimeDependentVelocity()
    initial = torch.zeros(2, 3, 100)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    result = integrate_flow(
        model,
        initial,
        torch.tensor([3, 4]),
        steps=2,
        method="heun",
        mask=mask,
    )
    torch.testing.assert_close(result[mask], torch.full_like(result[mask], 1.5))
    assert torch.count_nonzero(result[~mask]) == 0


def test_bidirectional_integrator_rejects_invalid_arguments():
    model = TimeDependentVelocity()
    initial = torch.zeros(1, 2, 100)
    nfp = torch.tensor([4])
    for kwargs in (
        {"steps": 0},
        {"start_time": 0.5, "end_time": 0.5},
        {"method": "euler"},
    ):
        try:
            integrate_flow(model, initial, nfp, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")
