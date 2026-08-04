from __future__ import annotations

from typing import Literal

import torch


GEOMETRY_DIM = 99
TOKEN_DIM = 100


def prepare_velocity_model(model, nfp: torch.Tensor):
    validate = getattr(model, "validate_nfp", None)
    unchecked = getattr(model, "forward_unchecked", None)
    if callable(validate) and callable(unchecked):
        validate(nfp)
        return unchecked
    return model


def parseval_geometry_weights(
    standard_deviation: torch.Tensor,
) -> torch.Tensor:
    """Return mean-one geometry weights for physical curve L2 error."""
    std = torch.as_tensor(standard_deviation, dtype=torch.float32)
    if std.ndim != 1 or std.numel() != TOKEN_DIM:
        raise ValueError(f"standard_deviation must have shape ({TOKEN_DIM},)")
    if not torch.all(torch.isfinite(std)) or torch.any(std <= 0.0):
        raise ValueError("standard_deviation must be finite and positive")
    parseval = torch.full_like(std[:GEOMETRY_DIM], 0.5)
    parseval[0::33] = 1.0
    weights = parseval * std[:GEOMETRY_DIM].square()
    return weights * (GEOMETRY_DIM / weights.sum())


def physical_flow_feature_weights(
    standard_deviation: torch.Tensor,
    *,
    relative_geometry_weight: float = 0.05,
    current_feature_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the mixed physical/relative metric in normalized coordinates."""
    if not 0.0 <= relative_geometry_weight <= 1.0:
        raise ValueError("relative_geometry_weight must be in [0, 1]")
    if not current_feature_weight > 0.0:
        raise ValueError("current_feature_weight must be positive")
    physical = parseval_geometry_weights(standard_deviation)
    geometry = (
        (1.0 - relative_geometry_weight) * physical
        + relative_geometry_weight * torch.ones_like(physical)
    )
    feature = torch.cat(
        [geometry, geometry.new_tensor([float(current_feature_weight)])]
    )
    return feature, physical


def flow_matching_batch(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    noise = torch.randn_like(data)
    time = torch.rand(data.shape[0], device=data.device, dtype=torch.float32)
    mixed = (1.0 - time[:, None, None]) * noise + time[:, None, None] * data
    target = data - noise
    return mixed, time, target


def flow_matching_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    feature_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    square = (prediction.float() - target.float()).square()
    if feature_weights is None:
        feature = torch.ones(square.shape[-1], dtype=square.dtype, device=square.device)
    else:
        feature = feature_weights.to(dtype=square.dtype, device=square.device)
        if feature.ndim != 1 or feature.numel() != square.shape[-1]:
            raise ValueError("feature_weights must match the final tensor dimension")
    weighted = square * feature
    if mask is None:
        observations = square.numel() // square.shape[-1]
        return weighted.sum() / (observations * feature.sum()).clamp_min(1.0)
    token_weights = mask[..., None].to(square.dtype)
    return torch.sum(weighted * token_weights) / (
        torch.sum(token_weights) * feature.sum()
    ).clamp_min(1.0)


def physical_flow_loss_components(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    feature_weights: torch.Tensor,
    physical_geometry_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the optimized loss and its three audit components in one pass."""
    square = (prediction.float() - target.float()).square()
    feature = feature_weights.to(dtype=square.dtype, device=square.device)
    physical = physical_geometry_weights.to(dtype=square.dtype, device=square.device)
    if feature.shape != (square.shape[-1],):
        raise ValueError("feature_weights must match the final tensor dimension")
    if physical.shape != (GEOMETRY_DIM,) or square.shape[-1] != TOKEN_DIM:
        raise ValueError("physical geometry weights require 100-dimensional tokens")
    total_loss = torch.sum(square * feature) / (
        (square.numel() // TOKEN_DIM) * feature.sum()
    ).clamp_min(1.0)
    geometry_square = square[..., :GEOMETRY_DIM]
    geometry_physical_loss = torch.sum(geometry_square * physical) / (
        (geometry_square.numel() // GEOMETRY_DIM) * physical.sum()
    ).clamp_min(1.0)
    return (
        total_loss,
        geometry_physical_loss,
        geometry_square.mean(),
        square[..., -1].mean(),
    )


@torch.no_grad()
def sample_heun(
    model,
    noise: torch.Tensor,
    nfp: torch.Tensor,
    *,
    steps: int = 32,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if steps < 1:
        raise ValueError("steps must be positive")
    velocity_model = prepare_velocity_model(model, nfp)
    state = noise
    dt = 1.0 / steps
    for index in range(steps):
        time = torch.full(
            (state.shape[0],), index / steps, device=state.device, dtype=torch.float32
        )
        velocity = velocity_model(state, time, nfp, mask)
        predicted = state + dt * velocity
        if index + 1 == steps:
            state = predicted
        else:
            next_time = torch.full_like(time, (index + 1) / steps)
            next_velocity = velocity_model(predicted, next_time, nfp, mask)
            state = state + 0.5 * dt * (velocity + next_velocity)
    return state


@torch.no_grad()
def integrate_flow(
    model,
    state: torch.Tensor,
    nfp: torch.Tensor,
    *,
    start_time: float = 0.0,
    end_time: float = 1.0,
    steps: int = 32,
    method: Literal["heun", "rk4"] = "rk4",
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Integrate the learned flow ODE in either time direction."""
    if steps < 1:
        raise ValueError("steps must be positive")
    if end_time == start_time:
        raise ValueError("start_time and end_time must differ")
    if method not in ("heun", "rk4"):
        raise ValueError(f"unsupported integration method {method!r}")
    if state.ndim != 3 or nfp.shape != (state.shape[0],):
        raise ValueError("state and nfp batch dimensions must match")

    velocity_model = prepare_velocity_model(model, nfp)
    value = state
    dt = (float(end_time) - float(start_time)) / steps

    def velocity(at: torch.Tensor, time_value: float) -> torch.Tensor:
        time = torch.full(
            (at.shape[0],),
            time_value,
            device=at.device,
            dtype=torch.float32,
        )
        return velocity_model(at, time, nfp, mask)

    for index in range(steps):
        time_value = float(start_time) + index * dt
        k1 = velocity(value, time_value)
        if method == "heun":
            predicted = value + dt * k1
            k2 = velocity(predicted, time_value + dt)
            value = value + 0.5 * dt * (k1 + k2)
        else:
            k2 = velocity(value + 0.5 * dt * k1, time_value + 0.5 * dt)
            k3 = velocity(value + 0.5 * dt * k2, time_value + 0.5 * dt)
            k4 = velocity(value + dt * k3, time_value + dt)
            value = value + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if mask is not None:
            value = value * mask[..., None]
    return value


def sample_rk4(
    model,
    noise: torch.Tensor,
    nfp: torch.Tensor,
    *,
    steps: int = 32,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return integrate_flow(
        model,
        noise,
        nfp,
        start_time=0.0,
        end_time=1.0,
        steps=steps,
        method="rk4",
        mask=mask,
    )
