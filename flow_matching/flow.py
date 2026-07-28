from __future__ import annotations

import torch


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
) -> torch.Tensor:
    square = (prediction.float() - target.float()).square()
    if mask is None:
        return square.mean()
    weights = mask[..., None].to(square.dtype)
    return torch.sum(square * weights) / (torch.sum(weights) * square.shape[-1]).clamp_min(1.0)


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
    state = noise
    dt = 1.0 / steps
    for index in range(steps):
        time = torch.full(
            (state.shape[0],), index / steps, device=state.device, dtype=torch.float32
        )
        velocity = model(state, time, nfp, mask)
        predicted = state + dt * velocity
        if index + 1 == steps:
            state = predicted
        else:
            next_time = torch.full_like(time, (index + 1) / steps)
            next_velocity = model(predicted, next_time, nfp, mask)
            state = state + 0.5 * dt * (velocity + next_velocity)
    return state
