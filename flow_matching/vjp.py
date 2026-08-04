from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Literal

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from flow_matching.data import CoilNormalizer
from flow_matching.flow import prepare_velocity_model


@dataclass(frozen=True)
class FlowVjpDiagnostics:
    decode_wall_s: float
    backward_wall_s: float
    use_checkpoint: bool
    checkpoint_steps: int
    rk4_steps: int
    baseline_memory_allocated_bytes: int
    forward_peak_memory_allocated_bytes: int
    total_peak_memory_allocated_bytes: int
    dominant_current_index: tuple[int, ...]
    dominant_current_sign: tuple[int, ...]


def canonicalize_currents_torch(
    tokens: torch.Tensor,
    target_l1_a: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if tokens.ndim != 3 or tokens.shape[-1] != 100:
        raise ValueError("tokens must have shape (batch, coils, 100)")
    current = tokens[..., -1]
    l1 = current.abs().sum(dim=1, keepdim=True)
    fallback = torch.full_like(current, float(target_l1_a) / current.shape[1])
    scaled = torch.where(
        l1 > 1.0e-12,
        current * (float(target_l1_a) / l1.clamp_min(1.0e-12)),
        fallback,
    )
    dominant = scaled.abs().argmax(dim=1)
    dominant_value = scaled.gather(1, dominant[:, None]).squeeze(1)
    dominant_sign = dominant_value.sign()
    dominant_sign = torch.where(
        dominant_sign == 0.0,
        torch.ones_like(dominant_sign),
        dominant_sign,
    )
    output = torch.cat(
        [tokens[..., :-1], (scaled * dominant_sign[:, None])[..., None]],
        dim=-1,
    )
    return output, dominant, dominant_sign


def inverse_normalizer_torch(
    normalized: torch.Tensor,
    normalizer: CoilNormalizer,
    key: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reference_key = f"{int(key[0])}:{int(key[1])}"
    if reference_key not in normalizer.current_l1_a:
        raise KeyError(f"normalizer has no current reference for {reference_key}")
    mean = torch.as_tensor(normalizer.mean, dtype=normalized.dtype, device=normalized.device)
    std = torch.as_tensor(normalizer.std, dtype=normalized.dtype, device=normalized.device)
    physical = normalized * std + mean
    return canonicalize_currents_torch(
        physical,
        normalizer.current_l1_a[reference_key],
    )


def integrate_flow_differentiable(
    model,
    state: torch.Tensor,
    nfp: torch.Tensor,
    *,
    start_time: float = 0.0,
    end_time: float = 1.0,
    steps: int = 256,
    method: Literal["rk4"] = "rk4",
    mask: torch.Tensor | None = None,
    checkpoint_steps: int = 8,
    use_checkpoint: bool = True,
) -> torch.Tensor:
    if method != "rk4":
        raise ValueError("the validated differentiable path currently requires RK4")
    if (
        steps < 1
        or checkpoint_steps < 1
        or (use_checkpoint and steps % checkpoint_steps != 0)
    ):
        raise ValueError(
            "steps must be positive; checkpointed steps must be divisible by checkpoint_steps"
        )
    if state.ndim != 3 or nfp.shape != (state.shape[0],):
        raise ValueError("state and nfp batch dimensions must match")
    velocity_model = prepare_velocity_model(model, nfp)
    dt = (float(end_time) - float(start_time)) / steps

    def integrate_chunk(
        value: torch.Tensor,
        first_step: int,
        chunk_steps: int,
    ) -> torch.Tensor:
        for local_step in range(chunk_steps):
            index = first_step + local_step
            time_value = float(start_time) + index * dt

            def velocity(at: torch.Tensor, at_time: float) -> torch.Tensor:
                times = torch.full(
                    (at.shape[0],),
                    at_time,
                    dtype=torch.float32,
                    device=at.device,
                )
                return velocity_model(at, times, nfp, mask)

            k1 = velocity(value, time_value)
            k2 = velocity(value + 0.5 * dt * k1, time_value + 0.5 * dt)
            k3 = velocity(value + 0.5 * dt * k2, time_value + 0.5 * dt)
            k4 = velocity(value + dt * k3, time_value + dt)
            value = value + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            if mask is not None:
                value = value * mask[..., None]
        return value

    value = state
    if use_checkpoint:
        for first_step in range(0, steps, checkpoint_steps):
            value = checkpoint(
                lambda current, offset=first_step: integrate_chunk(
                    current, offset, checkpoint_steps
                ),
                value,
                use_reentrant=False,
            )
    else:
        value = integrate_chunk(value, 0, steps)
    return value


def decode_physical_vjp(
    model,
    normalizer: CoilNormalizer,
    noise: np.ndarray,
    physical_cotangent: np.ndarray,
    *,
    nfp: int,
    device: torch.device,
    rk4_steps: int = 256,
    checkpoint_steps: int = 8,
    use_checkpoint: bool = False,
) -> tuple[np.ndarray, np.ndarray, FlowVjpDiagnostics]:
    values = np.asarray(noise, dtype=np.float32)
    cotangent = np.asarray(physical_cotangent, dtype=np.float32)
    if values.ndim == 2:
        values = values[None]
    if cotangent.ndim == 2:
        cotangent = cotangent[None]
    if values.shape != cotangent.shape or values.ndim != 3 or values.shape[-1] != 100:
        raise ValueError("noise and physical cotangent must share shape (batch, coils, 100)")
    if values.shape[0] != 1:
        raise ValueError("the first validated VJP path supports one center at a time")
    key = (int(nfp), int(values.shape[1]))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    latent = torch.from_numpy(values).to(device=device, dtype=torch.float32).requires_grad_(True)
    nfp_tensor = torch.full((1,), int(nfp), dtype=torch.long, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline_memory_allocated_bytes = int(torch.cuda.memory_allocated(device))
    else:
        baseline_memory_allocated_bytes = 0
    started = time.perf_counter()
    normalized = integrate_flow_differentiable(
        model,
        latent,
        nfp_tensor,
        steps=rk4_steps,
        checkpoint_steps=checkpoint_steps,
        use_checkpoint=use_checkpoint,
    )
    physical, dominant, dominant_sign = inverse_normalizer_torch(normalized, normalizer, key)
    torch.cuda.synchronize(device)
    decode_wall_s = time.perf_counter() - started
    forward_peak_memory_allocated_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    objective = torch.sum(
        physical * torch.from_numpy(cotangent).to(device=device, dtype=torch.float32)
    )
    started = time.perf_counter()
    (latent_gradient,) = torch.autograd.grad(objective, latent)
    torch.cuda.synchronize(device)
    backward_wall_s = time.perf_counter() - started
    total_peak_memory_allocated_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    diagnostics = FlowVjpDiagnostics(
        decode_wall_s=float(decode_wall_s),
        backward_wall_s=float(backward_wall_s),
        use_checkpoint=bool(use_checkpoint),
        checkpoint_steps=int(checkpoint_steps),
        rk4_steps=int(rk4_steps),
        baseline_memory_allocated_bytes=baseline_memory_allocated_bytes,
        forward_peak_memory_allocated_bytes=forward_peak_memory_allocated_bytes,
        total_peak_memory_allocated_bytes=total_peak_memory_allocated_bytes,
        dominant_current_index=tuple(int(value) for value in dominant.detach().cpu().tolist()),
        dominant_current_sign=tuple(int(value) for value in dominant_sign.detach().cpu().tolist()),
    )
    return (
        physical.detach().cpu().numpy(),
        latent_gradient.detach().cpu().numpy(),
        diagnostics,
    )
