from __future__ import annotations

"""First-order score-gradient Flow update utilities.

The native score gradient is measured in the exact data coordinates used by
the R04 Adam probe.  The Flow model uses the distilled-prior coordinates, so
the small current-channel canonicalization Jacobian is handled explicitly
instead of silently taking a dot product between incompatible coordinates.
"""

from typing import Callable

import numpy as np
import torch
from torch import nn

from flow_matching.data import CoilNormalizer, canonicalize_currents


N_BASE_COILS = 3
TOKEN_DIM = 100
CURRENT_INDEX = TOKEN_DIM - 1


def _optimizer_coordinates_from_flow(
    flow_tokens: np.ndarray,
    flow_normalizer: CoilNormalizer,
    optimizer_normalizer: CoilNormalizer,
    *,
    current_l1_a: float,
    nfp: int,
) -> np.ndarray:
    values = np.asarray(flow_tokens, dtype=np.float32)
    if values.shape != (N_BASE_COILS, TOKEN_DIM):
        raise ValueError("flow tokens must have shape (3,100)")
    physical = flow_normalizer.inverse(values[None], (nfp, N_BASE_COILS))[0]
    canonical = canonicalize_currents(physical[None], current_l1_a)[0]
    return ((canonical - optimizer_normalizer.mean) / optimizer_normalizer.std).astype(
        np.float64
    )


def map_score_gradient_to_flow(
    flow_tokens: np.ndarray,
    optimizer_gradient: np.ndarray,
    flow_normalizer: CoilNormalizer,
    optimizer_normalizer: CoilNormalizer,
    *,
    current_l1_a: float,
    nfp: int,
    current_fd_step: float = 1.0e-4,
) -> np.ndarray:
    """Map an Adam-coordinate score gradient into Flow coordinates.

    Geometry channels are affine and use the exact diagonal Jacobian.  The
    three current channels pass through L1/sign canonicalization, so their
    3x3 local Jacobian is evaluated by a cheap central difference of the
    coordinate transform itself.  No native score calls are made here.
    """

    x = np.asarray(flow_tokens, dtype=np.float64)
    gradient = np.asarray(optimizer_gradient, dtype=np.float64)
    if x.shape != (N_BASE_COILS, TOKEN_DIM) or gradient.shape != x.shape:
        raise ValueError("flow tokens and optimizer gradient must both have shape (3,100)")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(gradient)):
        raise ValueError("flow tokens and optimizer gradient must be finite")
    if current_fd_step <= 0.0:
        raise ValueError("current finite-difference step must be positive")

    mapped = np.zeros_like(gradient)
    geometry_ratio = flow_normalizer.std[:CURRENT_INDEX] / optimizer_normalizer.std[
        :CURRENT_INDEX
    ]
    mapped[:, :CURRENT_INDEX] = gradient[:, :CURRENT_INDEX] * geometry_ratio[None, :]

    base_current = x[:, CURRENT_INDEX].copy()

    def current_map(current_values: np.ndarray) -> np.ndarray:
        trial = x.copy()
        trial[:, CURRENT_INDEX] = current_values
        return _optimizer_coordinates_from_flow(
            trial.astype(np.float32),
            flow_normalizer,
            optimizer_normalizer,
            current_l1_a=current_l1_a,
            nfp=nfp,
        )[:, CURRENT_INDEX]

    jacobian = np.empty((N_BASE_COILS, N_BASE_COILS), dtype=np.float64)
    for column in range(N_BASE_COILS):
        plus = base_current.copy()
        minus = base_current.copy()
        plus[column] += current_fd_step
        minus[column] -= current_fd_step
        jacobian[:, column] = (current_map(plus) - current_map(minus)) / (
            2.0 * current_fd_step
        )
    mapped[:, CURRENT_INDEX] = jacobian.T @ gradient[:, CURRENT_INDEX]
    if not np.all(np.isfinite(mapped)):
        raise ValueError("mapped score gradient is nonfinite")
    return mapped.astype(np.float32)


def coordinate_gradient_directional_check(
    flow_tokens: np.ndarray,
    optimizer_gradient: np.ndarray,
    flow_gradient: np.ndarray,
    flow_normalizer: CoilNormalizer,
    optimizer_normalizer: CoilNormalizer,
    *,
    current_l1_a: float,
    nfp: int,
    direction: np.ndarray,
    step: float = 1.0e-5,
) -> float:
    """Return relative error of the coordinate-chain directional identity."""

    x = np.asarray(flow_tokens, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    if x.shape != (N_BASE_COILS, TOKEN_DIM) or direction.shape != x.shape:
        raise ValueError("flow tokens and direction must both have shape (3,100)")
    plus = _optimizer_coordinates_from_flow(
        (x + step * direction).astype(np.float32),
        flow_normalizer,
        optimizer_normalizer,
        current_l1_a=current_l1_a,
        nfp=nfp,
    )
    minus = _optimizer_coordinates_from_flow(
        (x - step * direction).astype(np.float32),
        flow_normalizer,
        optimizer_normalizer,
        current_l1_a=current_l1_a,
        nfp=nfp,
    )
    coordinate_direction = (plus - minus) / (2.0 * step)
    left = float(np.sum(np.asarray(optimizer_gradient, dtype=np.float64) * coordinate_direction))
    right = float(np.sum(np.asarray(flow_gradient, dtype=np.float64) * direction))
    return abs(left - right) / max(abs(left), abs(right), 1.0e-12)


def cfm_loss_terms(
    model: nn.Module,
    data: torch.Tensor,
    *,
    feature_weights: torch.Tensor,
    noise: torch.Tensor,
    time_value: torch.Tensor,
) -> torch.Tensor:
    """Per-sample Flow matching losses for supplied Monte Carlo draws."""

    if data.ndim != 3 or data.shape[1:] != (N_BASE_COILS, TOKEN_DIM):
        raise ValueError("data must have shape (batch,3,100)")
    if noise.shape != data.shape or time_value.shape != (len(data),):
        raise ValueError("noise/time shapes do not match data")
    mixed = (1.0 - time_value[:, None, None]) * noise + time_value[:, None, None] * data
    target = data - noise
    prediction = model(
        mixed,
        time_value,
        torch.full((len(data),), 8, dtype=torch.long, device=data.device),
    )
    weights = feature_weights.to(device=data.device, dtype=torch.float32)
    square = (prediction.float() - target.float()).square() * weights
    return square.sum(dim=(1, 2)) / (N_BASE_COILS * weights.sum()).clamp_min(1.0)


def _sample_noise_time(
    data: torch.Tensor, *, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    noise = torch.randn(
        data.shape,
        generator=generator,
        device=data.device,
        dtype=data.dtype,
    )
    time_value = torch.rand(
        (len(data),),
        generator=generator,
        device=data.device,
        dtype=torch.float32,
    )
    return noise, time_value


def valid_flow_terms_with_transport(
    model: nn.Module,
    data: torch.Tensor,
    score_gradient: torch.Tensor,
    *,
    feature_weights: torch.Tensor,
    monte_carlo_samples: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one ordinary and one transport term per valid sample.

    ``create_graph=True`` is intentional: the outer Flow optimizer needs the
    parameter derivative of ``g^T grad_x ell``.  The score gradient is treated
    as a detached numerical observation and receives no gradient.
    """

    if len(data) == 0 or score_gradient.shape != data.shape:
        raise ValueError("valid data and score gradient must be nonempty and aligned")
    if monte_carlo_samples < 1:
        raise ValueError("monte_carlo_samples must be positive")
    expanded = (
        data[:, None]
        .expand(-1, monte_carlo_samples, -1, -1)
        .reshape(-1, N_BASE_COILS, TOKEN_DIM)
        .detach()
        .requires_grad_(True)
    )
    expanded_gradient = (
        score_gradient.detach()[:, None]
        .expand(-1, monte_carlo_samples, -1, -1)
        .reshape_as(expanded)
    )
    noise, time_value = _sample_noise_time(expanded, generator=generator)
    losses = cfm_loss_terms(
        model,
        expanded,
        feature_weights=feature_weights,
        noise=noise,
        time_value=time_value,
    )
    input_gradient = torch.autograd.grad(
        losses.sum(), expanded, create_graph=True, retain_graph=True
    )[0]
    transport = (input_gradient * expanded_gradient).sum(dim=(1, 2))
    ordinary = losses.reshape(-1, monte_carlo_samples).mean(dim=1)
    transport = transport.reshape(-1, monte_carlo_samples).mean(dim=1)
    return ordinary, transport


def ordinary_flow_terms(
    model: nn.Module,
    data: torch.Tensor,
    *,
    feature_weights: torch.Tensor,
    monte_carlo_samples: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if len(data) == 0:
        raise ValueError("ordinary Flow terms require a nonempty batch")
    expanded = (
        data[:, None]
        .expand(-1, monte_carlo_samples, -1, -1)
        .reshape(-1, N_BASE_COILS, TOKEN_DIM)
    )
    noise, time_value = _sample_noise_time(expanded, generator=generator)
    losses = cfm_loss_terms(
        model,
        expanded,
        feature_weights=feature_weights,
        noise=noise,
        time_value=time_value,
    )
    return losses.reshape(-1, monte_carlo_samples).mean(dim=1)


__all__ = [
    "coordinate_gradient_directional_check",
    "map_score_gradient_to_flow",
    "cfm_loss_terms",
    "ordinary_flow_terms",
    "valid_flow_terms_with_transport",
]
