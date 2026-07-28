from __future__ import annotations

import math

import torch


def curve_metrics(tokens: torch.Tensor, *, samples: int = 64) -> dict[str, torch.Tensor]:
    values = tokens.float()
    coefficients = values[..., :99].view(*values.shape[:-1], 3, 33)
    device = values.device
    t = torch.arange(samples, device=device, dtype=torch.float32) / samples
    modes = torch.arange(1, 17, device=device, dtype=torch.float32)
    angle = 2.0 * math.pi * modes[:, None] * t[None]
    sine = torch.sin(angle)
    cosine = torch.cos(angle)
    omega = 2.0 * math.pi * modes

    constant = coefficients[..., 0, None]
    sine_coeff = coefficients[..., 1::2]
    cosine_coeff = coefficients[..., 2::2]
    position = (
        constant
        + torch.einsum("...cm,mt->...ct", sine_coeff, sine)
        + torch.einsum("...cm,mt->...ct", cosine_coeff, cosine)
    )
    first = (
        torch.einsum("...cm,mt->...ct", sine_coeff * omega, cosine)
        - torch.einsum("...cm,mt->...ct", cosine_coeff * omega, sine)
    )
    second = -(
        torch.einsum("...cm,mt->...ct", sine_coeff * omega.square(), sine)
        + torch.einsum("...cm,mt->...ct", cosine_coeff * omega.square(), cosine)
    )
    position = position.movedim(-2, -1)
    first = first.movedim(-2, -1)
    second = second.movedim(-2, -1)
    speed = torch.linalg.vector_norm(first, dim=-1)
    curvature = torch.linalg.vector_norm(torch.linalg.cross(first, second, dim=-1), dim=-1) / speed.clamp_min(1.0e-8).pow(3)
    radius = torch.linalg.vector_norm(position, dim=-1)
    cylindrical_radius = torch.linalg.vector_norm(position[..., :2], dim=-1)
    mode_energy = sine_coeff.square() + cosine_coeff.square()
    total_energy = mode_energy.sum(dim=(-3, -2, -1)).clamp_min(1.0e-20)
    high_energy = mode_energy[..., 9:].sum(dim=(-3, -2, -1))
    batch = values.shape[0]
    flat_curvature = curvature.reshape(batch, -1)
    return {
        "finite": torch.isfinite(values).all(dim=(-2, -1)),
        "length_mean": speed.mean(dim=(-2, -1)),
        "curvature_p95": torch.quantile(flat_curvature, 0.95, dim=1),
        "curvature_max": flat_curvature.max(dim=1).values,
        "radius_max": radius.amax(dim=(-2, -1)),
        "axis_distance_min": cylindrical_radius.amin(dim=(-2, -1)),
        "high_mode_fraction": high_energy / total_energy,
        "current_l1_a": values[..., -1].abs().sum(dim=1),
    }


def reference_bounds(metrics: dict[str, torch.Tensor]) -> dict[str, tuple[float, float]]:
    bounds = {}
    for name in ("length_mean", "curvature_p95", "radius_max", "axis_distance_min", "high_mode_fraction"):
        values = metrics[name][torch.isfinite(metrics[name])]
        low = float(torch.quantile(values, 0.005).cpu())
        high = float(torch.quantile(values, 0.995).cpu())
        bounds[name] = (low, high)
    return bounds


def geometry_eligible(
    metrics: dict[str, torch.Tensor], bounds: dict[str, tuple[float, float]]
) -> torch.Tensor:
    eligible = metrics["finite"].clone()
    for name in ("length_mean", "radius_max", "axis_distance_min"):
        low, high = bounds[name]
        eligible &= metrics[name] >= 0.5 * low
        eligible &= metrics[name] <= 1.5 * high
    for name in ("curvature_p95", "high_mode_fraction"):
        _, high = bounds[name]
        eligible &= metrics[name] <= 2.0 * high + 1.0e-12
    return eligible


def summarize_metrics(
    metrics: dict[str, torch.Tensor], bounds: dict[str, tuple[float, float]] | None = None
) -> dict[str, float]:
    summary = {"finite_rate": float(metrics["finite"].float().mean().cpu())}
    if bounds is not None:
        summary["geometry_eligible_rate"] = float(
            geometry_eligible(metrics, bounds).float().mean().cpu()
        )
    for name, values in metrics.items():
        if name == "finite":
            continue
        finite = values[torch.isfinite(values)]
        summary[f"{name}_median"] = float(torch.median(finite).cpu()) if len(finite) else float("nan")
    return summary
