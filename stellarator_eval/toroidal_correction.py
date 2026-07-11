from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class ToroidalMode:
    m: int
    n: int


@dataclass
class ToroidalCorrectionFit:
    modes: list[ToroidalMode]
    cos_coeffs: np.ndarray
    sin_coeffs: np.ndarray
    iota: float
    nfp: int
    mpol: int
    ntor: int
    regularization: float
    diagnostics: dict[str, float | int]


def build_toroidal_modes(mpol: int, ntor: int) -> list[ToroidalMode]:
    """Return a non-redundant real Fourier basis without the constant gauge."""
    if mpol < 0 or ntor < 0:
        raise ValueError("mpol and ntor must be non-negative")
    modes = [ToroidalMode(0, n) for n in range(1, ntor + 1)]
    modes.extend(
        ToroidalMode(m, n)
        for m in range(1, mpol + 1)
        for n in range(-ntor, ntor + 1)
    )
    return modes


def evaluate_toroidal_correction(
    fit: ToroidalCorrectionFit,
    theta,
    phi,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate nu, its coordinate derivatives, and D nu."""
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    theta, phi = np.broadcast_arrays(theta, phi)
    nu = np.zeros_like(theta)
    nu_theta = np.zeros_like(theta)
    nu_phi = np.zeros_like(theta)
    for mode, cosine_coeff, sine_coeff in zip(
        fit.modes, fit.cos_coeffs, fit.sin_coeffs
    ):
        phase = TWOPI * (mode.m * theta - mode.n * fit.nfp * phi)
        cosine = np.cos(phase)
        sine = np.sin(phase)
        value = cosine_coeff * cosine + sine_coeff * sine
        phase_theta = TWOPI * mode.m
        phase_phi = -TWOPI * mode.n * fit.nfp
        phase_derivative = -cosine_coeff * sine + sine_coeff * cosine
        nu += value
        nu_theta += phase_theta * phase_derivative
        nu_phi += phase_phi * phase_derivative
    along_field = nu_phi + fit.iota * nu_theta
    return nu, nu_theta, nu_phi, along_field


def fit_toroidal_correction(
    theta,
    phi,
    target,
    *,
    iota: float,
    nfp: int,
    mpol: int,
    ntor: int,
    regularization: float = 0.0,
    weight_power: float = 2.0,
    resonance_tolerance: float = 1e-10,
) -> ToroidalCorrectionFit:
    """Fit D nu = target by orthogonal Fourier least squares.

    The samples must cover a uniform tensor grid over one field period in phi
    and one full period in theta. The constant part of target is reported but
    omitted because a periodic magnetic differential equation cannot fit it.
    """
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    target = np.asarray(target, dtype=float)
    theta, phi, target = np.broadcast_arrays(theta, phi, target)
    modes = build_toroidal_modes(mpol, ntor)
    centered = target - np.mean(target)
    cos_coeffs = np.zeros(len(modes))
    sin_coeffs = np.zeros(len(modes))
    skipped_resonances = 0
    min_divisor = float("inf")

    for index, mode in enumerate(modes):
        phase = TWOPI * (mode.m * theta - mode.n * nfp * phi)
        target_cos = 2.0 * float(np.mean(centered * np.cos(phase)))
        target_sin = 2.0 * float(np.mean(centered * np.sin(phase)))
        divisor = TWOPI * (mode.m * iota - mode.n * nfp)
        min_divisor = min(min_divisor, abs(divisor))
        if abs(divisor) <= resonance_tolerance:
            skipped_resonances += 1
            continue
        mode_weight = (
            1.0 + mode.m**2 + (mode.n * nfp) ** 2
        ) ** (0.5 * weight_power)
        denominator = divisor * divisor + regularization * mode_weight * mode_weight
        cos_coeffs[index] = -divisor * target_sin / denominator
        sin_coeffs[index] = divisor * target_cos / denominator

    fit = ToroidalCorrectionFit(
        modes=modes,
        cos_coeffs=cos_coeffs,
        sin_coeffs=sin_coeffs,
        iota=float(iota),
        nfp=int(nfp),
        mpol=int(mpol),
        ntor=int(ntor),
        regularization=float(regularization),
        diagnostics={},
    )
    nu, _, _, fitted = evaluate_toroidal_correction(fit, theta, phi)
    residual = fitted - centered
    centered_norm = np.linalg.norm(centered)
    fit.diagnostics = {
        "sample_count": int(target.size),
        "mode_count": int(len(modes)),
        "target_mean": float(np.mean(target)),
        "target_rms": float(np.sqrt(np.mean(target * target))),
        "centered_target_rms": float(np.sqrt(np.mean(centered * centered))),
        "residual_rms": float(np.sqrt(np.mean(residual * residual))),
        "relative_l2": float(
            np.linalg.norm(residual) / max(centered_norm, 1e-30)
        ),
        "nu_rms_turns": float(np.sqrt(np.mean(nu * nu))),
        "nu_max_abs_turns": float(np.max(np.abs(nu))),
        "min_mode_divisor": float(min_divisor),
        "skipped_resonance_count": int(skipped_resonances),
    }
    return fit
