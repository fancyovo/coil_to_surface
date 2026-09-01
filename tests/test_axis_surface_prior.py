from __future__ import annotations

import numpy as np
import pytest

from flow_matching.axis_surface_prior import (
    TOKEN_DIM,
    evaluate_fourier,
    gauss_linking_number,
    sample_axis_surface_prior,
    supported_conditions,
)


def test_supported_conditions_exclude_five_coils() -> None:
    assert len(supported_conditions()) == 26
    assert all(n_base_coils <= 4 for _, n_base_coils in supported_conditions())
    with pytest.raises(ValueError, match="nc<=4"):
        sample_axis_surface_prior(seed=1, nfp=4, n_base_coils=5)


@pytest.mark.parametrize(
    ("nfp", "n_base_coils", "family"),
    ((4, 1, "near_circular"), (2, 4, "balanced"), (8, 4, "helical")),
)
def test_prior_is_finite_simple_and_linked(nfp: int, n_base_coils: int, family: str) -> None:
    sample = sample_axis_surface_prior(
        seed=391,
        nfp=nfp,
        n_base_coils=n_base_coils,
        family=family,
    )
    assert sample.tokens.shape == (n_base_coils, TOKEN_DIM)
    assert np.isfinite(sample.tokens).all()
    assert sample.metadata["independent_of_quasr"] is True
    assert sample.metadata["scalar_monotonicity_lower_bound"] > 0.5
    assert sample.metadata["curve_fit_rms_max_m"] < 2.0e-3
    assert sample.metadata["curve_fit_abs_max_m"] < 1.0e-2
    assert abs(sample.metadata["reference_linking_number"]) > 0.85
    curves = evaluate_fourier(sample.tokens, samples=128)
    axis = sample.reference_axis[::16]
    link = gauss_linking_number(curves[0], axis)
    assert abs(link) > 0.8

    coefficients = sample.tokens[:, :99].reshape(n_base_coils, 3, 33)
    mode_energy = coefficients[:, :, 1:] ** 2
    high_mode_fraction = mode_energy[:, :, 17:].sum() / mode_energy.sum()
    assert high_mode_fraction < 0.02


def test_prior_is_bitwise_reproducible() -> None:
    first = sample_axis_surface_prior(seed=20260901, nfp=6, n_base_coils=3, family="balanced")
    second = sample_axis_surface_prior(seed=20260901, nfp=6, n_base_coils=3, family="balanced")
    assert np.array_equal(first.tokens, second.tokens)
    assert first.metadata == second.metadata
