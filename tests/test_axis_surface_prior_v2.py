from __future__ import annotations

import numpy as np
import pytest

from flow_matching.axis_surface_prior import evaluate_fourier, gauss_linking_number
from flow_matching.axis_surface_prior_v2 import prototype_presets, sample_shaped_prior_prototype


@pytest.mark.parametrize("preset", prototype_presets())
def test_v2_prototype_has_visible_axis_and_surface_variation(preset: str) -> None:
    prototype = sample_shaped_prior_prototype(
        seed=20260901,
        nfp=4,
        n_base_coils=3,
        preset=preset,
        surface_phi_samples=96,
        surface_theta_samples=48,
    )
    assert prototype.tokens.shape == (3, 100)
    assert np.isfinite(prototype.tokens).all()
    assert prototype.metadata["status"] == "geometry_only_awaiting_user_acceptance"
    assert prototype.metadata["axis_R_peak_to_peak_m"] > 0.22
    assert prototype.metadata["axis_Z_peak_to_peak_m"] > 0.20
    assert prototype.metadata["winding_section_mean_radius_peak_to_peak_m"] > 0.04
    assert prototype.metadata["scalar_monotonicity_lower_bound"] > 0.5
    assert prototype.metadata["curve_fit_abs_max_m"] < 2.0e-3

    curves = evaluate_fourier(prototype.tokens, samples=128)
    link = gauss_linking_number(curves[0], prototype.reference_axis[::16])
    assert abs(link) > 0.8
