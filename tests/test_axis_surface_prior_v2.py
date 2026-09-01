from __future__ import annotations

import numpy as np
import pytest

from flow_matching.axis_surface_prior import evaluate_fourier, gauss_linking_number, supported_conditions
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
    monotonicity_floor = 0.3 if preset == "compact_flexible" else 0.5
    assert prototype.metadata["scalar_monotonicity_lower_bound"] > monotonicity_floor
    assert prototype.metadata["curve_fit_abs_max_m"] < 2.0e-3

    curves = evaluate_fourier(prototype.tokens, samples=128)
    link = gauss_linking_number(curves[0], prototype.reference_axis[::16])
    assert abs(link) > 0.8


@pytest.mark.parametrize("nfp,n_base_coils", supported_conditions())
def test_balanced_v2_scoring_mode_covers_registered_conditions(nfp: int, n_base_coils: int) -> None:
    sample = sample_shaped_prior_prototype(
        seed=20260901 + 17 * nfp + n_base_coils,
        nfp=nfp,
        n_base_coils=n_base_coils,
        preset="balanced_stellarator",
        curve_samples=128,
        frame_samples=512,
        surface_phi_samples=32,
        surface_theta_samples=24,
        sample_role="registered_scoring",
    )
    assert sample.tokens.shape == (n_base_coils, 100)
    assert sample.metadata["format"] == "axis_surface_contour_prior_balanced_v2"
    assert sample.metadata["status"] == "registered_experimental_scoring"
    assert sample.metadata["sample_role"] == "registered_scoring"
    assert sample.metadata["scalar_monotonicity_lower_bound"] > 0.5
    assert abs(sample.metadata["reference_linking_number"]) > 0.75


def test_compact_flexible_v3_has_compact_radius_and_wider_shape_range() -> None:
    compact = sample_shaped_prior_prototype(
        seed=20260924,
        nfp=5,
        n_base_coils=4,
        preset="compact_flexible",
        surface_phi_samples=64,
        surface_theta_samples=32,
        sample_role="registered_scoring",
    )
    balanced = sample_shaped_prior_prototype(
        seed=20260924,
        nfp=5,
        n_base_coils=4,
        preset="balanced_stellarator",
        surface_phi_samples=64,
        surface_theta_samples=32,
        sample_role="registered_scoring",
    )
    assert compact.metadata["format"] == "axis_surface_contour_prior_compact_flexible_v3"
    assert 0.18 <= compact.metadata["parameters"]["minor_radius"] <= 0.22
    assert 0.17 <= compact.metadata["winding_section_mean_radius_m"] <= 0.23
    assert compact.metadata["parameters"]["elongation_variation"] > balanced.metadata["parameters"]["elongation_variation"]
    assert compact.metadata["parameters"]["cross_section_rotation"] > balanced.metadata["parameters"]["cross_section_rotation"]
    assert compact.metadata["parameters"]["helical_ripple"] > balanced.metadata["parameters"]["helical_ripple"]
    assert compact.metadata["scalar_monotonicity_lower_bound"] > 0.25
    assert abs(compact.metadata["reference_linking_number"]) > 0.75
