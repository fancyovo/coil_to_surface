import numpy as np

from stellarator_eval.config import VolumeQSConfig
from stellarator_eval.psi import PolyMode, PsiModel, build_modes, psi_and_gradient
from stellarator_eval.volume_qs import (
    _volume_candidate_lattice_shape,
    FluxScaleFit,
    StraightFieldFit,
    calibrate_toroidal_flux_gpu,
    compute_f_c,
    evaluate_psi_tensor_numpy,
    _minimum_volume_sample_count,
    _physical_volume_weights,
    _surface_radius_on_rays,
    _budget_flux_levels,
    vacuum_G,
)


def test_volume_candidate_oversampling_preserves_default_and_scales_pool() -> None:
    default_shape = _volume_candidate_lattice_shape(180_000, 96, 1.25)
    expanded_shape = _volume_candidate_lattice_shape(180_000, 96, 1.6)

    assert np.prod(default_shape) == 225_792
    assert np.prod(expanded_shape) >= 288_000
    assert expanded_shape[0] == default_shape[0] == 96
    assert np.prod(expanded_shape) > np.prod(default_shape)


class UniformToroidalField:
    def __init__(self, magnitude):
        self.magnitude = magnitude

    def eval_B_grad(self, xyz, precision="fp32"):
        return self.eval_B(xyz, precision=precision), np.zeros((len(xyz), 3, 3))

    def eval_B(self, xyz, precision="fp32"):
        xyz = np.asarray(xyz)
        phi = np.arctan2(xyz[:, 1], xyz[:, 0])
        return self.magnitude * np.column_stack(
            [-np.sin(phi), np.cos(phi), np.zeros_like(phi)]
        )


def test_star_shaped_volume_weights_include_boundary_radius_squared():
    major_radius = np.asarray([0.8, 1.0, 1.2])
    boundary_radius = np.asarray([0.1, 0.2, 0.3])
    weights = _physical_volume_weights(major_radius, boundary_radius)
    np.testing.assert_allclose(weights, major_radius * boundary_radius**2)


def test_volume_sampler_requires_fixed_budget_and_95_percent_validity():
    assert _minimum_volume_sample_count(100_000, 30_000, 125_952) == 119_655
    assert _minimum_volume_sample_count(200_000, 30_000, 201_000) == 200_000


def test_volume_sampler_can_use_only_the_fixed_budget_for_full_evaluation():
    assert _minimum_volume_sample_count(
        180_000, 180_000, 225_792, minimum_candidate_valid_fraction=0.0
    ) == 180_000
    assert _minimum_volume_sample_count(
        180_000, 180_000, 225_792, minimum_candidate_valid_fraction=0.8
    ) == 180_634


def make_model():
    modes = build_modes(3, 1)
    phi_axis = np.linspace(0.0, 2.0 * np.pi / 3.0, 64, endpoint=False)
    return PsiModel(
        coeffs=np.linspace(-0.02, 0.03, len(modes)),
        modes=modes,
        nfp=3,
        a=0.2,
        phi_axis=phi_axis,
        R_axis=1.1 + 0.01 * np.cos(3.0 * phi_axis),
        Z_axis=0.01 * np.sin(3.0 * phi_axis),
        R_axis_phi=-0.03 * np.sin(3.0 * phi_axis),
        Z_axis_phi=0.03 * np.cos(3.0 * phi_axis),
        fit_info={},
    )


def test_tensor_psi_matches_reference_implementation():
    rng = np.random.default_rng(12)
    model = make_model()
    phi = rng.uniform(0.0, 2.0 * np.pi / model.nfp, 300)
    ra, za, _, _ = model.axis_at(phi)
    R = ra + rng.uniform(-0.1, 0.1, len(phi))
    Z = za + rng.uniform(-0.1, 0.1, len(phi))
    reference = psi_and_gradient(model, R, Z, phi)
    tensor = evaluate_psi_tensor_numpy(model, R, Z, phi)
    for actual, expected in zip(tensor, reference):
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


def test_flux_scale_polynomial_integrates_derivative():
    fit = FluxScaleFit(np.asarray([2.0, -0.5, 0.25]), 0.4, {})
    s = np.linspace(0.0, 0.4, 1000)
    numerical = np.gradient(fit.evaluate(s), s)
    np.testing.assert_allclose(numerical[1:-1], fit.derivative(s)[1:-1], rtol=2e-6, atol=2e-6)


def test_surface_radius_solver_lands_on_requested_level():
    phi_axis = np.linspace(0.0, 2.0 * np.pi / 3.0, 32, endpoint=False)
    model = PsiModel(
        coeffs=np.asarray([1.0]),
        modes=[PolyMode(0, 2, 0, "cos")],
        nfp=3,
        a=0.2,
        phi_axis=phi_axis,
        R_axis=np.ones_like(phi_axis),
        Z_axis=np.zeros_like(phi_axis),
        R_axis_phi=np.zeros_like(phi_axis),
        Z_axis_phi=np.zeros_like(phi_axis),
        fit_info={},
    )
    theta = np.linspace(0.0, 2.0 * np.pi, 257, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi / model.nfp, len(theta), endpoint=False)
    radius, residual = _surface_radius_on_rays(model, 0.04, theta, phi)
    ra, za, _, _ = model.axis_at(phi)
    values, *_ = evaluate_psi_tensor_numpy(
        model,
        ra + radius * np.cos(theta),
        za + radius * np.sin(theta),
        phi,
    )
    assert np.max(residual) < 1e-10
    np.testing.assert_allclose(values, 0.04, rtol=0.0, atol=1e-10)


def test_surface_radius_solver_handles_higher_order_radial_terms():
    phi_axis = np.linspace(0.0, 2.0 * np.pi / 4.0, 32, endpoint=False)
    model = PsiModel(
        coeffs=np.asarray([1.0, 0.08, -0.015]),
        modes=[
            PolyMode(0, 2, 0, "cos"),
            PolyMode(4, 0, 0, "cos"),
            PolyMode(2, 2, 1, "cos"),
        ],
        nfp=4,
        a=0.2,
        phi_axis=phi_axis,
        R_axis=np.ones_like(phi_axis),
        Z_axis=np.zeros_like(phi_axis),
        R_axis_phi=np.zeros_like(phi_axis),
        Z_axis_phi=np.zeros_like(phi_axis),
        fit_info={},
    )
    theta = np.linspace(0.0, 2.0 * np.pi, 513, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi / model.nfp, len(theta), endpoint=False)
    radius, residual = _surface_radius_on_rays(model, 0.36, theta, phi)
    ra, za, _, _ = model.axis_at(phi)
    values, *_ = evaluate_psi_tensor_numpy(
        model,
        ra + radius * np.cos(theta),
        za + radius * np.sin(theta),
        phi,
    )
    assert np.max(residual) < 1e-10
    np.testing.assert_allclose(values, 0.36, rtol=0.0, atol=1e-10)


def test_flux_level_budget_keeps_outer_neighbors_and_spaced_fallbacks():
    levels = [0.001, 0.002, 0.004, 0.008, 0.012, 0.02, 0.04, 0.08, 0.12, 0.16,
              0.25, 0.36, 0.49, 0.64, 0.81]
    selected = _budget_flux_levels(levels)
    assert selected == [0.81, 0.64, 0.36, 0.16, 0.08]


def test_batched_flux_calibration_for_circular_surfaces():
    phi_axis = np.linspace(0.0, 2.0 * np.pi / 3.0, 32, endpoint=False)
    model = PsiModel(
        coeffs=np.asarray([1.0]),
        modes=[PolyMode(0, 2, 0, "cos")],
        nfp=3,
        a=0.2,
        phi_axis=phi_axis,
        R_axis=np.ones_like(phi_axis),
        Z_axis=np.zeros_like(phi_axis),
        R_axis_phi=np.zeros_like(phi_axis),
        Z_axis_phi=np.zeros_like(phi_axis),
        fit_info={},
    )
    config = VolumeQSConfig(
        s_edge=0.25,
        flux_level_count=7,
        flux_phi_count=4,
        flux_theta_count=128,
        flux_radial_quadrature=16,
        flux_degree=3,
    )
    magnitude = 1.7
    calibration = calibrate_toroidal_flux_gpu(model, UniformToroidalField(magnitude), config)
    expected_derivative = magnitude * model.a**2 / 2.0
    np.testing.assert_allclose(calibration.derivative(config.s_edge), expected_derivative, rtol=2e-6)
    np.testing.assert_allclose(
        calibration.psi_edge, expected_derivative * config.s_edge, rtol=2e-6
    )
    assert calibration.diagnostics["section_relative_std_max"] < 1e-12
    expected_volume = 2.0 * np.pi**2 * 1.0 * 0.1**2
    np.testing.assert_allclose(
        calibration.diagnostics["boundary_volume_edge"], expected_volume, rtol=2e-6
    )
    np.testing.assert_allclose(
        calibration.diagnostics["boundary_effective_minor_radius_edge"], 0.1, rtol=2e-6
    )


def test_batched_flux_calibration_preserves_explicit_levels():
    phi_axis = np.linspace(0.0, 2.0 * np.pi / 3.0, 32, endpoint=False)
    model = PsiModel(
        coeffs=np.asarray([1.0]),
        modes=[PolyMode(0, 2, 0, "cos")],
        nfp=3,
        a=0.2,
        phi_axis=phi_axis,
        R_axis=np.ones_like(phi_axis),
        Z_axis=np.zeros_like(phi_axis),
        R_axis_phi=np.zeros_like(phi_axis),
        Z_axis_phi=np.zeros_like(phi_axis),
        fit_info={},
    )
    config = VolumeQSConfig(s_edge=0.25, flux_phi_count=2, flux_theta_count=32)
    levels = np.asarray([0.01, 0.04, 0.09, 0.16, 0.25])
    calibration = calibrate_toroidal_flux_gpu(
        model, UniformToroidalField(1.7), config, levels=levels
    )
    np.testing.assert_array_equal(calibration.s_knots, levels)


def test_compute_f_c_matches_direct_formula():
    count = 20
    rng = np.random.default_rng(91)
    B = rng.normal(size=(count, 3))
    grad_B = rng.normal(size=(count, 3, 3))
    grad_s = rng.normal(size=(count, 3))
    points = {
        "s": np.linspace(0.01, 0.16, count),
        "rho": np.linspace(0.1, 1.0, count),
        "grad_s": grad_s,
    }
    alpha = StraightFieldFit([], np.empty(0), np.asarray([-0.47]), {})
    flux = FluxScaleFit(np.asarray([0.002]), 0.16, {})
    result = compute_f_c(points, B, grad_B, alpha, flux, M=1, N=3, G=5.2)
    magnitude = np.linalg.norm(B, axis=1)
    grad_magnitude = np.einsum("nij,ni->nj", grad_B, B) / magnitude[:, None]
    grad_psi = 0.002 * grad_s
    A = np.sum(np.cross(B, grad_psi) * grad_magnitude, axis=1)
    C = np.sum(B * grad_magnitude, axis=1)
    expected = (-0.47 - 3.0) * A - 5.2 * C
    np.testing.assert_allclose(result["f_C"], expected)


def test_vacuum_G_uses_signed_toroidal_flux():
    currents = np.asarray([2.0, -3.0, 5.0])
    positive = vacuum_G(currents, 4, 0.02)
    negative = vacuum_G(currents, 4, -0.02)
    linked_current = 2 * 4 * np.sum(np.abs(currents))
    expected = 4.0e-7 * np.pi * linked_current / (2.0 * np.pi)
    np.testing.assert_allclose(positive, expected, rtol=0.0, atol=1.0e-20)
    np.testing.assert_allclose(negative, -positive)


def test_normalized_f_c_is_invariant_under_global_current_reversal():
    count = 30
    scale = 1.7
    rng = np.random.default_rng(918)
    B = rng.normal(size=(count, 3))
    grad_B = rng.normal(size=(count, 3, 3))
    points = {
        "s": np.linspace(0.01, 0.16, count),
        "rho": np.linspace(0.1, 1.0, count),
        "grad_s": rng.normal(size=(count, 3)),
    }
    alpha = StraightFieldFit([], np.empty(0), np.asarray([1.23]), {})
    flux = FluxScaleFit(np.asarray([0.004, -0.001]), 0.16, {})
    G = 3.8
    reference = compute_f_c(points, B, grad_B, alpha, flux, M=1, N=3, G=G)
    reversed_result = compute_f_c(
        points,
        -scale * B,
        -scale * grad_B,
        alpha,
        FluxScaleFit(-scale * flux.coefficients, flux.s_edge, {}),
        M=1,
        N=3,
        G=-scale * G,
    )
    np.testing.assert_allclose(
        reversed_result["f_C_over_B3"],
        reference["f_C_over_B3"],
        rtol=2.0e-14,
        atol=2.0e-14,
    )
