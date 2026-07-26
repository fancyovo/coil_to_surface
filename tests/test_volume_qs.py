import numpy as np

from stellarator_eval.config import VolumeQSConfig
from stellarator_eval.psi import PolyMode, PsiModel, build_modes, psi_and_gradient
from stellarator_eval.volume_qs import (
    FluxScaleFit,
    StraightFieldFit,
    calibrate_toroidal_flux_gpu,
    compute_f_c,
    evaluate_psi_tensor_numpy,
    _surface_radius_on_rays,
)


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
