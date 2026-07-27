import numpy as np

from stellarator_eval.toroidal_correction import (
    ToroidalCorrectionFit,
    build_toroidal_modes,
    evaluate_toroidal_correction,
    fit_toroidal_correction,
)


def test_toroidal_modes_have_no_constant_or_duplicates():
    modes = build_toroidal_modes(4, 3)
    assert modes
    assert all((mode.m, mode.n) != (0, 0) for mode in modes)
    assert len(modes) == len({(mode.m, mode.n) for mode in modes})
    assert all(mode.n > 0 for mode in modes if mode.m == 0)


def test_fourier_least_squares_recovers_exact_magnetic_derivative():
    nfp = 5
    iota = -0.43
    nphi = 31
    ntheta = 37
    phi = np.arange(nphi)[:, None] / (nfp * nphi)
    theta = np.arange(ntheta)[None, :] / ntheta
    modes = build_toroidal_modes(3, 2)
    rng = np.random.default_rng(17)
    expected = ToroidalCorrectionFit(
        modes=modes,
        cos_coeffs=rng.normal(scale=2e-3, size=len(modes)),
        sin_coeffs=rng.normal(scale=2e-3, size=len(modes)),
        iota=iota,
        nfp=nfp,
        mpol=3,
        ntor=2,
        regularization=0.0,
        diagnostics={},
    )
    expected_nu, _, _, target = evaluate_toroidal_correction(
        expected, theta, phi
    )
    fitted = fit_toroidal_correction(
        theta,
        phi,
        target,
        iota=iota,
        nfp=nfp,
        mpol=3,
        ntor=2,
    )
    actual_nu, _, _, actual_target = evaluate_toroidal_correction(
        fitted, theta, phi
    )
    np.testing.assert_allclose(actual_target, target, atol=2e-13)
    np.testing.assert_allclose(actual_nu, expected_nu, atol=2e-14)
    assert fitted.diagnostics["relative_l2"] < 1e-12


def test_coordinate_correction_preserves_alpha():
    rng = np.random.default_rng(23)
    phi = rng.uniform(0.0, 0.2, 100)
    theta = rng.uniform(0.0, 1.0, 100)
    nu = rng.normal(scale=0.01, size=100)
    iota = -0.57
    phi_b = phi + nu
    theta_b = theta + iota * nu
    np.testing.assert_allclose(theta_b - iota * phi_b, theta - iota * phi)
