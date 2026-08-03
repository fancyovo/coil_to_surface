import numpy as np

from stellarator_eval.alpha_clebsch import (
    AlphaFitResult,
    ClebschMode,
    alpha_coordinates_from_volume_points,
    build_clebsch_modes,
    disjoint_train_validation_indices,
    evaluate_alpha_fit,
    evaluate_lambda,
    zernike_radial,
)


def test_clebsch_basis_removes_flux_function_gauge():
    modes = build_clebsch_modes(6, 6, 4)
    assert modes
    assert all(not (mode.m == 0 and mode.n == 0) for mode in modes)
    assert len(modes) == len({(mode.l, mode.m, mode.n, mode.kind) for mode in modes})


def test_zernike_radial_known_polynomials():
    rho = np.linspace(0.0, 1.0, 9)
    np.testing.assert_allclose(zernike_radial(rho, 2, 0), 2.0 * rho**2 - 1.0)
    np.testing.assert_allclose(zernike_radial(rho, 3, 1), 3.0 * rho**3 - 2.0 * rho)
    np.testing.assert_allclose(zernike_radial(rho, 4, 2), 4.0 * rho**4 - 3.0 * rho**2)


def test_exact_alpha_reconstruction_has_zero_residual():
    count = 200
    rho = np.linspace(0.08, 1.0, count)
    theta = np.linspace(-np.pi, np.pi, count)
    phi = np.linspace(0.0, 2.0 * np.pi / 3.0, count, endpoint=False)
    modes = [
        ClebschMode(1, 1, 0, "sin"),
        ClebschMode(2, 0, 1, "cos"),
    ]
    result = AlphaFitResult(
        modes=modes,
        lambda_coeffs=np.asarray([0.12, -0.04]),
        iota_coeffs=np.asarray([-0.48, 0.03]),
        radial_order=2,
        poloidal_order=1,
        toroidal_order=1,
        iota_degree=1,
        diagnostics={},
    )
    _, lambda_theta, lambda_phi = evaluate_lambda(result, rho, theta, phi, nfp=3)
    cross_theta = np.column_stack(
        [np.ones(count), 0.2 * np.ones(count), -0.1 * np.ones(count)]
    )
    cross_phi = np.column_stack(
        [0.05 * np.ones(count), -0.4 * np.ones(count), 0.3 * np.ones(count)]
    )
    iota = result.iota(rho)
    B = (1.0 + lambda_theta)[:, None] * cross_theta + (
        lambda_phi - iota
    )[:, None] * cross_phi
    coordinates = {
        "rho": rho,
        "theta": theta,
        "phi": phi,
        "cross_theta": cross_theta,
        "cross_phi": cross_phi,
        "grad_psi": np.column_stack(
            [np.zeros(count), np.zeros(count), np.ones(count)]
        ),
    }
    metrics, _ = evaluate_alpha_fit(result, coordinates, B, nfp=3)
    assert metrics["relative_l2"] < 1e-14


def test_accelerated_alpha_sampling_uses_disjoint_splits():
    training, validation = disjoint_train_validation_indices(180, 120, 60)
    assert len(training) == 120
    assert len(validation) == 60
    assert len(np.intersect1d(training, validation)) == 0
    np.testing.assert_array_equal(np.sort(np.r_[training, validation]), np.arange(180))


def test_accelerated_alpha_coordinates_preserve_physical_flux_gradient():
    points = {
        "rho": np.asarray([0.2, 0.8]),
        "theta": np.asarray([0.1, -0.3]),
        "phi": np.asarray([0.0, 0.2]),
        "grad_psi": np.asarray([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]]),
        "grad_theta": np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        "grad_phi": np.asarray([[0.0, 0.0, 1.0], [0.0, 2.0, 0.0]]),
    }
    coordinates = alpha_coordinates_from_volume_points(points)
    np.testing.assert_array_equal(coordinates["grad_psi"], points["grad_psi"])
    np.testing.assert_allclose(
        coordinates["cross_theta"],
        np.cross(points["grad_psi"], points["grad_theta"]),
    )
    np.testing.assert_allclose(
        coordinates["cross_phi"],
        np.cross(points["grad_psi"], points["grad_phi"]),
    )
