import math

import numpy as np

from scripts.evaluate_qh_equal_s_surface_qs_gpu import periodic_surface_area_density, polynomial_derivative


def test_polynomial_derivative_matches_direct_expression():
    coefficients = np.asarray([2.0, -3.0, 4.0])
    value = 0.25
    assert polynomial_derivative(coefficients, value) == 2.0 - 6.0 * value + 12.0 * value**2


def test_periodic_surface_area_density_recovers_torus_area():
    nfp = 4
    n_phi = 160
    n_theta = 192
    major_radius = 1.1
    minor_radius = 0.18
    phi_values = (np.arange(n_phi) + 0.371) * 2.0 * np.pi / (nfp * n_phi)
    theta_values = (np.arange(n_theta) + 0.613) * 2.0 * np.pi / n_theta
    phi, theta = np.meshgrid(phi_values, theta_values, indexing="ij")
    R = major_radius + minor_radius * np.cos(theta)
    Z = minor_radius * np.sin(theta)
    density = periodic_surface_area_density(R, Z, phi)
    numerical = float(np.sum(density) * (2.0 * np.pi / n_theta) * (2.0 * np.pi / (nfp * n_phi)) * nfp)
    exact = 4.0 * np.pi**2 * major_radius * minor_radius
    assert math.isclose(numerical, exact, rel_tol=3e-4)
