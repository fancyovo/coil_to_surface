from __future__ import annotations

import numpy as np

from stellarator_eval.axis import (
    _classify_map_topology,
    _topology_stability_margin,
    interp_periodic_hermite,
    rk4_period_samples,
)
from stellarator_eval.psi import PolyMode, PsiModel, psi_and_gradient


def test_axis_topology_margin_is_quality_not_existence_gate():
    normalized_trace = 1.9808281169448664
    assert _classify_map_topology(-normalized_trace, 1.0) == "elliptic"
    assert np.isclose(
        _topology_stability_margin(-normalized_trace, 1.0),
        2.0 - normalized_trace,
    )
    assert _classify_map_topology(-2.0001, 1.0) == "hyperbolic"
    assert _classify_map_topology(-2.0, 1.0) == "parabolic"


def test_periodic_hermite_value_derivative_are_consistent():
    nfp = 3
    period = 2.0 * np.pi / nfp
    phi_axis = np.linspace(0.0, period, 48, endpoint=False)
    values = 1.2 + 0.08 * np.cos(3 * phi_axis) - 0.03 * np.sin(6 * phi_axis)
    derivatives = -0.24 * np.sin(3 * phi_axis) - 0.18 * np.cos(6 * phi_axis)
    phi = np.linspace(-0.2, period + 0.2, 200)

    value, derivative = interp_periodic_hermite(
        phi, phi_axis, values, derivatives, nfp
    )
    eps = 1e-7
    value_plus, _ = interp_periodic_hermite(
        phi + eps, phi_axis, values, derivatives, nfp
    )

    np.testing.assert_allclose((value_plus - value) / eps, derivative, atol=2e-6)


def test_periodic_hermite_is_continuous_across_period():
    nfp = 2
    period = np.pi
    phi_axis = np.linspace(0.0, period, 32, endpoint=False)
    values = np.cos(2.0 * phi_axis)
    derivatives = -2.0 * np.sin(2.0 * phi_axis)
    eps = 1e-9

    left = interp_periodic_hermite(-eps, phi_axis, values, derivatives, nfp)
    right = interp_periodic_hermite(eps, phi_axis, values, derivatives, nfp)

    np.testing.assert_allclose(left[0], right[0], atol=1e-8)
    np.testing.assert_allclose(left[1], right[1], atol=1e-8)


def test_psi_phi_gradient_matches_scalar_finite_difference():
    nfp = 3
    period = 2.0 * np.pi / nfp
    phi_axis = np.linspace(0.0, period, 64, endpoint=False)
    axis_R = 1.0 + 0.04 * np.cos(3.0 * phi_axis)
    axis_Z = 0.03 * np.sin(3.0 * phi_axis)
    model = PsiModel(
        coeffs=np.zeros(2),
        modes=[PolyMode(1, 1, 0, "cos"), PolyMode(0, 2, 0, "cos")],
        nfp=nfp,
        a=0.1,
        phi_axis=phi_axis,
        R_axis=axis_R,
        Z_axis=axis_Z,
        R_axis_phi=-0.12 * np.sin(3.0 * phi_axis),
        Z_axis_phi=0.09 * np.cos(3.0 * phi_axis),
        fit_info={"poly_degree": 2, "m_tor": 0},
    )
    phi = np.linspace(0.0, period, 80, endpoint=False)
    ra, za, _, _ = model.axis_at(phi)
    R = ra + 0.05
    Z = za + 0.02
    psi, _, _, grad_phi = psi_and_gradient(model, R, Z, phi)
    eps = 1e-7
    psi_plus, _, _, _ = psi_and_gradient(model, R, Z, phi + eps)

    np.testing.assert_allclose((psi_plus - psi) / eps, grad_phi, atol=2e-5)


def test_sampled_rk4_retains_fourth_stage_in_z_update():
    class HelicalField:
        def set_points(self, xyz):
            self.xyz = np.asarray(xyz)

        def B(self):
            phi = np.arctan2(self.xyz[:, 1], self.xyz[:, 0])
            return np.column_stack(
                [-np.sin(phi), np.cos(phi), np.full(len(phi), 0.2)]
            )

    r0 = np.array([0.8, 1.1])
    z0 = np.array([-0.1, 0.2])
    nfp = 2
    phi, r_hist, z_hist, r1, z1 = rk4_period_samples(
        HelicalField(), r0, z0, nfp, n_zeta=5, steps=40
    )

    np.testing.assert_allclose(r1, r0, atol=1e-14)
    np.testing.assert_allclose(z1, z0 + np.pi * 0.2 * r0, atol=1e-12)
    np.testing.assert_allclose(phi, np.arange(5) * np.pi / 5, atol=1e-14)
    np.testing.assert_allclose(r_hist, np.repeat(r0[:, None], 5, axis=1), atol=1e-14)
