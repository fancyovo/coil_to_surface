from __future__ import annotations

import numpy as np

from stellarator_eval.desc_joint_ls import (
    JointSpectralFitConfig,
    PhaseConstraintSet,
    WeightedSampleSet,
    _stack_phase_constraints,
    fit_joint_rzl_data,
)


class _Basis:
    def __init__(self, modes, evaluator):
        self.modes = np.asarray(modes, dtype=int)
        self.num_modes = len(self.modes)
        self._evaluator = evaluator

    def evaluate(self, nodes):
        return np.asarray(self._evaluator(np.asarray(nodes, dtype=float)), dtype=float)


def test_phase_beta_matrix_has_one_entry_per_sample():
    basis = _Basis([[1, 1, 0]], lambda nodes: np.sin(nodes[:, 1:2]))
    nodes = np.array(
        [
            [0.3, 0.1, 0.0],
            [0.3, 0.2, 0.4],
            [0.6, 0.3, 0.0],
            [0.6, 0.4, 0.4],
        ]
    )
    beta_ids = np.array([0, 0, 1, 2])
    phase = PhaseConstraintSet(
        nodes=nodes,
        theta=np.zeros(len(nodes)),
        beta_ids=beta_ids,
        beta_groups=np.array([0, 1, 1]),
    )

    _, A_beta, _, _, _, _, _, _ = _stack_phase_constraints(
        basis, [phase], iota_powers=(0, 2)
    )

    assert np.all(np.count_nonzero(A_beta, axis=1) == 1)
    np.testing.assert_array_equal(np.argmax(A_beta == -1.0, axis=1), beta_ids)


def test_joint_phase_fit_recovers_synthetic_beta_and_iota():
    scalar_basis = _Basis([[0, 0, 0]], lambda nodes: np.ones((len(nodes), 1)))
    lambda_basis = _Basis(
        [[1, -1, 0], [2, -1, 0]],
        lambda nodes: np.column_stack(
            [np.sin(nodes[:, 1]), nodes[:, 0] ** 2 * np.sin(nodes[:, 1])]
        ),
    )

    rho_levels = np.array([0.25, 0.7])
    zeta = np.linspace(0.0, 1.2, 7)
    beta_true = np.array([-0.4, 0.3, -0.2, 0.5])
    cL_true = np.array([0.12, -0.08])
    iota_true = np.array([0.45, 0.20])
    rows = []
    beta_ids = []
    beta_groups = []
    line = 0
    for group, rho in enumerate(rho_levels):
        for theta0 in (0.4, 2.1):
            for zz in zeta:
                theta_node = theta0 + 0.35 * zz + 0.12 * np.sin(2.0 * zz + theta0)
                rows.append([rho, theta_node, zz])
                beta_ids.append(line)
            beta_groups.append(group)
            line += 1
    nodes = np.asarray(rows)
    beta_ids = np.asarray(beta_ids)
    lambda_value = lambda_basis.evaluate(nodes) @ cL_true
    iota_value = iota_true[0] + iota_true[1] * nodes[:, 0] ** 2
    theta_target = -lambda_value + beta_true[beta_ids] + nodes[:, 2] * iota_value

    phase = PhaseConstraintSet(
        nodes=nodes,
        theta=theta_target,
        beta_ids=beta_ids,
        beta_groups=np.asarray(beta_groups),
    )
    eq = type(
        "FakeEquilibrium",
        (),
        {"R_basis": scalar_basis, "Z_basis": scalar_basis, "L_basis": lambda_basis},
    )()
    sample_nodes = np.array([[0.3, 0.0, 0.0], [0.8, 0.0, 0.0]])
    fit = fit_joint_rzl_data(
        eq,
        R_datasets=[WeightedSampleSet(sample_nodes, np.full(2, 2.0))],
        Z_datasets=[WeightedSampleSet(sample_nodes, np.full(2, -1.0))],
        phase_datasets=[phase],
        config=JointSpectralFitConfig(
            rz_ridge=0.0,
            l_ridge=0.0,
            beta_ridge=0.0,
            iota_ridge=0.0,
            beta_gauge_weight=0.0,
            iota_powers=(0, 2),
        ),
    )

    np.testing.assert_allclose(fit.L_lmn, cL_true, atol=1e-10)
    np.testing.assert_allclose(fit.beta, beta_true, atol=1e-10)
    np.testing.assert_allclose(fit.iota_coeffs, iota_true, atol=1e-10)
    assert fit.diagnostics["phase_fit_rms"] < 1e-11
