from __future__ import annotations

import numpy as np

from scripts.qh_blackbox_gradient_reference import (
    branch_fingerprint,
    case_rows,
    rms_orthogonal_basis,
)


def test_rms_orthogonal_basis_has_expected_normalization() -> None:
    basis = rms_orthogonal_basis(12, 73).astype(np.float64)
    np.testing.assert_allclose(basis @ basis.T, 12.0 * np.eye(12), atol=2.0e-6)
    np.testing.assert_allclose(np.sqrt(np.mean(basis * basis, axis=1)), 1.0, atol=1.0e-7)


def test_case_rows_include_one_center_and_antithetic_endpoints() -> None:
    centers = [{"center_id": "sample", "dimension": 12, "direction_count": 3}]
    rows = case_rows(centers, (0.1, 0.05))
    assert len(rows) == 1 + 2 * 3 * 2
    assert rows[0]["kind"] == "center"
    endpoints = rows[1:]
    assert {row["sign"] for row in endpoints} == {-1, 1}
    assert {row["scale"] for row in endpoints} == {0.1, 0.05}


def test_branch_fingerprint_tracks_discrete_score_state() -> None:
    result = {
        "status": "ok",
        "diagnostics": {
            "surface_level": 0.16,
            "stable_surface_count": 3,
            "surface_long_trace_rejected_count": 0,
            "flux_attempt_count": 1,
            "volume_candidate_count": 100000,
            "volume_available_count": 100000,
            "volume_point_count": 100000,
            "alpha_column_count": 48,
        },
    }
    assert branch_fingerprint(result) == (
        "ok",
        0.16,
        3,
        0,
        1,
        100000,
        100000,
        100000,
        48,
    )
