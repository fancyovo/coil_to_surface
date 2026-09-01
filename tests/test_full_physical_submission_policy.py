from __future__ import annotations

import pytest

from evaluation.full_physical.write_submission_policy import build_policy


def test_parallel_policy_records_independent_cross_pool_jobs() -> None:
    policy = build_policy(
        stage="surface_candidates",
        serial=False,
        serial_reason="",
        candidate_count=3,
        candidate_pools=["p107", "p107", "students"],
    )
    assert policy["mode"] == "parallel"
    assert policy["dependency_policy"] == "independent_jobs"
    assert policy["candidate_pools"] == ["p107", "p107", "students"]
    assert policy["serial_reason"] is None


def test_serial_policy_requires_and_records_reason() -> None:
    with pytest.raises(ValueError, match="SERIAL_REASON"):
        build_policy(
            stage="source_psi_candidates",
            serial=True,
            serial_reason="",
            candidate_count=1,
            candidate_pools=["p107"],
        )
    policy = build_policy(
        stage="source_psi_candidates",
        serial=True,
        serial_reason="only one verified GPU slot is available",
        candidate_count=2,
        candidate_pools=["p107", "p107"],
    )
    assert policy["mode"] == "serial"
    assert policy["dependency_policy"] == "afterany_chain"
    assert policy["serial_reason"] == "only one verified GPU slot is available"


def test_policy_rejects_pool_count_or_name_mismatch() -> None:
    with pytest.raises(ValueError, match="one entry per candidate"):
        build_policy(
            stage="surface_candidates",
            serial=False,
            serial_reason="",
            candidate_count=2,
            candidate_pools=["p107"],
        )
    with pytest.raises(ValueError, match="unsupported candidate pools"):
        build_policy(
            stage="surface_candidates",
            serial=False,
            serial_reason="",
            candidate_count=1,
            candidate_pools=["unknown"],
        )
