from __future__ import annotations

import json
from pathlib import Path

from evaluation.full_physical.select_largest_standard_surface import (
    apply_nested_volume_check,
    load_candidate_rows,
)


def _write_candidate(root: Path, name: str, target_s: float, volume: float) -> None:
    output = root / name / "standard_rho_1"
    output.mkdir(parents=True)
    (output / "boozer_standard.npz").touch()
    summary = {
        "target_s": target_s,
        "accepted_for_downstream": True,
        "newton": {
            "state": {"geometry": {"signed_volume_m3": -volume}},
        },
    }
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_nested_volume_check_rejects_inner_branch_solver_successes(tmp_path: Path) -> None:
    _write_candidate(tmp_path, "s_0p24", 0.24, 0.031)
    _write_candidate(tmp_path, "s_0p36", 0.36, 0.049)
    _write_candidate(tmp_path, "s_0p64", 0.64, 0.036)
    _write_candidate(tmp_path, "s_0p81", 0.81, 0.010)

    rows = load_candidate_rows(tmp_path)
    apply_nested_volume_check(rows)

    accepted = [row["target_s"] for row in rows if row["accepted"]]
    assert accepted == [0.24, 0.36]
    assert rows[2]["solver_accepted"] is True
    assert rows[2]["branch_consistency"]["rejection_reason"] == (
        "non_increasing_enclosed_volume"
    )
    assert rows[3]["branch_consistency"]["previous_largest_abs_volume_m3"] == 0.049


def test_completed_candidate_that_failed_before_standard_is_outer_failure(
    tmp_path: Path,
) -> None:
    _write_candidate(tmp_path, "s_0p49", 0.49, 0.065)
    failed = tmp_path / "s_0p64"
    failed.mkdir()
    (failed / "gpu_postflight.csv").touch()

    rows = load_candidate_rows(tmp_path)

    assert [row["target_s"] for row in rows] == [0.49, 0.64]
    assert rows[1]["accepted"] is False
    assert rows[1]["failure_stage"] == "completed_before_standard_acceptance"
