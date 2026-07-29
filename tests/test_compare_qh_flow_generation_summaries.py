import pytest

from scripts.compare_qh_flow_generation_summaries import comparison_summary


def test_comparison_accepts_legacy_and_current_summary_schemas() -> None:
    old = {
        "new_score_all": {"count": 10, "mean": 2.0, "median": 1.0, "p90": 4.0, "max": 5.0},
        "new_score_ok": {"count": 4, "mean": 4.0, "median": 3.0, "p90": 6.0, "max": 7.0},
        "status_transitions": {"ok->ok": 4, "ok->drift_rejected": 5, "geometry_rejected->geometry_rejected": 1},
    }
    new = {
        "score_all": {"count": 10, "mean": 3.0, "median": 2.0, "p90": 6.0, "max": 10.0},
        "score_ok": {"count": 5, "mean": 6.0, "median": 4.0, "p90": 9.0, "max": 14.0},
        "status_counts": {"ok": 5, "drift_rejected": 5},
    }

    summary = comparison_summary(old, new)

    assert summary["old_geometry_eligible_rate"] == pytest.approx(0.9)
    assert summary["new_geometry_eligible_rate"] == pytest.approx(1.0)
    assert summary["old_ok_rate"] == pytest.approx(0.4)
    assert summary["new_ok_rate"] == pytest.approx(0.5)
    assert summary["score_all_relative_change"]["mean"] == pytest.approx(0.5)
    assert summary["score_ok_relative_change"]["max"] == pytest.approx(1.0)
