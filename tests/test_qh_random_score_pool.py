from __future__ import annotations

from scripts.evaluate_qh_random_score_pool import summarize


def test_summary_reports_practical_and_extreme_score_tails() -> None:
    rows = [
        {"score": 2.0, "status": "no_axis"},
        {"score": 42.0, "status": "ok"},
        {"score": 55.0, "status": "ok"},
    ]

    summary = summarize(rows)

    assert summary["score_exceedance_counts"]["40"] == 2
    assert summary["score_exceedance_counts"]["50"] == 1
    assert summary["status_ok_rate"] == 2 / 3
