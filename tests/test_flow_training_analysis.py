import pytest

from scripts.analyze_qh_flow_training import summarize_rows, validation_rows


def test_validation_rows_deduplicates_resumed_steps() -> None:
    rows = [
        {"event": "validation", "step": 200, "validation_loss": 3.0},
        {"event": "validation", "step": 400, "validation_loss": 2.0},
        {"event": "resume_validation", "step": 400, "validation_loss": 1.9},
    ]

    selected = validation_rows(rows)

    assert [row["step"] for row in selected] == [200, 400]
    assert selected[-1]["validation_loss"] == 1.9


def test_summary_reports_tail_slope_per_thousand_steps() -> None:
    rows = []
    for step, value in ((1000, 3.0), (2000, 2.0), (3000, 1.0)):
        rows.append(
            {
                "event": "validation",
                "step": step,
                "validation_loss": value,
                "validation_geometry_physical_loss": value + 1.0,
                "validation_geometry_relative_loss": value + 2.0,
                "validation_current_loss": value + 3.0,
            }
        )

    summary = summarize_rows(rows, tail_points=2)

    total = summary["validation_loss"]
    assert total["minimum"] == 1.0
    assert total["minimum_step"] == 3000
    assert total["tail_mean"] == 1.5
    assert total["tail_slope_per_1000_steps"] == pytest.approx(-1.0)
