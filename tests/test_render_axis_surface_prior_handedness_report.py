import json
import math

import pytest

from scripts.render_axis_surface_prior_handedness_report import (
    sample_id_from_trajectory,
    summary_without_history,
)


def test_sample_id_from_negative_continuation_trajectory() -> None:
    assert (
        sample_id_from_trajectory("axisv3_case_01341_continue_adam200")
        == "axisv3_case_01341"
    )
    with pytest.raises(ValueError, match="unexpected trajectory id"):
        sample_id_from_trajectory("axisv3_case_01341")


def test_summary_excludes_nonfinite_raw_history() -> None:
    results = {
        "negative_run_root": "negative_run",
        "trajectories": [
            {
                "trajectory_id": "axisv3_case_01341_continue_adam200",
                "sample_id": "axisv3_case_01341",
                "history": [{"gradient_previous_moment_cosine": math.nan}],
            }
        ],
    }

    summary = summary_without_history(results)
    row = summary["trajectories"][0]
    assert "history" not in row
    assert row["history_jsonl"].replace("\\", "/").endswith(
        "negative_run/trajectories/axisv3_case_01341_continue_adam200/"
        "optimization/history.jsonl"
    )
    json.dumps(summary, allow_nan=False)
