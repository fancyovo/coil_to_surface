import numpy as np

from flow_matching.data import RawGroup
from scripts.evaluate_corrected_score_calibration import distribution, select_quasr_cases


def test_distribution_reports_fixed_percentiles():
    result = distribution([0.0, 1.0, 2.0, 3.0, 4.0])
    assert result["count"] == 5
    assert result["mean"] == 2.0
    assert result["p50"] == 2.0
    assert result["max"] == 4.0


def test_quasr_selection_is_without_replacement_and_preserves_condition():
    groups = {
        (4, 2): RawGroup(
            tokens=np.arange(5 * 2 * 100, dtype=np.float32).reshape(5, 2, 100),
            ids=np.arange(10, 15, dtype=np.int32),
        ),
        (3, 1): RawGroup(
            tokens=np.arange(4 * 100, dtype=np.float32).reshape(4, 1, 100),
            ids=np.arange(20, 24, dtype=np.int32),
        ),
    }
    rows = select_quasr_cases(groups, 8, np.random.default_rng(123))
    assert len({row["source_id"] for row in rows}) == 8
    assert all(row["tokens"].shape == (row["key"][1], 100) for row in rows)
