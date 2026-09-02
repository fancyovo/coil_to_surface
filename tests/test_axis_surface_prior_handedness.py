from __future__ import annotations

import math

from scripts.analyze_axis_surface_prior_handedness import (
    binomial_two_sided_log10,
    classify_iota_interval,
    sign_summary,
)
from scripts.score_axis_surface_prior_signed_mirrors import transform_raw


def test_classify_iota_interval_uses_the_whole_fitted_interval() -> None:
    assert classify_iota_interval(-1.2, -0.8) == "negative"
    assert classify_iota_interval(0.7, 1.1) == "positive"
    assert classify_iota_interval(-0.1, 0.2) == "crosses_zero"
    assert classify_iota_interval(float("nan"), 0.2) == "missing"


def test_binomial_log_probability_is_symmetric() -> None:
    assert math.isclose(
        binomial_two_sided_log10(9, 1),
        binomial_two_sided_log10(1, 9),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    assert math.isclose(binomial_two_sided_log10(5, 5), 0.0, abs_tol=1.0e-12)


def test_sign_summary_excludes_crossing_intervals_from_definite_fraction() -> None:
    summary = sign_summary(
        [
            {"iota_sign": "negative", "iota_midpoint": -1.0, "score": 2.0},
            {"iota_sign": "positive", "iota_midpoint": 1.0, "score": 3.0},
            {"iota_sign": "crosses_zero", "iota_midpoint": 0.0, "score": 4.0},
            {"iota_sign": "missing", "iota_midpoint": None, "score": None},
        ]
    )
    assert summary["definite_sign_count"] == 2
    assert summary["negative_fraction_of_definite"] == 0.5
    assert summary["sign_counts"] == {
        "negative": 1,
        "crosses_zero": 1,
        "positive": 1,
        "missing": 1,
    }


def test_coordinate_reflection_changes_only_requested_coefficients() -> None:
    raw = {
        "x": [[1.0, 2.0]],
        "y": [[3.0, 4.0]],
        "z": [[5.0, 6.0]],
        "current": [7.0],
    }
    mirrored = transform_raw(raw, (1.0, -1.0, 1.0))
    assert mirrored == {
        "x": [[1.0, 2.0]],
        "y": [[-3.0, -4.0]],
        "z": [[5.0, 6.0]],
        "current": [7.0],
    }
    assert raw["y"] == [[3.0, 4.0]]
