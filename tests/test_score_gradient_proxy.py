from __future__ import annotations

import pytest

from scripts.score_gradient_proxy import coordinate_omitted_gradient_score


def test_coordinate_omitted_proxy_preserves_formal_gate() -> None:
    components = {
        "axis": 100.0,
        "psi": 100.0,
        "surface": 100.0,
        "coordinate": 0.0,
        "volume_qs": 100.0,
        "iota": 100.0,
        "coil": 100.0,
    }
    # The formal weighted average is 90, so a formal score of 45 encodes gate=0.5.
    assert coordinate_omitted_gradient_score(45.0, components) == pytest.approx(50.0)


def test_coordinate_changes_do_not_change_proxy_when_gate_is_fixed() -> None:
    low_coordinate = {
        "axis": 80.0,
        "psi": 81.0,
        "surface": 82.0,
        "coordinate": 10.0,
        "volume_qs": 83.0,
        "iota": 84.0,
        "coil": 85.0,
    }
    high_coordinate = {**low_coordinate, "coordinate": 90.0}

    def formal_average(components: dict[str, float]) -> float:
        weights = {
            "axis": 10.0,
            "psi": 10.0,
            "surface": 10.0,
            "coordinate": 10.0,
            "volume_qs": 42.0,
            "iota": 10.0,
            "coil": 8.0,
        }
        return sum(weights[name] * value for name, value in components.items()) / 100.0

    low_proxy = coordinate_omitted_gradient_score(
        formal_average(low_coordinate), low_coordinate
    )
    high_proxy = coordinate_omitted_gradient_score(
        formal_average(high_coordinate), high_coordinate
    )
    assert low_proxy == pytest.approx(high_proxy)
