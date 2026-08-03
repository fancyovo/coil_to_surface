from __future__ import annotations

import numpy as np

from flow_matching.quasr_export import parse_simson_coils, stable_split


def ref(name: str) -> dict[str, str]:
    return {"$type": "ref", "value": name}


def test_parse_simson_coils_selects_base_curves_and_scales_current():
    curve_dofs = np.arange(99, dtype=float).tolist()
    payload = {
        "@class": "SIMSON",
        "graph": [[ref("surface")], [ref("base_coil"), ref("symmetric_coil")]],
        "simsopt_objs": {
            "surface": {"@class": "SurfaceXYZTensorFourier", "nfp": 4},
            "curve": {"@class": "CurveXYZFourier", "order": 16, "dofs": ref("curve_dofs")},
            "curve_dofs": {"@class": "DOFs", "x": {"data": curve_dofs}},
            "raw_current": {"@class": "Current", "current": -0.5},
            "scaled_current": {
                "@class": "ScaledCurrent",
                "current_to_scale": ref("raw_current"),
                "scale": 2.0e6,
            },
            "base_coil": {
                "@class": "Coil",
                "curve": ref("curve"),
                "current": ref("scaled_current"),
            },
            "rotated_curve": {"@class": "RotatedCurve"},
            "symmetric_coil": {
                "@class": "Coil",
                "curve": ref("rotated_curve"),
                "current": ref("scaled_current"),
            },
        },
    }
    parsed = parse_simson_coils(payload)
    assert parsed.nfp == 4
    assert parsed.curve_order == 16
    assert parsed.tokens.shape == (1, 100)
    np.testing.assert_array_equal(parsed.tokens[0, :99], np.arange(99, dtype=np.float32))
    assert parsed.tokens[0, 99] == -1.0e6


def test_parse_simson_coils_zero_pads_lower_orders_by_coordinate():
    order = 1
    source_count = 2 * order + 1
    coefficients = np.arange(3 * source_count, dtype=float)
    payload = {
        "@class": "SIMSON",
        "graph": [[ref("surface")], [ref("coil")]],
        "simsopt_objs": {
            "surface": {"@class": "SurfaceXYZTensorFourier", "nfp": 2},
            "curve": {"@class": "CurveXYZFourier", "order": order, "dofs": ref("dofs")},
            "dofs": {"@class": "DOFs", "x": {"data": coefficients.tolist()}},
            "current": {"@class": "Current", "current": 3.0},
            "coil": {"@class": "Coil", "curve": ref("curve"), "current": ref("current")},
        },
    }
    parsed = parse_simson_coils(payload)
    assert parsed.curve_order == 1
    for coordinate in range(3):
        start = coordinate * 33
        source_start = coordinate * source_count
        np.testing.assert_array_equal(
            parsed.tokens[0, start : start + source_count],
            coefficients[source_start : source_start + source_count],
        )
        assert np.count_nonzero(parsed.tokens[0, start + source_count : start + 33]) == 0


def test_stable_split_is_repeatable_and_in_range():
    values = [stable_split(index) for index in range(1000)]
    assert values == [stable_split(index) for index in range(1000)]
    assert set(values) == {0, 1, 2}
