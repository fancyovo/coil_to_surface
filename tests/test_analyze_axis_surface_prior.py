from __future__ import annotations

from scripts.analyze_axis_surface_prior import representative_rows


def _row(case_id: int, score: float, coil: float) -> dict[str, object]:
    return {
        "case_id": case_id,
        "family": "balanced_stellarator",
        "native": {
            "score": score,
            "components": {"coil": coil},
        },
    }


def test_representative_rows_supports_registered_v2_family() -> None:
    selected = representative_rows(
        [
            _row(1, 4.0, 70.0),
            _row(2, 8.0, 75.0),
            _row(3, 6.0, 80.0),
        ]
    )
    assert [label for label, _ in selected] == [
        "balanced_stellarator_top_score",
        "balanced_stellarator_median_engineering",
    ]
    assert int(selected[0][1]["case_id"]) == 2
    assert int(selected[1][1]["case_id"]) == 2
