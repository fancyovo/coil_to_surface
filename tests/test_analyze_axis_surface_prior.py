from __future__ import annotations

from pathlib import Path

from scripts.analyze_axis_surface_prior import _write_three_coil_html, representative_rows


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


def test_three_html_refits_camera_for_narrow_viewports(tmp_path: Path) -> None:
    output = tmp_path / "coils.html"
    row = {
        "case_id": 1,
        "family": "balanced_stellarator",
        "nfp": 1,
        "n_base_coils": 1,
        "tokens": [[0.0] * 100],
        "native": {"score": 1.0, "components": {"coil": 70.0}},
    }
    _write_three_coil_html(row, output, "responsive preview")
    html = output.read_text(encoding="utf-8")
    assert "halfHorizontal=Math.atan" in html
    assert "Math.min(halfVertical,halfHorizontal)" in html
    assert "renderer.setSize(innerWidth,innerHeight);fitCamera()" in html
