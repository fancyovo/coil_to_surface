import json
from pathlib import Path

import pytest

from evaluation.full_physical.select_source_psi_candidate import (
    select_source_candidate,
)


def write_candidate(
    root: Path,
    name: str,
    *,
    a: float,
    radius: float,
    validation_rms: float = 2e-4,
    outer_failure: bool = True,
) -> None:
    candidate = root / name
    candidate.mkdir()
    levels = [
        {
            "psi_level": 0.49,
            "ok": True,
            "verify_ok": True,
            "radius_mean": radius,
        }
    ]
    if outer_failure:
        levels.append({"psi_level": 0.64, "ok": False, "reason": "drift"})
    payload = {
        "axis": {"has_axis": True, "best_residual": 2e-8},
        "psi": {
            "a": a,
            "fit_info": {"train_rms": 1.5e-4, "validation_rms": validation_rms},
        },
        "surface_screen": {"levels": levels},
    }
    (candidate / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def test_selects_largest_verified_physical_radius(tmp_path: Path) -> None:
    write_candidate(tmp_path, "a_0p04", a=0.04, radius=0.035)
    write_candidate(tmp_path, "a_0p08", a=0.08, radius=0.062)

    result = select_source_candidate(tmp_path)

    assert result["selected"]["a_m"] == pytest.approx(0.08)
    assert result["selected"]["largest_verified_level"]["radius_mean"] == pytest.approx(
        0.062
    )


def test_rejects_unbracketed_or_bad_fit_candidates(tmp_path: Path) -> None:
    write_candidate(
        tmp_path, "a_0p08", a=0.08, radius=0.08, outer_failure=False
    )
    write_candidate(
        tmp_path, "a_0p06", a=0.06, radius=0.06, validation_rms=8e-4
    )

    with pytest.raises(ValueError, match="no source-psi candidate"):
        select_source_candidate(tmp_path)
