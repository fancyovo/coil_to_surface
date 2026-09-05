from __future__ import annotations

import numpy as np
import pytest

from stellarator_eval.surface_provenance import require_surface_kind


def test_surface_kind_accepts_exact_scalar_text() -> None:
    assert (
        require_surface_kind(
            {"kind": np.asarray("alpha_nu")},
            "alpha_nu",
            stage="test",
        )
        == "alpha_nu"
    )


@pytest.mark.parametrize("actual", ["alpha", "direct_gpu_point_cloud"])
def test_surface_kind_rejects_non_alpha_nu_routes(actual: str) -> None:
    with pytest.raises(ValueError, match="psi -> alpha -> nu"):
        require_surface_kind(
            {"kind": np.asarray(actual)},
            "alpha_nu",
            stage="test",
        )


def test_surface_kind_requires_provenance_field() -> None:
    with pytest.raises(ValueError, match="missing required provenance"):
        require_surface_kind({}, "alpha_nu", stage="test")
