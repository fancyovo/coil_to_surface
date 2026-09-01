from __future__ import annotations

import pytest

from scripts.prepare_axis_surface_prior_v3_adam200 import (
    EXPECTED_ROW_COUNT,
    SOURCE_PROTOCOL_ID,
    validate_source_rows,
)


def source_rows(*, nc: int = 4, protocol_id: str = SOURCE_PROTOCOL_ID):
    return [
        {
            "case_id": index,
            "protocol_id": protocol_id,
            "n_base_coils": nc,
        }
        for index in range(EXPECTED_ROW_COUNT)
    ]


def test_v3_source_validation_accepts_exact_registered_population() -> None:
    validate_source_rows(source_rows())


def test_v3_source_validation_rejects_wrong_protocol_or_nc5() -> None:
    with pytest.raises(ValueError, match="source protocol mismatch"):
        validate_source_rows(source_rows(protocol_id="historical-source"))
    with pytest.raises(ValueError, match="excluded nc"):
        validate_source_rows(source_rows(nc=5))
