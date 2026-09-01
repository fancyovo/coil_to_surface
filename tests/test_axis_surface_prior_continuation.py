from __future__ import annotations

import numpy as np
import pytest

from scripts.prepare_axis_surface_prior_continuation import continuation_payload


def test_continuation_payload_wraps_saved_direct_data_parameters() -> None:
    parameters = np.arange(200, dtype=np.float32).reshape(2, 100) / 100.0
    best = {
        "nfp": 5,
        "raw": {"current": [250000.0, 250000.0]},
        "original_space_local_gradient_adam": {
            "parameter_space": "data",
            "normalized_coil_tokens": parameters.tolist(),
            "best_iteration": 170,
            "native_score": {"status": "ok", "score": 69.0, "components": {"coil": 71.0}},
        },
    }
    source = {"data_prior_screening": {"current_l1_a": 500000.0}}
    prepared = continuation_payload(best, source, protocol_id="continuation-test")
    metadata = prepared["data_prior_screening"]
    assert metadata["protocol_id"] == "continuation-test"
    assert metadata["current_l1_a"] == 500000.0
    assert metadata["source_best_iteration"] == 170
    np.testing.assert_array_equal(metadata["normalized_coil_tokens"], parameters)


def test_continuation_payload_rejects_unwrapped_or_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="direct-data"):
        continuation_payload({"raw": {"current": [1.0]}}, {}, protocol_id="test")
