from __future__ import annotations

import json

import numpy as np

from scripts.flow_runtime import load_initial_noise


def test_data_prior_parameters_take_priority_over_retained_flow_metadata(tmp_path) -> None:
    data = np.arange(200, dtype=np.float32).reshape(2, 100)
    latent = np.full((2, 100), -3.0, dtype=np.float32)
    path = tmp_path / "mixed_start.json"
    path.write_text(
        json.dumps(
            {
                "data_prior_screening": {
                    "normalized_coil_tokens": data.tolist(),
                },
                "flow_prior_screening": {"noise": latent.tolist()},
            }
        ),
        encoding="utf-8",
    )

    loaded, payload = load_initial_noise(path)

    np.testing.assert_array_equal(loaded, data)
    assert "data_prior_screening" in payload
