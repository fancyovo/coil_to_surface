from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.run_qh_data_space_trajectory_rerun import (
    coil_tokens,
    make_data_start,
    worker_cases,
)


class IdentityNormalizer:
    def transform(self, values: np.ndarray, condition: tuple[int, int]):
        assert condition == (4, 2)
        return values.astype(np.float32), 0.0


def test_make_data_start_reuses_saved_physical_winner(tmp_path: Path) -> None:
    tokens = np.arange(200, dtype=float).reshape(2, 100)
    payload = {
        "raw": {
            "x": tokens[:, :33].tolist(),
            "y": tokens[:, 33:66].tolist(),
            "z": tokens[:, 66:99].tolist(),
            "current": tokens[:, 99].tolist(),
        },
        "flow_prior_screening": {
            "candidate_index": 7,
            "candidate_count": 32,
            "native_score": {"status": "ok", "score": 80.0},
        },
    }
    source = tmp_path / "start.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    result, clipped = make_data_start(
        payload,
        IdentityNormalizer(),
        nfp=4,
        n_base_coils=2,
        source_path=source,
    )

    assert clipped == 0.0
    assert np.array_equal(coil_tokens(result), tokens)
    assert np.array_equal(
        result["data_prior_screening"]["normalized_coil_tokens"], tokens
    )
    assert result["data_prior_screening"]["candidate_index"] == 7


def test_worker_cases_are_filtered_and_sorted() -> None:
    manifest = {
        "cases": [
            {"trajectory_id": "b", "worker_index": 1},
            {"trajectory_id": "c", "worker_index": 0},
            {"trajectory_id": "a", "worker_index": 1},
        ]
    }
    assert [row["trajectory_id"] for row in worker_cases(manifest, 1)] == ["a", "b"]
