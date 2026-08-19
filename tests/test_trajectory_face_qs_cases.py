from __future__ import annotations

import numpy as np

from scripts.prepare_qh_trajectory_face_qs_cases import (
    condition_quotas,
    select_trajectories,
    token_case,
)


def sample_rows() -> list[dict[str, str]]:
    rows = []
    for nfp, n_coils, count in ((3, 2, 8), (4, 3, 12), (5, 4, 5)):
        for index in range(count):
            rows.append(
                {
                    "trajectory_id": f"nfp{nfp}_nc{n_coils}_{index:02d}",
                    "nfp": str(nfp),
                    "n_base_coils": str(n_coils),
                    "best_global_score": str(60.0 + index),
                }
            )
    return rows


def test_condition_selection_is_proportional_and_covers_each_group() -> None:
    rows = sample_rows()
    quotas = condition_quotas(rows, 10)
    assert sum(quotas.values()) == 10
    assert all(value >= 1 for value in quotas.values())
    selected = select_trajectories(rows, 10)
    assert len(selected) == 10
    assert len({row["trajectory_id"] for row in selected}) == 10
    assert {(int(row["nfp"]), int(row["n_base_coils"])) for row in selected} == set(quotas)


def test_token_case_preserves_fourier_blocks_and_current() -> None:
    tokens = np.arange(200, dtype=np.float32).reshape(2, 100)
    case = token_case(tokens, nfp=4)
    assert case["nfp"] == 4
    assert np.array_equal(case["raw"]["x"], tokens[:, :33])
    assert np.array_equal(case["raw"]["y"], tokens[:, 33:66])
    assert np.array_equal(case["raw"]["z"], tokens[:, 66:99])
    assert np.array_equal(case["raw"]["current"], tokens[:, 99])
    assert case["raw"]["current_unit"] == "A"
