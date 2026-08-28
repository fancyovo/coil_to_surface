from __future__ import annotations

import json
from pathlib import Path

from flow_matching.trajectory_dataset import atomic_write_jsonl_gzip
from scripts.qh_data_space_random_survey import (
    manifest_conditions,
    parse_worker_counts,
    select_adam_followup,
    validate_existing_rows,
    wilson_interval,
)


def test_manifest_conditions_are_numeric_and_sorted() -> None:
    manifest = {
        "group_counts": {
            "nfp4_nc3": 20,
            "nfp2_nc5": 10,
            "nfp2_nc2": 5,
        }
    }
    assert manifest_conditions(manifest) == [
        {"nfp": 2, "n_base_coils": 2, "quasr_count": 5},
        {"nfp": 2, "n_base_coils": 5, "quasr_count": 10},
        {"nfp": 4, "n_base_coils": 3, "quasr_count": 20},
    ]


def test_worker_count_parsing_supports_uniform_and_calibrated_counts() -> None:
    assert parse_worker_counts("7", 3) == [7, 7, 7]
    assert parse_worker_counts("7,8,9", 3) == [7, 8, 9]


def test_existing_rows_must_be_contiguous_without_duplicates() -> None:
    conditions = [(2, 2), (3, 4)]
    targets = {(2, 2): 3, (3, 4): 2}
    rows = [
        {"nfp": 2, "n_base_coils": 2, "condition_sample_index": 0},
        {"nfp": 3, "n_base_coils": 4, "condition_sample_index": 0},
        {"nfp": 2, "n_base_coils": 2, "condition_sample_index": 1},
    ]
    assert validate_existing_rows(rows, conditions, targets) == {
        (2, 2): 2,
        (3, 4): 1,
    }


def test_followup_selection_is_bounded_and_probability_is_recorded() -> None:
    rows = []
    for index in range(100):
        rows.append(
            {
                "sample_id": f"sample_{index}",
                "score": 15.0 if index < 50 else 55.0,
                "status": "ok",
                "nfp": 4,
                "n_base_coils": 3,
                "worker_index": 0,
                "condition_index": 0,
                "condition_sample_index": index,
            }
        )
    selection = select_adam_followup(rows, seed=17)
    assert selection["selected_count"] == 28
    low, high = selection["bands"][0], selection["bands"][-1]
    assert low["selection_probability"] == 4 / 50
    assert high["selection_probability"] == 24 / 50


def test_wilson_zero_success_has_nonzero_finite_upper_bound() -> None:
    low, high = wilson_interval(0, 1000)
    assert low == 0.0
    assert high is not None and 0.0 < high < 0.01


def test_atomic_chunk_fixture_is_valid_gzip_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "chunk_000000.jsonl.gz"
    atomic_write_jsonl_gzip(path, [{"sample_id": "a", "score": 1.0}])
    assert path.is_file()
    assert json.loads(__import__("gzip").open(path, "rt", encoding="utf-8").read())["sample_id"] == "a"
