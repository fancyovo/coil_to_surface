from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flow_matching.data import CoilNormalizer
import scripts.qh_data_space_random_adam_followup as followup
from scripts.qh_data_space_random_adam_followup import (
    OPTIMIZER_LIBRARY_SYMBOLS,
    assign_workers,
    make_start_payload,
    parse_low_ok_quotas,
    reconstruct_sample,
    select_followup_cases,
    validate_optimizer_summary,
    validate_optimizer_library_api,
)


def make_row(
    sample_id: str,
    *,
    score: float,
    status: str,
    ncoils: int,
    sample_index: int,
) -> dict:
    return {
        "sample_id": sample_id,
        "score": score,
        "status": status,
        "nfp": 4,
        "n_base_coils": ncoils,
        "worker_index": 0,
        "condition_index": ncoils - 1,
        "condition_sample_index": sample_index,
        "sampled_standard_normal_rms": 1.0,
        "effective_normalized_rms": 1.0,
        "post_projection_clipped_fraction": 0.0,
    }


def test_expanded_selection_preserves_original_and_stratifies_low_ok() -> None:
    rows = []
    for ncoils in (1, 2):
        rows.append(
            make_row(
                f"high_nc{ncoils}",
                score=12.0 + ncoils,
                status="ok",
                ncoils=ncoils,
                sample_index=0,
            )
        )
        for index in range(1, 6):
            rows.append(
                make_row(
                    f"low_nc{ncoils}_{index}",
                    score=5.0,
                    status="ok",
                    ncoils=ncoils,
                    sample_index=index,
                )
            )
    rows.append(
        make_row(
            "original_invalid",
            score=0.1,
            status="no_axis",
            ncoils=2,
            sample_index=6,
        )
    )
    original = {
        "selected": [
            {
                "sample_id": "original_invalid",
                "score": 0.1,
                "status": "no_axis",
                "nfp": 4,
                "n_base_coils": 2,
                "within_band_selection_probability": 0.01,
            }
        ]
    }

    selection = select_followup_cases(
        rows, original, seed=17, low_ok_quotas={1: 2, 2: 3}
    )

    assert selection["selected_count"] == 8
    assert selection["adam_eligible_count"] == 7
    assert selection["diagnostic_only_count"] == 1
    selected = {case["sample_id"]: case for case in selection["cases"]}
    assert selected["high_nc1"]["selection_roles"][0]["analysis_weight"] == 1.0
    assert selected["original_invalid"]["original_selection_member"] is True
    assert selected["original_invalid"]["adam_eligible"] is False
    assert [row["selected_count"] for row in selection["low_ok_strata"]] == [2, 3]


def test_worker_assignment_balances_adam_eligible_cases() -> None:
    cases = [
        {
            "sample_id": f"case_{index}",
            "survey_score": float(index),
            "n_base_coils": index % 5 + 1,
            "adam_eligible": True,
            "estimated_adam_wall_s": 10.0,
        }
        for index in range(12)
    ]
    cases.append(
        {
            "sample_id": "diagnostic",
            "survey_score": 0.0,
            "n_base_coils": 2,
            "adam_eligible": False,
            "estimated_adam_wall_s": 1.0,
        }
    )

    loads = assign_workers(cases, worker_count=3)

    eligible_counts = [
        sum(case["adam_eligible"] and case["worker_index"] == worker for case in cases)
        for worker in range(3)
    ]
    assert eligible_counts == [4, 4, 4]
    assert max(loads) - min(loads) <= 1.0


def test_reconstruction_replays_the_condition_seed_stream() -> None:
    normalizer = CoilNormalizer(
        mean=np.zeros(100, dtype=np.float32),
        std=np.ones(100, dtype=np.float32),
        current_l1_a={"4:2": 2.0},
        clip=8.0,
    )
    manifest = {
        "sampling": {
            "seed_base": 123,
            "conditions": [{"nfp": 4, "n_base_coils": 2}],
        }
    }
    case = {
        "sample_id": "sample",
        "nfp": 4,
        "n_base_coils": 2,
        "reconstruction": {
            "worker_index": 3,
            "condition_index": 0,
            "condition_sample_index": 2,
        },
    }
    expected_rng = np.random.default_rng(np.random.SeedSequence([123, 3, 0]))
    expected = expected_rng.standard_normal((3, 2, 100), dtype=np.float32)[-1]

    sampled, effective, tokens, clipped = reconstruct_sample(
        normalizer, manifest, case
    )

    np.testing.assert_array_equal(sampled, expected)
    np.testing.assert_array_equal(tokens, normalizer.inverse(expected[None], (4, 2))[0])
    np.testing.assert_array_equal(effective, normalizer.transform(tokens[None], (4, 2))[0][0])
    assert clipped >= 0.0


def test_start_payload_does_not_reuse_a_survey_axis_hint() -> None:
    case = {
        "sample_id": "sample",
        "nfp": 4,
        "n_base_coils": 1,
        "survey_score": 21.0,
        "survey_status": "ok",
        "reconstruction": {
            "worker_index": 0,
            "condition_index": 0,
            "condition_sample_index": 0,
        },
    }
    payload = make_start_payload(case, np.zeros((1, 100), dtype=np.float32), 0.0)

    assert payload["noise"] == [[0.0] * 100]
    assert "native_score" not in payload["data_prior_screening"]
    assert payload["data_prior_screening"]["axis_hint_policy"].startswith("fresh")


def test_low_ok_quota_parser_requires_unique_nonnegative_entries() -> None:
    assert parse_low_ok_quotas("1:4,2:8") == {1: 4, 2: 8}


def test_optimizer_summary_must_match_the_frozen_64_direction_recipe() -> None:
    summary = {
        "status": "ok",
        "stop_reason": "completed_iterations",
        "completed_iterations": 200,
        "manifest": {
            "parameter_space": "data",
            "optimizer": "adam",
            "iterations": 200,
            "coordinate_gradient": {
                "mode": "random-orthogonal",
                "random_directions": 64,
                "difference": "centered",
                "perturbation": 0.0025,
                "formal_surface_theta_count": 128,
                "local_surface_theta_count": 64,
            },
            "adam": {
                "learning_rate": 0.01,
                "beta1": 0.7,
                "beta2": 0.999,
            },
        },
    }

    validate_optimizer_summary(summary)
    summary["manifest"]["coordinate_gradient"]["random_directions"] = 32
    with pytest.raises(RuntimeError, match="directions"):
        validate_optimizer_summary(summary)


def test_optimizer_library_api_gate_lists_missing_batch_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class IncompleteLibrary:
        pass

    monkeypatch.setattr(followup.ctypes, "CDLL", lambda _path: IncompleteLibrary())

    with pytest.raises(RuntimeError, match="sgpu_create_field_batch_f32"):
        validate_optimizer_library_api(Path("incomplete.so"))


def test_optimizer_library_api_gate_accepts_complete_symbol_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = type(
        "CompleteLibrary",
        (),
        {symbol: object() for symbol in OPTIMIZER_LIBRARY_SYMBOLS},
    )()
    monkeypatch.setattr(followup.ctypes, "CDLL", lambda _path: library)

    validate_optimizer_library_api(Path("complete.so"))
