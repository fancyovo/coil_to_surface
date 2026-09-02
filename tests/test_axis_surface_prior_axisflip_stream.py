from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_axis_surface_prior_axisflip_stream import (
    PROTOCOL_ID,
    TARGET_HELICITY_SIGN,
    case_id_for_worker,
    classify_iota_interval,
    discovery_is_open,
    formal_screening_score_config,
    optimizer_command,
)
from scripts.optimize_flow_latent import validate_recorded_initial_score
from scripts.sample_axis_surface_prior import compact_result


def test_case_streams_are_disjoint_and_cover_the_interleaved_population() -> None:
    ids = [
        case_id_for_worker(worker, 6, sequence)
        for sequence in range(4)
        for worker in range(6)
    ]
    assert ids == list(range(24))


def test_case_stream_rejects_invalid_indices() -> None:
    with pytest.raises(ValueError, match="worker-index"):
        case_id_for_worker(6, 6, 0)
    with pytest.raises(ValueError, match="sequence-index"):
        case_id_for_worker(0, 6, -1)


def test_discovery_deadline_is_soft_and_excludes_the_boundary() -> None:
    assert discovery_is_open(14399.9, 14400.0)
    assert not discovery_is_open(14400.0, 14400.0)
    assert not discovery_is_open(15000.0, 14400.0)


def test_iota_interval_classification() -> None:
    assert classify_iota_interval(-1.2, -0.8) == "negative"
    assert classify_iota_interval(0.4, 0.9) == "positive"
    assert classify_iota_interval(-0.1, 0.2) == "crosses_zero"
    assert classify_iota_interval(float("nan"), 0.2) == "missing"


def test_optimizer_command_pins_positive_hand_abi11_recipe() -> None:
    command = optimizer_command(
        checkpoint=Path("checkpoint.pt"),
        start_path=Path("start.json"),
        score_lib=Path("libscore.so"),
        output_dir=Path("optimization"),
        nfp=7,
        n_base_coils=4,
        iterations=200,
        max_wall_s=3000.0,
        optimizer_seed=123,
        device=0,
    )
    values = dict(zip(command[2::2], command[3::2]))
    assert TARGET_HELICITY_SIGN == 1
    assert values["--target-helicity-sign"] == "1"
    assert values["--iterations"] == "200"
    assert values["--random-directions"] == "64"
    assert values["--perturbation"] == "0.0025"
    assert values["--learning-rate"] == "0.01"
    assert values["--parameter-space"] == "data"
    assert values["--data-start-mode"] == "exact-unclipped"
    assert values["--recorded-initial-score-tolerance"] == "0.1"


def test_compact_result_retains_magnetic_axis_for_optimizer_continuation() -> None:
    compact = compact_result(
        {
            "score": 72.5,
            "status": "ok",
            "components": {},
            "diagnostics": {"axis_R": 1.25, "axis_Z": -0.125},
        }
    )
    assert compact["diagnostics"]["axis_R"] == 1.25
    assert compact["diagnostics"]["axis_Z"] == -0.125


def test_recorded_initial_score_gate_requires_axis_and_checks_before_updates() -> None:
    recorded = {
        "score": 72.5,
        "diagnostics": {"axis_R": 1.25, "axis_Z": -0.125},
    }
    current = {
        "score": 72.55,
        "diagnostics": {
            "axis_R": 1.2501,
            "axis_Z": -0.1249,
            "axis_hint_distance": 1.4e-4,
        },
    }
    gate = validate_recorded_initial_score(recorded, current, tolerance=0.1)
    assert gate["absolute_delta"] == pytest.approx(0.05)
    with pytest.raises(RuntimeError, match="differs from screening"):
        validate_recorded_initial_score(
            recorded, {**current, "score": 72.7}, tolerance=0.1
        )
    with pytest.raises(RuntimeError, match="axis_R/axis_Z"):
        validate_recorded_initial_score(
            {"score": 72.5, "diagnostics": {}},
            current,
            tolerance=0.1,
        )


def test_v1_through_v3_are_invalidated_and_v4_is_registered() -> None:
    v1 = json.loads(
        Path(
            "evaluation/axis_surface_contour_prior_compact_flexible_"
            "axisflip_stream_adam200_abi11_v1.json"
        ).read_text(encoding="utf-8")
    )
    v2 = json.loads(
        Path(
            "evaluation/axis_surface_contour_prior_compact_flexible_"
            "axisflip_stream_adam200_abi11_v2.json"
        ).read_text(encoding="utf-8")
    )
    assert v1["status"] == "invalidated"
    assert v1["replacement_protocol_id"].endswith("abi11-v2")
    assert v2["status"] == "invalidated"
    assert v2["replacement_protocol_id"].endswith("abi11-v3")
    assert v2["screening"]["pre_update_consistency_tolerance"] == 0.1
    v3 = json.loads(
        Path(
            "evaluation/axis_surface_contour_prior_compact_flexible_"
            "axisflip_stream_adam200_abi11_v3.json"
        ).read_text(encoding="utf-8")
    )
    assert v3["status"] == "invalidated"
    assert v3["replacement_protocol_id"] == PROTOCOL_ID
    v4 = json.loads(
        Path(
            "evaluation/axis_surface_contour_prior_compact_flexible_"
            "axisflip_stream_adam200_abi11_v4.json"
        ).read_text(encoding="utf-8")
    )
    assert v4["protocol_id"] == PROTOCOL_ID


def test_screening_uses_the_optimizer_formal_score_configuration() -> None:
    config = formal_screening_score_config()
    assert config == {
        "iota_degree": 3,
        "surface_selection_mode": 1,
        "surface_confidence_periods": 1,
        "surface_theta_count": 128,
        "surface_trace_steps": 400,
        "surface_flux_bisection_iters": 6,
    }
