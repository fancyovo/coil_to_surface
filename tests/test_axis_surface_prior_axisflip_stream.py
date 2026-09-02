from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_axis_surface_prior_axisflip_stream import (
    TARGET_HELICITY_SIGN,
    case_id_for_worker,
    classify_iota_interval,
    discovery_is_open,
    optimizer_command,
)


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
