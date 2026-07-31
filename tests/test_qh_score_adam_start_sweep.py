from __future__ import annotations

import json

import numpy as np

from scripts.optimize_flow_prior_zo_adam import load_initial_noise
from scripts.evaluate_qh_random_score_pool import summarize
from scripts.prepare_qh_score_adam_start_panel import (
    StartTarget,
    parse_target,
    select_targets,
)


def test_select_targets_does_not_reuse_row() -> None:
    rows = [
        {"score": value, "case_id": index, "status": "ok"}
        for index, value in enumerate((0.1, 0.2, 0.3))
    ]
    targets = [
        StartTarget("low_0", 0.19, "ok", 0.0, 1.0, 0),
        StartTarget("low_1", 0.21, "ok", 0.0, 1.0, 1),
    ]
    selected = select_targets(rows, targets)
    assert [row["case_id"] for _, row in selected] == [1, 2]


def test_parse_target_requires_score_inside_bounds() -> None:
    target = parse_target("high_0,45,ok,40,50,0")
    assert target.target_score == 45.0
    assert target.direction_seed_offset == 0


def test_load_initial_noise_accepts_generic_start(tmp_path) -> None:
    path = tmp_path / "start.json"
    noise = np.arange(300, dtype=np.float32).reshape(3, 100)
    path.write_text(
        json.dumps({"flow_prior_start": {"noise": noise.tolist()}}),
        encoding="utf-8",
    )
    loaded, payload = load_initial_noise(path)
    np.testing.assert_array_equal(loaded, noise)
    assert "flow_prior_start" in payload


def test_random_pool_summary_counts_high_score_tail() -> None:
    rows = [
        {"score": 2.0, "status": "no_axis"},
        {"score": 42.0, "status": "ok"},
        {"score": 55.0, "status": "ok"},
    ]
    summary = summarize(rows)
    assert summary["score_exceedance_counts"]["40"] == 2
    assert summary["score_exceedance_counts"]["50"] == 1
