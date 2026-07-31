from __future__ import annotations

import json

import numpy as np

from scripts.analyze_qh_score_adam_start_sweep import analyze
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


def test_analyze_tracks_gain_and_recorded_score_discrepancy(tmp_path) -> None:
    starts = []
    for start_id, initial in enumerate((5.0, 20.0, 45.0)):
        starts.append(
            {
                "start_id": start_id,
                "stratum": f"tier_{start_id}",
                "recorded_score": initial - 0.01,
            }
        )
        output = tmp_path / f"start_{start_id:02d}"
        output.mkdir()
        (output / "summary.json").write_text(
            json.dumps(
                {
                    "initial_score": initial,
                    "final_score": initial + 1.0 + 0.1 * start_id,
                    "best_score": initial + 1.5 + 0.2 * start_id,
                    "best_iteration": 2,
                    "completed_iterations": 2,
                    "total_wall_s": 20.0,
                    "mean_iteration_wall_s": 10.0,
                }
            ),
            encoding="utf-8",
        )
        history = [
            {
                "iteration": iteration,
                "current_score": initial + 0.5 * iteration,
                "best_score": initial + 0.75 * iteration,
                "valid_endpoint_fraction": 1.0,
            }
            for iteration in (1, 2)
        ]
        (output / "history.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in history),
            encoding="utf-8",
        )

    summary, rows = analyze({"starts": starts}, tmp_path)

    assert summary["count"] == 3
    assert rows[0]["best_gain"] == 1.5
    assert abs(rows[0]["initial_score_discrepancy"] - 0.01) < 1.0e-12
    assert rows[0]["checkpoints"]["1"]["current_gain"] == 0.5
