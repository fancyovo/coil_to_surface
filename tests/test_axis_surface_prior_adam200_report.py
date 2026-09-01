from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.render_axis_surface_prior_adam200_report_figures import (
    first_passage_summary,
    plot_success_rates,
)


ROOT = Path(__file__).resolve().parents[1]


def test_success_rate_plot_accepts_a_zero_success_group(tmp_path) -> None:
    rows = [
        {"n_base_coils": 1, "nfp": 3, "best_score": 4.0},
        {"n_base_coils": 1, "nfp": 3, "best_score": 8.0},
        {"n_base_coils": 2, "nfp": 4, "best_score": 55.0},
    ]
    output = tmp_path / "success_rates.png"
    plot_success_rates(rows, output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_first_passage_summary_separates_best_so_far_from_current_score() -> None:
    steps = np.arange(201)
    crosses_and_falls = np.full(201, 10.0)
    crosses_and_falls[40:80] = 55.0
    crosses_late = np.full(201, 5.0)
    crosses_late[180:] = 60.0
    never_crosses = np.full(201, 12.0)
    summary = first_passage_summary(
        [
            {"nc": 1, "steps": steps, "scores": crosses_and_falls},
            {"nc": 2, "steps": steps, "scores": crosses_late},
            {"nc": 2, "steps": steps, "scores": never_crosses},
        ]
    )
    assert summary["crossing_trajectories"] == 2
    assert summary["crossed_by_step"]["50"] == 1
    assert summary["crossed_by_step"]["200"] == 2
    assert summary["current_score_ge_threshold_at_step"]["50"] == 1
    assert summary["current_score_ge_threshold_at_step"]["100"] == 0


def test_full_evaluation_launchers_validate_gpu_abi_before_submission() -> None:
    for name in ("submit_source_psi_candidates.sh", "submit_surface_candidates.sh"):
        text = (ROOT / "evaluation" / "full_physical" / name).read_text(encoding="utf-8")
        guard = text.index("validate_gpu_library(Path(sys.argv[1]))")
        policy = text.index("write_submission_policy.py")
        manifest = text.index("manifest=$OUTPUT_ROOT/")
        assert "serial_candidates=${SERIAL_CANDIDATES:-0}" in text
        assert "serial_reason=${SERIAL_REASON:-}" in text
        assert "SERIAL_REASON is required when SERIAL_CANDIDATES=1" in text
        assert "CANDIDATE_POOLS" in text
        assert guard < policy < manifest
