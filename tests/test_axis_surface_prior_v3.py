from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_compact_flexible_v3_protocol_is_score_only_and_count_bounded() -> None:
    protocol = json.loads(
        (REPO_ROOT / "evaluation" / "axis_surface_contour_prior_compact_flexible_abi11_v3.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["protocol_id"] == "qh-axis-surface-contour-compact-flexible-score-abi11-v3"
    assert protocol["generator"]["winding_surface_minor_radius_range_m"] == [0.18, 0.22]
    assert protocol["sample_design"]["total_count"] == 3600
    assert protocol["sample_design"]["samples_per_shard"] == 600
    assert protocol["evaluation"]["optimizer"] is None


def test_compact_flexible_v3_launcher_uses_six_gpus_and_one_hour_limit() -> None:
    submit = (REPO_ROOT / "scripts" / "submit_axis_surface_prior_v3_experiment.sh").read_text(
        encoding="utf-8"
    )
    worker = (REPO_ROOT / "scripts" / "slurm_axis_surface_prior_v3_worker.sh").read_text(
        encoding="utf-8"
    )
    assert "--array=0-3" in submit
    assert "--array=0-1" in submit
    assert 'total_count=3600' in submit
    assert 'START_DEPENDENCY' in submit
    assert "#SBATCH --time=01:00:00" in worker
    assert "sample_axis_surface_prior_v3.py" in worker
    assert "adam" not in worker.lower()
