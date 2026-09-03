from __future__ import annotations

import json
from pathlib import Path

import pytest

from flow_matching.axis_surface_prior_v2 import sample_shaped_prior_prototype
from scripts.run_axis_surface_prior_axisflip_stream import (
    RADIUS012_GENERATOR_FORMAT,
    RADIUS012_PROTOCOL_ID,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_radius012_generator_override_is_explicit_and_bounded() -> None:
    sample = sample_shaped_prior_prototype(
        seed=20260906,
        nfp=8,
        n_base_coils=3,
        preset="compact_flexible",
        curve_samples=128,
        frame_samples=512,
        surface_phi_samples=48,
        surface_theta_samples=24,
        sample_role="registered_scoring",
        axis_chirality=-1,
        minor_radius_m=0.12,
    )
    metadata = sample.metadata
    assert metadata["format"] == RADIUS012_GENERATOR_FORMAT
    assert metadata["minor_radius_center_m"] == 0.12
    assert metadata["minor_radius_override"] is True
    assert 0.108 <= metadata["parameters"]["minor_radius"] <= 0.132
    assert metadata["nfp"] == 8
    assert metadata["n_base_coils"] == 3


def test_radius_override_rejects_unregistered_contexts() -> None:
    with pytest.raises(ValueError, match="registered only"):
        sample_shaped_prior_prototype(
            seed=1,
            nfp=8,
            n_base_coils=3,
            preset="compact_flexible",
            sample_role="geometry_review",
            axis_chirality=-1,
            minor_radius_m=0.12,
        )


def test_protocol_and_launchers_freeze_six_card_count_bounded_design() -> None:
    protocol = json.loads(
        (
            REPO_ROOT
            / "evaluation"
            / "axis_surface_contour_prior_r012_curvature_r04_axisflip_adam200_abi11_v1.json"
        ).read_text(encoding="utf-8")
    )
    worker = (
        REPO_ROOT / "scripts" / "slurm_axisflip_r012_curvature_r04_worker.sh"
    ).read_text(encoding="utf-8")
    submit = (
        REPO_ROOT / "scripts" / "submit_axisflip_r012_curvature_r04_experiment.sh"
    ).read_text(encoding="utf-8")
    build = (
        REPO_ROOT / "scripts" / "slurm_axisflip_r012_curvature_r04_build_smoke.sh"
    ).read_text(encoding="utf-8")

    assert protocol["protocol_id"] == RADIUS012_PROTOCOL_ID
    assert protocol["generator"]["fixed_condition"] == {
        "nfp": 8,
        "n_base_coils": 3,
    }
    assert protocol["legality_audit"]["fixed_total_samples"] == 384
    assert protocol["adam_selection"]["completed_trajectories_total"] == 12
    assert protocol["score_configuration"]["coil_curvature_p95_scale_m_inv"] == 25.0
    assert protocol["score_configuration"]["coil_curvature_max_scale_m_inv"] == 35.0
    assert "--array=0-3" in submit
    assert "--array=0-1" in submit
    assert "--max-completed-cases 2" in worker
    assert "--legality-audit-cases-per-worker 64" in worker
    assert "--fixed-nfp 8" in worker
    assert "--fixed-n-base-coils 3" in worker
    assert "--random-directions" not in worker
    assert "-DSGPU_COIL_CURVATURE_P95_SCALE=25.0" in build


def test_default_native_build_keeps_original_p95_scale() -> None:
    cmake = (REPO_ROOT / "gpu_backend" / "CMakeLists.txt").read_text(encoding="utf-8")
    source = (REPO_ROOT / "gpu_backend" / "src" / "score_pipeline.cu").read_text(
        encoding="utf-8"
    )
    assert 'set(SGPU_COIL_CURVATURE_P95_SCALE "10.0"' in cmake
    assert source.count("COIL_CURVATURE_P95_SCALE") >= 4
    assert "q_down(metrics.curvature_p95, 10.0" not in source
    assert "q_down_derivative(metrics.curvature_p95, 10.0" not in source
