import json
import os
from pathlib import Path
import sys

import pytest

from evaluation.full_physical.run_full_evaluation import (
    run_gpu_stage,
    surface_expected_rejection,
)


ROOT = Path(__file__).resolve().parents[1]


def test_standard_submitter_submits_exactly_one_job() -> None:
    text = (
        ROOT / "evaluation" / "full_physical" / "submit_full_evaluation.sh"
    ).read_text(encoding="utf-8")
    assert text.count("sbatch --parsable") == 1
    assert "scripts/slurm_full_physical_evaluation.sh" in text
    assert text.count("sbatch --test-only") == 1
    assert "students-default" in text


def test_in_allocation_runner_does_not_submit_nested_jobs() -> None:
    text = (
        ROOT / "evaluation" / "full_physical" / "run_full_evaluation.py"
    ).read_text(encoding="utf-8")
    assert "sbatch" not in text
    assert "ThreadPoolExecutor" in text
    assert "select_source_psi_candidate.py" in text
    assert "select_largest_standard_surface.py" in text
    assert "slurm_evaluate_saved_boozer_full_cpu.sh" in text


def test_retired_direct_full_evaluation_entrypoint_is_disabled() -> None:
    text = (ROOT / "scripts" / "evaluate_cem_candidate_full.py").read_text(
        encoding="utf-8"
    )
    assert "Direct point-cloud full evaluation is disabled" in text
    assert "submit_full_evaluation.sh" in text


def test_full_evaluation_requires_alpha_nu_at_both_boundaries() -> None:
    solver = (ROOT / "scripts" / "solve_boozer_from_alpha_nu.py").read_text(
        encoding="utf-8"
    )
    downstream = (
        ROOT / "scripts" / "evaluate_saved_boozer_surface_full.py"
    ).read_text(encoding="utf-8")
    assert "ALPHA_NU_INITIALIZER_KIND" in solver
    assert "require_surface_kind" in solver
    assert "STANDARD_ALPHA_NU_SURFACE_KIND" in downstream
    assert "require_surface_kind" in downstream


def test_surface_exit_three_requires_structured_physical_rejection(
    tmp_path: Path,
) -> None:
    assert surface_expected_rejection(tmp_path, 3) is None
    assert surface_expected_rejection(tmp_path, 1) is None

    rejection = tmp_path / "alpha" / "rejection.json"
    rejection.parent.mkdir()
    rejection.write_text(
        '{"status":"rejected","stage":"fixed_budget_volume_sampling"}',
        encoding="utf-8",
    )

    result = surface_expected_rejection(tmp_path, 3)
    assert result is not None
    assert result["kind"] == "physical_candidate_rejection"


def test_standard_solver_rejection_is_expected(tmp_path: Path) -> None:
    summary = tmp_path / "standard_rho_1" / "summary.json"
    summary.parent.mkdir()
    summary.write_text(
        '{"target_s":0.64,"accepted_for_downstream":false}', encoding="utf-8"
    )

    result = surface_expected_rejection(tmp_path, 3)
    assert result is not None
    assert result["kind"] == "standard_surface_rejection"


def test_gpu_stage_preserves_expected_rejection(tmp_path: Path) -> None:
    output = tmp_path / "candidate"
    rejection = output / "alpha" / "rejection.json"
    rejection.parent.mkdir(parents=True)
    rejection.write_text(
        json.dumps(
            {"status": "rejected", "stage": "fixed_budget_volume_sampling"}
        ),
        encoding="utf-8",
    )

    results = run_gpu_stage(
        project=tmp_path,
        stage="surface",
        values=["0.64"],
        gpu_ids=["0"],
        candidate_cpus=1,
        base_env=os.environ.copy(),
        command_for_value=lambda _value: [
            sys.executable,
            "-c",
            "raise SystemExit(3)",
        ],
        env_for_value=lambda _value: {},
        output_for_value=lambda _value: output,
        log_dir=tmp_path / "logs",
        classify_nonzero=surface_expected_rejection,
    )

    assert results[0]["returncode"] == 3
    assert results[0]["outcome"] == "rejected"


def test_gpu_stage_rejects_unstructured_nonzero_exit(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="surface candidate failures"):
        run_gpu_stage(
            project=tmp_path,
            stage="surface",
            values=["0.64"],
            gpu_ids=["0"],
            candidate_cpus=1,
            base_env=os.environ.copy(),
            command_for_value=lambda _value: [
                sys.executable,
                "-c",
                "raise SystemExit(3)",
            ],
            env_for_value=lambda _value: {},
            output_for_value=lambda _value: tmp_path / "candidate",
            log_dir=tmp_path / "logs",
            classify_nonzero=surface_expected_rejection,
        )
