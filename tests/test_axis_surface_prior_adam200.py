from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from flow_matching.data import CoilNormalizer
from scripts.optimize_flow_latent import recorded_axis_hint
from scripts.prepare_axis_surface_prior_adam200 import (
    assign_workers,
    exact_standardized_parameters,
    select_valid_rows,
)
from scripts.run_axis_surface_prior_adam200 import trajectory_wall_limit
from scripts.run_axis_surface_prior_adam200 import artifact_format


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_optimizer_repository_root_reference_is_defined() -> None:
    source = (REPO_ROOT / "scripts" / "optimize_flow_latent.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    loaded_names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    assert "REPO_ROOT" in loaded_names
    assert "PROJECT_ROOT" not in loaded_names


def test_recorded_axis_hint_requires_complete_finite_coordinates() -> None:
    assert recorded_axis_hint(None) is None
    assert recorded_axis_hint({"diagnostics": {"score": 12.0}}) is None
    assert recorded_axis_hint({"diagnostics": {"axis_R": 1.2, "axis_Z": float("nan")}}) is None
    assert recorded_axis_hint({"diagnostics": {"axis_R": 1.2, "axis_Z": -0.3}}) == (
        1.2,
        -0.3,
    )


def test_trajectory_wall_limit_allows_slow_nc4_without_exceeding_worker_budget() -> None:
    assert trajectory_wall_limit(worker_elapsed_s=0.0, worker_max_wall_s=17400.0) == 7200.0
    assert trajectory_wall_limit(worker_elapsed_s=14600.0, worker_max_wall_s=17400.0) == 2500.0


def test_artifact_format_defaults_to_frozen_v2_and_accepts_registered_override() -> None:
    assert artifact_format({}, "trajectory") == (
        "axis_surface_prior_balanced_v2_adam200_trajectory_v1"
    )
    manifest = {"artifact_formats": {"trajectory": "registered_v3_trajectory_v1"}}
    assert artifact_format(manifest, "trajectory") == "registered_v3_trajectory_v1"


def _row(case_id: int, status: str | None, nc: int) -> dict[str, object]:
    return {
        "case_id": case_id,
        "n_base_coils": nc,
        "native": {"status": status} if status is not None else None,
    }


def test_random_selection_uses_only_valid_rows_and_is_reproducible() -> None:
    rows = [
        _row(index, "ok" if index % 2 else (None if index == 0 else "no_axis"), 1 + index % 4)
        for index in range(30)
    ]
    first = select_valid_rows(rows, count=8, seed=73)
    second = select_valid_rows(rows, count=8, seed=73)
    assert [row["case_id"] for row in first] == [row["case_id"] for row in second]
    assert len({row["case_id"] for row in first}) == 8
    assert all(row["native"]["status"] == "ok" for row in first)


def test_worker_assignment_has_equal_counts_and_balances_expensive_cases() -> None:
    rows = [_row(index, "ok", 1 + index % 4) for index in range(84)]
    assignment = assign_workers(rows, worker_count=6)
    counts = [sum(worker == index for worker in assignment.values()) for index in range(6)]
    assert counts == [14] * 6


def test_exact_standardized_parameters_preserve_the_physical_start() -> None:
    rng = np.random.default_rng(17)
    tokens = rng.normal(size=(2, 100)).astype(np.float32)
    tokens[:, -1] = np.asarray([180000.0, 220000.0], dtype=np.float32)
    normalizer = CoilNormalizer(
        mean=np.zeros(100, dtype=np.float32),
        std=np.linspace(0.5, 2.0, 100, dtype=np.float32),
        current_l1_a={"7:2": 123.0},
    )
    parameters, current_l1_a, diagnostics = exact_standardized_parameters(
        tokens, normalizer, condition=(7, 2)
    )
    exact = CoilNormalizer(
        mean=normalizer.mean,
        std=normalizer.std,
        current_l1_a={"7:2": current_l1_a},
        clip=float("inf"),
    )
    reconstructed = exact.inverse(parameters[None], (7, 2))[0]
    np.testing.assert_allclose(reconstructed, tokens, rtol=2.0e-6, atol=2.0e-6)
    assert diagnostics["geometry_relative_rms"] < 2.0e-6
    assert diagnostics["current_relative_rms"] < 2.0e-6
