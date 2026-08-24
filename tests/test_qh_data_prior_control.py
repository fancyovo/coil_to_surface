from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_qh_data_prior_control import worker_cases
from scripts.prepare_qh_data_prior_control import (
    assign_workers,
    load_reference_cases,
    select_cases,
)


def write_reference(root: Path, name: str, *, nfp: int, coils: int, wall: float) -> None:
    path = root / "trajectories" / name / "trajectory_manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "trajectory_id": name,
                "condition": {
                    "nfp": nfp,
                    "n_base_coils": coils,
                    "group": f"nfp{nfp}_coils{coils}",
                },
                "seeds": {"screening": 11 + coils, "optimizer": 21 + coils},
                "timing": {"trajectory_wall_s": wall},
                "optimization": {
                    "initial_score": 70 + coils,
                    "best_score": 80 + coils,
                },
            }
        ),
        encoding="utf-8",
    )


def test_reference_loading_and_greedy_worker_assignment(tmp_path: Path) -> None:
    write_reference(tmp_path, "case_a", nfp=4, coils=1, wall=10.0)
    write_reference(tmp_path, "case_b", nfp=4, coils=2, wall=9.0)
    write_reference(tmp_path, "case_c", nfp=5, coils=3, wall=8.0)
    write_reference(tmp_path, "case_d", nfp=5, coils=4, wall=7.0)

    cases = load_reference_cases(tmp_path)
    assigned, loads = assign_workers(cases, 2)

    assert len(assigned) == 4
    assert sum(loads) == 34.0
    assert max(loads) - min(loads) <= 2.0
    assert {row["worker_index"] for row in assigned} == {0, 1}


def test_worker_cases_filters_and_sorts() -> None:
    manifest = {
        "cases": [
            {"trajectory_id": "b", "worker_index": 1},
            {"trajectory_id": "c", "worker_index": 0},
            {"trajectory_id": "a", "worker_index": 1},
        ]
    }

    assert [row["trajectory_id"] for row in worker_cases(manifest, 1)] == [
        "a",
        "b",
    ]


def test_case_selection_is_reproducible_without_replacement() -> None:
    cases = [{"trajectory_id": f"case_{index:02d}"} for index in range(20)]

    first = select_cases(cases, case_count=8, seed=73)
    second = select_cases(cases, case_count=8, seed=73)

    assert first == second
    assert len(first) == 8
    assert len({row["trajectory_id"] for row in first}) == 8
