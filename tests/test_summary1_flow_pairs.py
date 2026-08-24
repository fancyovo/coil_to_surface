from __future__ import annotations

import json
from pathlib import Path

from scripts.prepare_summary1_flow_pairs import (
    select_trajectory_manifests,
    selected_native_score,
)


def write_manifest(root: Path, trajectory_id: str, nfp: int, n_coils: int) -> None:
    path = root / "trajectories" / trajectory_id / "trajectory_manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "trajectory_id": trajectory_id,
                "condition": {"nfp": nfp, "n_base_coils": n_coils},
            }
        ),
        encoding="utf-8",
    )


def test_select_trajectory_manifests_balances_conditions(tmp_path: Path) -> None:
    for index in range(3):
        write_manifest(tmp_path, f"a{index}", 3, 2)
    for index in range(2):
        write_manifest(tmp_path, f"b{index}", 4, 3)

    selected = select_trajectory_manifests(tmp_path, case_count=4, seed=17)

    assert len(selected) == 4
    conditions = []
    for path in selected:
        payload = json.loads(path.read_text(encoding="utf-8"))
        condition = payload["condition"]
        conditions.append((condition["nfp"], condition["n_base_coils"]))
    assert conditions.count((3, 2)) == 2
    assert conditions.count((4, 3)) == 2


def test_selected_native_score_reads_screening_result() -> None:
    payload = {"flow_prior_screening": {"native_score": {"score": 81.25}}}
    assert selected_native_score(payload) == 81.25
