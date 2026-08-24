from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))



def select_trajectory_manifests(
    root: Path,
    *,
    case_count: int,
    seed: int,
) -> list[Path]:
    manifests = sorted(root.glob("trajectories/*/trajectory_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"no trajectories found below {root}")
    grouped: dict[tuple[int, int], list[Path]] = defaultdict(list)
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        condition = payload["condition"]
        grouped[(int(condition["nfp"]), int(condition["n_base_coils"]))].append(path)

    rng = np.random.default_rng(seed)
    for paths in grouped.values():
        rng.shuffle(paths)
    ordered_groups = sorted(grouped, key=lambda key: (-len(grouped[key]), key))
    selected: list[Path] = []
    round_index = 0
    target = min(case_count, len(manifests))
    while len(selected) < target:
        added = False
        for key in ordered_groups:
            paths = grouped[key]
            if round_index < len(paths):
                selected.append(paths[round_index])
                added = True
                if len(selected) == target:
                    break
        if not added:
            break
        round_index += 1
    return selected


def selected_native_score(payload: dict[str, Any]) -> float:
    screening = payload.get("flow_prior_screening", {})
    native = screening.get("native_score", {})
    if "score" in native:
        return float(native["score"])
    if "score" in payload:
        return float(payload["score"])
    return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-count", type=int, default=32)
    parser.add_argument("--selection-seed", type=int, default=2026082402)
    args = parser.parse_args()
    selected = select_trajectory_manifests(
        args.trajectory_root,
        case_count=args.case_count,
        seed=args.selection_seed,
    )
    case_dir = args.output_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, manifest_path in enumerate(selected):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        condition = manifest["condition"]
        trajectory_id = str(manifest["trajectory_id"])
        source_start = manifest_path.parent / "screening" / "selected_start.json"
        payload = json.loads(source_start.read_text(encoding="utf-8"))
        noise = np.asarray(
            payload.get("flow_prior_screening", {}).get("noise", payload.get("noise")),
            dtype=np.float32,
        )
        n_base_coils = int(condition["n_base_coils"])
        if noise.shape != (n_base_coils, 100):
            raise ValueError(
                f"{trajectory_id} selected start has noise shape {noise.shape}, "
                f"expected {(n_base_coils, 100)}"
            )
        case_id = f"{trajectory_id}_start"
        path = case_dir / f"case_{index:02d}.json"
        shutil.copyfile(source_start, path)
        rows.append(
            {
                "index": index,
                "case_id": case_id,
                "trajectory_id": trajectory_id,
                "source_stage": "screening_selected_start",
                "selected_candidate_index": int(
                    payload.get("flow_prior_screening", {}).get("candidate_index", -1)
                ),
                "selected_score": selected_native_score(payload),
                "nfp": int(condition["nfp"]),
                "n_base_coils": n_base_coils,
                "case_path": str(path.resolve()),
                "source_manifest": str(manifest_path.resolve()),
                "source_start": str(source_start.resolve()),
            }
        )
    manifest = {
        "format": "summary1_flow_parameterization_pairs_v2",
        "trajectory_root": str(args.trajectory_root.resolve()),
        "selection_seed": args.selection_seed,
        "selection_rule": (
            "one original 32-candidate screening winner from each distinct trajectory; "
            "conditions sampled in balanced round-robin order"
        ),
        "cases": rows,
    }
    (args.output_dir / "pair_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"case_count": len(rows), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
