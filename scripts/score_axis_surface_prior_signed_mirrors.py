from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "gpu_backend" / "python"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.sample_axis_surface_prior import compact_result


TRANSFORMS = {
    "identity": (1.0, 1.0, 1.0),
    "reflect_y": (1.0, -1.0, 1.0),
    "reflect_z": (1.0, 1.0, -1.0),
    "rotate_pi_about_x": (1.0, -1.0, -1.0),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transform_raw(raw: dict[str, Any], signs: tuple[float, float, float]) -> dict[str, Any]:
    transformed = copy.deepcopy(raw)
    for key, sign in zip(("x", "y", "z"), signs, strict=True):
        values = np.asarray(raw[key], dtype=float)
        transformed[key] = (sign * values).tolist()
    return transformed


def score_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    component_names = sorted(set(left.get("components", {})) & set(right.get("components", {})))
    return {
        "score_absolute_difference": abs(float(left["score"]) - float(right["score"])),
        "component_absolute_differences": {
            name: abs(float(left["components"][name]) - float(right["components"][name]))
            for name in component_names
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score analytic-prior endpoints under both target signs and mirror controls.")
    parser.add_argument("--case-file", type=Path, action="append", required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--expected-lib-sha", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    if commit != args.expected_commit:
        raise RuntimeError(f"repository commit {commit} != {args.expected_commit}")
    actual_lib_sha = file_sha256(args.lib)
    if actual_lib_sha != args.expected_lib_sha:
        raise RuntimeError("score library hash mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    from stellarator_gpu import score_coils_native

    cases = []
    for case_path in args.case_file:
        payload = json.loads(case_path.read_text(encoding="utf-8"))
        raw = payload["raw"]
        nfp = int(payload["nfp"])
        sample_id = case_path.parents[1].name
        scores: dict[str, dict[str, Any]] = {}
        for transform_name, signs in TRANSFORMS.items():
            transformed = transform_raw(raw, signs)
            determinant = float(np.prod(np.asarray(signs, dtype=float)))
            for target_sign in (1, -1):
                key = f"{transform_name}__target_{'positive' if target_sign > 0 else 'negative'}"
                result = score_coils_native(
                    args.lib,
                    transformed["x"],
                    transformed["y"],
                    transformed["z"],
                    transformed["current"],
                    nfp,
                    device_id=args.device,
                    target_helicity=(1, target_sign * nfp),
                )
                compact = compact_result(result)
                compact["coordinate_transform"] = {
                    "name": transform_name,
                    "diagonal": list(signs),
                    "determinant": determinant,
                }
                compact["target_helicity"] = [1, target_sign * nfp]
                scores[key] = compact

        comparisons = {}
        for mirror_name in ("reflect_y", "reflect_z"):
            comparisons[f"identity_negative_vs_{mirror_name}_positive"] = score_delta(
                scores["identity__target_negative"],
                scores[f"{mirror_name}__target_positive"],
            )
            comparisons[f"identity_positive_vs_{mirror_name}_negative"] = score_delta(
                scores["identity__target_positive"],
                scores[f"{mirror_name}__target_negative"],
            )
        comparisons["proper_rotation_same_positive_target"] = score_delta(
            scores["identity__target_positive"],
            scores["rotate_pi_about_x__target_positive"],
        )
        cases.append(
            {
                "sample_id": sample_id,
                "case_file": str(case_path.resolve()),
                "case_sha256": file_sha256(case_path),
                "nfp": nfp,
                "scores": scores,
                "comparisons": comparisons,
            }
        )

    output = {
        "format": "axis_surface_prior_signed_mirror_score_audit_v1",
        "code_commit": commit,
        "score_library": str(args.lib.resolve()),
        "score_library_sha256": actual_lib_sha,
        "interpretation": (
            "Improper coordinate reflections have determinant -1 and are expected to exchange helicity signs; "
            "the determinant +1 rotation is a same-sign invariance control."
        ),
        "cases": cases,
    }
    output_path = args.output_dir / "signed_mirror_scores.json"
    output_path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "signed_mirror_score_audit_complete", "output": str(output_path)}, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
