from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_PYTHON = REPO_ROOT / "gpu_backend" / "python"
if str(GPU_PYTHON) not in sys.path:
    sys.path.insert(0, str(GPU_PYTHON))

from stellarator_gpu import score_coils_native


PRODUCTION_SCORE = {
    "surface_selection_mode": 1,
    "surface_confidence_periods": 1,
    "surface_theta_count": 128,
    "surface_trace_steps": 400,
}


def load_case(path: Path) -> tuple[list, list, list, list[float], int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data["raw"]
    unit = str(raw.get("current_unit", "A")).lower()
    scale = 1.0 if unit in {"a", "amp", "amps"} else 1.0e6
    return (
        raw["x"],
        raw["y"],
        raw["z"],
        [float(value) * scale for value in raw["current"]],
        int(data.get("nfp", raw.get("nfp"))),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select good QUASR QH cases and record axis hints outside the timed benchmark."
    )
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--candidate-limit", type=int, default=48)
    parser.add_argument("--selected-count", type=int, default=8)
    args = parser.parse_args()

    metadata_rows = json.loads(args.metadata.read_text(encoding="utf-8"))
    candidates = [
        row for row in metadata_rows
        if int(row.get("helicity", -1)) == 1 and row.get("_split") == args.split
    ]
    candidates.sort(key=lambda row: float(row["qs_error"]))
    candidates = candidates[: args.candidate_limit]

    evaluated = []
    for row in candidates:
        case_id = int(row["ID"])
        path = args.case_dir / f"id_{case_id:07d}.json"
        x, y, z, currents, nfp = load_case(path)
        started = time.perf_counter()
        result = score_coils_native(
            args.lib,
            x,
            y,
            z,
            currents,
            nfp,
            device_id=args.device,
            target_helicity=(1, nfp),
            config_overrides=PRODUCTION_SCORE,
        )
        diagnostics = result["diagnostics"]
        evaluated.append(
            {
                "case_id": case_id,
                "case_path": str(path.resolve()),
                "nfp": nfp,
                "n_base_coils": len(currents),
                "metadata_qs_error": float(row["qs_error"]),
                "metadata_mean_iota": float(row["mean_iota"]),
                "selection_wall_s": time.perf_counter() - started,
                "selection_score": float(result["score"]),
                "selection_status": result["status"],
                "axis_R": float(diagnostics["axis_R"]),
                "axis_Z": float(diagnostics["axis_Z"]),
                "selection_result": result,
            }
        )

    valid = [
        row for row in evaluated
        if row["selection_status"] == "ok"
        and abs(row["axis_R"]) < float("inf")
        and abs(row["axis_Z"]) < float("inf")
    ]
    valid.sort(key=lambda row: (-row["selection_score"], row["metadata_qs_error"]))
    selected = valid[: args.selected_count]
    if len(selected) != args.selected_count:
        raise RuntimeError(
            f"only {len(selected)} valid QH cases among {len(candidates)} candidates; "
            f"requested {args.selected_count}"
        )

    payload = {
        "schema": "score-eval-profile-cases-v1",
        "selection_is_outside_timing": True,
        "selection_config": PRODUCTION_SCORE,
        "library_sha256": sha256(args.lib),
        "split": args.split,
        "candidate_limit": args.candidate_limit,
        "selected_count": len(selected),
        "cases": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps({"selected": len(selected), "output": str(args.output)}))


if __name__ == "__main__":
    main()
