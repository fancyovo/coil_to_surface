from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_case_shape(path: Path) -> tuple[int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data["raw"]
    nfp = int(data.get("nfp", raw.get("nfp")))
    return nfp, len(raw["current"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a strict-axis psi-grid holdout from an existing score validation run."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_files = sorted(args.source_dir.glob("worker_*.jsonl"))
    if not source_files:
        raise FileNotFoundError(f"no worker_*.jsonl files under {args.source_dir}")

    selected: dict[int, dict] = {}
    total_rows = 0
    for source_path in source_files:
        for line in source_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            total_rows += 1
            row = json.loads(line)
            legacy = row["legacy"]
            if legacy["status"] != "ok":
                continue
            case_id = int(row["case_id"])
            case_path = args.case_dir / f"id_{case_id:07d}.json"
            nfp, n_base_coils = load_case_shape(case_path)
            if nfp != int(row["nfp"]):
                raise ValueError(f"case {case_id}: nfp mismatch ({nfp} != {row['nfp']})")
            diagnostics = legacy["diagnostics"]
            selected[case_id] = {
                "case_id": case_id,
                "case_path": str(case_path.resolve()),
                "nfp": nfp,
                "n_base_coils": n_base_coils,
                "metadata_qs_error": float(row["metadata_qs_error"]),
                "metadata_mean_iota": float(row["metadata_mean_iota"]),
                "selection_score": float(legacy["score"]),
                "axis_R": float(diagnostics["axis_R"]),
                "axis_Z": float(diagnostics["axis_Z"]),
            }

    cases = [selected[case_id] for case_id in sorted(selected)]
    if not cases:
        raise RuntimeError("no legacy status=ok cases found")
    payload = {
        "schema": "psi-grid-strict-hint-holdout-v1",
        "source_dir": str(args.source_dir.resolve()),
        "source_files": [str(path.resolve()) for path in source_files],
        "source_row_count": total_rows,
        "selection_rule": "one row per case with legacy.status == 'ok'",
        "case_count": len(cases),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"source_rows": total_rows, "selected_cases": len(cases)}))


if __name__ == "__main__":
    main()
