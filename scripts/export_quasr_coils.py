from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.batch_volume_qs_quasr import stratified_rows
from stellarator_eval.quasr import (
    load_quasr_field_input,
    load_quasr_metadata,
    quasr_failure_case_payload,
)
from stellarator_eval.serialization import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quasr-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-helicity", type=int, default=20)
    args = parser.parse_args()

    metadata = load_quasr_metadata(args.metadata)
    selected = stratified_rows(metadata, 0, args.per_helicity) + stratified_rows(
        metadata, 1, args.per_helicity
    )
    case_dir = args.output_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, row in enumerate(selected, start=1):
        device_id = int(row["ID"])
        field_input, info = load_quasr_field_input(args.quasr_root, device_id)
        payload = quasr_failure_case_payload(
            field_input,
            device_id=device_id,
            metadata_row=row,
        )
        case_path = case_dir / f"id_{device_id:07d}.json"
        write_json(case_path, payload)
        manifest.append(
            {
                "id": device_id,
                "helicity": int(row["helicity"]),
                "nfp": int(field_input.nfp),
                "base_coils": int(info["nc_per_hp"]),
                "curve_order": int(info["curve_order"]),
                "file": str(case_path.relative_to(args.output_dir)),
            }
        )
        print(f"[{index:02d}/{len(selected):02d}] id={device_id}", flush=True)

    write_json(args.output_dir / "metadata_selected.json", selected)
    write_json(
        args.output_dir / "manifest.json",
        {
            "selection": "qs_error_rank_stratified_per_helicity",
            "per_helicity": int(args.per_helicity),
            "count": len(manifest),
            "samples": manifest,
        },
    )
    print(json.dumps({"count": len(manifest), "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
