from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


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
    parser.add_argument("--validation-per-helicity", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()

    metadata = load_quasr_metadata(args.metadata)
    calibration = stratified_rows(metadata, 0, args.per_helicity) + stratified_rows(
        metadata, 1, args.per_helicity
    )
    calibration_ids = {int(row["ID"]) for row in calibration}
    rng = np.random.default_rng(args.seed)
    validation = []
    for helicity in (0, 1):
        eligible = []
        for row in metadata:
            try:
                error = float(row["qs_error"])
                if (
                    int(row["helicity"]) == helicity
                    and int(row["ID"]) not in calibration_ids
                    and math.isfinite(error)
                    and error > 0.0
                ):
                    eligible.append(row)
            except (KeyError, TypeError, ValueError):
                continue
        if args.validation_per_helicity > len(eligible):
            raise ValueError(
                f"requested {args.validation_per_helicity} validation rows for helicity "
                f"{helicity}, only {len(eligible)} are eligible"
            )
        indices = rng.choice(
            len(eligible), size=args.validation_per_helicity, replace=False
        )
        validation.extend(eligible[int(index)] for index in indices)
    selected = [(row, "calibration") for row in calibration] + [
        (row, "validation") for row in validation
    ]
    case_dir = args.output_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    selected_metadata = []
    for index, (row, split) in enumerate(selected, start=1):
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
                "split": split,
            }
        )
        selected_metadata.append({**row, "_split": split})
        print(
            f"[{index:04d}/{len(selected):04d}] split={split} id={device_id}",
            flush=True,
        )

    write_json(args.output_dir / "metadata_selected.json", selected_metadata)
    write_json(
        args.output_dir / "manifest.json",
        {
            "selection": {
                "calibration": "qs_error_rank_stratified_per_helicity",
                "validation": "seeded_random_per_helicity_excluding_calibration",
                "calibration_per_helicity": int(args.per_helicity),
                "validation_per_helicity": int(args.validation_per_helicity),
                "seed": int(args.seed),
            },
            "count": len(manifest),
            "samples": manifest,
        },
    )
    print(json.dumps({"count": len(manifest), "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
