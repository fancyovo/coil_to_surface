from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
GPU_PYTHON = REPO_ROOT / "gpu_backend" / "python"
if str(GPU_PYTHON) not in sys.path:
    sys.path.insert(0, str(GPU_PYTHON))

from stellarator_eval.field import load_case_file
from stellarator_gpu import score_coils_native


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_file", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument(
        "--lib",
        type=Path,
        default=REPO_ROOT / "gpu_backend" / "build_native_score" / "libstellarator_gpu.so",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    field_input = load_case_file(args.case_file, "raw")
    match = re.search(r"(\d+)", args.case_file.stem)
    case_id = int(match.group(1)) if match else None
    metadata = None
    if args.metadata:
        rows = json.loads(args.metadata.read_text(encoding="utf-8"))
        metadata = next((row for row in rows if int(row["ID"]) == case_id), None)
    helicity = int(metadata["helicity"]) if metadata else 0
    target = (1, 0 if helicity == 0 else int(field_input.nfp))
    result = score_coils_native(
        args.lib,
        field_input.coeffs_x,
        field_input.coeffs_y,
        field_input.coeffs_z,
        field_input.currents,
        field_input.nfp,
        device_id=args.device,
        target_helicity=target,
    )
    payload = {
        "case_id": case_id,
        "helicity": helicity,
        "target_helicity": list(target),
        "metadata": metadata,
        "native_score": result,
    }
    rendered = json.dumps(payload, indent=2, allow_nan=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
