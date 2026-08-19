from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_qh_face_qs_case_gpu import validate_case  # noqa: E402


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dense face-QS validation for one GPU shard.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--gpu-lib", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()
    records = read_json(args.experiment_root / "cases.json")
    selected = [row for index, row in enumerate(records) if index % args.shard_count == args.shard_index]
    counts = {"ok": 0, "failed": 0}
    for index, row in enumerate(selected, start=1):
        result = validate_case(args.experiment_root / "cases" / row["case_id"], gpu_lib=args.gpu_lib, device=args.device)
        counts[result["status"]] += 1
        print(json.dumps({"event": "validated", "case_id": row["case_id"], "status": result["status"], "index": index, "count": len(selected)}), flush=True)
    write_json(args.experiment_root / f"validation_shard_{args.shard_index:02d}.json", {"shard_index": args.shard_index, "selected": len(selected), "counts": counts})


if __name__ == "__main__":
    main()
