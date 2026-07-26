from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_PYTHON = REPO_ROOT / "gpu_backend" / "python"
if str(GPU_PYTHON) not in sys.path:
    sys.path.insert(0, str(GPU_PYTHON))

from stellarator_gpu import score_coils_native


def load_coil_arrays(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data["raw"]
    unit = str(raw.get("current_unit", "A")).lower()
    if unit in {"a", "amp", "amps"}:
        current_scale = 1.0
    elif unit in {"ma", "megaamp", "megaamps"}:
        current_scale = 1.0e6
    else:
        raise ValueError(f"unsupported current unit {unit!r}")
    return (
        raw["x"],
        raw["y"],
        raw["z"],
        [float(value) * current_scale for value in raw["current"]],
        int(data.get("nfp", raw.get("nfp"))),
        raw.get("metadata", {}),
    )


def case_id(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if not match:
        raise ValueError(f"case filename has no numeric ID: {path}")
    return int(match.group(1))


def evaluate(path: Path, lib: Path, device: int) -> dict:
    coeffs_x, coeffs_y, coeffs_z, currents_a, nfp, metadata = load_coil_arrays(path)
    helicity = int(metadata.get("helicity", 0))
    started = time.perf_counter()
    native = score_coils_native(
        lib,
        coeffs_x,
        coeffs_y,
        coeffs_z,
        currents_a,
        nfp,
        device_id=device,
        target_helicity=(1, 0 if helicity == 0 else nfp),
    )
    return {
        "case_id": case_id(path),
        "helicity": helicity,
        "nfp": nfp,
        "metadata_qs_error": float(metadata["qs_error"]) if "qs_error" in metadata else None,
        "metadata_mean_iota": float(metadata["mean_iota"]) if "mean_iota" in metadata else None,
        "wall_s": time.perf_counter() - started,
        "native_score": native,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lib",
        type=Path,
        default=REPO_ROOT / "gpu_backend" / "build_native_score" / "libstellarator_gpu.so",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()

    if args.worker_count <= 0 or not 0 <= args.worker_index < args.worker_count:
        raise ValueError("worker-index must be in [0, worker-count)")
    paths = sorted(args.case_dir.glob("*.json"), key=case_id)
    paths = paths[args.worker_index :: args.worker_count]
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise ValueError("worker has no input cases")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.warmup:
        evaluate(paths[0], args.lib, args.device)
    batch_started = time.perf_counter()
    with args.output.open("w", encoding="utf-8") as stream:
        for path in paths:
            row = evaluate(path, args.lib, args.device)
            stream.write(json.dumps(row, allow_nan=True, separators=(",", ":")) + "\n")
            stream.flush()
    summary = {
        "worker_index": args.worker_index,
        "worker_count": args.worker_count,
        "count": len(paths),
        "wall_s": time.perf_counter() - batch_started,
    }
    print(json.dumps(summary, separators=(",", ":")))


if __name__ == "__main__":
    main()
