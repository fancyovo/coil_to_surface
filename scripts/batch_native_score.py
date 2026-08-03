from __future__ import annotations

import argparse
import json
import math
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


def dispersed(items: list[dict]) -> list[dict]:
    if len(items) < 2:
        return items
    step = len(items) // 2 + 1
    while math.gcd(step, len(items)) != 1:
        step += 1
    return [items[(index * step) % len(items)] for index in range(len(items))]


def select_paths(case_dir: Path, metadata_path: Path | None, split: str | None) -> list[Path]:
    if metadata_path is None:
        return sorted(case_dir.glob("*.json"), key=case_id)
    rows = json.loads(metadata_path.read_text(encoding="utf-8"))
    if split:
        rows = [row for row in rows if row.get("_split") == split]
    groups = []
    for helicity in (0, 1):
        group = [row for row in rows if int(row["helicity"]) == helicity]
        group.sort(key=lambda row: float(row["qs_error"]))
        groups.append(dispersed(group))
    ordered = []
    for index in range(max(map(len, groups), default=0)):
        group_order = groups if (index // 2) % 2 == 0 else reversed(groups)
        for group in group_order:
            if index < len(group):
                ordered.append(case_dir / f"id_{int(group[index]['ID']):07d}.json")
    missing = [path for path in ordered if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"metadata references missing case {missing[0]}")
    return ordered


def evaluate(path: Path, lib: Path, device: int, config_overrides: dict | None = None) -> dict:
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
        config_overrides=config_overrides,
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
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--split", choices=("calibration", "validation"))
    parser.add_argument("--case-id-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lib",
        type=Path,
        default=REPO_ROOT / "gpu_backend" / "build_native_score" / "libstellarator_gpu.so",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--psi-solver-mode", type=int, choices=(1, 2), default=2)
    parser.add_argument("--alpha-solver-mode", type=int, choices=(1, 2), default=2)
    parser.add_argument("--volume-point-count", type=int)
    parser.add_argument("--alpha-fit-point-count", type=int)
    parser.add_argument("--volume-phi-count", type=int)
    parser.add_argument("--axis-fallback-grid", type=int)
    parser.add_argument("--axis-fallback-max-candidates", type=int)
    parser.add_argument("--axis-fallback-newton-iters", type=int)
    parser.add_argument("--axis-fallback-max-nfp", type=int)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--total-limit", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()

    if args.worker_count <= 0 or not 0 <= args.worker_index < args.worker_count:
        raise ValueError("worker-index must be in [0, worker-count)")
    if args.case_id_file:
        ids = [int(line) for line in args.case_id_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        paths = [args.case_dir / f"id_{case_id_value:07d}.json" for case_id_value in ids]
    else:
        paths = select_paths(args.case_dir, args.metadata, args.split)
    if args.total_limit is not None:
        paths = paths[: args.total_limit]
    paths = paths[args.worker_index :: args.worker_count]
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise ValueError("worker has no input cases")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    config_overrides = {
        "psi_solver_mode": args.psi_solver_mode,
        "alpha_solver_mode": args.alpha_solver_mode,
    }
    for name in (
        "axis_fallback_grid",
        "axis_fallback_max_candidates",
        "axis_fallback_newton_iters",
        "axis_fallback_max_nfp",
        "volume_point_count",
        "alpha_fit_point_count",
        "volume_phi_count",
    ):
        value = getattr(args, name)
        if value is not None:
            config_overrides[name] = value
    if args.warmup:
        evaluate(paths[0], args.lib, args.device, config_overrides)
    batch_started = time.perf_counter()
    with args.output.open("w", encoding="utf-8") as stream:
        for path in paths:
            row = evaluate(path, args.lib, args.device, config_overrides)
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
