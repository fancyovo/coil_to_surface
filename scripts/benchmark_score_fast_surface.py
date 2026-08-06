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


VARIANTS = {
    "p1_t64_k400": {
        "surface_selection_mode": 1,
        "surface_confidence_periods": 1,
        "surface_theta_count": 64,
        "surface_trace_steps": 400,
    },
    "p1_t128_k400": {
        "surface_selection_mode": 1,
        "surface_confidence_periods": 1,
        "surface_theta_count": 128,
        "surface_trace_steps": 400,
    },
    "p1_t128_k800": {
        "surface_selection_mode": 1,
        "surface_confidence_periods": 1,
        "surface_theta_count": 128,
        "surface_trace_steps": 800,
    },
    "p1_t256_k800": {
        "surface_selection_mode": 1,
        "surface_confidence_periods": 1,
        "surface_theta_count": 256,
        "surface_trace_steps": 800,
    },
    "p2_t128_k800": {
        "surface_selection_mode": 1,
        "surface_confidence_periods": 2,
        "surface_theta_count": 128,
        "surface_trace_steps": 800,
    },
}


def case_id(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if not match:
        raise ValueError(f"case filename has no ID: {path}")
    return int(match.group(1))


def load_case(path: Path):
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


def dispersed(rows: list[dict]) -> list[dict]:
    if len(rows) < 2:
        return rows
    step = len(rows) // 2 + 1
    while math.gcd(step, len(rows)) != 1:
        step += 1
    return [rows[(index * step) % len(rows)] for index in range(len(rows))]


def evaluate(
    lib: Path,
    arrays,
    nfp: int,
    device: int,
    overrides: dict | None = None,
) -> dict:
    started = time.perf_counter()
    result = score_coils_native(
        lib,
        *arrays,
        nfp,
        device_id=device,
        target_helicity=(1, nfp),
        config_overrides=overrides,
    )
    result["caller_wall_s"] = time.perf_counter() - started
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--variant-profile", choices=("selected", "matrix"), default="matrix")
    parser.add_argument("--total-limit", type=int, default=128)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()

    rows = json.loads(args.metadata.read_text(encoding="utf-8"))
    rows = [
        row for row in rows
        if row.get("_split") == args.split and int(row["helicity"]) == 1
    ]
    rows.sort(key=lambda row: float(row["qs_error"]))
    rows = dispersed(rows)[args.sample_offset : args.sample_offset + args.total_limit]
    rows = rows[args.worker_index :: args.worker_count]
    if not rows:
        raise ValueError("worker has no selected rows")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for metadata in rows:
            path = args.case_dir / f"id_{int(metadata['ID']):07d}.json"
            x, y, z, currents, nfp = load_case(path)
            arrays = (x, y, z, currents)
            legacy = evaluate(args.lib, arrays, nfp, args.device)
            axis = legacy["diagnostics"]
            variants = {
                "p1_t128_k400_global": evaluate(
                    args.lib, arrays, nfp, args.device, VARIANTS["p1_t128_k400"]
                )
            }
            if legacy["status"] == "ok":
                axis_hint = {
                    "axis_hint_enabled": 1,
                    "axis_hint_require_continuation": 1,
                    "axis_hint_R": float(axis["axis_R"]),
                    "axis_hint_Z": float(axis["axis_Z"]),
                }
                selected_variants = (
                    {"p1_t128_k400": VARIANTS["p1_t128_k400"]}
                    if args.variant_profile == "selected" else VARIANTS
                )
                for name, settings in selected_variants.items():
                    variants[name] = evaluate(
                        args.lib, arrays, nfp, args.device, {**settings, **axis_hint}
                    )
            payload = {
                "case_id": case_id(path),
                "nfp": nfp,
                "metadata_qs_error": float(metadata["qs_error"]),
                "metadata_mean_iota": float(metadata["mean_iota"]),
                "legacy": legacy,
                "variants": variants,
            }
            stream.write(json.dumps(payload, allow_nan=True, separators=(",", ":")) + "\n")
            stream.flush()

    print(json.dumps({"worker": args.worker_index, "count": len(rows)}))


if __name__ == "__main__":
    main()
