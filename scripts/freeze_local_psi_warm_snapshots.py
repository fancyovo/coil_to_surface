from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_PYTHON = REPO_ROOT / "gpu_backend" / "python"
for path in (REPO_ROOT, GPU_PYTHON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from stellarator_gpu import score_coils_native


def score_arguments(tokens: np.ndarray) -> tuple[np.ndarray, ...]:
    values = np.asarray(tokens, dtype=np.float64)
    return values[..., :33], values[..., 33:66], values[..., 66:99], values[..., 99]


def exact_config(center: dict[str, Any]) -> dict[str, Any]:
    return {
        "iota_degree": 3,
        "surface_selection_mode": 1,
        "surface_confidence_periods": 1,
        "surface_theta_count": 128,
        "surface_trace_steps": 400,
        "surface_flux_bisection_iters": 6,
        "axis_hint_enabled": 1,
        "axis_hint_require_continuation": 2,
        "axis_hint_R": center["axis_R"],
        "axis_hint_Z": center["axis_Z"],
    }


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=0.005)
    parser.add_argument("--direction-count", type=int, default=4)
    args = parser.parse_args()
    if args.direction_count <= 0:
        raise ValueError("direction-count must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    manifest = json.loads(
        (args.candidate_dir / "candidates.json").read_text(encoding="utf-8")
    )
    arrays = np.load(args.candidate_dir / "candidates.npz")
    tokens = np.asarray(arrays["tokens"], dtype=np.float64)
    x, y, z, current = score_arguments(tokens)
    if len(manifest["centers"]) != 1:
        raise ValueError("warm-start snapshot calibration requires exactly one center")
    center = manifest["centers"][0]
    selected = [
        row for row in manifest["candidates"]
        if row["kind"] == "center" or (
            row["kind"] == "endpoint"
            and np.isclose(float(row["scale"]), args.scale, rtol=0.0, atol=1.0e-15)
            and int(row["direction_index"]) < args.direction_count
        )
    ]
    selected.sort(key=lambda row: int(row["candidate_index"]))
    expected = 1 + 2 * args.direction_count
    if len(selected) != expected:
        raise ValueError(f"expected {expected} selected candidates, found {len(selected)}")

    args.output_dir.mkdir(parents=True)
    rows = []
    try:
        for metadata in selected:
            index = int(metadata["candidate_index"])
            label = "center" if metadata["kind"] == "center" else (
                f"direction_{int(metadata['direction_index']):03d}_"
                f"{'plus' if int(metadata['sign']) > 0 else 'minus'}"
            )
            snapshot = args.output_dir / f"{label}.bin"
            os.environ["SGPU_PSI_QR_SNAPSHOT"] = str(snapshot)
            result = score_coils_native(
                args.lib,
                x[index], y[index], z[index], current[index],
                int(center["nfp"]),
                device_id=0,
                target_helicity=(1, int(center["nfp"])),
                config_overrides=exact_config(center),
            )
            if not snapshot.is_file():
                raise RuntimeError(f"score did not create snapshot {snapshot}")
            rows.append({
                **metadata,
                "label": label,
                "snapshot": str(snapshot),
                "snapshot_bytes": snapshot.stat().st_size,
                "snapshot_sha256": digest(snapshot),
                "score": float(result["score"]),
                "status": result["status"],
                "psi_train_rms": float(result["diagnostics"]["psi_train_rms"]),
                "psi_angle_p95": float(result["diagnostics"]["psi_angle_p95"]),
                "surface_level": float(result["diagnostics"]["surface_level"]),
                "psi_fit_s": float(result["timing"]["psi_fit_s"]),
                "total_s": float(result["timing"]["total_s"]),
            })
    finally:
        os.environ.pop("SGPU_PSI_QR_SNAPSHOT", None)

    output = {
        "format": "local_psi_warm_snapshots_v1",
        "candidate_dir": str(args.candidate_dir),
        "library": str(args.lib),
        "scale": args.scale,
        "direction_count": args.direction_count,
        "center": center,
        "rows": rows,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
