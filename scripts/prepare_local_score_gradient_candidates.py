from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.optimize_flow_prior_zo_adam import decode_noise_rk4, load_flow_checkpoint


def parse_center(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label:
        raise argparse.ArgumentTypeError("center must use LABEL=PATH")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"center does not exist: {path}")
    return label, path


def parse_scales(value: str) -> tuple[float, ...]:
    scales = tuple(float(item) for item in value.split(",") if item.strip())
    if not scales or any(scale <= 0.0 for scale in scales):
        raise argparse.ArgumentTypeError("scales must be positive")
    return scales


def load_center(path: Path) -> tuple[np.ndarray, int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    trajectory = payload.get("flow_prior_standard_adam_trajectory")
    if not isinstance(trajectory, dict) or "noise" not in trajectory:
        raise ValueError(f"unsupported center format: {path}")
    noise = np.asarray(trajectory["noise"], dtype=np.float32)
    if noise.ndim != 2 or noise.shape[1] != 100:
        raise ValueError(f"center noise must have shape (coils, 100): {path}")
    return noise, int(payload["nfp"]), trajectory


def rms_orthogonal_directions(dimension: int, count: int, seed: int) -> np.ndarray:
    if count < 1 or count > dimension:
        raise ValueError("random direction count must be in [1, dimension]")
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((dimension, dimension))
    basis, _ = np.linalg.qr(matrix)
    return (basis[:, :count].T * math.sqrt(dimension)).astype(np.float32)


def coordinate_directions(dimension: int) -> np.ndarray:
    return np.eye(dimension, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--center", action="append", type=parse_center, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--direction-mode", choices=("random", "coordinate"), default="random")
    parser.add_argument("--direction-count", type=int, default=32)
    parser.add_argument("--scales", type=parse_scales, default=(0.0025, 0.005, 0.01))
    parser.add_argument("--seed", type=int, default=2026081101)
    parser.add_argument("--flow-steps", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    centers = []
    nfp_values = set()
    noise_shapes = set()
    for label, path in args.center:
        noise, nfp, trajectory = load_center(path)
        centers.append((label, path, noise, nfp, trajectory))
        nfp_values.add(nfp)
        noise_shapes.add(noise.shape)
    if len(nfp_values) != 1 or len(noise_shapes) != 1:
        raise ValueError("all centers must share nfp and noise shape")

    noise_shape = next(iter(noise_shapes))
    dimension = int(np.prod(noise_shape))
    if args.direction_mode == "coordinate":
        directions = coordinate_directions(dimension)
    else:
        directions = rms_orthogonal_directions(
            dimension, args.direction_count, args.seed
        )

    candidate_noise: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    center_records: list[dict[str, Any]] = []
    for center_index, (label, path, noise, nfp, trajectory) in enumerate(centers):
        native = trajectory["native_score"]
        diagnostics = native["diagnostics"]
        center_records.append(
            {
                "center_index": center_index,
                "label": label,
                "path": str(path),
                "nfp": nfp,
                "saved_score": float(native["score"]),
                "saved_status": str(native["status"]),
                "axis_R": float(diagnostics["axis_R"]),
                "axis_Z": float(diagnostics["axis_Z"]),
                "surface_level": float(diagnostics["surface_level"]),
            }
        )
        metadata.append(
            {
                "candidate_index": len(candidate_noise),
                "center_index": center_index,
                "kind": "center",
                "direction_index": -1,
                "scale": 0.0,
                "sign": 0,
            }
        )
        candidate_noise.append(noise.copy())
        for direction_index, flat_direction in enumerate(directions):
            direction = flat_direction.reshape(noise_shape)
            for scale in args.scales:
                for sign in (-1, 1):
                    metadata.append(
                        {
                            "candidate_index": len(candidate_noise),
                            "center_index": center_index,
                            "kind": "endpoint",
                            "direction_index": direction_index,
                            "scale": float(scale),
                            "sign": sign,
                        }
                    )
                    candidate_noise.append(
                        (noise.astype(np.float64) + sign * scale * direction).astype(
                            np.float32
                        )
                    )

    device = torch.device(args.device)
    model, normalizer, checkpoint = load_flow_checkpoint(args.checkpoint, device)
    decoded, decode_wall_s = decode_noise_rk4(
        model,
        normalizer,
        np.asarray(candidate_noise, dtype=np.float32),
        nfp=next(iter(nfp_values)),
        steps=args.flow_steps,
        device=device,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "candidates.npz",
        noise=np.asarray(candidate_noise, dtype=np.float32),
        tokens=np.asarray(decoded, dtype=np.float64),
        directions=np.asarray(directions, dtype=np.float32),
    )
    manifest = {
        "format": "local_score_gradient_candidates_v1",
        "created_unix_s": time.time(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": int(checkpoint["step"]),
        "flow_steps": args.flow_steps,
        "flow_dtype": "float32",
        "decode_wall_s": decode_wall_s,
        "direction_mode": args.direction_mode,
        "direction_count": len(directions),
        "dimension": dimension,
        "scales": list(args.scales),
        "seed": args.seed,
        "centers": center_records,
        "candidates": metadata,
    }
    (args.output_dir / "candidates.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in (
        "flow_steps", "decode_wall_s", "direction_mode", "direction_count",
        "dimension", "scales"
    )}, indent=2))


if __name__ == "__main__":
    main()
