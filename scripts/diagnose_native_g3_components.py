from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_PYTHON = REPO_ROOT / "gpu_backend" / "python"
if str(GPU_PYTHON) not in sys.path:
    sys.path.insert(0, str(GPU_PYTHON))

from stellarator_gpu import (
    score_coils_g2_gradient_native,
    score_coils_g3_gradient_native,
    score_coils_native,
)


WEIGHTS = {
    "coordinate": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "coordinate_alpha": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "coordinate_normal": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "volume": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
}


def load_case(path: Path) -> tuple[dict[str, np.ndarray], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload["raw"]
    return (
        {
            "x": np.asarray(raw["x"], dtype=np.float64),
            "y": np.asarray(raw["y"], dtype=np.float64),
            "z": np.asarray(raw["z"], dtype=np.float64),
            "current": np.asarray(raw["current"], dtype=np.float64),
        },
        int(payload["nfp"]),
    )


def dot_gradient(gradient: dict[str, np.ndarray], direction: dict[str, np.ndarray]) -> float:
    return float(sum(np.sum(gradient[key] * direction[key]) for key in gradient))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 0.0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolate native G3 coordinate and volume-QS responses.")
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--gradient-lib", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=sorted(WEIGHTS), required=True)
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--geometry-rms", type=float, default=1.0e-6)
    parser.add_argument("--current-rms", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026080417)
    args = parser.parse_args()

    raw, nfp = load_case(args.case)
    overrides = {
        "score_weights": WEIGHTS[args.mode],
        "score_qh_total_iota_floor": 1.0,
        "score_qh_total_helicity_floor": 1.0,
    }
    if args.mode == "coordinate_alpha":
        overrides["score_alpha_normal_B_scale"] = 1.0e6
    elif args.mode == "coordinate_normal":
        overrides["score_alpha_relative_l2_scale"] = 1.0e6
    elif args.mode == "volume":
        overrides["score_volume_qs_iota_floor"] = 1.0

    call = {
        "nfp": nfp,
        "target_helicity": (1, nfp),
        "config_overrides": overrides,
    }
    g2 = score_coils_g2_gradient_native(
        args.gradient_lib, raw["x"], raw["y"], raw["z"], raw["current"], **call
    )
    g3 = score_coils_g3_gradient_native(
        args.gradient_lib, raw["x"], raw["y"], raw["z"], raw["current"], **call
    )
    increment = {key: g3["gradient"][key] - g2["gradient"][key] for key in g2["gradient"]}
    increment_values = np.concatenate([value.ravel() for value in increment.values()])

    rng = np.random.default_rng(args.seed)
    rows = []
    for index in range(args.directions):
        direction = {key: rng.standard_normal(raw[key].shape) for key in raw}
        geometry_rms = np.sqrt(
            np.mean(np.concatenate([direction[key].ravel() for key in ("x", "y", "z")]) ** 2)
        )
        current_rms = np.sqrt(np.mean(direction["current"] ** 2))
        for key in ("x", "y", "z"):
            direction[key] *= args.geometry_rms / max(float(geometry_rms), 1.0e-30)
        direction["current"] *= args.current_rms / max(float(current_rms), 1.0e-30)
        endpoints = []
        for sign in (-1.0, 1.0):
            perturbed = {key: raw[key] + sign * direction[key] for key in raw}
            endpoints.append(
                score_coils_native(
                    args.gradient_lib,
                    perturbed["x"], perturbed["y"], perturbed["z"], perturbed["current"],
                    **call,
                )
            )
        rows.append(
            {
                "direction": index,
                "g2_prediction": dot_gradient(g2["gradient"], direction),
                "g3_prediction": dot_gradient(g3["gradient"], direction),
                "g3_increment_prediction": dot_gradient(increment, direction),
                "blackbox_central": 0.5 * (endpoints[1]["score"] - endpoints[0]["score"]),
                "minus_status": endpoints[0]["status"],
                "plus_status": endpoints[1]["status"],
                "minus_components": endpoints[0]["components"],
                "plus_components": endpoints[1]["components"],
                "minus_diagnostics": endpoints[0]["diagnostics"],
                "plus_diagnostics": endpoints[1]["diagnostics"],
            }
        )
    observed = np.asarray([row["blackbox_central"] for row in rows])
    predicted_g2 = np.asarray([row["g2_prediction"] for row in rows])
    predicted_g3 = np.asarray([row["g3_prediction"] for row in rows])
    output = {
        "format": "native_g3_component_diagnosis_v1",
        "case": str(args.case),
        "mode": args.mode,
        "nfp": nfp,
        "config_overrides": overrides,
        "direction_count": args.directions,
        "direction_scales": {
            "geometry_m_rms": args.geometry_rms,
            "current_a_rms": args.current_rms,
        },
        "g2_score": g2["score_result"],
        "g3_score": g3["score_result"],
        "g2_diagnostics": g2["gradient_diagnostics"],
        "g3_diagnostics": g3["gradient_diagnostics"],
        "g3_increment_rms": float(np.sqrt(np.mean(increment_values * increment_values))),
        "g2_cosine": cosine(predicted_g2, observed),
        "g3_cosine": cosine(predicted_g3, observed),
        "g2_sign_rate": float(np.mean(np.sign(predicted_g2) == np.sign(observed))),
        "g3_sign_rate": float(np.mean(np.sign(predicted_g3) == np.sign(observed))),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
