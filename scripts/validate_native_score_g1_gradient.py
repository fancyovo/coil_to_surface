from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_PYTHON = REPO_ROOT / "gpu_backend" / "python"
if str(GPU_PYTHON) not in sys.path:
    sys.path.insert(0, str(GPU_PYTHON))

from stellarator_gpu import (
    coil_component_gradient_native,
    score_coils_g1_gradient_native,
    score_coils_g2_gradient_native,
    score_coils_native,
)


def load_raw_case(path: Path) -> tuple[dict[str, np.ndarray], int]:
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


def finite_difference_component(
    lib: Path,
    raw: dict[str, np.ndarray],
    nfp: int,
    direction: dict[str, np.ndarray],
    step: float,
) -> float:
    values = []
    for sign in (-1.0, 1.0):
        perturbed = {
            key: raw[key] + sign * step * direction[key]
            for key in ("x", "y", "z", "current")
        }
        values.append(
            coil_component_gradient_native(
                lib,
                perturbed["x"],
                perturbed["y"],
                perturbed["z"],
                perturbed["current"],
                nfp,
            )["component"]
        )
    return float((values[1] - values[0]) / (2.0 * step))


def dot_gradient(gradient: dict[str, np.ndarray], direction: dict[str, np.ndarray]) -> float:
    return float(sum(np.sum(gradient[key] * direction[key]) for key in gradient))


def compact_forward(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": float(result["score"]),
        "status": result["status"],
        "components": result["components"],
        "total_s": float(result["timing"]["total_s"]),
        "surface_level": float(result["diagnostics"]["surface_level"]),
        "iota_min": float(result["diagnostics"]["iota_min"]),
        "qs_global_error": float(result["diagnostics"]["qs_global_error"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and benchmark the opt-in native G1 gradient.")
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--baseline-lib", type=Path, required=True)
    parser.add_argument("--gradient-lib", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--directions", type=int, default=24)
    parser.add_argument("--steps", default="1,0.5,0.25")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--blackbox-directions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026080402)
    args = parser.parse_args()

    raw, nfp = load_raw_case(args.case)
    analytical_started = time.perf_counter()
    analytical = coil_component_gradient_native(
        args.gradient_lib, raw["x"], raw["y"], raw["z"], raw["current"], nfp
    )
    analytical_wall_s = time.perf_counter() - analytical_started
    rng = np.random.default_rng(args.seed)
    geometry_scale = 1.0e-6
    current_scale = 1.0
    steps = tuple(float(item) for item in args.steps.split(",") if item.strip())
    direction_rows = []
    for direction_index in range(args.directions):
        direction = {
            key: rng.standard_normal(raw[key].shape)
            for key in ("x", "y", "z", "current")
        }
        geometry_rms = np.sqrt(
            np.mean(np.concatenate([direction[key].ravel() for key in ("x", "y", "z")]) ** 2)
        )
        current_rms = np.sqrt(np.mean(direction["current"] ** 2))
        for key in ("x", "y", "z"):
            direction[key] *= geometry_scale / max(float(geometry_rms), 1.0e-30)
        direction["current"] *= current_scale / max(float(current_rms), 1.0e-30)
        predicted = dot_gradient(analytical["gradient"], direction)
        estimates = [
            finite_difference_component(args.gradient_lib, raw, nfp, direction, step)
            for step in steps
        ]
        reference = estimates[-1]
        direction_rows.append(
            {
                "direction": direction_index,
                "analytical": predicted,
                "finite_difference": estimates,
                "absolute_error": abs(predicted - reference),
                "relative_error": abs(predicted - reference) / max(abs(reference), abs(predicted), 1.0e-12),
            }
        )

    baseline_rows = []
    forward_rows = []
    gradient_rows = []
    g2_rows = []
    gradient_payload = None
    for repeat in range(args.repeats):
        started = time.perf_counter()
        baseline = score_coils_native(
            args.baseline_lib,
            raw["x"], raw["y"], raw["z"], raw["current"], nfp,
            target_helicity=(1, nfp),
        )
        baseline_rows.append({"wall_s": time.perf_counter() - started, "result": compact_forward(baseline)})
        started = time.perf_counter()
        forward = score_coils_native(
            args.gradient_lib,
            raw["x"], raw["y"], raw["z"], raw["current"], nfp,
            target_helicity=(1, nfp),
        )
        forward_rows.append({"wall_s": time.perf_counter() - started, "result": compact_forward(forward)})
        started = time.perf_counter()
        gradient_payload = score_coils_g1_gradient_native(
            args.gradient_lib,
            raw["x"], raw["y"], raw["z"], raw["current"], nfp,
            target_helicity=(1, nfp),
        )
        gradient_rows.append(
            {
                "wall_s": time.perf_counter() - started,
                "result": compact_forward(gradient_payload["score_result"]),
                "gradient_diagnostics": gradient_payload["gradient_diagnostics"],
            }
        )
        started = time.perf_counter()
        g2_payload = score_coils_g2_gradient_native(
            args.gradient_lib,
            raw["x"], raw["y"], raw["z"], raw["current"], nfp,
            target_helicity=(1, nfp),
        )
        g2_rows.append(
            {
                "wall_s": time.perf_counter() - started,
                "result": compact_forward(g2_payload["score_result"]),
                "gradient_diagnostics": g2_payload["gradient_diagnostics"],
            }
        )

    blackbox_rows = []
    g2_gradient = g2_payload["gradient"]
    for direction_index in range(args.blackbox_directions):
        direction = {
            key: rng.standard_normal(raw[key].shape)
            for key in ("x", "y", "z", "current")
        }
        geometry_rms = np.sqrt(
            np.mean(np.concatenate([direction[key].ravel() for key in ("x", "y", "z")]) ** 2)
        )
        current_rms = np.sqrt(np.mean(direction["current"] ** 2))
        for key in ("x", "y", "z"):
            direction[key] *= 1.0e-5 / max(float(geometry_rms), 1.0e-30)
        direction["current"] *= 10.0 / max(float(current_rms), 1.0e-30)
        endpoints = []
        for sign in (-1.0, 1.0):
            perturbed = {key: raw[key] + sign * direction[key] for key in direction}
            endpoint = score_coils_native(
                args.gradient_lib,
                perturbed["x"], perturbed["y"], perturbed["z"],
                perturbed["current"], nfp, target_helicity=(1, nfp),
            )
            endpoints.append(endpoint)
        blackbox_rows.append(
            {
                "direction": direction_index,
                "predicted_fixed_front": dot_gradient(g2_gradient, direction),
                "blackbox_central": 0.5 * (endpoints[1]["score"] - endpoints[0]["score"]),
                "minus": compact_forward(endpoints[0]),
                "plus": compact_forward(endpoints[1]),
            }
        )

    relative_errors = np.asarray([row["relative_error"] for row in direction_rows])
    baseline_wall = np.asarray([row["wall_s"] for row in baseline_rows])
    forward_wall = np.asarray([row["wall_s"] for row in forward_rows])
    gradient_wall = np.asarray([row["wall_s"] for row in gradient_rows])
    g2_wall = np.asarray([row["wall_s"] for row in g2_rows])
    forward_status_match = all(
        baseline["result"]["status"] == forward["result"]["status"]
        for baseline, forward in zip(baseline_rows, forward_rows, strict=True)
    )
    forward_score_max_abs_diff = max(
        abs(baseline["result"]["score"] - forward["result"]["score"])
        for baseline, forward in zip(baseline_rows, forward_rows, strict=True)
    )
    forward_component_max_abs_diff = max(
        abs(
            baseline["result"]["components"][component]
            - forward["result"]["components"][component]
        )
        for baseline, forward in zip(baseline_rows, forward_rows, strict=True)
        for component in baseline["result"]["components"]
    )
    output = {
        "format": "native_score_g1_validation_v1",
        "case": str(args.case),
        "nfp": nfp,
        "coil_component": analytical["component"],
        "analytical_component_wall_s": analytical_wall_s,
        "direction_scales": {"geometry_m_rms": geometry_scale, "current_a_rms": current_scale},
        "finite_difference_steps": steps,
        "direction_rows": direction_rows,
        "relative_error_median": float(np.median(relative_errors)),
        "relative_error_p95": float(np.percentile(relative_errors, 95.0)),
        "baseline_forward": baseline_rows,
        "experimental_forward": forward_rows,
        "g1_gradient": gradient_rows,
        "g1_g2_gradient": g2_rows,
        "blackbox_direction_rows": blackbox_rows,
        "baseline_forward_wall_median_s": float(np.median(baseline_wall)),
        "experimental_forward_wall_median_s": float(np.median(forward_wall)),
        "g1_wall_median_s": float(np.median(gradient_wall)),
        "g2_wall_median_s": float(np.median(g2_wall)),
        "forward_status_match": forward_status_match,
        "forward_score_max_abs_diff": float(forward_score_max_abs_diff),
        "forward_component_max_abs_diff": float(forward_component_max_abs_diff),
        "forward_only_overhead_fraction": float(np.median(forward_wall) / np.median(baseline_wall) - 1.0),
        "g1_overhead_fraction": float(np.median(gradient_wall) / np.median(forward_wall) - 1.0),
        "g2_overhead_fraction": float(np.median(g2_wall) / np.median(forward_wall) - 1.0),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    if (
        not forward_status_match
        or forward_score_max_abs_diff > 1.0e-10
        or forward_component_max_abs_diff > 1.0e-10
    ):
        raise RuntimeError(
            "experimental gradient library changed the production forward result: "
            f"status_match={forward_status_match}, score_diff={forward_score_max_abs_diff:.3e}, "
            f"component_diff={forward_component_max_abs_diff:.3e}"
        )


if __name__ == "__main__":
    main()
