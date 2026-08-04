from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_PYTHON = REPO_ROOT / "gpu_backend" / "python"
for path in (REPO_ROOT, GPU_PYTHON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.qh_blackbox_gradient_reference import file_sha256, write_json
from stellarator_gpu import score_coils_g4_fixed_branch_batch_native


COMPONENTS = ("axis", "psi", "surface", "coordinate", "volume_qs", "iota", "coil")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def score_arguments(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(tokens, dtype=np.float64)
    return values[..., :33], values[..., 33:66], values[..., 66:99], values[..., 99]


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 0.0 else float("nan")


def rms(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array)))


def normalized(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    scale = rms(array)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("cannot normalize a zero or non-finite vector")
    return array / scale


def projection(gradient: np.ndarray, direction: np.ndarray) -> float:
    return float(np.sum(np.asarray(gradient, dtype=np.float64) * direction))


def reconstruct_gradient(slopes: np.ndarray, directions: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(slopes)):
        return np.full(directions.shape[1], np.nan, dtype=np.float64)
    return np.mean(slopes[:, None] * directions, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare fixed-axis G4 against the existing full 300-D score reference."
    )
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--closure", type=Path, required=True)
    parser.add_argument("--gradient-lib", type=Path, required=True)
    parser.add_argument("--center-index", type=int, required=True)
    parser.add_argument("--scale", type=float, default=0.0025)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()
    if args.scale <= 0.0:
        raise ValueError("scale must be positive")

    manifest = read_json(args.reference_dir / "manifest.json")
    reference_summary = read_json(args.reference_dir / "summary.json")
    closure = read_json(args.closure)
    banks = np.load(args.reference_dir / "latent_banks.npz")
    reference = np.load(args.reference_dir / "reference_gradients.npz")
    raw = np.load(args.reference_dir / "raw_tokens.npy", mmap_mode="r")
    cases = load_jsonl(args.reference_dir / "cases.jsonl")

    center_index = int(args.center_index)
    if not 0 <= center_index < len(manifest["centers"]):
        raise ValueError("center-index is outside the reference manifest")
    center_info = manifest["centers"][center_index]
    expected_iteration = int(center_info["recorded_iteration"])
    if int(closure["iteration"]) != expected_iteration:
        raise ValueError(
            f"closure iteration {closure['iteration']} does not match center {expected_iteration}"
        )
    scales = np.asarray(reference["scales"], dtype=np.float64)
    matching_scales = np.flatnonzero(np.isclose(scales, args.scale, rtol=0.0, atol=1.0e-15))
    if matching_scales.size != 1:
        raise ValueError(f"scale {args.scale} is not unique in the reference")
    scale_index = int(matching_scales[0])

    center_rows = [row for row in cases if int(row["center_index"]) == center_index]
    center_row = next(row for row in center_rows if row["kind"] == "center")
    endpoint_rows = sorted(
        (
            row
            for row in center_rows
            if row["kind"] == "endpoint" and int(row["scale_index"]) == scale_index
        ),
        key=lambda row: (int(row["direction_index"]), int(row["sign"])),
    )
    direction_count = int(center_info["direction_count"])
    if len(endpoint_rows) != 2 * direction_count:
        raise RuntimeError("reference endpoint set is incomplete")

    query_rows = [center_row, *endpoint_rows]
    center_tokens = np.asarray(raw[int(center_row["case_id"])], dtype=np.float64)
    query_tokens = np.asarray(
        [raw[int(row["case_id"])] for row in query_rows], dtype=np.float64
    )
    center_x, center_y, center_z, center_current = score_arguments(center_tokens)
    query_x, query_y, query_z, query_current = score_arguments(query_tokens)

    started = time.perf_counter()
    g4 = score_coils_g4_fixed_branch_batch_native(
        args.gradient_lib,
        center_x,
        center_y,
        center_z,
        center_current,
        query_x,
        query_y,
        query_z,
        query_current,
        int(center_info["nfp"]),
        device_id=args.device_id,
        target_helicity=(1, int(center_info["nfp"])),
    )
    batch_wall_s = time.perf_counter() - started
    query_results = g4["query_score_results"]

    slopes = np.full(direction_count, np.nan, dtype=np.float64)
    component_slopes = np.full((direction_count, len(COMPONENTS)), np.nan, dtype=np.float64)
    status_counts: dict[str, int] = {}
    result_by_case = {
        int(row["case_id"]): result for row, result in zip(query_rows, query_results, strict=True)
    }
    for direction_index in range(direction_count):
        pair = [
            row for row in endpoint_rows if int(row["direction_index"]) == direction_index
        ]
        by_sign = {int(row["sign"]): result_by_case[int(row["case_id"])] for row in pair}
        minus = by_sign[-1]
        plus = by_sign[1]
        for result in (minus, plus):
            status = str(result["status"])
            status_counts[status] = status_counts.get(status, 0) + 1
        if minus["status"] != "ok" or plus["status"] != "ok":
            continue
        slopes[direction_index] = (float(plus["score"]) - float(minus["score"])) / (
            2.0 * args.scale
        )
        for component_index, name in enumerate(COMPONENTS):
            component_slopes[direction_index, component_index] = (
                float(plus["components"][name]) - float(minus["components"][name])
            ) / (2.0 * args.scale)

    directions = np.asarray(banks["directions"][center_index], dtype=np.float64)
    full_gradient = np.asarray(reference["gradients"][center_index, scale_index], dtype=np.float64)
    full_component_gradients = np.asarray(
        reference["component_gradients"][center_index, scale_index], dtype=np.float64
    )
    component_names = [str(value) for value in reference["component_names"]]
    g2_gradient = np.asarray(closure["latent_gradients"]["g2"], dtype=np.float64)
    g3_gradient = np.asarray(closure["latent_gradients"]["g3"], dtype=np.float64)
    full_slopes = directions @ full_gradient.reshape(-1)
    g2_slopes = directions @ g2_gradient.reshape(-1)
    g3_slopes = directions @ g3_gradient.reshape(-1)
    valid = np.isfinite(slopes) & np.isfinite(full_slopes)
    g4_gradient = reconstruct_gradient(slopes, directions).reshape(full_gradient.shape)

    trajectory_path = Path(center_info["source_path"])
    current_state = read_json(trajectory_path)
    next_path = trajectory_path.with_name(f"step_{expected_iteration + 1:04d}.json")
    next_state = read_json(next_path)
    adam_direction = normalized(
        np.asarray(next_state["noise"], dtype=np.float64)
        - np.asarray(current_state["noise"], dtype=np.float64)
    )

    component_alignment: dict[str, Any] = {}
    for component_index, name in enumerate(COMPONENTS):
        reference_index = component_names.index(name)
        full_component = full_component_gradients[reference_index]
        g4_component_gradient = reconstruct_gradient(
            component_slopes[:, component_index], directions
        ).reshape(full_component.shape)
        component_valid = np.isfinite(component_slopes[:, component_index])
        component_alignment[name] = {
            "valid_direction_count": int(np.sum(component_valid)),
            "full_g4_cosine": cosine(
                directions[component_valid] @ full_component.reshape(-1),
                component_slopes[component_valid, component_index],
            ),
            "full_on_adam": projection(full_component, adam_direction),
            "g4_on_adam": projection(g4_component_gradient, adam_direction),
            "full_gradient_rms": rms(full_component),
            "g4_gradient_rms": rms(g4_component_gradient),
        }

    center_query = query_results[0]
    output = {
        "format": "qh_g4_reference_alignment_v1",
        "center_index": center_index,
        "center_id": center_info["center_id"],
        "iteration": expected_iteration,
        "scale": float(args.scale),
        "nfp": int(center_info["nfp"]),
        "n_coils": int(center_info["n_coils"]),
        "gradient_lib": str(args.gradient_lib),
        "gradient_lib_sha256": file_sha256(args.gradient_lib),
        "reference_manifest_sha256": file_sha256(args.reference_dir / "manifest.json"),
        "closure_sha256": file_sha256(args.closure),
        "batch_wall_s": float(batch_wall_s),
        "mean_query_wall_s": float(batch_wall_s / len(query_rows)),
        "query_count": len(query_rows),
        "valid_direction_count": int(np.sum(valid)),
        "endpoint_status_counts": status_counts,
        "center_score": float(g4["center_score_result"]["score"]),
        "fixed_branch_center_score": float(center_query["score"]),
        "fixed_branch_center_delta": float(
            center_query["score"] - g4["center_score_result"]["score"]
        ),
        "gradient_alignment": {
            "full_g2_cosine": cosine(full_slopes[valid], g2_slopes[valid]),
            "full_g3_cosine": cosine(full_slopes[valid], g3_slopes[valid]),
            "full_g4_cosine": cosine(full_slopes[valid], slopes[valid]),
            "g2_g4_cosine": cosine(g2_slopes[valid], slopes[valid]),
            "g3_g4_cosine": cosine(g3_slopes[valid], slopes[valid]),
            "full_slope_rms": rms(full_slopes[valid]),
            "g2_slope_rms": rms(g2_slopes[valid]),
            "g3_slope_rms": rms(g3_slopes[valid]),
            "g4_slope_rms": rms(slopes[valid]),
        },
        "adam_projection": {
            "full": projection(full_gradient, adam_direction),
            "g2": projection(g2_gradient, adam_direction),
            "g3": projection(g3_gradient, adam_direction),
            "g4": projection(g4_gradient, adam_direction),
        },
        "component_alignment": component_alignment,
        "reference_metadata": reference_summary["centers"][center_index]["scales"][scale_index],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "summary.json", output)
    np.savez_compressed(
        args.output_dir / "directional_slopes.npz",
        directions=directions,
        full=full_slopes,
        g2=g2_slopes,
        g3=g3_slopes,
        g4=slopes,
        g4_components=component_slopes,
        valid=valid,
        component_names=np.asarray(COMPONENTS),
    )
    print(json.dumps(output["gradient_alignment"], indent=2), flush=True)
    print(json.dumps(output["adam_projection"], indent=2), flush=True)


if __name__ == "__main__":
    main()
