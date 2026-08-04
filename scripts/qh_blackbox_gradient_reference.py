from __future__ import annotations

import argparse
import hashlib
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

TOKEN_DIM = 100
SCORE_DEFINITION = "corrected_abi9_g_over_2pi_per_helicity"
DEFAULT_SCALES = (0.01, 0.005, 0.0025, 0.00125)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in value.split(",") if item.strip())
    if not values or any(item <= 0.0 for item in values):
        raise argparse.ArgumentTypeError("scales must be positive")
    return values


def parse_center(value: str) -> tuple[str, Path]:
    try:
        label, raw_path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("center must use LABEL=PATH") from exc
    if not label or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in label):
        raise argparse.ArgumentTypeError("center label must be filesystem safe")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"center path does not exist: {path}")
    return label, path


def load_center(path: Path) -> tuple[np.ndarray, int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    trajectory = payload.get("flow_prior_standard_adam_trajectory")
    if not isinstance(trajectory, dict) or "noise" not in trajectory:
        raise ValueError(f"{path} is not a saved standard-Adam trajectory case")
    noise = np.asarray(trajectory["noise"], dtype=np.float32)
    if noise.ndim != 2 or noise.shape[1] != TOKEN_DIM:
        raise ValueError(f"{path} noise must have shape (coils, {TOKEN_DIM})")
    return noise, int(payload["nfp"]), payload


def rms_orthogonal_basis(dimension: int, seed: int) -> np.ndarray:
    if dimension < 1:
        raise ValueError("dimension must be positive")
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((dimension, dimension))
    basis, _ = np.linalg.qr(matrix)
    return (basis.T * math.sqrt(dimension)).astype(np.float32)


def branch_fingerprint(result: dict[str, Any] | None) -> tuple[Any, ...]:
    if result is None:
        return ("worker_error",)
    diagnostics = result.get("diagnostics", {})

    def rounded(name: str, digits: int = 10) -> float | None:
        value = float(diagnostics.get(name, float("nan")))
        return round(value, digits) if math.isfinite(value) else None

    return (
        str(result.get("status")),
        rounded("surface_level", 8),
        int(diagnostics.get("stable_surface_count", -1)),
        int(diagnostics.get("surface_long_trace_rejected_count", -1)),
        int(diagnostics.get("flux_attempt_count", -1)),
        int(diagnostics.get("volume_candidate_count", -1)),
        int(diagnostics.get("volume_available_count", -1)),
        int(diagnostics.get("volume_point_count", -1)),
        int(diagnostics.get("alpha_column_count", -1)),
    )


def compact_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    diagnostics = result.get("diagnostics", {})
    diagnostic_names = (
        "stage_completed",
        "axis_R",
        "axis_Z",
        "axis_residual",
        "axis_topology_trace",
        "axis_topology_det",
        "axis_ellipse_aspect",
        "psi_angle_p95",
        "surface_level",
        "surface_drift_relative_p95",
        "surface_one_period_drift_relative_p95",
        "surface_inverse_aspect_ratio",
        "surface_volume",
        "flux_attempt_count",
        "flux_edge",
        "iota_min",
        "iota_max",
        "score_before_qh_iota_gate",
        "score_qh_total_iota_factor",
        "score_qh_helicity_advantage",
        "score_qh_helicity_quality",
        "score_qh_total_helicity_factor",
        "qs_global_error",
        "qs_qa_global_error",
        "qs_qp_global_error",
        "stable_surface_count",
        "volume_candidate_count",
        "volume_available_count",
        "volume_point_count",
        "alpha_column_count",
        "surface_long_trace_rejected_count",
    )
    return {
        "score": float(result["score"]),
        "status": str(result["status"]),
        "components": {key: float(value) for key, value in result["components"].items()},
        "timing": {key: float(value) for key, value in result["timing"].items()},
        "diagnostics": {key: diagnostics.get(key) for key in diagnostic_names},
        "branch_fingerprint": list(branch_fingerprint(result)),
    }


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, separators=(",", ":"), allow_nan=True) + "\n")
        stream.flush()


def case_rows(center_rows: list[dict[str, Any]], scales: tuple[float, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for center_index, center in enumerate(center_rows):
        rows.append(
            {
                "case_id": len(rows),
                "center_index": center_index,
                "center_id": center["center_id"],
                "kind": "center",
                "direction_index": None,
                "scale_index": None,
                "scale": 0.0,
                "sign": 0,
            }
        )
        for scale_index, scale in enumerate(scales):
            for direction_index in range(int(center["dimension"])):
                for sign in (-1, 1):
                    rows.append(
                        {
                            "case_id": len(rows),
                            "center_index": center_index,
                            "center_id": center["center_id"],
                            "kind": "endpoint",
                            "direction_index": direction_index,
                            "scale_index": scale_index,
                            "scale": float(scale),
                            "sign": sign,
                        }
                    )
    return rows


def prepare(args: argparse.Namespace) -> None:
    from scripts.optimize_flow_prior_zo_adam import decode_noise_rk4, load_flow_checkpoint

    protected_outputs = ("manifest.json", "raw_tokens.npy", "latent_banks.npz", "cases.jsonl")
    if any((args.output_dir / name).exists() for name in protected_outputs):
        raise FileExistsError(f"refusing to overwrite prepared data in {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    center_rows: list[dict[str, Any]] = []
    noises: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    dimensions = set()
    for center_index, (label, path) in enumerate(args.center):
        noise, nfp, payload = load_center(path)
        dimension = int(noise.size)
        dimensions.add(dimension)
        recorded = payload["flow_prior_standard_adam_trajectory"].get("native_score", {})
        center_rows.append(
            {
                "center_id": label,
                "source_path": str(path),
                "source_sha256": file_sha256(path),
                "nfp": nfp,
                "n_coils": int(noise.shape[0]),
                "dimension": dimension,
                "recorded_iteration": int(payload["flow_prior_standard_adam_trajectory"]["iteration"]),
                "recorded_score": float(recorded.get("score", float("nan"))),
                "direction_seed": int(args.seed + 1009 * center_index),
            }
        )
        noises.append(noise)
        directions.append(rms_orthogonal_basis(dimension, args.seed + 1009 * center_index))
    if len(dimensions) != 1:
        raise ValueError("the first reference batch requires centers with a common latent dimension")

    rows = case_rows(center_rows, args.scales)
    dimension = dimensions.pop()
    raw_shape = (len(rows), dimension // TOKEN_DIM, TOKEN_DIM)
    raw_tmp = args.output_dir / "raw_tokens.npy.tmp"
    raw = np.lib.format.open_memmap(raw_tmp, mode="w+", dtype=np.float64, shape=raw_shape)
    model, normalizer, checkpoint = load_flow_checkpoint(args.checkpoint, args.torch_device)
    import torch

    by_center: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_center.setdefault(int(row["center_index"]), []).append(row)
    decode_wall_s = 0.0
    for center_index, group in by_center.items():
        center = noises[center_index].reshape(-1)
        basis = directions[center_index]
        key = (int(center_rows[center_index]["nfp"]), int(center_rows[center_index]["n_coils"]))
        for start in range(0, len(group), args.decode_batch_size):
            batch_rows = group[start : start + args.decode_batch_size]
            latent = []
            for row in batch_rows:
                value = center.copy()
                if row["kind"] == "endpoint":
                    value += float(row["sign"]) * float(row["scale"]) * basis[int(row["direction_index"])]
                latent.append(value.reshape(raw_shape[1:]))
            decoded, wall_s = decode_noise_rk4(
                model,
                normalizer,
                np.asarray(latent, dtype=np.float32),
                nfp=key[0],
                steps=args.rk4_steps,
                device=torch.device(args.torch_device),
            )
            decode_wall_s += wall_s
            for row, tokens in zip(batch_rows, decoded, strict=True):
                raw[int(row["case_id"])] = tokens
        raw.flush()
    del raw
    raw_tmp.replace(args.output_dir / "raw_tokens.npy")
    np.savez_compressed(
        args.output_dir / "latent_banks.npz",
        centers=np.stack(noises),
        directions=np.stack(directions),
        scales=np.asarray(args.scales, dtype=np.float64),
    )
    with (args.output_dir / "cases.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    manifest = {
        "format": "qh_blackbox_gradient_reference_v1",
        "score_definition": SCORE_DEFINITION,
        "created_unix_s": time.time(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "flow_checkpoint_step": int(checkpoint["step"]),
        "score_library": str(args.lib),
        "score_library_sha256": file_sha256(args.lib),
        "rk4_steps": int(args.rk4_steps),
        "decode_batch_size": int(args.decode_batch_size),
        "decode_wall_s": float(decode_wall_s),
        "seed": int(args.seed),
        "scales": [float(value) for value in args.scales],
        "centers": center_rows,
        "case_count": len(rows),
        "endpoint_count": len(rows) - len(center_rows),
        "raw_shape": list(raw_shape),
        "state": "prepared",
    }
    write_json(args.output_dir / "manifest.json", manifest)


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def score_shard(args: argparse.Namespace) -> None:
    from scripts.optimize_native_score_cem import token_case
    from stellarator_gpu import score_coils_native

    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    if file_sha256(args.lib) != manifest["score_library_sha256"]:
        raise RuntimeError("score library changed after preparation")
    rows = load_rows(args.output_dir / "cases.jsonl")
    raw = np.load(args.output_dir / "raw_tokens.npy", mmap_mode="r")
    output = args.output_dir / f"scores_rank_{args.rank:02d}.jsonl"
    done = set()
    if output.exists():
        for row in load_rows(output):
            done.add(int(row["case_id"]))
    selected = [row for row in rows if int(row["case_id"]) % args.world_size == args.rank and int(row["case_id"]) not in done]
    center_info = {index: value for index, value in enumerate(manifest["centers"])}
    started_all = time.perf_counter()
    for ordinal, row in enumerate(selected, start=1):
        case_id = int(row["case_id"])
        center = center_info[int(row["center_index"])]
        case = token_case(raw[case_id], nfp=int(center["nfp"]), target="QH")
        result = None
        error = None
        started = time.perf_counter()
        try:
            coil = case["raw"]
            result = score_coils_native(
                args.lib,
                coil["x"],
                coil["y"],
                coil["z"],
                coil["current"],
                int(center["nfp"]),
                device_id=args.device_id,
                target_helicity=(1, int(center["nfp"])),
            )
        except Exception as exc:
            error = repr(exc)
        elapsed = time.perf_counter() - started
        append_jsonl(
            output,
            {
                **row,
                "rank": int(args.rank),
                "elapsed_s": float(elapsed),
                "error": error,
                "result": compact_result(result),
            },
        )
        if ordinal % 20 == 0 or ordinal == len(selected):
            print(
                f"rank={args.rank} complete={ordinal}/{len(selected)} "
                f"case_id={case_id} elapsed_s={elapsed:.3f}",
                flush=True,
            )
    write_json(
        args.output_dir / f"scores_rank_{args.rank:02d}.done.json",
        {
            "rank": int(args.rank),
            "new_cases": len(selected),
            "total_done": len(done) + len(selected),
            "wall_s": float(time.perf_counter() - started_all),
        },
    )


def analyze(args: argparse.Namespace) -> None:
    manifest_path = args.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = load_rows(args.output_dir / "cases.jsonl")
    scored: dict[int, dict[str, Any]] = {}
    for path in sorted(args.output_dir.glob("scores_rank_*.jsonl")):
        for row in load_rows(path):
            scored[int(row["case_id"])] = row
    missing = sorted(set(range(len(cases))) - set(scored))
    if missing:
        raise RuntimeError(f"reference score is incomplete: {len(missing)} cases missing")
    banks = np.load(args.output_dir / "latent_banks.npz")
    centers = np.asarray(banks["centers"], dtype=np.float64)
    directions = np.asarray(banks["directions"], dtype=np.float64)
    scales = np.asarray(banks["scales"], dtype=np.float64)
    gradients = np.full(
        (len(manifest["centers"]), len(scales), *centers.shape[1:]), np.nan
    )
    summary_centers = []
    for center_index, center in enumerate(manifest["centers"]):
        center_rows = [row for row in cases if int(row["center_index"]) == center_index]
        base_case = next(row for row in center_rows if row["kind"] == "center")
        base_result = scored[int(base_case["case_id"])]["result"]
        base_fingerprint = tuple(base_result["branch_fingerprint"]) if base_result else ("worker_error",)
        scale_rows = []
        for scale_index, scale in enumerate(scales):
            slopes = np.full(directions.shape[1], np.nan)
            smooth = np.zeros(directions.shape[1], dtype=bool)
            status_counts: dict[str, int] = {}
            for direction_index in range(directions.shape[1]):
                pair = [
                    row
                    for row in center_rows
                    if row["kind"] == "endpoint"
                    and int(row["scale_index"]) == scale_index
                    and int(row["direction_index"]) == direction_index
                ]
                by_sign = {int(row["sign"]): scored[int(row["case_id"])] for row in pair}
                minus = by_sign[-1]["result"]
                plus = by_sign[1]["result"]
                statuses = (
                    "worker_error" if minus is None else minus["status"],
                    "worker_error" if plus is None else plus["status"],
                )
                for status in statuses:
                    status_counts[status] = status_counts.get(status, 0) + 1
                same_branch = (
                    minus is not None
                    and plus is not None
                    and minus["status"] == "ok"
                    and plus["status"] == "ok"
                    and tuple(minus["branch_fingerprint"]) == base_fingerprint
                    and tuple(plus["branch_fingerprint"]) == base_fingerprint
                )
                if same_branch:
                    slopes[direction_index] = (float(plus["score"]) - float(minus["score"])) / (2.0 * float(scale))
                    smooth[direction_index] = True
            if np.all(smooth):
                gradients[center_index, scale_index] = np.mean(
                    slopes[:, None] * directions[center_index], axis=0
                ).reshape(center["n_coils"], TOKEN_DIM)
            scale_rows.append(
                {
                    "scale": float(scale),
                    "smooth_direction_count": int(np.sum(smooth)),
                    "smooth_fraction": float(np.mean(smooth)),
                    "endpoint_status_counts": status_counts,
                    "slope_rms": float(np.sqrt(np.nanmean(slopes * slopes))),
                    "gradient_rms": float(np.sqrt(np.nanmean(gradients[center_index, scale_index] ** 2))),
                }
            )
        convergence = []
        for index in range(len(scales) - 1):
            left = gradients[center_index, index].ravel()
            right = gradients[center_index, index + 1].ravel()
            denominator = np.linalg.norm(left) * np.linalg.norm(right)
            convergence.append(
                {
                    "large_scale": float(scales[index]),
                    "small_scale": float(scales[index + 1]),
                    "cosine": float(np.dot(left, right) / denominator) if denominator > 0.0 else float("nan"),
                    "relative_l2": float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1.0e-30)),
                }
            )
        summary_centers.append(
            {
                "center_id": center["center_id"],
                "recorded_score": center["recorded_score"],
                "reevaluated_center": base_result,
                "scales": scale_rows,
                "scale_convergence": convergence,
            }
        )
    np.savez_compressed(args.output_dir / "reference_gradients.npz", gradients=gradients, scales=scales)
    elapsed_values = np.asarray([float(row["elapsed_s"]) for row in scored.values()], dtype=np.float64)
    summary = {
        "format": "qh_blackbox_gradient_reference_summary_v1",
        "score_definition": SCORE_DEFINITION,
        "case_count": len(cases),
        "score_wall_sum_s": float(np.sum(elapsed_values)),
        "score_elapsed_mean_s": float(np.mean(elapsed_values)),
        "score_elapsed_p95_s": float(np.percentile(elapsed_values, 95.0)),
        "centers": summary_centers,
    }
    write_json(args.output_dir / "summary.json", summary)
    manifest["state"] = "complete"
    manifest["summary"] = "summary.json"
    write_json(manifest_path, manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a multi-scale black-box latent reference gradient.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--center", type=parse_center, action="append", default=[])
    parser.add_argument("--scales", type=parse_floats, default=DEFAULT_SCALES)
    parser.add_argument("--seed", type=int, default=2026080401)
    parser.add_argument("--rk4-steps", type=int, default=256)
    parser.add_argument("--decode-batch-size", type=int, default=32)
    parser.add_argument("--torch-device", default="cuda:0")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--device-id", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    modes = sum((args.prepare_only, args.score_only, args.analyze_only))
    if modes != 1:
        raise ValueError("select exactly one of --prepare-only, --score-only, --analyze-only")
    if args.rank < 0 or args.world_size < 1 or args.rank >= args.world_size:
        raise ValueError("rank must be in [0, world_size)")
    if args.rk4_steps < 1 or args.decode_batch_size < 1:
        raise ValueError("RK4 steps and decode batch size must be positive")
    if args.prepare_only:
        if not args.center:
            raise ValueError("--prepare-only requires at least one --center")
        prepare(args)
    elif args.score_only:
        score_shard(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
