from __future__ import annotations

import argparse
from collections import Counter
import csv
from functools import wraps
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

DEFAULT_IDS = (1446077, 1826200, 2419096)
SCORE_DEFINITION = "corrected_abi9_g_over_2pi_per_helicity"
SCORE_LABEL = "corrected ABI-9 score"
_POSITIVE_ALPHAS = (
    0.001,
    0.002,
    0.004,
    0.006,
    0.009,
    0.012,
    0.018,
    0.024,
    0.03,
    0.045,
    0.06,
    0.09,
    0.12,
    0.18,
    0.24,
)
DEFAULT_ALPHAS = tuple(-value for value in reversed(_POSITIVE_ALPHAS)) + (
    0.0,
) + _POSITIVE_ALPHAS


def torch_no_grad(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        import torch

        with torch.no_grad():
            return function(*args, **kwargs)

    return wrapped


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def finite_json(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: finite_json(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [finite_json(value) for value in payload]
    if isinstance(payload, float) and not math.isfinite(payload):
        return None
    return payload


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def normalized_error(reference: np.ndarray, value: np.ndarray) -> dict[str, float]:
    delta = np.asarray(value, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return {
        "normalized_rms": float(np.sqrt(np.mean(delta * delta))),
        "normalized_max": float(np.max(np.abs(delta))),
        "normalized_relative_l2": float(
            np.linalg.norm(delta) / max(np.linalg.norm(reference), 1.0e-30)
        ),
    }


def normalize_direction(direction: np.ndarray) -> np.ndarray:
    value = np.asarray(direction, dtype=np.float64)
    rms = float(np.sqrt(np.mean(value * value)))
    if not math.isfinite(rms) or rms <= 0.0:
        raise ValueError("direction must have finite nonzero RMS")
    return (value / rms).astype(np.float32)


def first_drop_width(
    alphas: np.ndarray,
    scores: np.ndarray,
    *,
    drop: float,
) -> dict[str, float | bool]:
    if drop <= 0.0:
        raise ValueError("drop must be positive")
    zero = np.flatnonzero(np.isclose(alphas, 0.0, atol=1.0e-12))
    if len(zero) != 1:
        raise ValueError("alphas must contain exactly one zero")
    center_score = float(scores[zero[0]])
    threshold = center_score - drop

    def side_width(sign: int) -> tuple[float, bool]:
        selected = np.flatnonzero(alphas * sign > 0.0)
        selected = selected[np.argsort(np.abs(alphas[selected]))]
        previous_x = 0.0
        previous_score = center_score
        for index in selected:
            current_x = float(abs(alphas[index]))
            current_score = float(scores[index])
            if current_score <= threshold:
                denominator = previous_score - current_score
                fraction = (
                    1.0
                    if denominator <= 0.0
                    else np.clip((previous_score - threshold) / denominator, 0.0, 1.0)
                )
                return previous_x + float(fraction) * (current_x - previous_x), False
            previous_x = current_x
            previous_score = current_score
        return previous_x, True

    negative, negative_censored = side_width(-1)
    positive, positive_censored = side_width(1)
    return {
        "negative": negative,
        "positive": positive,
        "total": negative + positive,
        "negative_censored": negative_censored,
        "positive_censored": positive_censored,
    }


def first_drop_radius(
    alphas: np.ndarray,
    radii: np.ndarray,
    scores: np.ndarray,
    *,
    drop: float,
) -> dict[str, float | bool]:
    if len(alphas) != len(radii) or len(alphas) != len(scores):
        raise ValueError("alpha, radius, and score lengths must match")
    zero = np.flatnonzero(np.isclose(alphas, 0.0, atol=1.0e-12))
    if len(zero) != 1:
        raise ValueError("alphas must contain exactly one zero")
    center_score = float(scores[zero[0]])
    threshold = center_score - drop

    def side_radius(sign: int) -> tuple[float, bool]:
        selected = np.flatnonzero(alphas * sign > 0.0)
        selected = selected[np.argsort(np.abs(alphas[selected]))]
        previous_radius = 0.0
        previous_score = center_score
        for index in selected:
            current_radius = float(radii[index])
            current_score = float(scores[index])
            if current_score <= threshold:
                denominator = previous_score - current_score
                fraction = (
                    1.0
                    if denominator <= 0.0
                    else np.clip((previous_score - threshold) / denominator, 0.0, 1.0)
                )
                return previous_radius + float(fraction) * (
                    current_radius - previous_radius
                ), False
            previous_radius = current_radius
            previous_score = current_score
        return previous_radius, True

    negative, negative_censored = side_radius(-1)
    positive, positive_censored = side_radius(1)
    return {
        "negative": negative,
        "positive": positive,
        "mean": 0.5 * (negative + positive),
        "negative_censored": negative_censored,
        "positive_censored": positive_censored,
    }


def curve_roughness(alphas: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    order = np.argsort(alphas)
    x = np.asarray(alphas, dtype=np.float64)[order]
    y = np.asarray(scores, dtype=np.float64)[order]
    if len(x) < 3 or np.any(np.diff(x) <= 0.0):
        raise ValueError("roughness requires at least three distinct ordered points")
    adjacent = np.diff(y)
    intervals = np.diff(x)
    slopes = adjacent / intervals
    second = 2.0 * np.diff(slopes) / (intervals[:-1] + intervals[1:])
    second_weights = 0.5 * (intervals[:-1] + intervals[1:])
    return {
        "total_variation": float(np.sum(np.abs(adjacent))),
        "max_adjacent_jump": float(np.max(np.abs(adjacent))),
        "second_derivative_rms": float(
            np.sqrt(np.average(second * second, weights=second_weights))
        ),
    }


class CaseRegistry:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.by_hash: dict[str, int] = {}

    def register(self, tokens: np.ndarray, nfp: int) -> int:
        values = np.ascontiguousarray(tokens, dtype=np.float32)
        digest = hashlib.sha256()
        digest.update(int(nfp).to_bytes(2, "little", signed=False))
        digest.update(values.tobytes())
        key = digest.hexdigest()
        if key not in self.by_hash:
            case_id = len(self.rows)
            self.by_hash[key] = case_id
            self.rows.append(
                {
                    "case_id": case_id,
                    "sha256": key,
                    "nfp": int(nfp),
                    "n_coils": int(values.shape[0]),
                    "tokens": values.tolist(),
                }
            )
        return self.by_hash[key]


def make_point(
    registry: CaseRegistry,
    tokens: np.ndarray,
    *,
    source_id: int,
    nfp: int,
    kind: str,
    center: np.ndarray,
    path: str | None = None,
    direction: int | None = None,
    alpha: float | None = None,
    normalized_delta_rms: float | None = None,
) -> dict[str, Any]:
    from scripts.qh_score_noise_sensitivity import perturbation_metrics

    return {
        "point_id": -1,
        "case_id": registry.register(tokens, nfp),
        "source_id": int(source_id),
        "kind": kind,
        "path": path,
        "direction": direction,
        "alpha": alpha,
        "normalized_delta_rms": normalized_delta_rms,
        "perturbation": perturbation_metrics(tokens, center),
    }


@torch_no_grad
def prepare_cases(args: argparse.Namespace) -> None:
    import torch

    from flow_matching.data import CoilNormalizer
    from flow_matching.flow import integrate_flow
    from flow_matching.model import CoilFlowTransformer
    from scripts.qh_score_noise_sensitivity import find_sources, perturbation_metrics

    if not torch.cuda.is_available():
        raise RuntimeError("flow landscape preparation requires CUDA")
    source_ids = parse_ints(args.source_ids)
    alphas = np.asarray(parse_floats(args.alphas), dtype=np.float32)
    if len(source_ids) < 1 or len(alphas) < 3:
        raise ValueError("at least one source and three alpha values are required")
    if np.count_nonzero(np.isclose(alphas, 0.0, atol=1.0e-12)) != 1:
        raise ValueError("alphas must contain exactly one zero")
    if len(np.unique(alphas)) != len(alphas):
        raise ValueError("alphas must be unique")
    closure_steps = parse_ints(args.closure_steps)
    if (
        not closure_steps
        or any(steps < 1 for steps in closure_steps)
        or tuple(sorted(set(closure_steps))) != closure_steps
    ):
        raise ValueError("closure steps must be positive, unique, and increasing")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    model = CoilFlowTransformer(**checkpoint["model_config"]).to(device=device, dtype=torch.float32)
    model.load_state_dict(checkpoint["ema"])
    model.eval()
    normalizer = CoilNormalizer.from_dict(checkpoint["normalizer"])
    sources = find_sources(args.data_dir, source_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    max_coils = max(int(sources[source_id]["key"][1]) for source_id in source_ids)
    reference = np.zeros((len(source_ids), max_coils, 100), dtype=np.float32)
    masks = np.zeros((len(source_ids), max_coils), dtype=bool)
    canonical_raw: dict[int, np.ndarray] = {}
    source_rows = []
    for batch_index, source_id in enumerate(source_ids):
        source = sources[source_id]
        key = tuple(source["key"])
        normalized, clipped = normalizer.transform(source["tokens"][None], key)
        count = key[1]
        reference[batch_index, :count] = normalized[0]
        masks[batch_index, :count] = True
        canonical_raw[source_id] = normalizer.inverse(normalized, key)[0]
        source_rows.append(
            {
                "source_id": source_id,
                "split": source["split"],
                "nfp": key[0],
                "n_coils": key[1],
                "normalizer_clipped_fraction": clipped,
                "source_to_canonical": perturbation_metrics(
                    canonical_raw[source_id], source["tokens"]
                ),
            }
        )

    reference_tensor = torch.from_numpy(reference).to(device)
    mask_tensor = torch.from_numpy(masks).to(device)
    nfp_tensor = torch.tensor(
        [int(sources[source_id]["key"][0]) for source_id in source_ids],
        dtype=torch.long,
        device=device,
    )
    closure_rows = []
    selected_noise = None
    selected_reconstruction = None
    integration_started = time.perf_counter()
    for steps in closure_steps:
        started = time.perf_counter()
        noise = integrate_flow(
            model,
            reference_tensor,
            nfp_tensor,
            start_time=1.0,
            end_time=0.0,
            steps=steps,
            method="rk4",
            mask=mask_tensor,
        )
        reconstruction = integrate_flow(
            model,
            noise,
            nfp_tensor,
            start_time=0.0,
            end_time=1.0,
            steps=steps,
            method="rk4",
            mask=mask_tensor,
        )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        noise_numpy = noise.cpu().numpy()
        reconstruction_numpy = reconstruction.cpu().numpy()
        for batch_index, source_id in enumerate(source_ids):
            key = tuple(sources[source_id]["key"])
            count = key[1]
            reconstructed_raw = normalizer.inverse(
                reconstruction_numpy[batch_index, None, :count], key
            )[0]
            closure_rows.append(
                {
                    "source_id": source_id,
                    "steps": steps,
                    "batch_wall_s": elapsed,
                    **normalized_error(
                        reference[batch_index, :count],
                        reconstruction_numpy[batch_index, :count],
                    ),
                    **{
                        f"raw_{name}": value
                        for name, value in perturbation_metrics(
                            reconstructed_raw, canonical_raw[source_id]
                        ).items()
                    },
                }
            )
        selected_noise = noise_numpy
        selected_reconstruction = reconstruction_numpy
    assert selected_noise is not None and selected_reconstruction is not None

    registry = CaseRegistry()
    points: list[dict[str, Any]] = []
    direction_manifest = []
    for batch_index, source_id in enumerate(source_ids):
        source = sources[source_id]
        key = tuple(source["key"])
        nfp, count = key
        source_raw = np.asarray(source["tokens"], dtype=np.float32)
        source_flipped = source_raw.copy()
        source_flipped[:, -1] *= -1.0
        x_reference = reference[batch_index, :count]
        x_center = selected_reconstruction[batch_index, :count]
        z_center = selected_noise[batch_index, :count]
        center_raw = normalizer.inverse(x_center[None], key)[0]
        points.append(
            make_point(
                registry,
                source_raw,
                source_id=source_id,
                nfp=nfp,
                kind="source_raw",
                center=center_raw,
            )
        )
        points.append(
            make_point(
                registry,
                source_flipped,
                source_id=source_id,
                nfp=nfp,
                kind="source_flipped",
                center=source_raw,
            )
        )
        points.append(
            make_point(
                registry,
                canonical_raw[source_id],
                source_id=source_id,
                nfp=nfp,
                kind="source_canonical",
                center=center_raw,
            )
        )
        points.append(
            make_point(
                registry,
                center_raw,
                source_id=source_id,
                nfp=nfp,
                kind="reconstruction",
                center=center_raw,
                normalized_delta_rms=0.0,
            )
        )

        rng = np.random.default_rng(args.seed + source_id * 1000003)
        directions = np.stack(
            [normalize_direction(rng.standard_normal(z_center.shape)) for _ in range(args.directions)]
        )
        derivative_states = np.concatenate(
            [
                z_center[None] - args.derivative_step * directions,
                z_center[None] + args.derivative_step * directions,
            ],
            axis=0,
        )
        derivative_tensor = torch.from_numpy(derivative_states).to(device)
        derivative_nfp = torch.full(
            (len(derivative_states),), nfp, dtype=torch.long, device=device
        )
        derivative_decoded = integrate_flow(
            model,
            derivative_tensor,
            derivative_nfp,
            steps=closure_steps[-1],
            method="rk4",
        ).cpu().numpy()
        tangent = (
            derivative_decoded[args.directions :]
            - derivative_decoded[: args.directions]
        ) / (2.0 * args.derivative_step)
        random_data_directions = np.stack(
            [
                normalize_direction(rng.standard_normal(x_center.shape))
                * float(np.sqrt(np.mean(tangent[direction] ** 2)))
                for direction in range(args.directions)
            ]
        )

        latent_states = (
            z_center[None, None]
            + alphas[None, :, None, None] * directions[:, None]
        ).reshape(-1, count, 100)
        latent_tensor = torch.from_numpy(latent_states).to(device)
        latent_nfp = torch.full(
            (len(latent_states),), nfp, dtype=torch.long, device=device
        )
        latent_decoded = integrate_flow(
            model,
            latent_tensor,
            latent_nfp,
            steps=closure_steps[-1],
            method="rk4",
        ).cpu().numpy()
        tangent_direct_decoded = (
            x_center[None, None]
            + alphas[None, :, None, None] * tangent[:, None]
        ).reshape(-1, count, 100)
        random_direct_decoded = (
            x_center[None, None]
            + alphas[None, :, None, None] * random_data_directions[:, None]
        ).reshape(-1, count, 100)
        latent_raw = normalizer.inverse(latent_decoded, key)
        tangent_direct_raw = normalizer.inverse(tangent_direct_decoded, key)
        random_direct_raw = normalizer.inverse(random_direct_decoded, key)

        np.savez_compressed(
            args.output_dir / f"directions_{source_id}.npz",
            source_reference=x_reference,
            reconstruction=x_center,
            noise_center=z_center,
            latent_directions=directions,
            transported_tangents=tangent,
            random_data_directions=random_data_directions,
            alphas=alphas,
        )
        for direction in range(args.directions):
            tangent_rms = float(np.sqrt(np.mean(tangent[direction] ** 2)))
            direction_manifest.append(
                {
                    "source_id": source_id,
                    "direction": direction,
                    "latent_direction_rms": float(
                        np.sqrt(np.mean(directions[direction] ** 2))
                    ),
                    "transported_tangent_rms": tangent_rms,
                }
            )
            for alpha_index, alpha in enumerate(alphas):
                flat_index = direction * len(alphas) + alpha_index
                for path, normalized, raw in (
                    ("latent", latent_decoded[flat_index], latent_raw[flat_index]),
                    (
                        "tangent_direct",
                        tangent_direct_decoded[flat_index],
                        tangent_direct_raw[flat_index],
                    ),
                    (
                        "random_direct",
                        random_direct_decoded[flat_index],
                        random_direct_raw[flat_index],
                    ),
                ):
                    points.append(
                        make_point(
                            registry,
                            raw,
                            source_id=source_id,
                            nfp=nfp,
                            kind="landscape",
                            center=center_raw,
                            path=path,
                            direction=direction,
                            alpha=float(alpha),
                            normalized_delta_rms=float(
                                np.sqrt(np.mean((normalized - x_center) ** 2))
                            ),
                        )
                    )

    for point_id, point in enumerate(points):
        point["point_id"] = point_id
    write_jsonl(args.output_dir / "cases.jsonl", registry.rows)
    write_jsonl(args.output_dir / "points.jsonl", points)
    write_json(
        args.output_dir / "manifest.json",
        {
            "source_ids": list(source_ids),
            "alphas": alphas.tolist(),
            "directions": args.directions,
            "seed": args.seed,
            "derivative_step": args.derivative_step,
            "closure_steps": list(closure_steps),
            "selected_steps": closure_steps[-1],
            "checkpoint_step": int(checkpoint["step"]),
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "score_definition": SCORE_DEFINITION,
            "score_library_sha256": hashlib.sha256(args.lib.read_bytes()).hexdigest(),
            "model_parameter_dtype": str(next(model.parameters()).dtype),
            "autocast": False,
            "sources": source_rows,
            "closure": closure_rows,
            "direction_metrics": direction_manifest,
            "point_count": len(points),
            "unique_case_count": len(registry.rows),
            "deduplicated_count": len(points) - len(registry.rows),
            "flow_preparation_wall_s": time.perf_counter() - integration_started,
        },
    )


def score_partition(args: argparse.Namespace) -> None:
    from stellarator_gpu import score_coils_native
    from scripts.optimize_native_score_cem import token_case

    cases = [
        row
        for row in load_jsonl(args.output_dir / "cases.jsonl")
        if int(row["case_id"]) % args.world_size == args.rank
    ]
    started = time.perf_counter()
    score_wall_s = 0.0
    output_path = args.output_dir / f"score_rank_{args.rank:02d}.jsonl"
    with output_path.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(cases, start=1):
            case = token_case(
                np.asarray(row["tokens"], dtype=np.float64),
                nfp=int(row["nfp"]),
                target="QH",
                metadata={"flow_landscape_case_id": int(row["case_id"])},
            )
            result = None
            error = None
            score_started = time.perf_counter()
            try:
                raw = case["raw"]
                result = score_coils_native(
                    args.lib,
                    raw["x"],
                    raw["y"],
                    raw["z"],
                    raw["current"],
                    int(row["nfp"]),
                    device_id=args.rank,
                    target_helicity=(1, int(row["nfp"])),
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - score_started
            score_wall_s += elapsed
            stream.write(
                json.dumps(
                    {
                        "case_id": int(row["case_id"]),
                        "score_wall_s": elapsed,
                        "native_score": result,
                        "error": error,
                    },
                    separators=(",", ":"),
                    allow_nan=True,
                )
                + "\n"
            )
            stream.flush()
            if index % 20 == 0:
                print(
                    json.dumps(
                        {
                            "rank": args.rank,
                            "completed": index,
                            "assigned": len(cases),
                            "elapsed_s": time.perf_counter() - started,
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
    write_json(
        args.output_dir / f"runtime_rank_{args.rank:02d}.json",
        {
            "rank": args.rank,
            "assigned": len(cases),
            "wall_s": time.perf_counter() - started,
            "score_wall_s": score_wall_s,
        },
    )


def compact_score(scored: dict[str, Any]) -> dict[str, Any]:
    native = scored.get("native_score")
    if native is None:
        return {
            "score": 0.0,
            "status": "error",
            "error": scored.get("error"),
            "score_wall_s": float(scored["score_wall_s"]),
        }
    return {
        "score": float(native["score"]),
        "status": str(native["status"]),
        "components": native["components"],
        "diagnostics": native["diagnostics"],
        "score_wall_s": float(scored["score_wall_s"]),
    }


def plot_results(rows: list[dict[str, Any]], manifest: dict, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source_ids = manifest["source_ids"]
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, manifest["directions"]))
    figure, axes = plt.subplots(
        len(source_ids), 1, figsize=(11.0, 3.8 * len(source_ids)), squeeze=False
    )
    for axis, source_id in zip(axes[:, 0], source_ids, strict=True):
        selected = [
            row for row in rows if row["source_id"] == source_id and row["kind"] == "landscape"
        ]
        for direction, color in enumerate(colors):
            for path, linestyle in (
                ("latent", "-"),
                ("tangent_direct", "--"),
                ("random_direct", ":"),
            ):
                curve = sorted(
                    [
                        row
                        for row in selected
                        if row["direction"] == direction and row["path"] == path
                    ],
                    key=lambda row: row["alpha"],
                )
                axis.plot(
                    [row["alpha"] for row in curve],
                    [row["score"] for row in curve],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.5,
                    alpha=0.9,
                    label=f"d{direction} {path}" if source_id == source_ids[0] else None,
                )
        axis.axvline(0.0, color="#444444", linewidth=0.8)
        axis.set(
            title=f"QUASR {source_id}",
            xlabel="paired direction coordinate alpha",
            ylabel=SCORE_LABEL,
        )
        axis.grid(alpha=0.2)
    axes[0, 0].legend(ncol=2, fontsize=7)
    figure.suptitle(
        "QH score landscapes: latent, transported tangent, and random data directions"
    )
    figure.tight_layout()
    figure.savefig(output_dir / "landscape_score_vs_alpha.png", dpi=190)

    for axis in axes[:, 0]:
        axis.set_xlim(-0.035, 0.035)
    figure.suptitle("QH score landscapes near the reference coils")
    figure.tight_layout()
    figure.savefig(output_dir / "landscape_score_vs_alpha_zoom.png", dpi=190)
    plt.close(figure)

    figure, axes = plt.subplots(
        len(source_ids), 1, figsize=(11.0, 3.8 * len(source_ids)), squeeze=False
    )
    for axis, source_id in zip(axes[:, 0], source_ids, strict=True):
        selected = [
            row for row in rows if row["source_id"] == source_id and row["kind"] == "landscape"
        ]
        for direction, color in enumerate(colors):
            for path, marker in (
                ("latent", "o"),
                ("tangent_direct", "x"),
                ("random_direct", "+"),
            ):
                curve = [
                    row
                    for row in selected
                    if row["direction"] == direction and row["path"] == path
                ]
                axis.scatter(
                    [row["perturbation"]["position_delta_rms_m"] for row in curve],
                    [row["score"] for row in curve],
                    color=color,
                    marker=marker,
                    s=14,
                    alpha=0.65,
                )
        axis.set(
            title=f"QUASR {source_id}",
            xlabel="coil position RMS displacement [m]",
            ylabel=SCORE_LABEL,
        )
        axis.set_xscale("symlog", linthresh=5.0e-4)
        axis.grid(alpha=0.2)
    from matplotlib.lines import Line2D

    path_handles = [
        Line2D([], [], color="black", marker=marker, linestyle="none", label=path)
        for path, marker in (
            ("latent", "o"),
            ("tangent direct", "x"),
            ("random direct", "+"),
        )
    ]
    direction_handles = [
        Line2D([], [], color=color, linewidth=2.0, label=f"direction {direction}")
        for direction, color in enumerate(colors)
    ]
    axes[0, 0].legend(
        handles=path_handles + direction_handles,
        ncol=4,
        fontsize=7,
        loc="lower left",
    )
    figure.suptitle("Score retention at matched physical displacement")
    figure.tight_layout()
    figure.savefig(output_dir / "landscape_score_vs_displacement.png", dpi=190)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    for source_id in source_ids:
        closure = sorted(
            [row for row in manifest["closure"] if row["source_id"] == source_id],
            key=lambda row: row["steps"],
        )
        axis.loglog(
            [row["steps"] for row in closure],
            [row["raw_position_delta_rms_m"] for row in closure],
            marker="o",
            label=str(source_id),
        )
    axis.set(
        xlabel="RK4 steps in each direction",
        ylabel="round-trip coil position RMS error [m]",
        title="Reverse-forward closure convergence",
    )
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "closure_convergence.png", dpi=190)
    plt.close(figure)


def analyze(args: argparse.Namespace) -> None:
    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    cases = load_jsonl(args.output_dir / "cases.jsonl")
    points = load_jsonl(args.output_dir / "points.jsonl")
    scored_rows = []
    for path in sorted(args.output_dir.glob("score_rank_*.jsonl")):
        scored_rows.extend(load_jsonl(path))
    score_by_case = {int(row["case_id"]): compact_score(row) for row in scored_rows}
    if len(score_by_case) != len(cases):
        raise RuntimeError(f"expected {len(cases)} unique scores, found {len(score_by_case)}")
    rows = [{**point, **score_by_case[int(point["case_id"])]} for point in points]

    baselines = {}
    curve_metrics = []
    for source_id in manifest["source_ids"]:
        source_rows = [row for row in rows if row["source_id"] == source_id]
        baselines[str(source_id)] = {
            kind: next(row for row in source_rows if row["kind"] == kind)
            for kind in (
                "source_raw",
                "source_flipped",
                "source_canonical",
                "reconstruction",
            )
        }
        for direction in range(manifest["directions"]):
            paired = {}
            for path in ("latent", "tangent_direct", "random_direct"):
                curve = sorted(
                    [
                        row
                        for row in source_rows
                        if row["kind"] == "landscape"
                        and row["direction"] == direction
                        and row["path"] == path
                    ],
                    key=lambda row: row["alpha"],
                )
                alphas = np.asarray([row["alpha"] for row in curve])
                scores = np.asarray([row["score"] for row in curve])
                radii = np.asarray(
                    [row["perturbation"]["position_delta_rms_m"] for row in curve]
                )
                paired[path] = {
                    "source_id": source_id,
                    "direction": direction,
                    "path": path,
                    "status_counts": dict(Counter(row["status"] for row in curve)),
                    "drop_5_width": first_drop_width(alphas, scores, drop=5.0),
                    "drop_10_width": first_drop_width(alphas, scores, drop=10.0),
                    "drop_5_physical_radius_m": first_drop_radius(
                        alphas, radii, scores, drop=5.0
                    ),
                    "drop_10_physical_radius_m": first_drop_radius(
                        alphas, radii, scores, drop=10.0
                    ),
                    "roughness": curve_roughness(alphas, scores),
                }
                curve_metrics.append(paired[path])
            for comparator in ("tangent_direct", "random_direct"):
                for drop_name in ("drop_5_width", "drop_10_width"):
                    direct = float(paired[comparator][drop_name]["total"])
                    latent = float(paired["latent"][drop_name]["total"])
                    paired["latent"][f"ratio_to_{comparator}_{drop_name}"] = (
                        latent / direct if direct > 0.0 else math.inf
                    )
                for radius_name in (
                    "drop_5_physical_radius_m",
                    "drop_10_physical_radius_m",
                ):
                    direct = float(paired[comparator][radius_name]["mean"])
                    latent = float(paired["latent"][radius_name]["mean"])
                    paired["latent"][f"ratio_to_{comparator}_{radius_name}"] = (
                        latent / direct if direct > 0.0 else math.inf
                    )
                for roughness_name in (
                    "total_variation",
                "max_adjacent_jump",
                "second_derivative_rms",
                ):
                    direct = float(paired[comparator]["roughness"][roughness_name])
                    latent = float(paired["latent"]["roughness"][roughness_name])
                    paired["latent"][f"ratio_to_{comparator}_{roughness_name}"] = (
                        latent / direct if direct > 0.0 else math.inf
                    )

    fieldnames = [
        "point_id",
        "case_id",
        "source_id",
        "kind",
        "path",
        "direction",
        "alpha",
        "normalized_delta_rms",
        "position_delta_rms_m",
        "coefficient_relative_l2",
        "status",
        "score",
        "score_wall_s",
    ]
    with (args.output_dir / "landscape_rows.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{name: row.get(name) for name in fieldnames},
                    "position_delta_rms_m": row["perturbation"]["position_delta_rms_m"],
                    "coefficient_relative_l2": row["perturbation"]["coefficient_relative_l2"],
                }
            )

    runtime_rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.output_dir.glob("runtime_rank_*.json"))
    ]
    width_names = (
        "drop_5_width",
        "drop_10_width",
        "drop_5_physical_radius_m",
        "drop_10_physical_radius_m",
    )
    comparisons = ("tangent_direct", "random_direct")
    ratios = {
        comparator: {
            drop_name: [
                row[f"ratio_to_{comparator}_{drop_name}"]
                for row in curve_metrics
                if row["path"] == "latent"
                and math.isfinite(row[f"ratio_to_{comparator}_{drop_name}"])
            ]
            for drop_name in width_names
        }
        for comparator in comparisons
    }
    roughness_names = (
        "total_variation",
        "max_adjacent_jump",
        "second_derivative_rms",
    )
    roughness_ratios = {
        comparator: {
            name: [
                row[f"ratio_to_{comparator}_{name}"]
                for row in curve_metrics
                if row["path"] == "latent"
                and math.isfinite(row[f"ratio_to_{comparator}_{name}"])
            ]
            for name in roughness_names
        }
        for comparator in comparisons
    }
    summary = {
        "manifest": manifest,
        "baselines": baselines,
        "curve_metrics": curve_metrics,
        "aggregate": {
            comparator: {
                name: {
                    "count": len(values),
                    "median_latent_to_comparator_width": float(np.median(values)) if values else None,
                    "p25_latent_to_comparator_width": float(np.percentile(values, 25)) if values else None,
                    "p75_latent_to_comparator_width": float(np.percentile(values, 75)) if values else None,
                    "latent_wider_fraction": float(np.mean(np.asarray(values) > 1.0)) if values else None,
                }
                for name, values in comparison_values.items()
            }
            for comparator, comparison_values in ratios.items()
        },
        "roughness": {
            comparator: {
                name: {
                    "count": len(values),
                    "median_latent_to_comparator": float(np.median(values)) if values else None,
                    "p25_latent_to_comparator": float(np.percentile(values, 25)) if values else None,
                    "p75_latent_to_comparator": float(np.percentile(values, 75)) if values else None,
                    "latent_smoother_fraction": float(np.mean(np.asarray(values) < 1.0)) if values else None,
                }
                for name, values in comparison_values.items()
            }
            for comparator, comparison_values in roughness_ratios.items()
        },
        "runtime": {
            "flow_preparation_wall_s": manifest["flow_preparation_wall_s"],
            "rank": runtime_rows,
            "score_wall_s": max(row["wall_s"] for row in runtime_rows),
            "sum_score_wall_s": sum(row["score_wall_s"] for row in runtime_rows),
        },
        "library_sha256": hashlib.sha256(args.lib.read_bytes()).hexdigest(),
    }
    write_json(args.output_dir / "summary.json", finite_json(summary))
    plot_results(rows, manifest, args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare local QH score landscapes in flow latent and data spaces."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--source-ids", default=",".join(map(str, DEFAULT_IDS)))
    parser.add_argument(
        "--alphas", default=",".join(f"{value:.8g}" for value in DEFAULT_ALPHAS)
    )
    parser.add_argument("--directions", type=int, default=4)
    parser.add_argument("--derivative-step", type=float, default=0.01)
    parser.add_argument("--closure-steps", default="32,64,128,256")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.directions < 1 or args.derivative_step <= 0.0:
        raise ValueError("directions and derivative step must be positive")
    if args.prepare_only and args.analyze_only:
        raise ValueError("prepare-only and analyze-only are mutually exclusive")
    if args.prepare_only:
        prepare_cases(args)
    elif args.analyze_only:
        analyze(args)
    else:
        if not 0 <= args.rank < args.world_size:
            raise ValueError("rank must satisfy 0 <= rank < world_size")
        score_partition(args)


if __name__ == "__main__":
    main()
