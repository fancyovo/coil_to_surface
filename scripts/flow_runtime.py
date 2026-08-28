from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import torch

from flow_matching.data import CoilNormalizer
from flow_matching.flow import integrate_flow
from flow_matching.model import CoilFlowTransformer
from scripts.native_score_runtime import NativeScorePool, token_case


TOKEN_DIM = 100


def repository_provenance(repo_root: Path) -> dict[str, Any]:
    """Return stable tracked-source identity without enumerating artifacts."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked_status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=no"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        return {
            "commit": None,
            "tracked_dirty": None,
            "available": False,
            "error": type(exc).__name__,
        }
    return {
        "commit": commit,
        "tracked_dirty": bool(tracked_status.strip()),
        "available": True,
    }


def rms(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array)))


def orthogonal_directions(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    count: int,
) -> np.ndarray:
    dimension = int(np.prod(shape))
    if not 1 <= count <= dimension:
        raise ValueError("direction count must be in [1, latent dimension]")
    matrix = rng.standard_normal((dimension, count))
    basis, _ = np.linalg.qr(matrix, mode="reduced")
    directions = basis.T.reshape(count, *shape) * math.sqrt(dimension)
    return directions.astype(np.float32)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = np.asarray(left, dtype=np.float64).ravel()
    right_flat = np.asarray(right, dtype=np.float64).ravel()
    denominator = np.linalg.norm(left_flat) * np.linalg.norm(right_flat)
    if denominator <= 1.0e-30:
        return float("nan")
    return float(np.dot(left_flat, right_flat) / denominator)


def result_score(result: dict[str, Any] | None) -> float:
    if result is None:
        return 0.0
    score = float(result.get("score", 0.0))
    return score if math.isfinite(score) else 0.0


def result_valid(result: dict[str, Any] | None) -> bool:
    return result is not None and result.get("status") == "ok"


def diagnostics_value(result: dict[str, Any], name: str) -> float:
    return float(result.get("diagnostics", {}).get(name, float("nan")))


def load_flow_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[CoilFlowTransformer, CoilNormalizer, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {"ema", "model_config", "normalizer", "step"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"flow checkpoint is missing keys: {sorted(missing)}")
    model = CoilFlowTransformer(**checkpoint["model_config"]).to(
        device=device, dtype=torch.float32
    )
    model.load_state_dict(checkpoint["ema"])
    model.eval()
    return model, CoilNormalizer.from_dict(checkpoint["normalizer"]), checkpoint


def load_initial_noise(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for section in (
        "flow_prior_start",
        "flow_prior_screening",
        "flow_prior_local_full_gradient_adam",
        "flow_prior_standard_adam",
    ):
        if section in payload:
            noise = payload[section]["noise"]
            break
    else:
        if "noise" not in payload:
            raise ValueError("initial case does not contain flow-prior noise")
        noise = payload["noise"]
    value = np.asarray(noise, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != TOKEN_DIM:
        raise ValueError(f"initial noise must have shape (coils, {TOKEN_DIM})")
    return value, payload


@torch.inference_mode()
def decode_noise_rk4(
    model: CoilFlowTransformer,
    normalizer: CoilNormalizer,
    noise: np.ndarray,
    *,
    nfp: int,
    steps: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    values = np.asarray(noise, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != TOKEN_DIM:
        raise ValueError(f"noise must have shape (batch, coils, {TOKEN_DIM})")
    started = time.perf_counter()
    state = torch.from_numpy(values).to(device=device, dtype=torch.float32)
    nfp_tensor = torch.full(
        (len(values),), int(nfp), dtype=torch.long, device=device
    )
    decoded = integrate_flow(
        model,
        state,
        nfp_tensor,
        start_time=0.0,
        end_time=1.0,
        steps=steps,
        method="rk4",
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    normalized = decoded.cpu().numpy()
    key = (int(nfp), int(values.shape[1]))
    raw = normalizer.inverse(normalized, key).astype(np.float64, copy=False)
    return raw, float(time.perf_counter() - started)


def score_tokens(
    pool: NativeScorePool,
    tokens: np.ndarray,
    *,
    nfp: int,
    target: str,
    timeout_s: float,
    metadata: dict[str, Any],
    config_overrides: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any] | None], list[float], list[str | None], float]:
    cases = [
        token_case(
            value,
            nfp=nfp,
            target=target,
            metadata={**metadata, "batch_index": index},
        )
        for index, value in enumerate(tokens)
    ]
    started = time.perf_counter()
    evaluated = pool.map(
        cases,
        target=target,
        timeout_s=timeout_s,
        config_overrides=config_overrides,
    )
    return (
        [item[0] for item in evaluated],
        [float(item[1]) for item in evaluated],
        [item[2] for item in evaluated],
        float(time.perf_counter() - started),
    )
