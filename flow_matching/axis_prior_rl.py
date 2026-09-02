from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from flow_matching.data import CoilNormalizer, canonicalize_currents
from flow_matching.flow import physical_flow_feature_weights
from flow_matching.model import CoilFlowTransformer


PROTOCOL_ID = "qh-axisflip-prior-distilled-online-adam20-rwcfm-abi11-v1"
FORMAT = "axisflip_prior_distilled_online_adam20_rwcfm_v1"
TEACHER_FORMAT = "axisflip_compact_prior_teacher_dataset_v1"
NFP = 8
N_BASE_COILS = 3
TOKEN_DIM = 100
CURRENT_REFERENCE_KEY = f"{NFP}:{N_BASE_COILS}"


def model_config(
    *, width: int = 256, layers: int = 6, heads: int = 8, hidden: int = 704
) -> dict[str, int]:
    return {
        "token_dim": TOKEN_DIM,
        "width": int(width),
        "layers": int(layers),
        "heads": int(heads),
        "hidden": int(hidden),
        "max_nfp": 16,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_teacher_dataset(
    dataset_dir: Path, *, verify_hashes: bool = False
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != TEACHER_FORMAT or manifest.get("status") != "complete":
        raise ValueError("teacher dataset is incomplete or has the wrong format")
    if manifest.get("condition") != {"nfp": NFP, "n_base_coils": N_BASE_COILS}:
        raise ValueError("teacher dataset condition is not fixed nfp=8,nc=3")
    parts = []
    seed_parts = []
    for shard in manifest["shards"]:
        path = dataset_dir / shard["tokens_file"]
        if verify_hashes and file_sha256(path) != shard["tokens_sha256"]:
            raise ValueError(f"teacher shard hash mismatch: {path}")
        values = np.load(path, mmap_mode="r")
        expected = (int(shard["count"]), N_BASE_COILS, TOKEN_DIM)
        if values.shape != expected or values.dtype != np.float32:
            raise ValueError(f"teacher shard shape or dtype mismatch: {path}")
        parts.append(np.asarray(values))
        seed_parts.append(
            np.arange(
                int(shard["seed_start"]),
                int(shard["seed_stop_exclusive"]),
                dtype=np.int64,
            )
        )
    tokens = np.concatenate(parts)
    seeds = np.concatenate(seed_parts)
    if len(tokens) != int(manifest["sample_count"]):
        raise ValueError("teacher manifest sample count mismatch")
    return tokens, seeds, manifest


def split_masks(seeds: np.ndarray, first_seed: int) -> dict[str, np.ndarray]:
    remainder = (np.asarray(seeds, dtype=np.int64) - int(first_seed)) % 20
    return {
        "train": remainder < 18,
        "validation": remainder == 18,
        "test": remainder == 19,
    }


def fit_prior_normalizer(tokens: np.ndarray) -> CoilNormalizer:
    values = np.asarray(tokens, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (N_BASE_COILS, TOKEN_DIM):
        raise ValueError("teacher tokens must have shape (samples,3,100)")
    current_l1 = np.sum(np.abs(values[..., -1]), axis=1, dtype=np.float64)
    reference_l1 = float(np.median(current_l1))
    canonical = canonicalize_currents(values, reference_l1).astype(np.float64)
    flat = canonical.reshape(-1, TOKEN_DIM)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[:99] = np.maximum(std[:99], 1.0e-7)
    current_reference = reference_l1 / N_BASE_COILS
    if not current_reference > 0.0:
        raise ValueError("teacher current reference must be positive")
    # q0 has equal currents. A physical reference scale keeps later Adam-produced
    # relative current allocations representable without inventing q0 variance.
    std[-1] = current_reference
    return CoilNormalizer(
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        current_l1_a={CURRENT_REFERENCE_KEY: reference_l1},
        clip=float("inf"),
    )


def transform_tokens(tokens: np.ndarray, normalizer: CoilNormalizer) -> np.ndarray:
    normalized, clipped = normalizer.transform(
        np.asarray(tokens, dtype=np.float32), (NFP, N_BASE_COILS)
    )
    if clipped != 0.0:
        raise RuntimeError("unclipped prior normalizer unexpectedly clipped tokens")
    if not np.all(np.isfinite(normalized)):
        raise ValueError("normalized tokens contain nonfinite values")
    return normalized


def inverse_tokens(normalized: np.ndarray, normalizer: CoilNormalizer) -> np.ndarray:
    values = normalizer.inverse(
        np.asarray(normalized, dtype=np.float32), (NFP, N_BASE_COILS)
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("decoded tokens contain nonfinite values")
    return values


def random_permute_coils(
    values: torch.Tensor, *, generator: torch.Generator
) -> torch.Tensor:
    if values.ndim != 3 or values.shape[1:] != (N_BASE_COILS, TOKEN_DIM):
        raise ValueError("values must have shape (batch,3,100)")
    order = torch.argsort(
        torch.rand((len(values), N_BASE_COILS), generator=generator), dim=1
    )
    return torch.gather(values, 1, order[..., None].expand(-1, -1, TOKEN_DIM))


def per_sample_flow_terms(
    model: nn.Module,
    data: torch.Tensor,
    *,
    feature_weights: torch.Tensor,
) -> torch.Tensor:
    noise = torch.randn_like(data)
    time_value = torch.rand(len(data), device=data.device, dtype=torch.float32)
    mixed = (1.0 - time_value[:, None, None]) * noise + time_value[:, None, None] * data
    target = data - noise
    prediction = model(
        mixed,
        time_value,
        torch.full((len(data),), NFP, dtype=torch.long, device=data.device),
    )
    weights = feature_weights.to(device=data.device, dtype=torch.float32)
    square = (prediction.float() - target.float()).square() * weights
    return square.sum(dim=(1, 2)) / (N_BASE_COILS * weights.sum()).clamp_min(1.0)


def online_objective(
    model: nn.Module,
    *,
    current_data: torch.Tensor,
    improved_data: torch.Tensor,
    improved_weights: torch.Tensor,
    q0_data: torch.Tensor,
    feature_weights: torch.Tensor,
    improvement_fraction: float = 0.10,
    q0_fraction: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if improvement_fraction < 0.0 or q0_fraction < 0.0:
        raise ValueError("mixture fractions must be nonnegative")
    current_fraction = 1.0 - improvement_fraction - q0_fraction
    if current_fraction < 0.0:
        raise ValueError("mixture fractions sum to more than one")
    if improved_weights.shape != (len(improved_data),):
        raise ValueError("improved weights do not match improved samples")
    if torch.any(improved_weights <= 0.0):
        raise ValueError("improved weights must be positive")
    current_loss = per_sample_flow_terms(
        model, current_data, feature_weights=feature_weights
    ).mean()
    improvement_terms = per_sample_flow_terms(
        model, improved_data, feature_weights=feature_weights
    )
    improvement_loss = torch.sum(improvement_terms * improved_weights) / improved_weights.sum()
    q0_loss = per_sample_flow_terms(model, q0_data, feature_weights=feature_weights).mean()
    total = (
        current_fraction * current_loss
        + improvement_fraction * improvement_loss
        + q0_fraction * q0_loss
    )
    return total, {
        "loss": total.detach(),
        "current_loss": current_loss.detach(),
        "improvement_loss": improvement_loss.detach(),
        "q0_loss": q0_loss.detach(),
    }


def feature_weights(normalizer: CoilNormalizer, device: torch.device) -> torch.Tensor:
    weights, _ = physical_flow_feature_weights(
        torch.from_numpy(normalizer.std),
        relative_geometry_weight=0.05,
        current_feature_weight=1.0,
    )
    return weights.to(device)


def initialize_model(config: dict[str, int], *, seed: int) -> CoilFlowTransformer:
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        return CoilFlowTransformer(**config)


def invariant_descriptors(tokens: np.ndarray) -> np.ndarray:
    values = np.asarray(tokens, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (N_BASE_COILS, TOKEN_DIM):
        raise ValueError("tokens must have shape (samples,3,100)")
    return np.concatenate((values.mean(axis=1), values.std(axis=1)), axis=1)


def diversity_summary(tokens: np.ndarray, *, seed: int = 0, maximum: int = 512) -> dict[str, Any]:
    descriptors = invariant_descriptors(tokens)
    if len(descriptors) == 0:
        return {"count": 0}
    rng = np.random.default_rng(seed)
    if len(descriptors) > maximum:
        descriptors = descriptors[rng.choice(len(descriptors), maximum, replace=False)]
    centered = descriptors.astype(np.float64) - descriptors.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    trace = float(np.trace(covariance))
    square = float(np.sum(covariance * covariance))
    projection = rng.standard_normal((descriptors.shape[1], 32)) / math.sqrt(32.0)
    embedded = descriptors.astype(np.float64) @ projection
    norms = np.sum(embedded * embedded, axis=1)
    distances = np.maximum(norms[:, None] + norms[None, :] - 2.0 * embedded @ embedded.T, 0.0)
    np.fill_diagonal(distances, np.inf)
    nearest = np.sqrt(np.min(distances, axis=1)) if len(embedded) > 1 else np.asarray([math.nan])
    finite = nearest[np.isfinite(nearest)]
    return {
        "count": int(len(tokens)),
        "sampled_count": int(len(descriptors)),
        "descriptor_effective_rank": trace * trace / square if square > 0.0 else 0.0,
        "descriptor_total_variance": trace,
        "nearest_distance_p10": float(np.percentile(finite, 10)) if finite.size else None,
        "nearest_distance_median": float(np.median(finite)) if finite.size else None,
        "near_duplicate_rate_1e-4": float(np.mean(finite < 1.0e-4)) if finite.size else None,
    }
