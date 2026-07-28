from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


GroupKey = tuple[int, int]
SPLIT_CODES = {"train": 0, "validation": 1, "test": 2}
TOKEN_DIM = 100


@dataclass
class RawGroup:
    tokens: np.ndarray
    ids: np.ndarray


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_raw_groups(
    data_dir: str | Path,
    split: str,
    *,
    verify_hashes: bool = False,
) -> tuple[dict[GroupKey, RawGroup], dict]:
    if split not in SPLIT_CODES:
        raise ValueError(f"unknown split {split!r}")
    root = Path(data_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    token_parts: dict[GroupKey, list[np.ndarray]] = {}
    id_parts: dict[GroupKey, list[np.ndarray]] = {}
    code = SPLIT_CODES[split]
    for shard in manifest["shards"]:
        path = root / shard["file"]
        if verify_hashes and file_sha256(path) != shard["sha256"]:
            raise ValueError(f"SHA256 mismatch for {path}")
        with np.load(path, allow_pickle=False) as payload:
            selected = np.asarray(payload["split"]) == code
            if not np.any(selected):
                continue
            key = (int(shard["nfp"]), int(shard["n_coils"]))
            token_parts.setdefault(key, []).append(
                np.asarray(payload["tokens"][selected], dtype=np.float32)
            )
            id_parts.setdefault(key, []).append(np.asarray(payload["ids"][selected], dtype=np.int32))
    groups = {
        key: RawGroup(tokens=np.concatenate(parts), ids=np.concatenate(id_parts[key]))
        for key, parts in token_parts.items()
    }
    if not groups:
        raise ValueError(f"no samples found for split {split!r} in {root}")
    return groups, manifest


def _current_reference_key(key: GroupKey) -> str:
    return f"{key[0]}:{key[1]}"


def canonicalize_currents(tokens: np.ndarray, target_l1_a: float) -> np.ndarray:
    output = np.asarray(tokens, dtype=np.float32).copy()
    current = output[..., -1]
    l1 = np.sum(np.abs(current), axis=1, keepdims=True)
    fallback = np.full_like(current, float(target_l1_a) / current.shape[1])
    current = np.where(
        l1 > 1.0e-12,
        current * (float(target_l1_a) / np.maximum(l1, 1.0e-12)),
        fallback,
    )
    dominant = np.argmax(np.abs(current), axis=1)
    dominant_sign = np.sign(current[np.arange(current.shape[0]), dominant])
    dominant_sign[dominant_sign == 0.0] = 1.0
    output[..., -1] = current * dominant_sign[:, None]
    return output


@dataclass
class CoilNormalizer:
    mean: np.ndarray
    std: np.ndarray
    current_l1_a: dict[str, float]
    clip: float = 8.0

    @classmethod
    def fit(cls, groups: dict[GroupKey, RawGroup], *, minimum_std: float = 1.0e-7) -> "CoilNormalizer":
        references = {
            _current_reference_key(key): float(
                np.median(np.sum(np.abs(group.tokens[..., -1]), axis=1))
            )
            for key, group in groups.items()
        }
        total = 0
        value_sum = np.zeros(TOKEN_DIM, dtype=np.float64)
        square_sum = np.zeros(TOKEN_DIM, dtype=np.float64)
        for key, group in groups.items():
            canonical = canonicalize_currents(
                group.tokens, references[_current_reference_key(key)]
            ).astype(np.float64)
            flat = canonical.reshape(-1, TOKEN_DIM)
            total += flat.shape[0]
            value_sum += np.sum(flat, axis=0)
            square_sum += np.sum(flat * flat, axis=0)
        mean = value_sum / total
        variance = np.maximum(square_sum / total - mean * mean, 0.0)
        std = np.maximum(np.sqrt(variance), float(minimum_std))
        return cls(
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            current_l1_a=references,
        )

    def transform(self, tokens: np.ndarray, key: GroupKey) -> tuple[np.ndarray, float]:
        canonical = canonicalize_currents(
            tokens, self.current_l1_a[_current_reference_key(key)]
        )
        normalized = (canonical - self.mean) / self.std
        clipped = np.mean(np.abs(normalized) > self.clip)
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32), float(clipped)

    def inverse(self, normalized: np.ndarray, key: GroupKey) -> np.ndarray:
        tokens = np.asarray(normalized, dtype=np.float32) * self.std + self.mean
        return canonicalize_currents(tokens, self.current_l1_a[_current_reference_key(key)])

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "current_l1_a": self.current_l1_a,
            "clip": self.clip,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CoilNormalizer":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
            current_l1_a={str(key): float(value) for key, value in payload["current_l1_a"].items()},
            clip=float(payload.get("clip", 8.0)),
        )


class GroupStore:
    def __init__(self, groups: dict[GroupKey, RawGroup], normalizer: CoilNormalizer):
        self.groups: dict[GroupKey, torch.Tensor] = {}
        self.ids: dict[GroupKey, np.ndarray] = {}
        self.clip_fractions: dict[GroupKey, float] = {}
        for key, group in groups.items():
            normalized, clipped = normalizer.transform(group.tokens, key)
            self.groups[key] = torch.from_numpy(normalized)
            self.ids[key] = group.ids
            self.clip_fractions[key] = clipped
        self.keys = sorted(self.groups)
        counts = np.asarray([len(self.groups[key]) for key in self.keys], dtype=np.float64)
        self.probabilities = counts / np.sum(counts)

    def choose_key(self, generator: np.random.Generator) -> GroupKey:
        return self.keys[int(generator.choice(len(self.keys), p=self.probabilities))]

    def batch(
        self,
        key: GroupKey,
        batch_size: int,
        *,
        device: torch.device,
        generator: torch.Generator,
        permute: bool,
    ) -> torch.Tensor:
        source = self.groups[key]
        indices = torch.randint(len(source), (int(batch_size),), generator=generator)
        batch = source[indices]
        if permute and key[1] > 1:
            order = torch.argsort(
                torch.rand((batch.shape[0], key[1]), generator=generator), dim=1
            )
            batch = torch.gather(batch, 1, order[..., None].expand(-1, -1, TOKEN_DIM))
        return batch.to(device=device, non_blocking=True)


def group_counts(groups: dict[GroupKey, RawGroup] | GroupStore) -> dict[str, int]:
    values = groups.groups if isinstance(groups, GroupStore) else groups
    return {f"nfp{key[0]}_nc{key[1]}": len(value if torch.is_tensor(value) else value.tokens) for key, value in values.items()}
