from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import gzip
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np

from flow_matching.data import GroupKey, file_sha256


_GROUP_LABEL = re.compile(r"^nfp(?P<nfp>[1-9][0-9]*)_nc(?P<n_coils>[1-9][0-9]*)$")


def group_label(key: GroupKey) -> str:
    return f"nfp{key[0]}_nc{key[1]}"


def parse_group_label(label: str) -> GroupKey:
    match = _GROUP_LABEL.fullmatch(str(label))
    if match is None:
        raise ValueError(f"invalid condition group label {label!r}")
    return int(match.group("nfp")), int(match.group("n_coils"))


@dataclass(frozen=True)
class ConditionPrior:
    keys: tuple[GroupKey, ...]
    counts: tuple[int, ...]
    source_path: Path
    source_sha256: str
    source_kind: str

    def __post_init__(self) -> None:
        if not self.keys or len(self.keys) != len(self.counts):
            raise ValueError("condition prior must contain matching nonempty keys and counts")
        if tuple(sorted(self.keys)) != self.keys or len(set(self.keys)) != len(self.keys):
            raise ValueError("condition prior keys must be sorted and unique")
        if any(count <= 0 for count in self.counts):
            raise ValueError("condition prior counts must be positive")

    @property
    def total(self) -> int:
        return int(sum(self.counts))

    @property
    def probabilities(self) -> np.ndarray:
        counts = np.asarray(self.counts, dtype=np.float64)
        return counts / counts.sum()

    def sample_indices(self, generator: np.random.Generator, size: int) -> np.ndarray:
        if size < 1:
            raise ValueError("sample size must be positive")
        return generator.choice(len(self.keys), size=int(size), p=self.probabilities)

    def to_dict(self) -> dict[str, Any]:
        probabilities = self.probabilities
        return {
            "definition": "empirical_joint_distribution_over_quasr_qh_training_samples",
            "source_path": str(self.source_path.resolve()),
            "source_sha256": self.source_sha256,
            "source_kind": self.source_kind,
            "total_train_samples": self.total,
            "groups": [
                {
                    "label": group_label(key),
                    "nfp": key[0],
                    "n_coils": key[1],
                    "train_count": int(count),
                    "probability": float(probability),
                }
                for key, count, probability in zip(
                    self.keys, self.counts, probabilities, strict=True
                )
            ],
        }


def _prior_from_counts(
    counts: dict[GroupKey, int],
    *,
    source_path: Path,
    source_kind: str,
    supported_keys: set[GroupKey] | None,
) -> ConditionPrior:
    positive = {key: int(value) for key, value in counts.items() if int(value) > 0}
    if not positive:
        raise ValueError(f"no positive training counts found in {source_path}")
    if supported_keys is not None:
        missing = set(positive) - supported_keys
        extra = supported_keys - set(positive)
        if missing or extra:
            raise ValueError(
                "condition prior and checkpoint normalizer differ: "
                f"unsupported_prior={sorted(missing)}, missing_prior={sorted(extra)}"
            )
    keys = tuple(sorted(positive))
    return ConditionPrior(
        keys=keys,
        counts=tuple(positive[key] for key in keys),
        source_path=source_path,
        source_sha256=file_sha256(source_path),
        source_kind=source_kind,
    )


def load_train_condition_prior(
    data_dir: str | Path,
    *,
    training_run_manifest: str | Path | None = None,
    supported_keys: set[GroupKey] | None = None,
) -> ConditionPrior:
    """Load the exact empirical joint condition prior used for flow training."""
    if training_run_manifest is not None:
        run_manifest_path = Path(training_run_manifest)
        if run_manifest_path.is_file():
            payload = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            raw_counts = payload.get("train_counts")
            if not isinstance(raw_counts, dict):
                raise ValueError(f"train_counts is absent from {run_manifest_path}")
            counts = {
                parse_group_label(label): int(count)
                for label, count in raw_counts.items()
            }
            return _prior_from_counts(
                counts,
                source_path=run_manifest_path,
                source_kind="flow_training_run_manifest.train_counts",
                supported_keys=supported_keys,
            )

    root = Path(data_dir)
    dataset_manifest_path = root / "manifest.json"
    manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    counts: Counter[GroupKey] = Counter()
    for shard in manifest["shards"]:
        path = root / shard["file"]
        with np.load(path, allow_pickle=False) as payload:
            counts[(int(shard["nfp"]), int(shard["n_coils"]))] += int(
                np.count_nonzero(np.asarray(payload["split"]) == 0)
            )
    return _prior_from_counts(
        dict(counts),
        source_path=dataset_manifest_path,
        source_kind="quasr_qh_shards.split_train",
        supported_keys=supported_keys,
    )


def derive_stream_seed(seed_base: int, rank: int, *, max_ranks: int = 16) -> int:
    if seed_base < 0 or rank < 0 or rank >= max_ranks:
        raise ValueError("seed_base and rank must define a valid disjoint stream")
    return int(seed_base) * int(max_ranks) + int(rank)


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    if target.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {target} or {partial}")
    partial.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(partial, target)


def replace_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(partial, target)


def write_jsonl_gzip_atomic(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    *,
    compresslevel: int = 1,
) -> tuple[int, str]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    if target.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {target} or {partial}")
    count = 0
    with gzip.open(
        partial, "wt", encoding="utf-8", compresslevel=int(compresslevel)
    ) as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=True))
            stream.write("\n")
            count += 1
    os.replace(partial, target)
    return count, file_sha256(target)
