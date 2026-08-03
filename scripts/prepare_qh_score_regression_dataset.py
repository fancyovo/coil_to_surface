from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch


CORPUS_FORMAT = "qh_flow_iid_native_score_corpus_v1"
DATASET_FORMAT = "qh_latent_score_regression_dataset_v1"
SCORE_BINS = (-math.inf, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, math.inf)
SPLITS = ("train", "validation", "test")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


def score_bin_index(score: float) -> int:
    return int(np.searchsorted(np.asarray(SCORE_BINS[1:-1]), score, side="right"))


def score_bin_label(index: int) -> str:
    lower = SCORE_BINS[index]
    upper = SCORE_BINS[index + 1]
    if not math.isfinite(lower):
        return f"lt_{upper:g}"
    if not math.isfinite(upper):
        return f"ge_{lower:g}"
    return f"{lower:g}_to_{upper:g}"


def stable_order_key(sample_id: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()


def assign_stratified_splits(
    records: list[dict[str, Any]],
    *,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> list[str]:
    if validation_fraction <= 0.0 or test_fraction <= 0.0:
        raise ValueError("validation and test fractions must be positive")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation and test fractions leave no training data")
    strata: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        strata[(record["status"], score_bin_index(record["score"]))].append(index)
    split = ["train"] * len(records)
    for indices in strata.values():
        indices.sort(key=lambda index: stable_order_key(records[index]["sample_id"], seed))
        count = len(indices)
        if count < 3:
            continue
        validation_count = max(1, int(round(count * validation_fraction)))
        test_count = max(1, int(round(count * test_fraction)))
        while validation_count + test_count >= count:
            if validation_count >= test_count and validation_count > 1:
                validation_count -= 1
            elif test_count > 1:
                test_count -= 1
            else:
                break
        for index in indices[:test_count]:
            split[index] = "test"
        for index in indices[test_count : test_count + validation_count]:
            split[index] = "validation"
    return split


def distribution_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    score = np.asarray([record["score"] for record in records], dtype=np.float64)
    if score.size == 0:
        return {"count": 0}
    return {
        "count": int(score.size),
        "score": {
            "mean": float(np.mean(score)),
            "std": float(np.std(score)),
            "min": float(np.min(score)),
            "p10": float(np.quantile(score, 0.10)),
            "median": float(np.median(score)),
            "p90": float(np.quantile(score, 0.90)),
            "p95": float(np.quantile(score, 0.95)),
            "p99": float(np.quantile(score, 0.99)),
            "max": float(np.max(score)),
        },
        "status_counts": dict(sorted(Counter(record["status"] for record in records).items())),
        "condition_counts": dict(
            sorted(
                Counter(
                    f"nfp{record['nfp']}_nc{record['n_coils']}" for record in records
                ).items()
            )
        ),
        "score_bin_counts": dict(
            sorted(
                Counter(
                    score_bin_label(score_bin_index(record["score"])) for record in records
                ).items()
            )
        ),
        "score_tail_counts": {
            f"gt_{threshold:g}": int(np.sum(score > threshold))
            for threshold in (10.0, 20.0, 30.0, 40.0, 50.0)
        },
    }


def plot_dataset_distribution(records: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    score = np.asarray([record["score"] for record in records], dtype=np.float64)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].hist(score, bins=np.linspace(0.0, max(60.0, float(np.max(score)) + 1.0), 61), color="#24735a", alpha=0.86)
    for threshold in (20.0, 30.0, 40.0):
        axes[0].axvline(threshold, color="#9b3d32", ls="--", lw=1)
    axes[0].set(xlabel="native score", ylabel="sample count", title="Frozen current-version score distribution")
    ordered = np.sort(score)
    survival = (len(ordered) - np.arange(len(ordered))) / len(ordered)
    axes[1].plot(ordered, survival, color="#245b85", lw=2)
    axes[1].set(yscale="log", xlabel="native score threshold", ylabel="fraction with score at least threshold", title="Empirical score-tail survival")
    axes[1].grid(axis="y", alpha=0.25)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def write_split(
    output_dir: Path,
    split: str,
    records: list[dict[str, Any]],
    *,
    max_coils: int,
    status_codes: dict[str, int],
) -> dict[str, Any]:
    count = len(records)
    tokens = torch.zeros((count, max_coils, 100), dtype=torch.float32)
    mask = torch.zeros((count, max_coils), dtype=torch.bool)
    nfp = torch.empty(count, dtype=torch.int64)
    n_coils = torch.empty(count, dtype=torch.int64)
    target = torch.empty(count, dtype=torch.float32)
    status = torch.empty(count, dtype=torch.int16)
    for index, record in enumerate(records):
        coil_count = int(record["n_coils"])
        tokens[index, :coil_count] = torch.from_numpy(record["latent"])
        mask[index, :coil_count] = True
        nfp[index] = int(record["nfp"])
        n_coils[index] = coil_count
        target[index] = float(record["score"]) / 100.0
        status[index] = status_codes[record["status"]]
    payload_path = output_dir / f"{split}.pt"
    torch.save(
        {
            "format": DATASET_FORMAT,
            "split": split,
            "tokens": tokens,
            "mask": mask,
            "nfp": nfp,
            "n_coils": n_coils,
            "target": target,
            "status": status,
        },
        payload_path,
    )
    ids_path = output_dir / f"{split}_sample_ids.txt"
    ids_path.write_text(
        "".join(f"{record['sample_id']}\n" for record in records), encoding="utf-8"
    )
    return {
        "payload": payload_path.name,
        "payload_sha256": file_sha256(payload_path),
        "sample_ids": ids_path.name,
        "sample_ids_sha256": file_sha256(ids_path),
        **distribution_summary(records),
    }


def prepare_dataset(
    corpus_root: Path,
    output_dir: Path,
    *,
    score_library_sha256: str,
    seed: int = 20260802,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    verify_shard_hashes: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    shard_dir = corpus_root / "shards"
    stream_dir = corpus_root / "streams"
    frozen_meta_paths = sorted(shard_dir.glob("*.meta.json"))
    if not frozen_meta_paths:
        raise ValueError(f"no completed shard metadata under {shard_dir}")

    records: list[dict[str, Any]] = []
    included_shards: list[dict[str, Any]] = []
    excluded_library_rows: Counter[str] = Counter()
    excluded_invalid_rows: Counter[str] = Counter()
    seen_ids: set[str] = set()
    flow_checkpoint_hashes: Counter[str] = Counter()
    manifest_cache: dict[str, dict[str, Any]] = {}
    for meta_path in frozen_meta_paths:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("format") != CORPUS_FORMAT:
            raise ValueError(f"unsupported metadata format in {meta_path}")
        stream_id = str(metadata["stream_id"])
        stream_manifest = manifest_cache.get(stream_id)
        if stream_manifest is None:
            manifest_path = stream_dir / stream_id / "manifest.json"
            stream_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_cache[stream_id] = stream_manifest
        library_hash = str(stream_manifest["native_score"]["library_sha256"])
        if library_hash != score_library_sha256:
            excluded_library_rows[library_hash] += int(metadata["row_count"])
            continue
        flow_hash = str(stream_manifest["flow_checkpoint"]["sha256"])
        flow_checkpoint_hashes[flow_hash] += int(metadata["row_count"])
        shard_path = shard_dir / str(metadata["file"])
        actual_hash = file_sha256(shard_path) if verify_shard_hashes else str(metadata["sha256"])
        if actual_hash != str(metadata["sha256"]):
            raise ValueError(f"SHA256 mismatch for {shard_path}")
        row_count = 0
        with gzip.open(shard_path, "rt", encoding="utf-8") as stream:
            for line in stream:
                row_count += 1
                row = json.loads(line)
                if row.get("format") != CORPUS_FORMAT:
                    raise ValueError(f"unsupported row format in {shard_path}")
                if row["provenance"]["score_library_sha256"] != score_library_sha256:
                    raise ValueError(f"row provenance disagrees with stream in {shard_path}")
                native = row.get("native_score")
                if native is None or not math.isfinite(float(native.get("score", math.nan))):
                    excluded_invalid_rows["missing_or_nonfinite_score"] += 1
                    continue
                score = float(native["score"])
                if not 0.0 <= score <= 100.0:
                    excluded_invalid_rows["score_out_of_range"] += 1
                    continue
                nfp = int(row["condition"]["nfp"])
                n_coils = int(row["condition"]["n_base_coils"])
                latent = np.asarray(row["latent"], dtype=np.float32)
                if latent.shape != (n_coils, 100) or not np.isfinite(latent).all():
                    excluded_invalid_rows["invalid_latent"] += 1
                    continue
                sample_id = str(row["sample_id"])
                if sample_id in seen_ids:
                    raise ValueError(f"duplicate sample id {sample_id}")
                seen_ids.add(sample_id)
                records.append(
                    {
                        "sample_id": sample_id,
                        "nfp": nfp,
                        "n_coils": n_coils,
                        "latent": latent,
                        "score": score,
                        "status": str(native["status"]),
                    }
                )
        if row_count != int(metadata["row_count"]):
            raise ValueError(f"row count mismatch for {shard_path}")
        included_shards.append(
            {
                "meta": meta_path.name,
                "file": shard_path.name,
                "sha256": actual_hash,
                "row_count": row_count,
                "stream_id": stream_id,
            }
        )
    if not records:
        raise ValueError("no rows matched the requested score-library hash")

    assignments = assign_stratified_splits(
        records,
        seed=seed,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    split_records = {
        split: [record for record, assigned in zip(records, assignments, strict=True) if assigned == split]
        for split in SPLITS
    }
    if any(not values for values in split_records.values()):
        raise ValueError("stratification produced an empty split")
    split_ids = [
        {record["sample_id"] for record in split_records[split]} for split in SPLITS
    ]
    if any(split_ids[i] & split_ids[j] for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError("split sample ids overlap")

    status_codes = {
        status: index
        for index, status in enumerate(sorted({record["status"] for record in records}))
    }
    max_coils = max(record["n_coils"] for record in records)
    plot_dataset_distribution(records, output_dir / "dataset_score_distribution.png")
    split_summaries = {
        split: write_split(
            output_dir,
            split,
            split_records[split],
            max_coils=max_coils,
            status_codes=status_codes,
        )
        for split in SPLITS
    }
    snapshot_digest = hashlib.sha256(
        "".join(f"{row['file']}:{row['sha256']}\n" for row in included_shards).encode("utf-8")
    ).hexdigest()
    manifest = {
        "format": DATASET_FORMAT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(),
        "source_corpus_root": str(corpus_root.resolve()),
        "score_library_sha256": score_library_sha256,
        "flow_checkpoint_row_counts": dict(sorted(flow_checkpoint_hashes.items())),
        "snapshot": {
            "metadata_files_seen": len(frozen_meta_paths),
            "included_shards": len(included_shards),
            "included_shard_digest": snapshot_digest,
            "included_rows": len(records),
            "excluded_library_rows": dict(sorted(excluded_library_rows.items())),
            "excluded_invalid_rows": dict(sorted(excluded_invalid_rows.items())),
            "shards": included_shards,
        },
        "split": {
            "algorithm": "deterministic_sha256_order_within_status_and_score_bin",
            "seed": seed,
            "fractions": {
                "train": 1.0 - validation_fraction - test_fraction,
                "validation": validation_fraction,
                "test": test_fraction,
            },
            "score_bins": ["-inf", 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, "inf"],
            "disjoint_sample_ids_verified": True,
        },
        "representation": {
            "input": "flow latent before decoding",
            "token_dim": 100,
            "max_coils": max_coils,
            "padding": "zero with boolean attention mask",
            "additional_normalization": "none; flow latent uses its native standard-normal scale",
            "target": "native score divided by 100",
            "target_range": [0.0, 1.0],
            "status_codes": status_codes,
        },
        "all_included_rows": distribution_summary(records),
        "distribution_plot": {
            "file": "dataset_score_distribution.png",
            "sha256": file_sha256(output_dir / "dataset_score_distribution.png"),
        },
        "splits": split_summaries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a score-regression dataset from the QH IID corpus.")
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-library-sha256", required=True)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--skip-shard-hash-verification", action="store_true")
    args = parser.parse_args()
    manifest = prepare_dataset(
        args.corpus_root,
        args.output_dir,
        score_library_sha256=args.score_library_sha256,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        verify_shard_hashes=not args.skip_shard_hash_verification,
    )
    print(json.dumps({"output": str(args.output_dir), "splits": manifest["splits"]}, indent=2))


if __name__ == "__main__":
    main()
