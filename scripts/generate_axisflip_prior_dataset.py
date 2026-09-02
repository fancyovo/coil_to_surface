from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flow_matching.axis_surface_prior_v2 import sample_shaped_prior_prototype


FORMAT = "axisflip_compact_prior_teacher_dataset_v1"
SHARD_FORMAT = "axisflip_compact_prior_teacher_shard_v1"
NFP = 8
N_BASE_COILS = 3
TOKEN_DIM = 100
PRESET = "compact_flexible"
GENERATOR_FORMAT = "axis_surface_contour_prior_compact_flexible_axis_flip_v4"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def repository_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def require_clean_expected_commit(expected: str) -> str:
    commit = repository_commit()
    if commit != expected:
        raise RuntimeError(f"repository commit {commit} != expected {expected}")
    dirty = subprocess.check_output(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=REPO_ROOT,
        text=True,
    )
    if dirty.strip():
        raise RuntimeError("tracked experiment worktree is dirty")
    return commit


def generate_block(task: tuple[int, int, int]) -> tuple[int, np.ndarray]:
    offset, seed_start, count = task
    tokens = np.empty((count, N_BASE_COILS, TOKEN_DIM), dtype=np.float32)
    for local_index in range(count):
        generated = sample_shaped_prior_prototype(
            seed=seed_start + local_index,
            nfp=NFP,
            n_base_coils=N_BASE_COILS,
            preset=PRESET,
            surface_phi_samples=96,
            surface_theta_samples=48,
            sample_role="registered_scoring",
            axis_chirality=-1,
        )
        if generated.metadata.get("format") != GENERATOR_FORMAT:
            raise RuntimeError("generator format changed during teacher synthesis")
        if generated.metadata.get("construction_axis_chirality") != -1:
            raise RuntimeError("teacher generator did not use positive-hand axis flip")
        tokens[local_index] = generated.tokens
    return offset, tokens


def block_tasks(seed_start: int, count: int, block_size: int) -> list[tuple[int, int, int]]:
    if count <= 0 or block_size <= 0:
        raise ValueError("count and block size must be positive")
    return [
        (offset, seed_start + offset, min(block_size, count - offset))
        for offset in range(0, count, block_size)
    ]


def generate_shard(args: argparse.Namespace) -> None:
    commit = require_clean_expected_commit(args.expected_commit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = args.output_dir / f"{args.shard_name}.tokens.npy"
    metadata_path = args.output_dir / f"{args.shard_name}.json"
    if tokens_path.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite shard {args.shard_name}")
    temporary = tokens_path.with_name(f".{tokens_path.name}.{os.getpid()}.tmp")
    mapped = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(args.count, N_BASE_COILS, TOKEN_DIM),
    )
    tasks = block_tasks(args.seed_start, args.count, args.block_size)
    started = time.perf_counter()
    completed = 0
    next_report = max(1, args.count // 20)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(generate_block, task) for task in tasks]
        for future in as_completed(futures):
            offset, block = future.result()
            mapped[offset : offset + len(block)] = block
            completed += len(block)
            if completed >= next_report or completed == args.count:
                print(
                    json.dumps(
                        {
                            "event": "teacher_progress",
                            "shard": args.shard_name,
                            "completed": completed,
                            "count": args.count,
                            "elapsed_s": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
                next_report += max(1, args.count // 20)
    mapped.flush()
    del mapped
    os.replace(temporary, tokens_path)
    values = np.load(tokens_path, mmap_mode="r")
    if values.shape != (args.count, N_BASE_COILS, TOKEN_DIM):
        raise RuntimeError("written teacher shard has the wrong shape")
    elapsed = time.perf_counter() - started
    current = np.asarray(values[..., -1], dtype=np.float64)
    payload = {
        "format": SHARD_FORMAT,
        "dataset_format": FORMAT,
        "shard_name": args.shard_name,
        "repository_commit": commit,
        "generator": {
            "format": GENERATOR_FORMAT,
            "preset": PRESET,
            "sample_role": "registered_scoring",
            "axis_chirality": -1,
            "surface_phi_samples": 96,
            "surface_theta_samples": 48,
        },
        "condition": {"nfp": NFP, "n_base_coils": N_BASE_COILS},
        "seed_start": args.seed_start,
        "seed_stop_exclusive": args.seed_start + args.count,
        "count": args.count,
        "workers": args.workers,
        "block_size": args.block_size,
        "tokens_file": tokens_path.name,
        "tokens_sha256": file_sha256(tokens_path),
        "tokens_dtype": str(values.dtype),
        "tokens_shape": list(values.shape),
        "current_min_a": float(np.min(current)),
        "current_max_a": float(np.max(current)),
        "wall_s": elapsed,
        "samples_per_s": args.count / elapsed,
        "finished_unix_s": time.time(),
    }
    atomic_write_json(metadata_path, payload)
    print(json.dumps({"event": "teacher_shard_complete", **payload}), flush=True)


def finalize_dataset(args: argparse.Namespace) -> None:
    commit = require_clean_expected_commit(args.expected_commit)
    manifest_path = args.output_dir / "dataset_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite {manifest_path}")
    sidecars = []
    for name in args.shard_name:
        path = args.output_dir / f"{name}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != SHARD_FORMAT:
            raise ValueError(f"invalid teacher shard metadata: {path}")
        if payload.get("repository_commit") != commit:
            raise ValueError(f"teacher shard commit mismatch: {path}")
        if payload.get("condition") != {"nfp": NFP, "n_base_coils": N_BASE_COILS}:
            raise ValueError(f"teacher shard condition mismatch: {path}")
        tokens_path = args.output_dir / payload["tokens_file"]
        if file_sha256(tokens_path) != payload["tokens_sha256"]:
            raise ValueError(f"teacher shard hash mismatch: {tokens_path}")
        values = np.load(tokens_path, mmap_mode="r")
        if list(values.shape) != payload["tokens_shape"] or str(values.dtype) != "float32":
            raise ValueError(f"teacher shard array contract mismatch: {tokens_path}")
        sidecars.append(payload)
    sidecars.sort(key=lambda row: int(row["seed_start"]))
    cursor = int(sidecars[0]["seed_start"])
    first_seed = cursor
    for row in sidecars:
        if int(row["seed_start"]) != cursor:
            raise ValueError("teacher shard seed ranges overlap or contain a gap")
        cursor = int(row["seed_stop_exclusive"])
    total = sum(int(row["count"]) for row in sidecars)
    if total != args.expected_total:
        raise ValueError(f"teacher sample count {total} != expected {args.expected_total}")
    manifest = {
        "format": FORMAT,
        "status": "complete",
        "repository_commit": commit,
        "condition": {"nfp": NFP, "n_base_coils": N_BASE_COILS},
        "generator": sidecars[0]["generator"],
        "sample_count": total,
        "seed_start": first_seed,
        "seed_stop_exclusive": cursor,
        "split": {
            "rule": "seed offset modulo 20",
            "train_remainders": list(range(18)),
            "validation_remainder": 18,
            "test_remainder": 19,
            "fractions": {"train": 0.9, "validation": 0.05, "test": 0.05},
        },
        "shards": sidecars,
        "finished_unix_s": time.time(),
    }
    atomic_write_json(manifest_path, manifest)
    print(json.dumps({"event": "teacher_dataset_complete", **manifest}), flush=True)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Synthesize the fixed nfp=8,nc=3 axis-flip teacher prior.")
    commands = value.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--shard-name", required=True)
    generate.add_argument("--seed-start", type=int, required=True)
    generate.add_argument("--count", type=int, required=True)
    generate.add_argument("--workers", type=int, required=True)
    generate.add_argument("--block-size", type=int, default=128)
    generate.add_argument("--expected-commit", required=True)
    generate.set_defaults(func=generate_shard)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--output-dir", type=Path, required=True)
    finalize.add_argument("--shard-name", action="append", required=True)
    finalize.add_argument("--expected-total", type=int, default=200_000)
    finalize.add_argument("--expected-commit", required=True)
    finalize.set_defaults(func=finalize_dataset)
    return value


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
