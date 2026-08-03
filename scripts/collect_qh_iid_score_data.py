from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "gpu_backend" / "python"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from flow_matching.collection import (
    atomic_write_json,
    derive_stream_seed,
    group_label,
    load_train_condition_prior,
    replace_json,
    write_jsonl_gzip_atomic,
)
from flow_matching.data import CoilNormalizer, GroupKey, file_sha256
from flow_matching.flow import integrate_flow
from flow_matching.model import CoilFlowTransformer
from stellarator_gpu import native_score_config_snapshot, score_coils_native


FORMAT = "qh_flow_iid_native_score_corpus_v1"
TOKEN_DIM = 100
COEFF_COUNT = 33


def normalizer_keys(normalizer: CoilNormalizer) -> set[GroupKey]:
    keys = set()
    for text in normalizer.current_l1_a:
        nfp, n_coils = text.split(":", maxsplit=1)
        keys.add((int(nfp), int(n_coils)))
    return keys


def load_flow(
    checkpoint_path: Path, device: torch.device
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


@torch.inference_mode()
def decode_mixed_conditions(
    model: CoilFlowTransformer,
    normalizer: CoilNormalizer,
    condition_keys: list[GroupKey],
    generator: np.random.Generator,
    *,
    flow_steps: int,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    batch = len(condition_keys)
    max_coils = max(key[1] for key in condition_keys)
    latent = np.zeros((batch, max_coils, TOKEN_DIM), dtype=np.float32)
    mask = np.zeros((batch, max_coils), dtype=bool)
    for index, (_, n_coils) in enumerate(condition_keys):
        latent[index, :n_coils] = generator.standard_normal(
            (n_coils, TOKEN_DIM), dtype=np.float32
        )
        mask[index, :n_coils] = True

    state = torch.from_numpy(latent).to(device=device)
    nfp = torch.tensor([key[0] for key in condition_keys], dtype=torch.long, device=device)
    token_mask = torch.from_numpy(mask).to(device=device)
    started = time.perf_counter()
    decoded = integrate_flow(
        model,
        state,
        nfp,
        start_time=0.0,
        end_time=1.0,
        steps=flow_steps,
        method="rk4",
        mask=token_mask,
    )
    torch.cuda.synchronize(device)
    decode_wall_s = time.perf_counter() - started
    decoded_np = decoded.cpu().numpy()

    latent_rows = []
    raw_rows = []
    grouped: dict[GroupKey, list[int]] = {}
    for index, key in enumerate(condition_keys):
        grouped.setdefault(key, []).append(index)
        latent_rows.append(latent[index, : key[1]].copy())
    raw_by_index: dict[int, np.ndarray] = {}
    for key, indices in grouped.items():
        normalized = np.stack([decoded_np[index, : key[1]] for index in indices])
        raw = normalizer.inverse(normalized, key)
        raw_by_index.update(
            (index, raw[local_index]) for local_index, index in enumerate(indices)
        )
    raw_rows.extend(raw_by_index[index] for index in range(batch))
    return latent_rows, raw_rows, decode_wall_s


def score_tokens(
    lib_path: Path,
    tokens: np.ndarray,
    *,
    nfp: int,
    device_id: int,
) -> tuple[dict[str, Any] | None, str | None, float]:
    started = time.perf_counter()
    try:
        result = score_coils_native(
            lib_path,
            tokens[:, :COEFF_COUNT],
            tokens[:, COEFF_COUNT : 2 * COEFF_COUNT],
            tokens[:, 2 * COEFF_COUNT : 3 * COEFF_COUNT],
            tokens[:, -1],
            nfp,
            device_id=device_id,
            target_helicity=(1, nfp),
        )
        return result, None, time.perf_counter() - started
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}", time.perf_counter() - started


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def ensure_dataset_manifest(dataset_root: Path) -> None:
    path = dataset_root / "dataset_manifest.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != FORMAT:
            raise ValueError(f"unexpected score corpus format in {path}")
        return
    atomic_write_json(
        path,
        {
            "format": FORMAT,
            "description": "Append-only QH flow IID latent, decoded coil, and native-score corpus",
            "shard_glob": "shards/*.jsonl.gz",
            "shard_metadata_glob": "shards/*.meta.json",
            "stream_manifest_glob": "streams/*/manifest.json",
            "token_schema": {
                "shape": "n_base_coils x 100",
                "columns": "x_fourier[33], y_fourier[33], z_fourier[33], current_A",
            },
        },
    )


def should_stop(
    *,
    elapsed_s: float,
    max_wall_s: float,
    shard_durations: list[float],
) -> bool:
    if max_wall_s <= 0.0:
        return False
    reserve = 300.0
    if shard_durations:
        reserve = max(reserve, 1.25 * float(np.mean(shard_durations[-3:])))
    return elapsed_s + reserve >= max_wall_s


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuously decode empirical-condition QH flow latents and native-score them."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--training-run-manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--flow-steps", type=int, default=256)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--max-wall-s", type=float, default=85800.0)
    parser.add_argument("--seed-base", type=int)
    parser.add_argument("--job-id", default=os.environ.get("SLURM_JOB_ID", "local"))
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", str(local_rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.cuda.is_available():
        raise RuntimeError("QH IID score collection requires CUDA")
    if args.flow_steps < 1 or args.shard_size < 1 or args.max_shards < 0:
        raise ValueError("flow steps and shard size must be positive; max shards nonnegative")
    try:
        numeric_job_id = int(args.job_id)
    except ValueError as exc:
        if args.seed_base is None:
            raise ValueError("non-numeric job ID requires --seed-base") from exc
        numeric_job_id = int(args.seed_base)
    seed_base = numeric_job_id if args.seed_base is None else int(args.seed_base)
    stream_seed = derive_stream_seed(seed_base, rank)
    stream_id = f"slurm_{args.job_id}_rank_{rank}"

    torch.cuda.set_device(local_rank)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda", local_rank)
    model, normalizer, checkpoint = load_flow(args.checkpoint, device)
    supported = normalizer_keys(normalizer)
    prior_manifest = args.training_run_manifest
    if prior_manifest is None:
        candidate = args.checkpoint.parent / "run_manifest.json"
        prior_manifest = candidate if candidate.is_file() else None
    prior = load_train_condition_prior(
        args.data_dir,
        training_run_manifest=prior_manifest,
        supported_keys=supported,
    )

    args.dataset_root.mkdir(parents=True, exist_ok=True)
    (args.dataset_root / "shards").mkdir(exist_ok=True)
    stream_dir = args.dataset_root / "streams" / stream_id
    stream_dir.mkdir(parents=True, exist_ok=False)
    if rank == 0:
        ensure_dataset_manifest(args.dataset_root)

    checkpoint_sha = file_sha256(args.checkpoint)
    library_sha = file_sha256(args.lib)
    prior_probabilities = {
        key: float(probability)
        for key, probability in zip(prior.keys, prior.probabilities, strict=True)
    }
    score_configs = {
        str(nfp): native_score_config_snapshot(
            args.lib, device_id=local_rank, target_helicity=(1, nfp)
        )
        for nfp in sorted({key[0] for key in prior.keys})
    }
    manifest = {
        "format": FORMAT,
        "stage": "running",
        "stream_id": stream_id,
        "job_id": str(args.job_id),
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "stream_seed": stream_seed,
        "condition_prior": prior.to_dict(),
        "generation": {
            "latent_prior": "independent_standard_normal_per_real_coil_token",
            "condition_draw": "joint_empirical_quasr_qh_training_distribution",
            "flow_method": "rk4",
            "flow_steps": args.flow_steps,
            "flow_dtype": "float32",
            "flow_autocast": False,
            "mixed_condition_batching": "zero_padded_with_attention_mask",
            "shard_size": args.shard_size,
        },
        "flow_checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": checkpoint_sha,
            "state": "ema",
            "step": int(checkpoint["step"]),
            "model_config": checkpoint["model_config"],
            "normalizer": checkpoint["normalizer"],
        },
        "native_score": {
            "library_path": str(args.lib.resolve()),
            "library_sha256": library_sha,
            "target": "QH",
            "target_helicity_by_nfp": "(1, nfp)",
            "config_by_nfp": score_configs,
        },
        "code_commit": git_commit(),
        "started_unix_s": time.time(),
    }
    manifest_path = stream_dir / "manifest.json"
    atomic_write_json(manifest_path, manifest)

    rng = np.random.default_rng(stream_seed)
    job_started = time.perf_counter()
    sample_index = 0
    shard_index = 0
    shard_durations: list[float] = []
    status_counts: Counter[str] = Counter()
    while not should_stop(
        elapsed_s=time.perf_counter() - job_started,
        max_wall_s=args.max_wall_s,
        shard_durations=shard_durations,
    ):
        if args.max_shards and shard_index >= args.max_shards:
            break
        shard_started = time.perf_counter()
        condition_indices = prior.sample_indices(rng, args.shard_size)
        conditions = [prior.keys[int(index)] for index in condition_indices]
        latent_rows, raw_rows, decode_wall_s = decode_mixed_conditions(
            model,
            normalizer,
            conditions,
            rng,
            flow_steps=args.flow_steps,
            device=device,
        )
        rows = []
        score_wall_s = 0.0
        shard_status: Counter[str] = Counter()
        for offset, (key, latent, raw) in enumerate(
            zip(conditions, latent_rows, raw_rows, strict=True)
        ):
            result, error, elapsed = score_tokens(
                args.lib, raw, nfp=key[0], device_id=local_rank
            )
            score_wall_s += elapsed
            status = result["status"] if result is not None else "python_error"
            shard_status[status] += 1
            rows.append(
                {
                    "format": FORMAT,
                    "sample_id": f"{stream_id}_{sample_index + offset:012d}",
                    "stream_id": stream_id,
                    "stream_sample_index": sample_index + offset,
                    "shard_index": shard_index,
                    "condition": {
                        "nfp": key[0],
                        "n_base_coils": key[1],
                        "group": group_label(key),
                        "joint_prior_probability": prior_probabilities[key],
                    },
                    "latent": latent.tolist(),
                    "decoded_tokens": raw.tolist(),
                    "decoded_token_schema": "x[33],y[33],z[33],current_A",
                    "native_score": result,
                    "score_error": error,
                    "score_wall_s": elapsed,
                    "provenance": {
                        "stream_manifest": str(manifest_path.resolve()),
                        "flow_checkpoint_sha256": checkpoint_sha,
                        "score_library_sha256": library_sha,
                        "score_config_nfp": key[0],
                    },
                }
            )

        shard_stem = f"{stream_id}_shard_{shard_index:06d}"
        shard_path = args.dataset_root / "shards" / f"{shard_stem}.jsonl.gz"
        row_count, shard_sha = write_jsonl_gzip_atomic(shard_path, rows)
        shard_wall_s = time.perf_counter() - shard_started
        metadata = {
            "format": FORMAT,
            "stream_id": stream_id,
            "shard_index": shard_index,
            "file": shard_path.name,
            "sha256": shard_sha,
            "row_count": row_count,
            "first_stream_sample_index": sample_index,
            "last_stream_sample_index": sample_index + row_count - 1,
            "condition_counts": dict(sorted(Counter(map(group_label, conditions)).items())),
            "status_counts": dict(sorted(shard_status.items())),
            "timing": {
                "decode_wall_s": decode_wall_s,
                "native_score_sum_wall_s": score_wall_s,
                "shard_wall_s": shard_wall_s,
            },
        }
        atomic_write_json(
            args.dataset_root / "shards" / f"{shard_stem}.meta.json", metadata
        )
        sample_index += row_count
        shard_index += 1
        shard_durations.append(shard_wall_s)
        status_counts.update(shard_status)
        progress = {
            "format": FORMAT,
            "stage": "running",
            "stream_id": stream_id,
            "completed_shards": shard_index,
            "completed_samples": sample_index,
            "status_counts": dict(sorted(status_counts.items())),
            "elapsed_s": time.perf_counter() - job_started,
            "last_shard": metadata,
            "updated_unix_s": time.time(),
        }
        replace_json(stream_dir / "progress.json", progress)
        print(json.dumps({"event": "shard_complete", **progress}), flush=True)

    manifest["stage"] = "complete"
    manifest["completed_shards"] = shard_index
    manifest["completed_samples"] = sample_index
    manifest["status_counts"] = dict(sorted(status_counts.items()))
    manifest["elapsed_s"] = time.perf_counter() - job_started
    manifest["finished_unix_s"] = time.time()
    replace_json(manifest_path, manifest)
    replace_json(
        stream_dir / "progress.json",
        {
            "format": FORMAT,
            "stage": "complete",
            "stream_id": stream_id,
            "completed_shards": shard_index,
            "completed_samples": sample_index,
            "status_counts": dict(sorted(status_counts.items())),
            "elapsed_s": time.perf_counter() - job_started,
            "updated_unix_s": time.time(),
        },
    )


if __name__ == "__main__":
    main()
