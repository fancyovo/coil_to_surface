from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flow_matching.data import CoilNormalizer, load_raw_groups
from flow_matching.flow import integrate_flow
from flow_matching.model import CoilFlowTransformer


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def setup() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.cuda.is_available():
        raise RuntimeError("latent inversion requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl", device_id=device)
    return rank, local_rank, world_size, device


def barrier(world_size: int, local_rank: int) -> None:
    if world_size > 1:
        dist.barrier(device_ids=[local_rank])


def curve_position_rms(delta_tokens: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta_tokens, dtype=np.float64)[..., :99].reshape(
        *delta_tokens.shape[:-1], 3, 33
    )
    squared = delta[..., 0] ** 2 + 0.5 * np.sum(delta[..., 1:] ** 2, axis=-1)
    return np.sqrt(np.mean(np.sum(squared, axis=-1), axis=-1))


@torch.inference_mode()
def invert_group(
    model: CoilFlowTransformer,
    normalizer: CoilNormalizer,
    raw_tokens: np.ndarray,
    ids: np.ndarray,
    *,
    key: tuple[int, int],
    split: str,
    rank: int,
    world_size: int,
    steps: int,
    batch_size: int,
    closure_count: int,
    device: torch.device,
    output_dir: Path,
) -> tuple[dict[str, Any] | None, list[dict[str, float | int | str]]]:
    selected = np.arange(len(ids), dtype=np.int64)[rank::world_size]
    if not len(selected):
        return None, []
    normalized, clipped_fraction = normalizer.transform(raw_tokens[selected], key)
    latent_parts = []
    started = time.perf_counter()
    for start in range(0, len(selected), batch_size):
        stop = min(start + batch_size, len(selected))
        state = torch.from_numpy(normalized[start:stop]).to(device=device)
        nfp = torch.full((stop - start,), key[0], dtype=torch.long, device=device)
        latent = integrate_flow(
            model,
            state,
            nfp,
            start_time=1.0,
            end_time=0.0,
            steps=steps,
            method="rk4",
        )
        latent_parts.append(latent.cpu())
    torch.cuda.synchronize(device)
    inversion_s = time.perf_counter() - started
    latents = torch.cat(latent_parts, dim=0).contiguous()

    closure_rows: list[dict[str, float | int | str]] = []
    checked = min(int(closure_count), len(selected))
    if checked:
        latent = latents[:checked].to(device=device)
        nfp = torch.full((checked,), key[0], dtype=torch.long, device=device)
        closure_started = time.perf_counter()
        reconstructed = integrate_flow(
            model,
            latent,
            nfp,
            start_time=0.0,
            end_time=1.0,
            steps=steps,
            method="rk4",
        )
        torch.cuda.synchronize(device)
        reconstructed_np = reconstructed.cpu().numpy()
        normalized_delta = reconstructed_np - normalized[:checked]
        original_raw = normalizer.inverse(normalized[:checked], key)
        reconstructed_raw = normalizer.inverse(reconstructed_np, key)
        raw_rms = curve_position_rms(reconstructed_raw - original_raw)
        normalized_rms = np.sqrt(np.mean(normalized_delta.astype(np.float64) ** 2, axis=(1, 2)))
        for index in range(checked):
            closure_rows.append(
                {
                    "split": split,
                    "nfp": key[0],
                    "n_coils": key[1],
                    "rank": rank,
                    "id": int(ids[selected[index]]),
                    "normalized_rms": float(normalized_rms[index]),
                    "curve_position_rms_m": float(raw_rms[index]),
                    "batch_wall_s": float(time.perf_counter() - closure_started),
                }
            )

    filename = f"{split}_nfp{key[0]:02d}_nc{key[1]:02d}_rank{rank:02d}.pt"
    path = output_dir / filename
    torch.save(
        {
            "format": "qh_flow_latents_v1",
            "split": split,
            "key": key,
            "rank": rank,
            "ids": torch.from_numpy(np.asarray(ids[selected], dtype=np.int32)),
            "latents": latents,
        },
        path,
    )
    row = {
        "file": filename,
        "split": split,
        "nfp": key[0],
        "n_coils": key[1],
        "rank": rank,
        "count": len(selected),
        "clip_fraction": float(clipped_fraction),
        "inversion_wall_s": float(inversion_s),
        "samples_per_s": float(len(selected) / max(inversion_s, 1.0e-9)),
    }
    print(json.dumps({"event": "inverted_shard", **row}), flush=True)
    return row, closure_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Invert QUASR QH samples into flow latent space.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--closure-count", type=int, default=8)
    parser.add_argument("--limit-per-group", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.closure_count < 0:
        raise ValueError("steps and batch size must be positive; closure count must be nonnegative")
    if args.limit_per_group is not None and args.limit_per_group < 1:
        raise ValueError("limit per group must be positive")
    process_started = time.perf_counter()
    rank, local_rank, world_size, device = setup()
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    barrier(world_size, local_rank)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = CoilFlowTransformer(**checkpoint["model_config"]).to(device=device, dtype=torch.float32)
    model.load_state_dict(checkpoint["ema"])
    model.eval()
    normalizer = CoilNormalizer.from_dict(checkpoint["normalizer"])

    local_shards = []
    local_closure = []
    for split in ("train", "validation", "test"):
        groups, _ = load_raw_groups(args.data_dir, split)
        for key in sorted(groups):
            count = args.limit_per_group
            shard, closure = invert_group(
                model,
                normalizer,
                groups[key].tokens[:count],
                groups[key].ids[:count],
                key=key,
                split=split,
                rank=rank,
                world_size=world_size,
                steps=args.steps,
                batch_size=args.batch_size,
                closure_count=args.closure_count,
                device=device,
                output_dir=args.output_dir,
            )
            if shard is not None:
                local_shards.append(shard)
            local_closure.extend(closure)

    local_result = {
        "rank": rank,
        "shards": local_shards,
        "closure": local_closure,
        "wall_s": time.perf_counter() - process_started,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    if world_size > 1:
        gathered: list[dict[str, Any]] | None = [None] * world_size if rank == 0 else None
        dist.gather_object(local_result, gathered, dst=0)
    else:
        gathered = [local_result]
    barrier(world_size, local_rank)

    if rank == 0:
        assert gathered is not None
        shards = sorted(
            [row for result in gathered for row in result["shards"]],
            key=lambda row: row["file"],
        )
        for row in shards:
            row["sha256"] = file_sha256(args.output_dir / row["file"])
        closure = [row for result in gathered for row in result["closure"]]
        normalized = np.asarray([row["normalized_rms"] for row in closure], dtype=float)
        physical = np.asarray([row["curve_position_rms_m"] for row in closure], dtype=float)
        split_counts: dict[str, int] = {}
        group_counts: dict[str, int] = {}
        for row in shards:
            split_counts[row["split"]] = split_counts.get(row["split"], 0) + row["count"]
            group = f"nfp{row['nfp']}_nc{row['n_coils']}"
            group_counts[group] = group_counts.get(group, 0) + row["count"]
        manifest = {
            "format": "qh_flow_latents_v1",
            "args": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
            "world_size": world_size,
            "checkpoint_step": int(checkpoint["step"]),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "source_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
            "model_config": checkpoint["model_config"],
            "split_counts": split_counts,
            "group_counts": dict(sorted(group_counts.items())),
            "shards": shards,
            "closure": {
                "count": len(closure),
                "normalized_rms_mean": float(np.mean(normalized)),
                "normalized_rms_max": float(np.max(normalized)),
                "curve_position_rms_m_mean": float(np.mean(physical)),
                "curve_position_rms_m_max": float(np.max(physical)),
                "rows": closure,
            },
            "runtime": {
                "rank": [
                    {
                        "rank": result["rank"],
                        "wall_s": result["wall_s"],
                        "peak_gpu_bytes": result["peak_gpu_bytes"],
                    }
                    for result in gathered
                ],
                "wall_s": max(result["wall_s"] for result in gathered),
            },
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"event": "complete", **manifest["runtime"]}), flush=True)
    barrier(world_size, local_rank)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
