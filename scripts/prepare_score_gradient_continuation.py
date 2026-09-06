"""Freeze a complete online state for a single-change score-weighted continuation."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import shutil
import time

import torch

from scripts import score_gradient_flow_rl as rl


def prepare(args: argparse.Namespace) -> None:
    provenance = rl.require_clean_repository(args.expected_commit)
    protocol = rl.load_json(args.protocol)
    source = Path(protocol["source_run"])
    start = protocol["start_round"]
    for relative, expected in protocol["source_sha256"].items():
        if rl.file_sha256(source / relative) != expected:
            raise ValueError(f"frozen source hash changed: {relative}")
    manifest = rl.load_json(source / "manifest.json")
    if manifest["protocol"]["id"] != protocol["source_protocol_id"]:
        raise ValueError("source protocol mismatch")
    if manifest["repository"]["commit"] != protocol["source_commit"]:
        raise ValueError("source code mismatch")
    progress = rl.load_json(source / "progress.json")
    replay_metadata = rl.load_json(source / "replay_pool.json")
    if (progress["stage"] != "round_trained" or progress["completed_round"] != start - 1
            or progress["next_round"] != start or replay_metadata["source_round"] != start - 1):
        raise ValueError("checkpoint and replay are not from one completed round")
    checkpoint_relative = f"checkpoints/round_{start:04d}.pt"
    checkpoint = torch.load(source / checkpoint_relative, map_location="cpu", weights_only=False)
    required = {"model", "ema", "normalizer", "model_config", "online_optimizer"}
    if (not required.issubset(checkpoint) or checkpoint.get("outer_round") != start
            or checkpoint.get("stage") != "score_gradient_online_policy"
            or checkpoint.get("code_commit") != protocol["source_commit"]):
        raise ValueError("continuation requires a complete online checkpoint")
    groups = checkpoint["online_optimizer"]["param_groups"]
    for group in groups:
        if (group["lr"] != 5e-5 or tuple(group["betas"]) != (0.9, 0.95)
                or group["weight_decay"] != 0.01):
            raise ValueError("source AdamW settings changed")
    if not checkpoint["online_optimizer"]["state"]:
        raise ValueError("source optimizer has no accumulated state")
    pool = rl.load_replay_pool(source / "replay_pool.npz")
    if pool is None or len(pool["current"]) != rl.REPLAY_CAPACITY:
        raise ValueError("source replay pool is incomplete")
    strategy = manifest["strategy"]
    for key, expected in {
        "flow_optimizer_steps_per_round": rl.FLOW_OPTIMIZER_STEPS_PER_ROUND,
        "ema_lerp": rl.EMA_LERP, "beta": protocol["beta"],
        "gradient_directions": 64, "monte_carlo_samples": 4, "invalid_alpha": 0.05,
        "samples_per_round": 64, "world_size": 2,
    }.items():
        if strategy[key] != expected:
            raise ValueError(f"source setting mismatch: {key}")
    rl.validate_r04_score(Path(manifest["evaluator"]["library"]),
                          Path(manifest["evaluator"]["manifest"]), rl.R04_SCORE_SHA)
    for path, expected in [
        (manifest["evaluator"]["manifest"], manifest["evaluator"]["manifest_sha256"]),
        (manifest["paths"]["optimizer_checkpoint"], manifest["paths"]["optimizer_checkpoint_sha256"]),
    ]:
        if rl.file_sha256(Path(path)) != expected:
            raise ValueError("frozen scorer or coordinate normalizer changed")
    if args.run_root.exists():
        raise FileExistsError(args.run_root)
    for directory in ("checkpoints", "rounds", "logs", "source_snapshot"):
        (args.run_root / directory).mkdir(parents=True, exist_ok=False)
    for relative in (checkpoint_relative, "replay_pool.npz", "replay_pool.json"):
        shutil.copyfile(source / relative, args.run_root / relative)
    for relative in ("manifest.json", "progress.json", "replay_pool.json"):
        shutil.copyfile(source / relative, args.run_root / "source_snapshot" / relative)
    # Verify copied bytes as well as source bytes before publishing a new manifest.
    for relative, expected in protocol["source_sha256"].items():
        target = args.run_root / ("source_snapshot/manifest.json" if relative == "manifest.json" else relative)
        if rl.file_sha256(target) != expected or rl.file_sha256(source / relative) != expected:
            raise ValueError(f"source changed while freezing: {relative}")
    updated = copy.deepcopy(manifest)
    updated.update(format=protocol["format"], repository=provenance, created_unix_s=time.time())
    updated["protocol"] = {"id": protocol["protocol_id"], "status": "registered-experimental",
                           "relationship_to_default": "single-loss-change continuation; no default promotion"}
    updated["strategy"]["valid_score_weighting"] = protocol["valid_score_weighting"]
    updated["strategy"]["loss"] = protocol["loss"]
    updated["continuation"] = {"start_round": start, "source_run": str(source),
        "source_sha256": protocol["source_sha256"], "inherited": protocol["inherit"],
        "strategy_changed_keys": ["loss", "valid_score_weighting"],
        "optimizer_state_entries": len(checkpoint["online_optimizer"]["state"]),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    differences = {key for key in set(strategy) | set(updated["strategy"])
                   if strategy.get(key) != updated["strategy"].get(key)}
    if differences != {"loss", "valid_score_weighting"}:
        raise ValueError(f"unexpected strategy change: {differences}")
    shutil.copyfile(args.protocol, args.run_root / "protocol.json")
    rl.atomic_write_json(args.run_root / "manifest.json", updated)
    rl.atomic_write_json(args.run_root / "progress.json", {
        "format": protocol["format"], "stage": "prepared_continuation", "next_round": start,
        "updated_unix_s": time.time()})
    print(f"continuation_prepared start_round={start} replay_size={len(pool['current'])}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    prepare(parser.parse_args())
