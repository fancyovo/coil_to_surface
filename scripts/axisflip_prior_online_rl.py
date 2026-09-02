from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_PYTHON = REPO_ROOT / "gpu_backend" / "python"
for search_path in (REPO_ROOT, GPU_PYTHON):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from flow_matching.axis_prior_rl import (  # noqa: E402
    FORMAT,
    NFP,
    N_BASE_COILS,
    PROTOCOL_ID,
    diversity_summary,
    feature_weights,
    file_sha256,
    inverse_tokens,
    load_teacher_dataset,
    online_objective,
    random_permute_coils,
    split_masks,
    transform_tokens,
)
from flow_matching.data import CoilNormalizer  # noqa: E402
from flow_matching.flow import integrate_flow  # noqa: E402
from flow_matching.model import CoilFlowTransformer  # noqa: E402
from flow_matching.optimization import (  # noqa: E402
    CURRENT_NATIVE_SCORE_ABI,
    CURRENT_NATIVE_SCORE_LIBRARY_SHA256,
)
from scripts.flow_runtime import repository_provenance  # noqa: E402
from scripts.native_score_runtime import token_case, write_json  # noqa: E402
from scripts.optimize_flow_latent import score_config  # noqa: E402
from scripts.prepare_axis_surface_prior_adam200 import exact_standardized_start  # noqa: E402


WORKER_COUNT = 4
SAMPLES_PER_WORKER = 16
SAMPLES_PER_ROUND = WORKER_COUNT * SAMPLES_PER_WORKER
REPLAY_CAPACITY = 512
INVALID_WEIGHT = 0.05
FLOW_STEPS = 32


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_clean_repository(expected_commit: str) -> dict[str, Any]:
    provenance = repository_provenance(REPO_ROOT)
    if not provenance["available"] or provenance["tracked_dirty"]:
        raise RuntimeError("experiment requires a clean tracked worktree")
    if provenance["commit"] != expected_commit:
        raise RuntimeError(
            f"repository commit {provenance['commit']} != expected {expected_commit}"
        )
    return provenance


def load_policy_checkpoint(
    path: Path, device: torch.device
) -> tuple[CoilFlowTransformer, CoilNormalizer, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("policy checkpoint format mismatch")
    required = {"model_config", "ema", "normalizer", "step"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"policy checkpoint lacks {sorted(missing)}")
    model = CoilFlowTransformer(**checkpoint["model_config"]).to(device=device)
    model.load_state_dict(checkpoint["ema"])
    model.eval()
    return model, CoilNormalizer.from_dict(checkpoint["normalizer"]), checkpoint


def prepare(args: argparse.Namespace) -> None:
    provenance = require_clean_repository(args.expected_commit)
    if file_sha256(args.score_lib) != CURRENT_NATIVE_SCORE_LIBRARY_SHA256:
        raise ValueError("score library is not the promoted ABI-11 build")
    if file_sha256(args.optimizer_checkpoint) != args.expected_optimizer_checkpoint_sha:
        raise ValueError("optimizer normalizer checkpoint hash mismatch")
    convergence = load_json(args.distillation_dir / "convergence.json")
    if convergence.get("status") != "converged":
        raise ValueError("q0 distillation has not converged")
    q0_path = args.distillation_dir / "checkpoint_q0.pt"
    q0 = torch.load(q0_path, map_location="cpu", weights_only=False)
    if not q0.get("converged") or q0.get("format") != FORMAT:
        raise ValueError("q0 checkpoint is not an accepted converged checkpoint")
    _, _, teacher_manifest = load_teacher_dataset(args.dataset_dir, verify_hashes=True)
    for name in ("rounds", "checkpoints", "audit", "logs"):
        (args.run_root / name).mkdir(parents=True, exist_ok=True)
    checkpoint_zero = args.run_root / "checkpoints" / "round_000.pt"
    shutil.copy2(q0_path, checkpoint_zero)
    manifest = {
        "format": FORMAT,
        "protocol": {
            "id": PROTOCOL_ID,
            "status": "registered-experimental",
            "relationship_to_default": "Analytic-prior online Flow exploration; current QH default is unchanged.",
        },
        "repository": provenance,
        "created_unix_s": time.time(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "condition": {"nfp": NFP, "n_base_coils": N_BASE_COILS},
        "teacher": {
            "dataset": str(args.dataset_dir.resolve()),
            "dataset_repository_commit": teacher_manifest["repository_commit"],
            "sample_count": teacher_manifest["sample_count"],
        },
        "distillation": {
            "directory": str(args.distillation_dir.resolve()),
            "checkpoint": str(q0_path.resolve()),
            "checkpoint_sha256": file_sha256(q0_path),
            "convergence": convergence,
        },
        "evaluator": {
            "abi": CURRENT_NATIVE_SCORE_ABI,
            "library": str(args.score_lib.resolve()),
            "library_sha256": file_sha256(args.score_lib),
            "target_helicity": [1, NFP],
            "configuration": score_config(
                iota_degree=3, surface_theta_count=128, axis_hint=None
            ),
        },
        "adam20": {
            "optimizer_checkpoint": str(args.optimizer_checkpoint.resolve()),
            "optimizer_checkpoint_sha256": file_sha256(args.optimizer_checkpoint),
            "parameter_space": "exact-unclipped standardized data coordinates",
            "iterations": 20,
            "directions": 64,
            "difference": "centered",
            "perturbation": 0.0025,
            "learning_rate": 0.01,
            "beta": [0.7, 0.999],
            "axis_policy": "screen globally, then reuse the selected axis from optimizer step 0",
            "initial_score_tolerance": 0.1,
            "target": "best valid point among steps 0 through 20",
        },
        "round": {
            "workers": WORKER_COUNT,
            "samples_per_worker": SAMPLES_PER_WORKER,
            "samples": SAMPLES_PER_ROUND,
            "flow_steps": FLOW_STEPS,
            "seed": args.sample_seed,
        },
        "update": {
            "current_policy_fraction": 0.85,
            "improved_fraction": 0.10,
            "q0_fraction": 0.05,
            "invalid_weight": INVALID_WEIGHT,
            "valid_weight": "1 plus within-round best-score percentile",
            "replay_capacity": REPLAY_CAPACITY,
            "replay_policy": "FIFO over valid Adam20 best endpoints",
            "train_steps": args.train_steps,
            "batch_per_gpu": args.train_batch_per_gpu,
            "learning_rate": args.learning_rate,
        },
        "parallelism": {
            "collection": "four independent one-GPU workers, 16 samples each",
            "training": "four-GPU DDP",
            "serial_reason": "Flow q(k+1) depends on the complete scored Adam20 batch from q(k)",
        },
    }
    atomic_write_json(args.run_root / "manifest.json", manifest)
    atomic_savez(
        args.run_root / "replay_pool.npz",
        improved=np.empty((0, N_BASE_COILS, 100), dtype=np.float32),
        weights=np.empty(0, dtype=np.float32),
        best_scores=np.empty(0, dtype=np.float64),
        sample_ids=np.empty(0, dtype="U64"),
    )
    atomic_write_json(
        args.run_root / "progress.json",
        {
            "format": FORMAT,
            "stage": "prepared",
            "next_round": 0,
            "updated_unix_s": time.time(),
        },
    )
    print(json.dumps({"event": "online_prepared", "run_root": str(args.run_root)}))


def load_manifest(run_root: Path) -> dict[str, Any]:
    manifest = load_json(run_root / "manifest.json")
    if manifest.get("format") != FORMAT or manifest.get("protocol", {}).get("id") != PROTOCOL_ID:
        raise ValueError("online run manifest has the wrong protocol")
    if manifest.get("condition") != {"nfp": NFP, "n_base_coils": N_BASE_COILS}:
        raise ValueError("online run condition changed")
    if manifest["repository"] != repository_provenance(REPO_ROOT):
        raise RuntimeError("online run repository provenance changed")
    return manifest


@torch.inference_mode()
def generate_policy_batch(
    model: CoilFlowTransformer,
    normalizer: CoilNormalizer,
    *,
    count: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    generator = torch.Generator(device=device).manual_seed(int(seed))
    noise = torch.randn((count, N_BASE_COILS, 100), generator=generator, device=device)
    nfp = torch.full((count,), NFP, dtype=torch.long, device=device)
    started = time.perf_counter()
    normalized = integrate_flow(
        model,
        noise,
        nfp,
        start_time=0.0,
        end_time=1.0,
        steps=FLOW_STEPS,
        method="rk4",
    ).float().cpu().numpy()
    physical = inverse_tokens(normalized, normalizer).astype(np.float64)
    return normalized, physical, time.perf_counter() - started


def native_score(tokens: np.ndarray, *, lib: Path, device: int) -> tuple[dict[str, Any], float]:
    from stellarator_gpu import score_coils_native

    started = time.perf_counter()
    result = score_coils_native(
        str(lib.resolve()),
        tokens[:, :33],
        tokens[:, 33:66],
        tokens[:, 66:99],
        tokens[:, 99],
        NFP,
        device_id=device,
        target_helicity=(1, NFP),
        config_overrides=score_config(
            iota_degree=3, surface_theta_count=128, axis_hint=None
        ),
    )
    abi = int(result.get("diagnostics", {}).get("abi_version", -1))
    if abi != CURRENT_NATIVE_SCORE_ABI:
        raise RuntimeError(f"native evaluator returned ABI {abi}")
    return result, time.perf_counter() - started


def compact_native(result: dict[str, Any]) -> dict[str, Any]:
    components = result.get("components") or {}
    diagnostics = result.get("diagnostics") or {}

    def finite(value: Any) -> float | None:
        try:
            converted = float(value)
        except (TypeError, ValueError):
            return None
        return converted if math.isfinite(converted) else None

    return {
        "status": str(result.get("status", "missing")),
        "score": finite(result.get("score")) or 0.0,
        "components": {
            "volume_qs": finite(components.get("volume_qs")),
            "coil": finite(components.get("coil")),
        },
        "iota_min": finite(diagnostics.get("iota_min")),
        "iota_max": finite(diagnostics.get("iota_max")),
    }


def raw_tokens_from_case(payload: dict[str, Any]) -> np.ndarray:
    raw = payload["raw"]
    return np.column_stack(
        (
            np.asarray(raw["x"], dtype=np.float64),
            np.asarray(raw["y"], dtype=np.float64),
            np.asarray(raw["z"], dtype=np.float64),
            np.asarray(raw["current"], dtype=np.float64),
        )
    )


def run_adam20(
    *,
    start_path: Path,
    optimizer_dir: Path,
    log_path: Path,
    manifest: dict[str, Any],
    seed: int,
    device: int,
) -> tuple[dict[str, Any], float]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "optimize_flow_latent.py"),
        "--checkpoint",
        manifest["adam20"]["optimizer_checkpoint"],
        "--initial-case",
        str(start_path),
        "--lib",
        manifest["evaluator"]["library"],
        "--out-dir",
        str(optimizer_dir),
        "--nfp",
        str(NFP),
        "--n-base-coils",
        str(N_BASE_COILS),
        "--target-helicity-sign",
        "1",
        "--iterations",
        "20",
        "--max-wall-s",
        "900",
        "--parameter-space",
        "data",
        "--data-start-mode",
        "exact-unclipped",
        "--recorded-initial-score-tolerance",
        "0.1",
        "--perturbation",
        "0.0025",
        "--gradient-mode",
        "random-orthogonal",
        "--random-directions",
        "64",
        "--seed",
        str(seed),
        "--optimizer",
        "adam",
        "--learning-rate",
        "0.01",
        "--beta1",
        "0.7",
        "--beta2",
        "0.999",
        "--flow-device",
        str(device),
        "--score-device",
        str(device),
        "--plot-every",
        "0",
        "--progress-every",
        "10",
        "--trajectory-every",
        "0",
        "--state-every",
        "20",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=1200,
            text=True,
        )
    wall = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(f"Adam20 subprocess exited with {completed.returncode}")
    summary = load_json(optimizer_dir / "summary.json")
    if (
        summary.get("status") != "ok"
        or summary.get("stop_reason") != "completed_iterations"
        or int(summary.get("completed_iterations", -1)) != 20
    ):
        raise RuntimeError("Adam20 did not complete all requested updates")
    gate = load_json(optimizer_dir / "manifest.json").get("initial_consistency_gate")
    if (
        not gate
        or float(gate.get("absolute_delta", math.inf))
        > float(gate.get("tolerance", 0.1))
    ):
        raise RuntimeError("Adam20 initial score/axis consistency gate did not pass")
    return summary, wall


def collect_worker(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.run_root)
    if not 0 <= args.worker_index < WORKER_COUNT or args.device != args.worker_index:
        raise ValueError("collector worker/device assignment is invalid")
    if file_sha256(Path(manifest["evaluator"]["library"])) != manifest["evaluator"]["library_sha256"]:
        raise ValueError("score library changed after prepare")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    checkpoint_path = args.run_root / "checkpoints" / f"round_{args.round_index:03d}.pt"
    model, prior_normalizer, checkpoint = load_policy_checkpoint(checkpoint_path, device)
    checkpoint_round = int(checkpoint.get("outer_round", 0))
    if checkpoint_round != args.round_index:
        raise ValueError("policy checkpoint outer round mismatch")
    optimizer_checkpoint = torch.load(
        manifest["adam20"]["optimizer_checkpoint"], map_location="cpu", weights_only=False
    )
    optimizer_normalizer = CoilNormalizer.from_dict(optimizer_checkpoint["normalizer"])
    round_dir = args.run_root / "rounds" / f"round_{args.round_index:03d}"
    worker_dir = round_dir / f"worker_{args.worker_index:02d}"
    if worker_dir.exists():
        raise FileExistsError(worker_dir)
    for name in ("starts", "optimizations", "logs"):
        (worker_dir / name).mkdir(parents=True, exist_ok=True)
    batch_seed = int(manifest["round"]["seed"]) + args.round_index * 100000 + args.worker_index * 1000
    generated_normalized, generated_physical, flow_wall = generate_policy_batch(
        model,
        prior_normalizer,
        count=SAMPLES_PER_WORKER,
        seed=batch_seed,
        device=device,
    )
    current = np.empty_like(generated_normalized)
    improved = np.empty_like(generated_normalized)
    valid = np.zeros(SAMPLES_PER_WORKER, dtype=np.bool_)
    initial_scores = np.zeros(SAMPLES_PER_WORKER, dtype=np.float64)
    best_scores = np.zeros(SAMPLES_PER_WORKER, dtype=np.float64)
    initial_volume_qs = np.full(SAMPLES_PER_WORKER, np.nan, dtype=np.float64)
    initial_coil = np.full(SAMPLES_PER_WORKER, np.nan, dtype=np.float64)
    best_volume_qs = np.full(SAMPLES_PER_WORKER, np.nan, dtype=np.float64)
    best_coil = np.full(SAMPLES_PER_WORKER, np.nan, dtype=np.float64)
    statuses = []
    sample_ids = []
    records = []
    score_wall = 0.0
    adam_wall = 0.0
    for index, physical in enumerate(generated_physical):
        sample_id = f"r{args.round_index:03d}_w{args.worker_index:02d}_i{index:02d}"
        sample_ids.append(sample_id)
        parameters, current_l1, represented, roundtrip = exact_standardized_start(
            physical,
            optimizer_normalizer,
            condition=(NFP, N_BASE_COILS),
        )
        current[index] = transform_tokens(represented[None], prior_normalizer)[0]
        result, one_score_wall = native_score(
            represented, lib=Path(manifest["evaluator"]["library"]), device=args.device
        )
        score_wall += one_score_wall
        compact_initial = compact_native(result)
        status = compact_initial["status"]
        statuses.append(status)
        initial_scores[index] = compact_initial["score"]
        initial_volume_qs[index] = (
            compact_initial["components"]["volume_qs"]
            if compact_initial["components"]["volume_qs"] is not None
            else np.nan
        )
        initial_coil[index] = (
            compact_initial["components"]["coil"]
            if compact_initial["components"]["coil"] is not None
            else np.nan
        )
        improved[index] = current[index]
        best_scores[index] = initial_scores[index]
        best_volume_qs[index] = initial_volume_qs[index]
        best_coil[index] = initial_coil[index]
        adam_record = None
        if status == "ok":
            valid[index] = True
            start = token_case(
                represented,
                nfp=NFP,
                target="QH",
                metadata={"sample_id": sample_id, "online_round": args.round_index},
            )
            start["data_prior_screening"] = {
                "format": FORMAT,
                "protocol_id": PROTOCOL_ID,
                "sample_id": sample_id,
                "normalized_coil_tokens": parameters.tolist(),
                "current_l1_a": current_l1,
                "native_score": result,
                "roundtrip": roundtrip,
            }
            start_path = worker_dir / "starts" / f"{sample_id}.json"
            write_json(start_path, start)
            optimizer_dir = worker_dir / "optimizations" / sample_id
            summary, one_adam_wall = run_adam20(
                start_path=start_path,
                optimizer_dir=optimizer_dir,
                log_path=worker_dir / "logs" / f"{sample_id}.log",
                manifest=manifest,
                seed=batch_seed + index + 1,
                device=args.device,
            )
            adam_wall += one_adam_wall
            best_payload = load_json(optimizer_dir / "best.json")
            best_tokens = raw_tokens_from_case(best_payload)
            improved[index] = transform_tokens(best_tokens[None], prior_normalizer)[0]
            best_native = best_payload["original_space_local_gradient_adam"]["native_score"]
            compact_best = compact_native(best_native)
            if compact_best["status"] != "ok":
                raise RuntimeError("Adam20 best endpoint is not valid")
            best_scores[index] = compact_best["score"]
            best_volume_qs[index] = (
                compact_best["components"]["volume_qs"]
                if compact_best["components"]["volume_qs"] is not None
                else np.nan
            )
            best_coil[index] = (
                compact_best["components"]["coil"]
                if compact_best["components"]["coil"] is not None
                else np.nan
            )
            adam_record = {
                "best_score": best_scores[index],
                "best_iteration": int(summary["best_iteration"]),
                "wall_s": one_adam_wall,
                "optimizer_dir": str(optimizer_dir.relative_to(args.run_root)),
            }
        records.append(
            {
                "sample_id": sample_id,
                "initial": compact_initial,
                "valid": bool(valid[index]),
                "best_score": float(best_scores[index]),
                "best_volume_qs": (
                    float(best_volume_qs[index]) if math.isfinite(best_volume_qs[index]) else None
                ),
                "best_coil": float(best_coil[index]) if math.isfinite(best_coil[index]) else None,
                "adam20": adam_record,
            }
        )
        atomic_write_json(
            worker_dir / "progress.json",
            {
                "format": FORMAT,
                "stage": "collecting",
                "round": args.round_index,
                "worker": args.worker_index,
                "completed": index + 1,
                "valid": int(np.count_nonzero(valid[: index + 1])),
                "updated_unix_s": time.time(),
            },
        )
    data_path = worker_dir / "batch.npz"
    atomic_savez(
        data_path,
        current=current.astype(np.float32),
        improved=improved.astype(np.float32),
        valid=valid,
        initial_scores=initial_scores,
        best_scores=best_scores,
        initial_volume_qs=initial_volume_qs,
        initial_coil=initial_coil,
        best_volume_qs=best_volume_qs,
        best_coil=best_coil,
        statuses=np.asarray(statuses, dtype="U64"),
        sample_ids=np.asarray(sample_ids, dtype="U64"),
    )
    summary = {
        "format": FORMAT,
        "stage": "worker_complete",
        "round": args.round_index,
        "worker": args.worker_index,
        "sample_count": SAMPLES_PER_WORKER,
        "valid_count": int(np.count_nonzero(valid)),
        "status_counts": dict(Counter(statuses)),
        "flow_wall_s": flow_wall,
        "score_wall_s": score_wall,
        "adam20_wall_s": adam_wall,
        "batch_file": str(data_path.relative_to(args.run_root)),
        "records": records,
        "finished_unix_s": time.time(),
    }
    atomic_write_json(worker_dir / "summary.json", summary)
    atomic_write_json(worker_dir / "progress.json", summary)
    print(json.dumps({"event": "worker_complete", **summary}, separators=(",", ":")))


def finite_median(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else None


def finite_percentile(values: np.ndarray, percentile: float) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, percentile)) if finite.size else None


def finite_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    keep = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(keep) < 2:
        return None
    if np.std(left[keep]) == 0.0 or np.std(right[keep]) == 0.0:
        return None
    return float(np.corrcoef(left[keep], right[keep])[0, 1])


def score_rank_weights(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if len(values) == 0:
        return np.empty(0, dtype=np.float32)
    if len(values) == 1:
        return np.asarray([1.5], dtype=np.float32)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        average_rank = 0.5 * (start + stop - 1) / (len(values) - 1)
        ranks[order[start:stop]] = average_rank
        start = stop
    return (1.0 + ranks).astype(np.float32)


def summarize_round(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.run_root)
    round_dir = args.run_root / "rounds" / f"round_{args.round_index:03d}"
    summaries = []
    batches = []
    for worker in range(WORKER_COUNT):
        worker_dir = round_dir / f"worker_{worker:02d}"
        summary = load_json(worker_dir / "summary.json")
        if summary.get("stage") != "worker_complete" or int(summary["round"]) != args.round_index:
            raise ValueError(f"worker {worker} did not complete the requested round")
        with np.load(args.run_root / summary["batch_file"], allow_pickle=False) as data:
            if len(data["current"]) != SAMPLES_PER_WORKER:
                raise ValueError(f"worker {worker} wrote the wrong batch size")
            batch = {name: np.asarray(data[name]) for name in data.files}
        summaries.append(summary)
        batches.append(batch)
    keys = batches[0].keys()
    combined = {name: np.concatenate([batch[name] for batch in batches]) for name in keys}
    if len(set(combined["sample_ids"].tolist())) != SAMPLES_PER_ROUND:
        raise ValueError("round sample IDs are not unique")
    valid = combined["valid"].astype(bool)
    replay_path = args.run_root / "replay_pool.npz"
    with np.load(replay_path, allow_pickle=False) as replay:
        old_replay = {name: np.asarray(replay[name]) for name in replay.files}
    reward_improved = np.concatenate((combined["improved"], old_replay["improved"]))
    reward_scores = np.concatenate((combined["best_scores"], old_replay["best_scores"]))
    reward_valid = np.concatenate(
        (valid, np.ones(len(old_replay["improved"]), dtype=np.bool_))
    )
    reward_weights = np.full(len(reward_improved), INVALID_WEIGHT, dtype=np.float32)
    reward_weights[reward_valid] = score_rank_weights(reward_scores[reward_valid])

    new_replay_improved = np.concatenate((old_replay["improved"], combined["improved"][valid]))
    new_replay_scores = np.concatenate((old_replay["best_scores"], combined["best_scores"][valid]))
    new_replay_ids = np.concatenate((old_replay["sample_ids"], combined["sample_ids"][valid]))
    if len(new_replay_improved) > REPLAY_CAPACITY:
        keep = slice(len(new_replay_improved) - REPLAY_CAPACITY, None)
        new_replay_improved = new_replay_improved[keep]
        new_replay_scores = new_replay_scores[keep]
        new_replay_ids = new_replay_ids[keep]
    new_replay_weights = score_rank_weights(new_replay_scores)
    atomic_savez(
        replay_path,
        improved=new_replay_improved.astype(np.float32),
        weights=new_replay_weights.astype(np.float32),
        best_scores=new_replay_scores.astype(np.float64),
        sample_ids=new_replay_ids.astype("U64"),
    )

    training_path = round_dir / "training_data.npz"
    atomic_savez(
        training_path,
        current=combined["current"].astype(np.float32),
        improved=reward_improved.astype(np.float32),
        improved_weights=reward_weights.astype(np.float32),
        current_valid=valid,
        initial_scores=combined["initial_scores"].astype(np.float64),
        best_scores=combined["best_scores"].astype(np.float64),
        initial_volume_qs=combined["initial_volume_qs"].astype(np.float64),
        initial_coil=combined["initial_coil"].astype(np.float64),
        best_volume_qs=combined["best_volume_qs"].astype(np.float64),
        best_coil=combined["best_coil"].astype(np.float64),
        sample_ids=combined["sample_ids"].astype("U64"),
        statuses=combined["statuses"].astype("U64"),
    )
    initial = combined["initial_scores"]
    best = combined["best_scores"]
    gain = best[valid] - initial[valid]
    status_counts = dict(Counter(combined["statuses"].tolist()))
    summary = {
        "format": FORMAT,
        "stage": "round_collected",
        "round": args.round_index,
        "sample_count": SAMPLES_PER_ROUND,
        "valid_count": int(np.count_nonzero(valid)),
        "valid_rate": float(np.mean(valid)),
        "status_counts": status_counts,
        "initial": {
            "score_median_all": finite_median(initial),
            "score_median_valid": finite_median(initial[valid]),
            "score_p90_all": finite_percentile(initial, 90),
            "volume_qs_median_valid": finite_median(combined["initial_volume_qs"][valid]),
            "coil_median_valid": finite_median(combined["initial_coil"][valid]),
            "volume_qs_coil_correlation_valid": finite_correlation(
                combined["initial_volume_qs"][valid], combined["initial_coil"][valid]
            ),
        },
        "adam20": {
            "best_score_median_valid": finite_median(best[valid]),
            "best_score_p90_valid": finite_percentile(best[valid], 90),
            "gain_median_valid": finite_median(gain),
            "volume_qs_median_valid": finite_median(combined["best_volume_qs"][valid]),
            "coil_median_valid": finite_median(combined["best_coil"][valid]),
            "volume_qs_coil_correlation_valid": finite_correlation(
                combined["best_volume_qs"][valid], combined["best_coil"][valid]
            ),
            "threshold_counts": {
                str(threshold): int(np.count_nonzero(best[valid] >= threshold))
                for threshold in (50, 70, 80)
            },
        },
        "training": {
            "data_file": str(training_path.relative_to(args.run_root)),
            "improvement_count": int(len(reward_improved)),
            "invalid_weight": INVALID_WEIGHT,
            "valid_weight_min": float(np.min(reward_weights[reward_valid])) if np.any(reward_valid) else None,
            "valid_weight_max": float(np.max(reward_weights[reward_valid])) if np.any(reward_valid) else None,
            "replay_size_after_round": int(len(new_replay_improved)),
        },
        "diversity": {
            "current": diversity_summary(combined["current"], seed=1000 + args.round_index),
            "improved_valid": diversity_summary(
                combined["improved"][valid], seed=2000 + args.round_index
            ),
        },
        "runtime": {
            "flow_gpu_s": float(sum(row["flow_wall_s"] for row in summaries)),
            "score_gpu_s": float(sum(row["score_wall_s"] for row in summaries)),
            "adam20_gpu_s": float(sum(row["adam20_wall_s"] for row in summaries)),
        },
        "finished_unix_s": time.time(),
    }
    atomic_write_json(round_dir / "round_summary.json", summary)
    atomic_write_json(
        args.run_root / "progress.json",
        {
            "format": FORMAT,
            "stage": "round_collected",
            "round": args.round_index,
            "next_round": args.round_index,
            "valid_rate": summary["valid_rate"],
            "initial_score_median_all": summary["initial"]["score_median_all"],
            "initial_score_median_valid": summary["initial"]["score_median_valid"],
            "adam20_best_score_median_valid": summary["adam20"]["best_score_median_valid"],
            "replay_size": summary["training"]["replay_size_after_round"],
            "updated_unix_s": time.time(),
        },
    )
    print(json.dumps({"event": "round_collected", **summary}, separators=(",", ":")))


def distributed_setup() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("online Flow update requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl", device_id=device)
    return rank, local_rank, world_size, device


def barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier(device_ids=[torch.cuda.current_device()])


@torch.inference_mode()
def paired_policy_move(
    before: CoilFlowTransformer,
    after: CoilFlowTransformer,
    *,
    device: torch.device,
    seed: int,
) -> float:
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn((64, N_BASE_COILS, 100), generator=generator, device=device)
    nfp = torch.full((64,), NFP, dtype=torch.long, device=device)
    left = integrate_flow(before, noise, nfp, steps=FLOW_STEPS, method="rk4")
    right = integrate_flow(after, noise, nfp, steps=FLOW_STEPS, method="rk4")
    return float(torch.sqrt(torch.mean((right.float() - left.float()).square())).cpu())


def save_online_checkpoint(
    path: Path,
    *,
    model: CoilFlowTransformer,
    ema: CoilFlowTransformer,
    optimizer: torch.optim.Optimizer,
    normalizer: CoilNormalizer,
    checkpoint: dict[str, Any],
    outer_round: int,
    train_steps: int,
    code_commit: str,
) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(
        {
            "format": FORMAT,
            "stage": "online_policy",
            "model_config": model.config,
            "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "ema": {name: value.detach().cpu() for name, value in ema.state_dict().items()},
            "online_optimizer": optimizer.state_dict(),
            "normalizer": normalizer.to_dict(),
            "outer_round": int(outer_round),
            "step": int(checkpoint["step"]) + int(train_steps),
            "distillation_step": int(checkpoint.get("distillation_step", checkpoint["step"])),
            "code_commit": code_commit,
        },
        temporary,
    )
    os.replace(temporary, path)


def train_round(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.run_root)
    rank, local_rank, world_size, device = distributed_setup()
    if world_size != WORKER_COUNT:
        raise ValueError("online update requires four-GPU DDP")
    checkpoint_path = args.run_root / "checkpoints" / f"round_{args.round_index:03d}.pt"
    before_model, normalizer, checkpoint = load_policy_checkpoint(checkpoint_path, device)
    base_model = CoilFlowTransformer(**checkpoint["model_config"]).to(device)
    base_state = (
        checkpoint["model"]
        if checkpoint.get("stage") == "online_policy"
        else checkpoint["ema"]
    )
    base_model.load_state_dict(base_state)
    ema_model = CoilFlowTransformer(**checkpoint["model_config"]).to(device)
    ema_model.load_state_dict(checkpoint["ema"])
    ema_model.eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    initial_state = {
        name: value.detach().cpu().clone() for name, value in base_model.state_dict().items()
    }
    train_model: torch.nn.Module = base_model
    if world_size > 1:
        train_model = torch.nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
    update = manifest["update"]
    optimizer = torch.optim.AdamW(
        train_model.parameters(),
        lr=float(update["learning_rate"]),
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=True,
    )
    if checkpoint.get("stage") == "online_policy":
        optimizer_state = checkpoint.get("online_optimizer")
        if optimizer_state is None:
            raise ValueError("online policy checkpoint lacks optimizer state")
        optimizer.load_state_dict(optimizer_state)
    round_dir = args.run_root / "rounds" / f"round_{args.round_index:03d}"
    data_file = args.run_root / load_json(round_dir / "round_summary.json")["training"]["data_file"]
    data = np.load(data_file, allow_pickle=False)
    current = np.asarray(data["current"], dtype=np.float32)
    improved = np.asarray(data["improved"], dtype=np.float32)
    improved_weights = np.asarray(data["improved_weights"], dtype=np.float32)
    q0_path = Path(manifest["distillation"]["directory"]) / "q0_train_normalized.npy"
    q0 = np.load(q0_path, mmap_mode="r")
    if q0.ndim != 3 or q0.shape[1:] != (N_BASE_COILS, 100) or q0.dtype != np.float32:
        raise ValueError("q0 normalized training cache has the wrong contract")
    weights = feature_weights(normalizer, device)
    rng = np.random.default_rng(2026090400 + args.round_index * 100 + rank)
    torch.manual_seed(2026090500 + args.round_index * 100 + rank)
    permutation_generator = torch.Generator().manual_seed(
        2026090600 + args.round_index * 100 + rank
    )
    metrics = []
    started = time.perf_counter()
    for step in range(1, int(update["train_steps"]) + 1):
        batch_size = int(update["batch_per_gpu"])
        current_batch = torch.from_numpy(
            current[rng.integers(0, len(current), size=batch_size)]
        )
        improved_indices = rng.integers(0, len(improved), size=batch_size)
        improved_batch = torch.from_numpy(improved[improved_indices])
        weight_batch = torch.from_numpy(improved_weights[improved_indices]).to(device=device)
        q0_batch = torch.from_numpy(q0[rng.integers(0, len(q0), size=batch_size)])
        current_batch = random_permute_coils(
            current_batch, generator=permutation_generator
        ).to(device=device)
        improved_batch = random_permute_coils(
            improved_batch, generator=permutation_generator
        ).to(device=device)
        q0_batch = random_permute_coils(
            q0_batch, generator=permutation_generator
        ).to(device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, terms = online_objective(
                train_model,
                current_data=current_batch,
                improved_data=improved_batch,
                improved_weights=weight_batch,
                q0_data=q0_batch,
                feature_weights=weights,
                improvement_fraction=float(update["improved_fraction"]),
                q0_fraction=float(update["q0_fraction"]),
            )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(train_model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            for target, source in zip(
                ema_model.parameters(), base_model.parameters(), strict=True
            ):
                target.lerp_(source.detach(), 0.01)
        if step == 1 or step % 25 == 0 or step == int(update["train_steps"]):
            values = torch.stack(
                [
                    terms["loss"],
                    terms["current_loss"],
                    terms["improvement_loss"],
                    terms["q0_loss"],
                    torch.as_tensor(gradient_norm, dtype=torch.float32, device=device),
                ]
            ).float()
            if world_size > 1:
                dist.all_reduce(values, op=dist.ReduceOp.SUM)
                values /= world_size
            if rank == 0:
                row = {
                    "step": step,
                    "loss": float(values[0].cpu()),
                    "current_loss": float(values[1].cpu()),
                    "improvement_loss": float(values[2].cpu()),
                    "q0_loss": float(values[3].cpu()),
                    "gradient_norm": float(values[4].cpu()),
                }
                metrics.append(row)
                print(json.dumps({"event": "online_train", "round": args.round_index, **row}))
    barrier(world_size)
    if rank == 0:
        delta_square = 0.0
        base_square = 0.0
        for name, value in base_model.state_dict().items():
            current_value = value.detach().cpu().float()
            previous = initial_state[name].float()
            delta_square += float(torch.sum((current_value - previous).square()))
            base_square += float(torch.sum(previous.square()))
        move = paired_policy_move(
            before_model,
            ema_model,
            device=device,
            seed=2026090700 + args.round_index,
        )
        next_path = args.run_root / "checkpoints" / f"round_{args.round_index + 1:03d}.pt"
        save_online_checkpoint(
            next_path,
            model=base_model,
            ema=ema_model,
            optimizer=optimizer,
            normalizer=normalizer,
            checkpoint=checkpoint,
            outer_round=args.round_index + 1,
            train_steps=int(update["train_steps"]),
            code_commit=manifest["repository"]["commit"],
        )
        training_summary = {
            "format": FORMAT,
            "stage": "round_trained",
            "round": args.round_index,
            "world_size": world_size,
            "steps": int(update["train_steps"]),
            "wall_s": time.perf_counter() - started,
            "relative_parameter_update_l2": math.sqrt(delta_square / max(base_square, 1.0e-30)),
            "paired_generated_normalized_rms_move": move,
            "metrics": metrics,
            "next_checkpoint": str(next_path.relative_to(args.run_root)),
            "finished_unix_s": time.time(),
        }
        atomic_write_json(round_dir / "training_summary.json", training_summary)
        collected = load_json(round_dir / "round_summary.json")
        atomic_write_json(
            args.run_root / "progress.json",
            {
                "format": FORMAT,
                "stage": "round_trained",
                "completed_round": args.round_index,
                "next_round": args.round_index + 1,
                "valid_rate": collected["valid_rate"],
                "initial_score_median_all": collected["initial"]["score_median_all"],
                "initial_score_median_valid": collected["initial"]["score_median_valid"],
                "adam20_best_score_median_valid": collected["adam20"]["best_score_median_valid"],
                "policy_rms_move": move,
                "updated_unix_s": time.time(),
            },
        )
    barrier(world_size)
    if world_size > 1:
        dist.destroy_process_group()


def audit_worker(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.run_root)
    if not 0 <= args.worker_index < WORKER_COUNT or args.device != args.worker_index:
        raise ValueError("audit worker/device assignment is invalid")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    q0_path = args.run_root / "checkpoints" / "round_000.pt"
    model, prior_normalizer, _ = load_policy_checkpoint(q0_path, device)
    optimizer_checkpoint = torch.load(
        manifest["adam20"]["optimizer_checkpoint"], map_location="cpu", weights_only=False
    )
    optimizer_normalizer = CoilNormalizer.from_dict(optimizer_checkpoint["normalizer"])
    teacher, seeds, teacher_manifest = load_teacher_dataset(
        Path(manifest["teacher"]["dataset"]), verify_hashes=False
    )
    masks = split_masks(seeds, int(teacher_manifest["seed_start"]))
    test = teacher[np.flatnonzero(masks["test"])]
    count = args.samples_per_source
    start = args.worker_index * count
    direct_physical = np.asarray(test[start : start + count], dtype=np.float64)
    generated_normalized, generated_physical, flow_wall = generate_policy_batch(
        model,
        prior_normalizer,
        count=count,
        seed=args.seed + args.worker_index,
        device=device,
    )
    source_names = []
    statuses = []
    scores = []
    volume_qs = []
    coil = []
    represented_prior = []
    score_wall = 0.0
    for source, values in (("teacher", direct_physical), ("flow", generated_physical)):
        for physical in values:
            _, _, represented, _ = exact_standardized_start(
                physical,
                optimizer_normalizer,
                condition=(NFP, N_BASE_COILS),
            )
            result, elapsed = native_score(
                represented,
                lib=Path(manifest["evaluator"]["library"]),
                device=args.device,
            )
            score_wall += elapsed
            compact = compact_native(result)
            source_names.append(source)
            statuses.append(compact["status"])
            scores.append(compact["score"])
            volume_qs.append(compact["components"]["volume_qs"])
            coil.append(compact["components"]["coil"])
            represented_prior.append(
                transform_tokens(represented[None], prior_normalizer)[0]
            )
    audit_dir = args.run_root / "audit" / "q0"
    audit_dir.mkdir(parents=True, exist_ok=True)
    data_path = audit_dir / f"worker_{args.worker_index:02d}.npz"
    atomic_savez(
        data_path,
        source=np.asarray(source_names, dtype="U16"),
        statuses=np.asarray(statuses, dtype="U64"),
        scores=np.asarray(scores, dtype=np.float64),
        volume_qs=np.asarray(volume_qs, dtype=np.float64),
        coil=np.asarray(coil, dtype=np.float64),
        normalized=np.asarray(represented_prior, dtype=np.float32),
    )
    summary = {
        "format": FORMAT,
        "stage": "q0_audit_worker_complete",
        "worker": args.worker_index,
        "samples_per_source": count,
        "flow_wall_s": flow_wall,
        "score_wall_s": score_wall,
        "data_file": str(data_path.relative_to(args.run_root)),
        "finished_unix_s": time.time(),
    }
    atomic_write_json(audit_dir / f"worker_{args.worker_index:02d}.json", summary)
    print(json.dumps({"event": "q0_audit_worker_complete", **summary}))


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return center - radius, center + radius


def summarize_audit(args: argparse.Namespace) -> None:
    load_manifest(args.run_root)
    audit_dir = args.run_root / "audit" / "q0"
    parts = []
    for worker in range(WORKER_COUNT):
        summary = load_json(audit_dir / f"worker_{worker:02d}.json")
        if summary.get("stage") != "q0_audit_worker_complete":
            raise ValueError(f"q0 audit worker {worker} is incomplete")
        with np.load(args.run_root / summary["data_file"], allow_pickle=False) as values:
            parts.append({name: np.asarray(values[name]) for name in values.files})
    combined = {
        name: np.concatenate([part[name] for part in parts]) for name in parts[0]
    }
    by_source = {}
    for source in ("teacher", "flow"):
        select = combined["source"] == source
        status = combined["statuses"][select]
        valid = status == "ok"
        lower, upper = wilson_interval(int(np.count_nonzero(valid)), int(np.count_nonzero(select)))
        by_source[source] = {
            "count": int(np.count_nonzero(select)),
            "valid_count": int(np.count_nonzero(valid)),
            "valid_rate": float(np.mean(valid)),
            "valid_rate_wilson95": [lower, upper],
            "status_counts": dict(Counter(status.tolist())),
            "score_median_all": finite_median(combined["scores"][select]),
            "score_median_valid": finite_median(combined["scores"][select][valid]),
            "volume_qs_median_valid": finite_median(combined["volume_qs"][select][valid]),
            "coil_median_valid": finite_median(combined["coil"][select][valid]),
            "diversity": diversity_summary(combined["normalized"][select], seed=3000),
        }
    teacher_interval = by_source["teacher"]["valid_rate_wilson95"]
    flow_interval = by_source["flow"]["valid_rate_wilson95"]
    intervals_overlap = max(teacher_interval[0], flow_interval[0]) <= min(
        teacher_interval[1], flow_interval[1]
    )
    flow_not_worse = by_source["flow"]["valid_rate"] >= by_source["teacher"]["valid_rate"]
    teacher_rank = by_source["teacher"]["diversity"]["descriptor_effective_rank"]
    flow_rank = by_source["flow"]["diversity"]["descriptor_effective_rank"]
    diversity_ratio = flow_rank / teacher_rank if teacher_rank > 0.0 else 1.0
    passed = bool((intervals_overlap or flow_not_worse) and diversity_ratio >= 0.5)
    summary = {
        "format": FORMAT,
        "stage": "q0_audit_complete" if passed else "q0_audit_failed",
        "passed": passed,
        "by_source": by_source,
        "gate": {
            "valid_rate_intervals_overlap_or_flow_is_better": bool(
                intervals_overlap or flow_not_worse
            ),
            "descriptor_effective_rank_ratio": diversity_ratio,
            "minimum_rank_ratio": 0.5,
        },
        "finished_unix_s": time.time(),
    }
    atomic_write_json(audit_dir / "summary.json", summary)
    if not passed:
        raise RuntimeError("distilled q0 failed the teacher-vs-Flow audit")
    atomic_write_json(
        args.run_root / "progress.json",
        {
            "format": FORMAT,
            "stage": "q0_audit_complete",
            "next_round": 0,
            "q0_teacher_valid_rate": by_source["teacher"]["valid_rate"],
            "q0_flow_valid_rate": by_source["flow"]["valid_rate"],
            "updated_unix_s": time.time(),
        },
    )
    print(json.dumps({"event": "q0_audit_complete", **summary}, separators=(",", ":")))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Fixed-condition analytic-prior online Adam20 Flow matching.")
    commands = value.add_subparsers(dest="command", required=True)
    prepare_command = commands.add_parser("prepare")
    prepare_command.add_argument("--run-root", type=Path, required=True)
    prepare_command.add_argument("--distillation-dir", type=Path, required=True)
    prepare_command.add_argument("--dataset-dir", type=Path, required=True)
    prepare_command.add_argument("--score-lib", type=Path, required=True)
    prepare_command.add_argument("--optimizer-checkpoint", type=Path, required=True)
    prepare_command.add_argument("--expected-optimizer-checkpoint-sha", required=True)
    prepare_command.add_argument("--expected-commit", required=True)
    prepare_command.add_argument("--sample-seed", type=int, default=2026090302)
    prepare_command.add_argument("--train-steps", type=int, default=250)
    prepare_command.add_argument("--train-batch-per-gpu", type=int, default=256)
    prepare_command.add_argument("--learning-rate", type=float, default=5.0e-5)
    prepare_command.set_defaults(func=prepare)

    collect = commands.add_parser("collect-worker")
    collect.add_argument("--run-root", type=Path, required=True)
    collect.add_argument("--round-index", type=int, required=True)
    collect.add_argument("--worker-index", type=int, required=True)
    collect.add_argument("--device", type=int, required=True)
    collect.set_defaults(func=collect_worker)

    summarize = commands.add_parser("summarize-round")
    summarize.add_argument("--run-root", type=Path, required=True)
    summarize.add_argument("--round-index", type=int, required=True)
    summarize.set_defaults(func=summarize_round)

    train = commands.add_parser("train-round")
    train.add_argument("--run-root", type=Path, required=True)
    train.add_argument("--round-index", type=int, required=True)
    train.set_defaults(func=train_round)

    audit = commands.add_parser("audit-worker")
    audit.add_argument("--run-root", type=Path, required=True)
    audit.add_argument("--worker-index", type=int, required=True)
    audit.add_argument("--device", type=int, required=True)
    audit.add_argument("--samples-per-source", type=int, default=32)
    audit.add_argument("--seed", type=int, default=2026090303)
    audit.set_defaults(func=audit_worker)

    audit_summary = commands.add_parser("summarize-audit")
    audit_summary.add_argument("--run-root", type=Path, required=True)
    audit_summary.set_defaults(func=summarize_audit)
    return value


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
