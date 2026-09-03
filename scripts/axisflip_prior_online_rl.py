from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
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
    per_sample_flow_terms,
    random_permute_coils,
    split_masks,
    transform_tokens,
)
from flow_matching.data import CoilNormalizer  # noqa: E402
from flow_matching.flow import integrate_flow  # noqa: E402
from flow_matching.model import CoilFlowTransformer  # noqa: E402
from flow_matching.optimization import (  # noqa: E402
    CURRENT_NATIVE_SCORE_ABI,
)
from scripts.flow_runtime import repository_provenance  # noqa: E402
from scripts.native_score_runtime import token_case, write_json  # noqa: E402
from scripts.optimize_flow_latent import score_config  # noqa: E402
from scripts.prepare_axis_surface_prior_adam200 import exact_standardized_start  # noqa: E402


WORKER_COUNT = 4
SAMPLES_PER_WORKER = 16
SAMPLES_PER_ROUND = WORKER_COUNT * SAMPLES_PER_WORKER
REPLAY_CAPACITY = 512
FLOW_STEPS = 32
REWARD_EPSILON = 0.01
REWARD_TEMPERATURE = 7.5
MAX_TRAJECTORY_POINTS = 21
R04_CURVATURE_P95_SCALE_M_INV = 25.0
R04_CURVATURE_MAX_SCALE_M_INV = 35.0


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
    if args.reward_temperature <= 0.0 or args.reward_epsilon <= 0.0:
        raise ValueError("reward temperature and epsilon must be positive")
    score_sha = file_sha256(args.score_lib)
    if score_sha != args.expected_score_lib_sha:
        raise ValueError("score library SHA-256 differs from the frozen R04 build")
    score_manifest = load_json(args.score_library_manifest)
    if (
        score_manifest.get("interface_abi") != CURRENT_NATIVE_SCORE_ABI
        or score_manifest.get("sha256") != score_sha
        or score_manifest.get("cmake_overrides", {}).get(
            "SGPU_COIL_CURVATURE_P95_SCALE"
        )
        != R04_CURVATURE_P95_SCALE_M_INV
        or score_manifest.get("coil_curvature_p95_radius_m") != 0.04
        or score_manifest.get("coil_curvature_max_scale_m_inv")
        != R04_CURVATURE_MAX_SCALE_M_INV
    ):
        raise ValueError("score library manifest does not describe ABI-11 R04")
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
            "generator": teacher_manifest["generator"],
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
            "score_library_manifest": str(args.score_library_manifest.resolve()),
            "coil_curvature_p95_scale_m_inv": R04_CURVATURE_P95_SCALE_M_INV,
            "coil_curvature_p95_radius_m": 0.04,
            "coil_curvature_max_scale_m_inv": R04_CURVATURE_MAX_SCALE_M_INV,
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
            "replay_target": "all formal centers from step 0 through step 20",
            "gradient_endpoints_enter_replay": False,
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
            "reward_temperature": args.reward_temperature,
            "reward_epsilon": args.reward_epsilon,
            "weight_rule": "(epsilon + exp((clipped_score - pool_max_score) / tau)) / rollout_length",
            "replay_capacity_rollouts": REPLAY_CAPACITY,
            "replay_policy": "FIFO over whole rollouts; valid rollouts have 21 centers and invalid rollouts retain step 0",
            "train_steps": args.train_steps,
            "base_batch_per_gpu": args.train_batch_per_gpu,
            "trajectory_batch_rule": "base batch multiplied by ceiling(mean replay points per rollout), capped at 21x",
            "trajectory_microbatch_per_gpu": args.train_batch_per_gpu,
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
        targets=np.empty((0, N_BASE_COILS, 100), dtype=np.float32),
        scores=np.empty(0, dtype=np.float64),
        volume_qs=np.empty(0, dtype=np.float64),
        coil=np.empty(0, dtype=np.float64),
        rollout_ids=np.empty(0, dtype="U64"),
        steps=np.empty(0, dtype=np.int16),
        rollout_lengths=np.empty(0, dtype=np.int16),
        rollout_valid=np.empty(0, dtype=np.bool_),
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


def load_adam20_centers(
    optimizer_dir: Path, prior_normalizer: CoilNormalizer
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    paths = sorted((optimizer_dir / "trajectory").glob("step_*.json"))
    if len(paths) != MAX_TRAJECTORY_POINTS:
        raise RuntimeError(
            f"Adam20 center trajectory has {len(paths)} points, expected {MAX_TRAJECTORY_POINTS}"
        )
    targets = []
    scores = []
    volume_qs = []
    coil = []
    steps = []
    for expected_step, path in enumerate(paths):
        payload = load_json(path)
        metadata = payload.get("original_space_local_gradient_adam", {})
        if int(metadata.get("iteration", -1)) != expected_step:
            raise RuntimeError("Adam20 center trajectory step sequence is not contiguous")
        compact = compact_native(metadata.get("native_score", {}))
        if compact["status"] != "ok":
            raise RuntimeError("Adam20 center trajectory contains an invalid formal center")
        physical = raw_tokens_from_case(payload)
        targets.append(transform_tokens(physical[None], prior_normalizer)[0])
        scores.append(compact["score"])
        volume_qs.append(
            compact["components"]["volume_qs"]
            if compact["components"]["volume_qs"] is not None
            else math.nan
        )
        coil.append(
            compact["components"]["coil"]
            if compact["components"]["coil"] is not None
            else math.nan
        )
        steps.append(expected_step)
    return (
        np.asarray(targets, dtype=np.float32),
        np.asarray(scores, dtype=np.float64),
        np.asarray(volume_qs, dtype=np.float64),
        np.asarray(coil, dtype=np.float64),
        np.asarray(steps, dtype=np.int16),
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
        "1",
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
    trajectory_targets = []
    trajectory_scores = []
    trajectory_volume_qs = []
    trajectory_coil = []
    trajectory_ids = []
    trajectory_steps = []
    trajectory_lengths = []
    trajectory_valid = []
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
            centers = load_adam20_centers(optimizer_dir, prior_normalizer)
            center_count = len(centers[0])
            trajectory_targets.append(centers[0])
            trajectory_scores.append(centers[1])
            trajectory_volume_qs.append(centers[2])
            trajectory_coil.append(centers[3])
            trajectory_ids.append(np.full(center_count, sample_id, dtype="U64"))
            trajectory_steps.append(centers[4])
            trajectory_lengths.append(
                np.full(center_count, center_count, dtype=np.int16)
            )
            trajectory_valid.append(np.ones(center_count, dtype=np.bool_))
        else:
            trajectory_targets.append(current[index : index + 1].astype(np.float32))
            trajectory_scores.append(initial_scores[index : index + 1].astype(np.float64))
            trajectory_volume_qs.append(
                initial_volume_qs[index : index + 1].astype(np.float64)
            )
            trajectory_coil.append(initial_coil[index : index + 1].astype(np.float64))
            trajectory_ids.append(np.asarray([sample_id], dtype="U64"))
            trajectory_steps.append(np.asarray([0], dtype=np.int16))
            trajectory_lengths.append(np.asarray([1], dtype=np.int16))
            trajectory_valid.append(np.asarray([False], dtype=np.bool_))
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
        trajectory_targets=np.concatenate(trajectory_targets).astype(np.float32),
        trajectory_scores=np.concatenate(trajectory_scores).astype(np.float64),
        trajectory_volume_qs=np.concatenate(trajectory_volume_qs).astype(np.float64),
        trajectory_coil=np.concatenate(trajectory_coil).astype(np.float64),
        trajectory_ids=np.concatenate(trajectory_ids).astype("U64"),
        trajectory_steps=np.concatenate(trajectory_steps).astype(np.int16),
        trajectory_lengths=np.concatenate(trajectory_lengths).astype(np.int16),
        trajectory_valid=np.concatenate(trajectory_valid).astype(np.bool_),
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


def trajectory_reward_weights(
    scores: np.ndarray,
    rollout_lengths: np.ndarray,
    *,
    temperature: float = REWARD_TEMPERATURE,
    epsilon: float = REWARD_EPSILON,
) -> tuple[np.ndarray, float]:
    values = np.nan_to_num(
        np.asarray(scores, dtype=np.float64), nan=0.0, posinf=100.0, neginf=0.0
    )
    lengths = np.asarray(rollout_lengths, dtype=np.float64)
    if values.ndim != 1 or lengths.shape != values.shape or len(values) == 0:
        raise ValueError("trajectory scores and rollout lengths must be nonempty vectors")
    if temperature <= 0.0 or epsilon <= 0.0 or np.any(lengths < 1.0):
        raise ValueError("reward temperature, epsilon, and rollout lengths must be positive")
    clipped = np.clip(values, 0.0, 100.0)
    reference = float(np.max(clipped))
    weights = (epsilon + np.exp((clipped - reference) / temperature)) / lengths
    return weights.astype(np.float32), reference


def ordered_unique(values: np.ndarray) -> list[str]:
    return list(dict.fromkeys(np.asarray(values).astype(str).tolist()))


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
    current_trajectory = {
        "targets": combined["trajectory_targets"],
        "scores": combined["trajectory_scores"],
        "volume_qs": combined["trajectory_volume_qs"],
        "coil": combined["trajectory_coil"],
        "rollout_ids": combined["trajectory_ids"],
        "steps": combined["trajectory_steps"],
        "rollout_lengths": combined["trajectory_lengths"],
        "rollout_valid": combined["trajectory_valid"],
    }
    reward = {
        name: np.concatenate((current_trajectory[name], old_replay[name]))
        for name in current_trajectory
    }
    update = manifest["update"]
    reward_weights, reward_score_reference = trajectory_reward_weights(
        reward["scores"],
        reward["rollout_lengths"],
        temperature=float(update["reward_temperature"]),
        epsilon=float(update["reward_epsilon"]),
    )

    replay_combined = {
        name: np.concatenate((old_replay[name], current_trajectory[name]))
        for name in current_trajectory
    }
    replay_rollout_order = ordered_unique(replay_combined["rollout_ids"])
    retained_rollouts = replay_rollout_order[-REPLAY_CAPACITY:]
    keep = np.isin(replay_combined["rollout_ids"], retained_rollouts)
    new_replay = {name: values[keep] for name, values in replay_combined.items()}
    atomic_savez(
        replay_path,
        targets=new_replay["targets"].astype(np.float32),
        scores=new_replay["scores"].astype(np.float64),
        volume_qs=new_replay["volume_qs"].astype(np.float64),
        coil=new_replay["coil"].astype(np.float64),
        rollout_ids=new_replay["rollout_ids"].astype("U64"),
        steps=new_replay["steps"].astype(np.int16),
        rollout_lengths=new_replay["rollout_lengths"].astype(np.int16),
        rollout_valid=new_replay["rollout_valid"].astype(np.bool_),
    )

    reward_rollout_count = len(ordered_unique(reward["rollout_ids"]))
    mean_points_per_rollout = len(reward["targets"]) / reward_rollout_count
    base_batch = int(update["base_batch_per_gpu"])
    trajectory_batch = base_batch * min(
        MAX_TRAJECTORY_POINTS, max(1, math.ceil(mean_points_per_rollout))
    )
    weight_sum = float(np.sum(reward_weights, dtype=np.float64))
    weight_square_sum = float(np.sum(reward_weights.astype(np.float64) ** 2))
    weight_ess = weight_sum * weight_sum / max(weight_square_sum, 1.0e-30)
    top_threshold = float(np.percentile(reward["scores"], 90.0))
    top_weight_share = float(
        np.sum(reward_weights[reward["scores"] >= top_threshold], dtype=np.float64)
        / weight_sum
    )

    training_path = round_dir / "training_data.npz"
    atomic_savez(
        training_path,
        current=combined["current"].astype(np.float32),
        targets=reward["targets"].astype(np.float32),
        target_weights=reward_weights.astype(np.float32),
        target_scores=reward["scores"].astype(np.float64),
        target_rollout_ids=reward["rollout_ids"].astype("U64"),
        target_steps=reward["steps"].astype(np.int16),
        target_rollout_lengths=reward["rollout_lengths"].astype(np.int16),
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
    trajectory_by_step = {}
    for step in range(MAX_TRAJECTORY_POINTS):
        select = current_trajectory["steps"] == step
        if np.any(select):
            trajectory_by_step[str(step)] = {
                "count": int(np.count_nonzero(select)),
                "score_median": finite_median(current_trajectory["scores"][select]),
                "volume_qs_median": finite_median(
                    current_trajectory["volume_qs"][select]
                ),
                "coil_median": finite_median(current_trajectory["coil"][select]),
            }
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
            "center_trajectory_by_step": trajectory_by_step,
        },
        "training": {
            "data_file": str(training_path.relative_to(args.run_root)),
            "reward_point_count": int(len(reward["targets"])),
            "reward_rollout_count": int(reward_rollout_count),
            "mean_points_per_rollout": float(mean_points_per_rollout),
            "reward_temperature": float(update["reward_temperature"]),
            "reward_epsilon": float(update["reward_epsilon"]),
            "reward_score_reference": reward_score_reference,
            "reward_weight_min": float(np.min(reward_weights)),
            "reward_weight_max": float(np.max(reward_weights)),
            "reward_weight_effective_sample_size": weight_ess,
            "reward_top_score_decile_weight_share": top_weight_share,
            "base_batch_per_gpu": base_batch,
            "trajectory_batch_per_gpu": trajectory_batch,
            "trajectory_microbatch_per_gpu": int(
                update["trajectory_microbatch_per_gpu"]
            ),
            "replay_rollout_count_after_round": int(len(retained_rollouts)),
            "replay_point_count_after_round": int(len(new_replay["targets"])),
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
            "replay_rollouts": summary["training"]["replay_rollout_count_after_round"],
            "replay_points": summary["training"]["replay_point_count_after_round"],
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
    targets = np.asarray(data["targets"], dtype=np.float32)
    target_weights = np.asarray(data["target_weights"], dtype=np.float32)
    q0_path = Path(manifest["distillation"]["directory"]) / "q0_train_normalized.npy"
    q0 = np.load(q0_path, mmap_mode="r")
    if q0.ndim != 3 or q0.shape[1:] != (N_BASE_COILS, 100) or q0.dtype != np.float32:
        raise ValueError("q0 normalized training cache has the wrong contract")
    feature_weight_values = feature_weights(normalizer, device)
    rng = np.random.default_rng(2026090400 + args.round_index * 100 + rank)
    torch.manual_seed(2026090500 + args.round_index * 100 + rank)
    permutation_generator = torch.Generator().manual_seed(
        2026090600 + args.round_index * 100 + rank
    )
    metrics = []
    started = time.perf_counter()
    round_training = load_json(round_dir / "round_summary.json")["training"]
    base_batch_size = int(update["base_batch_per_gpu"])
    target_batch_size = int(round_training["trajectory_batch_per_gpu"])
    target_microbatch_size = int(update["trajectory_microbatch_per_gpu"])
    if target_batch_size % target_microbatch_size:
        raise ValueError("trajectory batch must be divisible by its microbatch")
    improvement_fraction = float(update["improved_fraction"])
    q0_fraction = float(update["q0_fraction"])
    current_fraction = 1.0 - improvement_fraction - q0_fraction
    for step in range(1, int(update["train_steps"]) + 1):
        current_batch = torch.from_numpy(
            current[rng.integers(0, len(current), size=base_batch_size)]
        )
        q0_batch = torch.from_numpy(
            q0[rng.integers(0, len(q0), size=base_batch_size)]
        )
        current_batch = random_permute_coils(
            current_batch, generator=permutation_generator
        ).to(device=device)
        q0_batch = random_permute_coils(
            q0_batch, generator=permutation_generator
        ).to(device=device)
        target_indices = rng.integers(0, len(targets), size=target_batch_size)
        sampled_target_weights = target_weights[target_indices].astype(
            np.float64, copy=False
        )
        target_denominator = float(np.sum(sampled_target_weights))
        if not math.isfinite(target_denominator) or target_denominator <= 0.0:
            raise RuntimeError("sampled trajectory weights have no positive mass")
        optimizer.zero_grad(set_to_none=True)
        accumulation_context = (
            train_model.no_sync if world_size > 1 else lambda: nullcontext()
        )
        with accumulation_context():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                current_loss = per_sample_flow_terms(
                    train_model,
                    current_batch,
                    feature_weights=feature_weight_values,
                ).mean()
            (current_fraction * current_loss).backward()
        with accumulation_context():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                q0_loss = per_sample_flow_terms(
                    train_model,
                    q0_batch,
                    feature_weights=feature_weight_values,
                ).mean()
            (q0_fraction * q0_loss).backward()
        target_loss_value = torch.zeros((), dtype=torch.float32, device=device)
        for offset in range(0, target_batch_size, target_microbatch_size):
            stop = offset + target_microbatch_size
            selected = target_indices[offset:stop]
            target_batch = torch.from_numpy(targets[selected])
            target_batch = random_permute_coils(
                target_batch, generator=permutation_generator
            ).to(device=device)
            weight_batch = torch.from_numpy(
                target_weights[selected].astype(np.float32, copy=False)
            ).to(device=device)
            is_last = stop == target_batch_size
            sync_context = nullcontext() if is_last else accumulation_context()
            with sync_context:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    target_terms = per_sample_flow_terms(
                        train_model,
                        target_batch,
                        feature_weights=feature_weight_values,
                    )
                    target_numerator = torch.sum(target_terms * weight_batch)
                    target_loss_part = target_numerator / target_denominator
                (improvement_fraction * target_loss_part).backward()
            target_loss_value += target_numerator.detach() / target_denominator
        loss = (
            current_fraction * current_loss.detach()
            + improvement_fraction * target_loss_value
            + q0_fraction * q0_loss.detach()
        )
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
                    loss,
                    current_loss.detach(),
                    target_loss_value,
                    q0_loss.detach(),
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
                    "base_batch_per_gpu": base_batch_size,
                    "trajectory_batch_per_gpu": target_batch_size,
                    "trajectory_microbatches_per_gpu": (
                        target_batch_size // target_microbatch_size
                    ),
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
            "base_batch_per_gpu": base_batch_size,
            "trajectory_batch_per_gpu": target_batch_size,
            "trajectory_microbatch_per_gpu": target_microbatch_size,
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
    prepare_command.add_argument("--score-library-manifest", type=Path, required=True)
    prepare_command.add_argument("--expected-score-lib-sha", required=True)
    prepare_command.add_argument("--optimizer-checkpoint", type=Path, required=True)
    prepare_command.add_argument("--expected-optimizer-checkpoint-sha", required=True)
    prepare_command.add_argument("--expected-commit", required=True)
    prepare_command.add_argument("--sample-seed", type=int, default=2026090302)
    prepare_command.add_argument("--train-steps", type=int, default=250)
    prepare_command.add_argument("--train-batch-per-gpu", type=int, default=256)
    prepare_command.add_argument("--learning-rate", type=float, default=5.0e-5)
    prepare_command.add_argument(
        "--reward-temperature", type=float, default=REWARD_TEMPERATURE
    )
    prepare_command.add_argument("--reward-epsilon", type=float, default=REWARD_EPSILON)
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
