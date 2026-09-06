from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import math
import os
from pathlib import Path
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
    N_BASE_COILS,
    NFP,
    feature_weights,
    file_sha256,
    inverse_tokens,
    transform_tokens,
)
from flow_matching.data import CoilNormalizer  # noqa: E402
from flow_matching.model import CoilFlowTransformer  # noqa: E402
from flow_matching.score_gradient_rl import (  # noqa: E402
    coordinate_gradient_directional_check,
    map_score_gradient_to_flow,
    ordinary_flow_terms,
    valid_flow_terms_with_transport,
    valid_score_weights,
)
from scripts.axisflip_prior_online_rl import (  # noqa: E402
    atomic_savez,
    atomic_write_json,
    compact_native,
    generate_policy_batch,
    native_score,
)
from scripts.flow_runtime import repository_provenance, result_score, result_valid  # noqa: E402
from scripts.native_score_runtime import token_case  # noqa: E402
from scripts.optimize_flow_latent import (  # noqa: E402
    LocalFullGradientEstimator,
    gradient_probe,
    random_direction_gradient,
)
from scripts.prepare_axis_surface_prior_adam200 import exact_standardized_start  # noqa: E402


PROTOCOL_ID = os.environ.get(
    "SCORE_GRADIENT_PROTOCOL_ID",
    "qh-axisflip-r012-score-gradient-replay50-rl-r04-abi11-v1",
)
FORMAT = os.environ.get(
    "SCORE_GRADIENT_FORMAT",
    "axisflip_r012_score_gradient_replay50_rl_r04_v1",
)
SAMPLES_PER_ROUND = 64
SAMPLES_PER_RANK = SAMPLES_PER_ROUND // 2
WORLD_SIZE = 2
REPLAY_CAPACITY = 512


def _env_positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _env_unit_interval(name: str, default: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be in (0, 1]")
    return value


FLOW_OPTIMIZER_STEPS_PER_ROUND = _env_positive_int(
    "SCORE_GRADIENT_FLOW_OPTIMIZER_STEPS_PER_ROUND", 50
)
EMA_LERP = _env_unit_interval("SCORE_GRADIENT_EMA_LERP", 0.01)
PRIOR_RADIUS_M = os.environ.get("SCORE_GRADIENT_PRIOR_RADIUS_M")
if PRIOR_RADIUS_M is not None:
    PRIOR_RADIUS_M = float(PRIOR_RADIUS_M)
PRIOR_RADIUS_RANGE_M = os.environ.get("SCORE_GRADIENT_PRIOR_RADIUS_RANGE_M")
FLOW_STEPS = 32
GRADIENT_DIRECTIONS = 64
GRADIENT_PERTURBATION = 0.0025
MONTE_CARLO_SAMPLES = 4
INVALID_ALPHA = 0.05
R04_SCORE_SHA = "7b21e66329a23f18ef3cfab408d0a21410acea3c8d10d8b2463ccee7f5a3969f"
R04_CURVATURE_P95_SCALE_M_INV = 25.0
R04_CURVATURE_MAX_SCALE_M_INV = 35.0
INITIAL_SCORE_TOLERANCE = 0.1
COORDINATE_CHECK_TOLERANCE = 5.0e-2


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def require_clean_repository(expected_commit: str) -> dict[str, Any]:
    provenance = repository_provenance(REPO_ROOT)
    if not provenance["available"] or provenance["tracked_dirty"]:
        raise RuntimeError("score-gradient experiment requires a clean tracked worktree")
    if provenance["commit"] != expected_commit:
        raise RuntimeError(
            f"repository commit {provenance['commit']} != expected {expected_commit}"
        )
    return provenance


def validate_r04_score(
    score_lib: Path, score_manifest_path: Path, expected_sha: str
) -> dict[str, Any]:
    actual_sha = file_sha256(score_lib)
    if actual_sha != expected_sha or actual_sha != R04_SCORE_SHA:
        raise ValueError("score library is not the frozen ABI-11 R04 build")
    score_manifest = load_json(score_manifest_path)
    overrides = score_manifest.get("cmake_overrides", {})
    if (
        score_manifest.get("interface_abi") != 11
        or score_manifest.get("sha256") != actual_sha
        or overrides.get("SGPU_COIL_CURVATURE_P95_SCALE")
        != R04_CURVATURE_P95_SCALE_M_INV
        or score_manifest.get("coil_curvature_p95_radius_m") != 0.04
        or score_manifest.get("coil_curvature_max_scale_m_inv")
        != R04_CURVATURE_MAX_SCALE_M_INV
    ):
        raise ValueError("score manifest does not describe ABI-11 R04")
    return score_manifest


def load_checkpoint(path: Path, device: torch.device) -> tuple[CoilFlowTransformer, CoilFlowTransformer, CoilNormalizer, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model_config", "ema", "normalizer", "step"}
    if checkpoint.get("format") is None or not required.issubset(checkpoint):
        raise ValueError("Flow checkpoint lacks the required q0/online fields")
    base = CoilFlowTransformer(**checkpoint["model_config"]).to(device=device)
    ema = CoilFlowTransformer(**checkpoint["model_config"]).to(device=device)
    base_state = checkpoint.get("model", checkpoint["ema"])
    base.load_state_dict(base_state)
    ema.load_state_dict(checkpoint["ema"])
    ema.eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    normalizer = CoilNormalizer.from_dict(checkpoint["normalizer"])
    return base, ema, normalizer, checkpoint


def inherit_reference_strategy(manifest: dict[str, Any], reference: dict[str, Any]) -> None:
    """Pin a radius control to the completed Students pilot and its full policy."""
    if reference["protocol"]["id"] != "qh-axisflip-r012-score-gradient-replay10-ema10-rl-r04-abi11-v1":
        raise ValueError("radius control requires the original unweighted Students reference")
    for section in ("condition", "evaluator", "paths"):
        if manifest[section] != reference[section]:
            raise ValueError(f"radius control differs from reference: {section}")
    actual_baseline = {k: v for k, v in manifest["baseline_r04"].items() if k != "q0_checkpoint_sha256"}
    expected_baseline = {k: v for k, v in reference["baseline_r04"].items() if k != "q0_checkpoint_sha256"}
    if actual_baseline != expected_baseline:
        raise ValueError("radius control scorer or coordinate baseline differs")
    if manifest["q0"]["model_config"] != reference["q0"]["model_config"]:
        raise ValueError("radius control model architecture differs")
    if reference["strategy"].get("valid_score_weighting") is not None:
        raise ValueError("weighted continuation cannot become the radius reference")
    ignored = {"beta", "beta_calibration", "beta_calibration_result"}
    actual = {k: v for k, v in manifest["strategy"].items() if k not in ignored}
    expected = {k: v for k, v in reference["strategy"].items() if k not in ignored}
    if actual != expected:
        raise ValueError("radius control policy differs from reference")
    beta = reference["strategy"].get("beta")
    if beta is None or not math.isfinite(beta) or beta <= 0:
        raise ValueError("reference must contain the frozen positive calibrated beta")
    manifest["strategy"] = copy.deepcopy(reference["strategy"])


def prepare(args: argparse.Namespace) -> None:
    provenance = require_clean_repository(args.expected_commit)
    score_manifest = validate_r04_score(
        args.score_lib, args.score_library_manifest, args.expected_score_lib_sha
    )
    if file_sha256(args.q0_checkpoint) != args.expected_q0_sha:
        raise ValueError("q0 checkpoint hash mismatch")
    if file_sha256(args.optimizer_checkpoint) != args.expected_optimizer_sha:
        raise ValueError("optimizer normalizer checkpoint hash mismatch")
    checkpoint = torch.load(args.q0_checkpoint, map_location="cpu", weights_only=False)
    if not {"model_config", "ema", "normalizer", "step"}.issubset(checkpoint):
        raise ValueError("q0 checkpoint is incomplete")
    if checkpoint["model_config"].get("width") != 256:
        raise ValueError("q0 model is not the R04 256-wide Flow")
    if args.run_root.exists():
        raise FileExistsError(args.run_root)
    (args.run_root / "checkpoints").mkdir(parents=True, exist_ok=False)
    (args.run_root / "rounds").mkdir(parents=True, exist_ok=False)
    (args.run_root / "logs").mkdir(parents=True, exist_ok=False)
    q0_copy = args.run_root / "checkpoints" / "round_0000.pt"
    q0_copy.write_bytes(args.q0_checkpoint.read_bytes())
    manifest = {
        "format": FORMAT,
        "protocol": {
            "id": PROTOCOL_ID,
            "status": "registered-experimental",
            "relationship_to_default": "student score-gradient policy comparison; R04 and QH defaults unchanged",
        },
        "repository": provenance,
        "created_unix_s": time.time(),
        "condition": {"nfp": NFP, "n_base_coils": N_BASE_COILS},
        "prior": {
            "minor_radius_center_m": PRIOR_RADIUS_M,
            "minor_radius_range_m": PRIOR_RADIUS_RANGE_M,
        },
        "baseline_r04": {
            "protocol_id": "qh-axisflip-r012-distilled-online-adam20-trajectory-rwcfm-r04-abi11-v1",
            "score_library_sha256": R04_SCORE_SHA,
            "score_library_manifest": str(args.score_library_manifest.resolve()),
            "score_manifest_sha256": file_sha256(args.score_library_manifest),
            "q0_checkpoint_sha256": file_sha256(args.q0_checkpoint),
            "optimizer_checkpoint_sha256": file_sha256(args.optimizer_checkpoint),
            "flow_steps": FLOW_STEPS,
            "target_helicity": [1, NFP],
            "curvature_p95_scale_m_inv": R04_CURVATURE_P95_SCALE_M_INV,
            "curvature_p95_radius_m": 0.04,
            "curvature_max_scale_m_inv": R04_CURVATURE_MAX_SCALE_M_INV,
            "score_configuration": {
                "iota_degree": 3,
                "surface_selection_mode": 1,
                "surface_confidence_periods": 1,
                "surface_theta_count": 128,
                "surface_trace_steps": 400,
                "surface_flux_bisection_iters": 6,
                "gradient_segments_per_coil": 256,
                "gradient_psi_iterations": 4,
                "gradient_alpha_iterations": 4,
                "gradient_local_surface_theta_count": 64,
            },
        },
        "q0": {
            "source_checkpoint": str(args.q0_checkpoint.resolve()),
            "copied_checkpoint": str(q0_copy.resolve()),
            "checkpoint_sha256": file_sha256(args.q0_checkpoint),
            "normalizer": {
                "source": "embedded in the pinned q0 checkpoint",
                "checkpoint_sha256": file_sha256(args.q0_checkpoint),
            },
            "model_config": checkpoint["model_config"],
        },
        "strategy": {
            "samples_per_round": SAMPLES_PER_ROUND,
            "world_size": WORLD_SIZE,
            "samples_per_rank": SAMPLES_PER_RANK,
            "gradient_parameter_space": "R04 exact-unclipped optimizer data coordinates",
            "flow_parameter_space": "R04 q0 prior-normalized coordinates",
            "gradient_directions": GRADIENT_DIRECTIONS,
            "gradient_difference": "centered random-orthogonal",
            "gradient_perturbation": GRADIENT_PERTURBATION,
            "coordinate_directional_check_step": 1.0e-3,
            "coordinate_directional_check_tolerance": COORDINATE_CHECK_TOLERANCE,
            "monte_carlo_samples": MONTE_CARLO_SAMPLES,
            "loss": "L_valid + alpha*L_invalid + beta*mean(g_flow^T grad_x ell)",
            "invalid_alpha": INVALID_ALPHA,
            "beta": None,
            "beta_calibration": "q0 no-update pilot; freeze before round 0",
            "ema_lerp": EMA_LERP,
            "adam20_rollout": False,
            "replay": {
                "enabled": True,
                "capacity": REPLAY_CAPACITY,
                "sampling": "uniform with replacement",
                "record": "one scored center with its fixed 64-direction score gradient",
            },
            "rho": None,
            "epsilon_transport": None,
            "flow_optimizer_steps_per_round": FLOW_OPTIMIZER_STEPS_PER_ROUND,
        },
        "parallelism": {
            "collection": "two independent one-GPU ranks, 32 centers each; one BatchCoilFieldGpu with 128 endpoints per valid center",
            "training": (
                "two-GPU DDP, "
                f"{FLOW_OPTIMIZER_STEPS_PER_ROUND} global optimizer steps per round; "
                "each step samples 32 records per rank from the synchronized replay pool"
            ),
            "center_endpoint_flattening": False,
        },
        "evaluator": {
            "abi": 11,
            "library": str(args.score_lib.resolve()),
            "library_sha256": file_sha256(args.score_lib),
            "manifest": str(args.score_library_manifest.resolve()),
            "manifest_sha256": file_sha256(args.score_library_manifest),
            "manifest_summary": score_manifest,
        },
        "paths": {
            "optimizer_checkpoint": str(args.optimizer_checkpoint.resolve()),
            "optimizer_checkpoint_sha256": file_sha256(args.optimizer_checkpoint),
        },
    }
    if getattr(args, "reference_manifest", None) is not None:
        if file_sha256(args.reference_manifest) != args.expected_reference_sha:
            raise ValueError("reference manifest hash mismatch")
        reference = load_json(args.reference_manifest)
        inherit_reference_strategy(manifest, reference)
        manifest["reference"] = {"manifest": str(args.reference_manifest.resolve()),
            "sha256": args.expected_reference_sha, "protocol_id": reference["protocol"]["id"],
            "strategy": "exact copy, including frozen beta; no new calibration"}
        atomic_write_json(args.run_root / "reference_manifest.json", reference)
    atomic_write_json(args.run_root / "manifest.json", manifest)
    atomic_write_json(
        args.run_root / "progress.json",
        {"format": FORMAT, "stage": "prepared", "next_round": 0, "updated_unix_s": time.time()},
    )
    print(json.dumps({"event": "score_gradient_prepared", "run_root": str(args.run_root)}))


def load_manifest(run_root: Path) -> dict[str, Any]:
    manifest = load_json(run_root / "manifest.json")
    if manifest.get("format") != FORMAT or manifest.get("protocol", {}).get("id") != PROTOCOL_ID:
        raise ValueError("score-gradient manifest has the wrong protocol")
    if manifest.get("condition") != {"nfp": NFP, "n_base_coils": N_BASE_COILS}:
        raise ValueError("score-gradient condition changed")
    strategy = manifest["strategy"]
    if "reference" in manifest:
        reference = load_json(run_root / "reference_manifest.json")
        if strategy != reference["strategy"]:
            raise ValueError("radius-control strategy differs from its frozen reference")
    if (strategy["flow_optimizer_steps_per_round"] != FLOW_OPTIMIZER_STEPS_PER_ROUND
            or strategy["ema_lerp"] != EMA_LERP):
        raise ValueError("runtime schedule differs from the frozen manifest")
    if "continuation" in manifest:
        protocol = load_json(run_root / "protocol.json")
        if (strategy.get("valid_score_weighting") != protocol["valid_score_weighting"]
                or strategy["beta"] != protocol["beta"]
                or manifest["continuation"]["start_round"] != protocol["start_round"]):
            raise ValueError("continuation loss or starting round changed")
        for relative, expected in protocol["source_sha256"].items():
            if relative.startswith("checkpoints/") and file_sha256(run_root / relative) != expected:
                raise ValueError("continuation checkpoint bytes changed")
    current = repository_provenance(REPO_ROOT)
    if manifest.get("repository") != current:
        raise RuntimeError("score-gradient repository provenance changed")
    return manifest


def permute_pair(
    data: torch.Tensor, gradient: torch.Tensor | None, *, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if data.ndim != 3 or data.shape[1:] != (N_BASE_COILS, 100):
        raise ValueError("data must have shape (batch,3,100)")
    order = torch.argsort(
        torch.rand((len(data), N_BASE_COILS), generator=generator, device=data.device), dim=1
    )
    expanded = order[..., None].expand(-1, -1, 100)
    permuted_data = torch.gather(data, 1, expanded)
    if gradient is None:
        return permuted_data, None
    return permuted_data, torch.gather(gradient, 1, expanded)


def collect_one(
    physical: np.ndarray,
    *,
    flow_normalizer: CoilNormalizer,
    optimizer_normalizer: CoilNormalizer,
    estimator: LocalFullGradientEstimator,
    score_lib: Path,
    rank: int,
    round_index: int,
    sample_index: int,
    seed: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    parameters, current_l1, represented, roundtrip = exact_standardized_start(
        physical, optimizer_normalizer, condition=(NFP, N_BASE_COILS)
    )
    current = transform_tokens(represented[None], flow_normalizer)[0]
    initial_result, initial_wall = native_score(represented, lib=score_lib, device=rank)
    compact_initial = compact_native(initial_result)
    record: dict[str, Any] = {
        "sample_id": f"r{round_index:04d}_r{rank:02d}_i{sample_index:03d}",
        "valid": compact_initial["status"] == "ok",
        "initial": compact_initial,
        "current": current,
        "score_gradient_flow": np.zeros_like(current, dtype=np.float32),
        "gradient_ok": False,
        "gradient_status": "not_attempted",
        "coordinate_check_relative_error": None,
        "timing_s": {"initial_score": initial_wall, "gradient": 0.0, "total": 0.0},
        "roundtrip": roundtrip,
    }
    if record["valid"]:
        direction_rng = np.random.default_rng(
            int(seed) + round_index * 1000003 + rank * 100003 + sample_index
        )
        directions, endpoint_data = gradient_probe(
            parameters,
            mode="random-orthogonal",
            perturbation=GRADIENT_PERTURBATION,
            random_direction_count=GRADIENT_DIRECTIONS,
            rng=direction_rng,
        )
        if directions is None:
            raise RuntimeError("random-orthogonal gradient unexpectedly returned no directions")
        endpoint_normalizer = copy.deepcopy(optimizer_normalizer)
        endpoint_normalizer.current_l1_a[f"{NFP}:{N_BASE_COILS}"] = current_l1
        endpoint_normalizer.clip = float("inf")
        endpoint_tokens = inverse_tokens(endpoint_data, endpoint_normalizer)
        gradient_started = time.perf_counter()
        local_scores, captured_result, details, _ = estimator.evaluate(
            represented, endpoint_tokens, initial_result
        )
        gradient_wall = time.perf_counter() - gradient_started
        statuses = details["status_counts"]
        all_endpoints_valid = statuses == {"ok": len(endpoint_tokens)}
        capture_delta = abs(
            result_score(captured_result) - result_score(initial_result)
        )
        center_ok = result_valid(captured_result) and capture_delta <= INITIAL_SCORE_TOLERANCE
        if all_endpoints_valid and center_ok:
            gradient_optimizer = random_direction_gradient(
                local_scores, GRADIENT_PERTURBATION, directions
            )
            gradient_flow = map_score_gradient_to_flow(
                current,
                gradient_optimizer,
                flow_normalizer,
                endpoint_normalizer,
                current_l1_a=current_l1,
                nfp=NFP,
            )
            check_direction = direction_rng.standard_normal(current.shape)
            check_direction /= max(float(np.linalg.norm(check_direction)), 1.0e-30)
            check_error = coordinate_gradient_directional_check(
                current,
                gradient_optimizer,
                gradient_flow,
                flow_normalizer,
                endpoint_normalizer,
                current_l1_a=current_l1,
                nfp=NFP,
                direction=check_direction,
                step=1.0e-3,
            )
            record["score_gradient_flow"] = gradient_flow
            record["gradient_ok"] = check_error <= COORDINATE_CHECK_TOLERANCE
            record["gradient_status"] = "ok" if record["gradient_ok"] else "coordinate_check_failed"
            record["coordinate_check_relative_error"] = float(check_error)
            record["gradient_norm_optimizer"] = float(np.linalg.norm(gradient_optimizer))
            record["gradient_norm_flow"] = float(np.linalg.norm(gradient_flow))
        else:
            record["gradient_status"] = "endpoint_invalid" if not all_endpoints_valid else "center_gate_failed"
        record["timing_s"]["gradient"] = gradient_wall
        record["gradient_details"] = details
    record["timing_s"]["total"] = time.perf_counter() - started
    return record


def stack_records(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    def component(row: dict[str, Any], name: str) -> float:
        value = row["initial"]["components"].get(name)
        return float(value) if value is not None else math.nan

    return {
        "current": np.asarray([row["current"] for row in records], dtype=np.float32),
        "score_gradient_flow": np.asarray(
            [row["score_gradient_flow"] for row in records], dtype=np.float32
        ),
        "valid": np.asarray([row["valid"] for row in records], dtype=np.bool_),
        "gradient_ok": np.asarray([row["gradient_ok"] for row in records], dtype=np.bool_),
        "scores": np.asarray([row["initial"]["score"] for row in records], dtype=np.float64),
        "volume_qs": np.asarray(
            [component(row, "volume_qs") for row in records],
            dtype=np.float64,
        ),
        "coil": np.asarray(
            [component(row, "coil") for row in records],
            dtype=np.float64,
        ),
        "initial_score_wall_s": np.asarray(
            [row["timing_s"]["initial_score"] for row in records], dtype=np.float64
        ),
        "gradient_wall_s": np.asarray(
            [row["timing_s"]["gradient"] for row in records], dtype=np.float64
        ),
        "coordinate_check_error": np.asarray(
            [row["coordinate_check_relative_error"] or np.nan for row in records],
            dtype=np.float64,
        ),
    }


def finite_median(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else None


def finite_percentile(values: np.ndarray, percentile: float) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, percentile)) if values.size else None


def finite_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    keep = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(keep) < 2:
        return None
    if np.std(left[keep]) == 0.0 or np.std(right[keep]) == 0.0:
        return None
    return float(np.corrcoef(left[keep], right[keep])[0, 1])


def combine_round(round_dir: Path) -> dict[str, np.ndarray]:
    parts = []
    for rank in range(WORLD_SIZE):
        path = round_dir / f"rank_{rank:02d}.npz"
        with np.load(path, allow_pickle=False) as payload:
            parts.append({name: np.asarray(payload[name]) for name in payload.files})
    return {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}


def summarize_collection(
    combined: dict[str, np.ndarray], records: list[dict[str, Any]], *, round_index: int, round_dir: Path
) -> dict[str, Any]:
    valid = combined["valid"]
    gradient_ok = combined["gradient_ok"]
    valid_scores = combined["scores"][valid]
    initial = {
        "score_median_all": finite_median(combined["scores"]),
        "score_p90_all": finite_percentile(combined["scores"], 90.0),
        "score_median_valid": finite_median(valid_scores),
        "volume_qs_median_valid": finite_median(combined["volume_qs"][valid]),
        "coil_median_valid": finite_median(combined["coil"][valid]),
        "volume_qs_coil_correlation_valid": finite_correlation(
            combined["volume_qs"][valid], combined["coil"][valid]
        ),
    }
    summary = {
        "format": FORMAT,
        "stage": "collection_complete",
        "round": round_index,
        "sample_count": int(len(combined["scores"])),
        "valid_count": int(np.count_nonzero(valid)),
        "valid_rate": float(np.mean(valid)),
        "gradient_ok_count": int(np.count_nonzero(gradient_ok)),
        "gradient_ok_rate_all": float(np.mean(gradient_ok)),
        "gradient_ok_rate_valid": (
            float(np.mean(gradient_ok[valid])) if np.any(valid) else None
        ),
        "initial": initial,
        "timing": {
            "rank_wall_s": [
                float(load_json(round_dir / f"rank_{rank:02d}.json")["wall_s"])
                for rank in range(WORLD_SIZE)
            ],
            "collection_wall_s": max(
                float(load_json(round_dir / f"rank_{rank:02d}.json")["wall_s"])
                for rank in range(WORLD_SIZE)
            ),
            "initial_score_sum_s": float(np.sum(combined["initial_score_wall_s"])),
            "gradient_sum_s": float(np.sum(combined["gradient_wall_s"])),
            "initial_score_median_s": finite_median(combined["initial_score_wall_s"]),
            "gradient_median_s_valid": finite_median(combined["gradient_wall_s"][valid]),
        },
        "status_counts": dict(Counter(row["initial"]["status"] for row in records)),
        "coordinate_check": {
            "relative_error_median": finite_median(combined["coordinate_check_error"]),
            "relative_error_p99": finite_percentile(combined["coordinate_check_error"], 99.0),
        },
        "records": [
            {
                key: value
                for key, value in row.items()
                if key not in {"current", "score_gradient_flow", "gradient_details"}
            }
            for row in records
        ],
        "finished_unix_s": time.time(),
    }
    atomic_write_json(round_dir / "collection_summary.json", summary)
    return summary


def distributed_setup() -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size != WORLD_SIZE or not torch.cuda.is_available():
        raise RuntimeError("score-gradient run requires two CUDA DDP ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    return rank, local_rank, device


def all_reduce_scalar(value: torch.Tensor) -> torch.Tensor:
    result = value.detach().clone()
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


REPLAY_ARRAY_KEYS = (
    "current",
    "score_gradient_flow",
    "valid",
    "gradient_ok",
    "scores",
    "volume_qs",
    "coil",
)


def append_replay_pool(
    previous: dict[str, np.ndarray] | None, incoming: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Append scored centers and keep the newest fixed-size FIFO window."""
    incoming_subset = {key: np.asarray(incoming[key]) for key in REPLAY_ARRAY_KEYS}
    if previous is None:
        combined = incoming_subset
    else:
        if set(previous) != set(REPLAY_ARRAY_KEYS):
            raise ValueError("replay pool has an unexpected schema")
        combined = {
            key: np.concatenate((np.asarray(previous[key]), incoming_subset[key]), axis=0)
            for key in REPLAY_ARRAY_KEYS
        }
    if len(combined["current"]) > REPLAY_CAPACITY:
        combined = {key: values[-REPLAY_CAPACITY:] for key, values in combined.items()}
    if len(combined["current"]) == 0:
        raise ValueError("replay pool cannot be empty after appending a round")
    return combined


def load_replay_pool(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != set(REPLAY_ARRAY_KEYS):
            raise ValueError("replay pool has an unexpected schema")
        pool = {key: np.asarray(payload[key]) for key in REPLAY_ARRAY_KEYS}
    lengths = {len(values) for values in pool.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) > REPLAY_CAPACITY:
        raise ValueError("replay pool lengths or capacity are invalid")
    return pool


def pool_to_device(pool: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    if len(pool["current"]) == 0:
        raise ValueError("cannot train from an empty replay pool")
    return {
        "current": torch.from_numpy(pool["current"]).to(device=device),
        "score_gradient_flow": torch.from_numpy(pool["score_gradient_flow"]).to(device=device),
        "valid": torch.from_numpy(pool["valid"]).to(device=device),
        "gradient_ok": torch.from_numpy(pool["gradient_ok"]).to(device=device),
        "scores": torch.from_numpy(pool["scores"]).to(device=device),
    }


def train_step_from_replay(
    *,
    model: torch.nn.Module,
    ema_model: CoilFlowTransformer,
    optimizer: torch.optim.Optimizer,
    normalizer: CoilNormalizer,
    pool: dict[str, torch.Tensor],
    device: torch.device,
    beta: float,
    sample_generator: torch.Generator,
    permutation_generator: torch.Generator,
    loss_generator: torch.Generator,
    valid_weighting: dict[str, Any] | None = None,
) -> dict[str, Any]:
    feature_weight_values = feature_weights(normalizer, device)
    sample_indices = torch.randint(
        len(pool["current"]),
        (SAMPLES_PER_RANK,),
        generator=sample_generator,
        device=device,
    )
    batch_data = pool["current"][sample_indices]
    batch_gradient = pool["score_gradient_flow"][sample_indices]
    batch_valid = pool["valid"][sample_indices]
    batch_gradient_ok = pool["gradient_ok"][sample_indices]
    valid_data = batch_data[batch_valid]
    valid_gradient = batch_gradient[batch_valid]
    valid_gradient = valid_gradient * batch_gradient_ok[batch_valid, None, None].to(
        dtype=valid_gradient.dtype
    )
    invalid_data = batch_data[~batch_valid]
    if len(valid_data) == 0:
        valid_data = batch_data[:1]
        valid_gradient = torch.zeros_like(valid_data)
        local_valid_count = 0
    else:
        local_valid_count = len(valid_data)
    if len(invalid_data) == 0:
        invalid_data = batch_data[:1]
        local_invalid_count = 0
    else:
        local_invalid_count = len(invalid_data)
    valid_data, valid_gradient = permute_pair(
        valid_data, valid_gradient, generator=permutation_generator
    )
    invalid_data, _ = permute_pair(invalid_data, None, generator=permutation_generator)
    valid_count_tensor = torch.tensor(local_valid_count, dtype=torch.float32, device=device)
    invalid_count_tensor = torch.tensor(local_invalid_count, dtype=torch.float32, device=device)
    global_valid = all_reduce_scalar(valid_count_tensor)
    global_invalid = all_reduce_scalar(invalid_count_tensor)
    score_weights = None
    weight_diagnostics = {}
    if valid_weighting is not None:
        valid_scores = pool["scores"][sample_indices][batch_valid].detach().float()
        maximum = valid_scores.max() if len(valid_scores) else torch.tensor(
            -float("inf"), device=device
        )
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        if global_valid.item() == 0:
            maximum.fill_(0.0)
        score_weights = valid_score_weights(
            valid_scores, reference_max=maximum,
            tau=valid_weighting["tau"], epsilon=valid_weighting["epsilon"],
        )
        weight_sum = all_reduce_scalar(score_weights.sum())
        weight_square_sum = all_reduce_scalar(score_weights.square().sum())
        weight_diagnostics = {
            "valid_weight_ess": float((weight_sum.square() / weight_square_sum.clamp_min(1e-12)).item()),
            "valid_weight_max_fraction": float(((1.0 + valid_weighting["epsilon"]) / weight_sum).item()) if global_valid.item() else 0.0,
            "valid_score_max": float(maximum.item()) if global_valid.item() else None,
        }
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    valid_started = time.perf_counter()
    valid_terms, transport_terms = valid_flow_terms_with_transport(
        model,
        valid_data,
        valid_gradient,
        feature_weights=feature_weight_values,
        monte_carlo_samples=MONTE_CARLO_SAMPLES,
        generator=loss_generator,
    )
    valid_forward_wall = time.perf_counter() - valid_started
    invalid_started = time.perf_counter()
    invalid_terms = ordinary_flow_terms(
        model,
        invalid_data,
        feature_weights=feature_weight_values,
        monte_carlo_samples=MONTE_CARLO_SAMPLES,
        generator=loss_generator,
    )
    invalid_forward_wall = time.perf_counter() - invalid_started
    valid_sum = valid_terms.sum()
    transport_sum = transport_terms.sum()
    invalid_sum = invalid_terms.sum()
    world_factor = float(WORLD_SIZE)
    objective = torch.zeros((), dtype=torch.float32, device=device)
    if float(global_valid.item()) > 0.0:
        if score_weights is None:
            objective = objective + world_factor * (valid_sum + beta * transport_sum) / global_valid
        else:
            # The dummy forward on an empty rank keeps DDP graphs aligned.
            weighted_sum = (valid_terms[:local_valid_count] * score_weights).sum()
            objective = objective + world_factor * weighted_sum / weight_sum
            objective = objective + world_factor * beta * transport_sum / global_valid
            weight_diagnostics["valid_loss_weighted"] = float(
                (all_reduce_scalar(weighted_sum.detach()) / weight_sum).item()
            )
            weight_diagnostics["valid_loss_unweighted"] = float(
                (all_reduce_scalar(valid_terms[:local_valid_count].detach().sum()) / global_valid).item()
            )
    if float(global_invalid.item()) > 0.0:
        objective = objective + world_factor * INVALID_ALPHA * invalid_sum / global_invalid
    backward_started = time.perf_counter()
    objective.backward()
    backward_wall = time.perf_counter() - backward_started
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer_started = time.perf_counter()
    optimizer.step()
    with torch.no_grad():
        for target, source in zip(ema_model.parameters(), model.module.parameters(), strict=True):
            target.lerp_(source.detach(), EMA_LERP)
    optimizer_wall = time.perf_counter() - optimizer_started
    detached_values = torch.stack(
        [
            valid_sum.detach(),
            transport_sum.detach(),
            invalid_sum.detach(),
            objective.detach(),
            torch.as_tensor(gradient_norm, dtype=torch.float32, device=device),
        ]
    )
    reduced_values = all_reduce_scalar(detached_values) / WORLD_SIZE
    return {
        "wall_s": time.perf_counter() - started,
        "valid_forward_wall_s": valid_forward_wall,
        "invalid_forward_wall_s": invalid_forward_wall,
        "backward_wall_s": backward_wall,
        "optimizer_wall_s": optimizer_wall,
        "global_valid_count": int(global_valid.item()),
        "global_invalid_count": int(global_invalid.item()),
        "valid_loss_sum_per_rank": float(reduced_values[0].cpu()),
        "transport_sum_per_rank": float(reduced_values[1].cpu()),
        "invalid_loss_sum_per_rank": float(reduced_values[2].cpu()),
        "objective": float(reduced_values[3].cpu()),
        "gradient_norm": float(reduced_values[4].cpu()),
        "beta": beta,
        **weight_diagnostics,
    }


def train_round(
    *,
    model: torch.nn.Module,
    ema_model: CoilFlowTransformer,
    optimizer: torch.optim.Optimizer,
    normalizer: CoilNormalizer,
    replay_pool: dict[str, np.ndarray],
    rank: int,
    device: torch.device,
    beta: float,
    round_index: int,
    valid_weighting: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run fixed-size replay updates with the manifest's optional valid weighting."""
    pool = pool_to_device(replay_pool, device)
    sample_generator = torch.Generator(device=device).manual_seed(
        2026090700 + round_index * 1000 + rank
    )
    permutation_generator = torch.Generator(device=device).manual_seed(
        2026090800 + round_index * 1000 + rank
    )
    loss_generator = torch.Generator(device=device).manual_seed(
        2026090900 + round_index * 1000 + rank
    )
    started = time.perf_counter()
    step_summaries = []
    for update_index in range(FLOW_OPTIMIZER_STEPS_PER_ROUND):
        step_summary = train_step_from_replay(
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            normalizer=normalizer,
            pool=pool,
            device=device,
            beta=beta,
            sample_generator=sample_generator,
            permutation_generator=permutation_generator,
            loss_generator=loss_generator,
            valid_weighting=valid_weighting,
        )
        step_summary["update"] = update_index
        step_summaries.append(step_summary)

    def statistic(key: str, percentile: float | None = None) -> float | None:
        values = np.asarray([row[key] for row in step_summaries], dtype=np.float64)
        if percentile is None:
            return float(np.median(values))
        return float(np.percentile(values, percentile))

    return {
        "updates": FLOW_OPTIMIZER_STEPS_PER_ROUND,
        "pool_size": len(replay_pool["current"]),
        "wall_s": time.perf_counter() - started,
        "step_wall_s_median": statistic("wall_s"),
        "step_wall_s_p90": statistic("wall_s", 90.0),
        "objective_first": float(step_summaries[0]["objective"]),
        "objective_last": float(step_summaries[-1]["objective"]),
        "objective_median": statistic("objective"),
        "objective_p90": statistic("objective", 90.0),
        "global_valid_count_median": statistic("global_valid_count"),
        "global_invalid_count_median": statistic("global_invalid_count"),
        "valid_forward_wall_s_total": float(
            sum(row["valid_forward_wall_s"] for row in step_summaries)
        ),
        "invalid_forward_wall_s_total": float(
            sum(row["invalid_forward_wall_s"] for row in step_summaries)
        ),
        "backward_wall_s_total": float(sum(row["backward_wall_s"] for row in step_summaries)),
        "optimizer_wall_s_total": float(sum(row["optimizer_wall_s"] for row in step_summaries)),
        "steps": step_summaries,
    }


def calibrate_beta(
    *,
    model: torch.nn.Module,
    normalizer: CoilNormalizer,
    current: np.ndarray,
    valid: np.ndarray,
    gradient_flow: np.ndarray,
    gradient_ok: np.ndarray,
    rank: int,
    device: torch.device,
    target_transport_fraction: float = 0.10,
) -> tuple[float, dict[str, float]]:
    """Calibrate one fixed beta from the first real scored batch.

    This pass creates the input gradient graph but performs no parameter update.
    The ratio is global across DDP ranks and therefore independent of their
    local valid counts.
    """

    if not 0.0 < target_transport_fraction < 1.0:
        raise ValueError("target transport fraction must be in (0,1)")
    feature_weight_values = feature_weights(normalizer, device)
    generator = torch.Generator(device=device).manual_seed(2026091200 + rank)
    local_data = torch.from_numpy(current[valid]).to(device=device)
    local_gradient = torch.from_numpy(gradient_flow[valid]).to(device=device)
    local_gradient = local_gradient * torch.from_numpy(
        gradient_ok[valid, None, None]
    ).to(device=device, dtype=local_gradient.dtype)
    local_count = len(local_data)
    if local_count == 0:
        local_data = torch.from_numpy(current[:1]).to(device=device)
        local_gradient = torch.zeros_like(local_data)
    ordinary, transport = valid_flow_terms_with_transport(
        model,
        local_data,
        local_gradient,
        feature_weights=feature_weight_values,
        monte_carlo_samples=MONTE_CARLO_SAMPLES,
        generator=generator,
    )
    valid_sum = ordinary.sum()
    transport_abs_sum = transport.abs().sum()
    count = all_reduce_scalar(torch.tensor(local_count, dtype=torch.float32, device=device))
    values = all_reduce_scalar(
        torch.stack([valid_sum.detach(), transport_abs_sum.detach()])
    )
    if float(count.item()) <= 0.0:
        raise RuntimeError("q0 calibration batch contains no valid samples")
    valid_mean = float(values[0].cpu()) / float(count.item())
    transport_abs_mean = float(values[1].cpu()) / float(count.item())
    beta_raw = target_transport_fraction * valid_mean / max(transport_abs_mean, 1.0e-12)
    beta = float(np.clip(beta_raw, 1.0e-5, 0.1))
    return beta, {
        "target_transport_fraction": target_transport_fraction,
        "valid_loss_mean": valid_mean,
        "transport_abs_mean": transport_abs_mean,
        "beta_raw": beta_raw,
        "beta": beta,
    }


def save_online_checkpoint(
    path: Path,
    *,
    model: CoilFlowTransformer,
    ema: CoilFlowTransformer,
    optimizer: torch.optim.Optimizer,
    normalizer: CoilNormalizer,
    checkpoint: dict[str, Any],
    outer_round: int,
    code_commit: str,
) -> None:
    atomic_torch_save(
        path,
        {
            "format": FORMAT,
            "stage": "score_gradient_online_policy",
            "model_config": model.config,
            "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "ema": {name: value.detach().cpu() for name, value in ema.state_dict().items()},
            "online_optimizer": optimizer.state_dict(),
            "normalizer": normalizer.to_dict(),
            "outer_round": int(outer_round),
            "step": int(checkpoint.get("step", 0)) + 1,
            "distillation_step": int(checkpoint.get("distillation_step", checkpoint.get("step", 0))),
            "code_commit": code_commit,
        },
    )


def run(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.run_root)
    rank, local_rank, device = distributed_setup()
    score_lib = Path(manifest["evaluator"]["library"])
    optimizer_checkpoint = torch.load(
        manifest["paths"]["optimizer_checkpoint"], map_location="cpu", weights_only=False
    )
    optimizer_normalizer = CoilNormalizer.from_dict(optimizer_checkpoint["normalizer"])
    started_job = time.perf_counter()
    round_index = int(manifest.get("continuation", {}).get("start_round", 0))
    while True:
        if args.max_rounds is not None and round_index >= args.max_rounds:
            break
        if (args.run_root / "STOP_AFTER_ROUND").exists():
            break
        if time.perf_counter() - started_job + args.reserve_s >= args.max_wall_s:
            break
        checkpoint_path = args.run_root / "checkpoints" / f"round_{round_index:04d}.pt"
        base_model, ema_model, normalizer, checkpoint = load_checkpoint(checkpoint_path, device)
        if rank == 0:
            expected_round = int(checkpoint.get("outer_round", 0))
            if expected_round != round_index:
                raise RuntimeError("checkpoint outer round does not match run loop")
        train_model = torch.nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
        optimizer = torch.optim.AdamW(
            train_model.parameters(),
            lr=5.0e-5,
            betas=(0.9, 0.95),
            weight_decay=0.01,
            fused=True,
        )
        if checkpoint.get("stage") == "score_gradient_online_policy":
            optimizer.load_state_dict(checkpoint["online_optimizer"])
        generated_normalized, generated_physical, flow_wall = generate_policy_batch(
            ema_model,
            normalizer,
            count=SAMPLES_PER_RANK,
            seed=2026091000 + round_index * 100 + rank,
            device=device,
        )
        estimator = LocalFullGradientEstimator(
            score_lib,
            nfp=NFP,
            score_device=local_rank,
            segments_per_coil=256,
            psi_iterations=4,
            alpha_iterations=4,
            formal_surface_theta_count=128,
            local_surface_theta_count=64,
            iota_degree=3,
            target_helicity=(1, NFP),
        )
        records = []
        collection_started = time.perf_counter()
        for index, physical in enumerate(generated_physical):
            records.append(
                collect_one(
                    physical,
                    flow_normalizer=normalizer,
                    optimizer_normalizer=optimizer_normalizer,
                    estimator=estimator,
                    score_lib=score_lib,
                    rank=local_rank,
                    round_index=round_index,
                    sample_index=index,
                    seed=2026091100,
                )
            )
        collection_wall = time.perf_counter() - collection_started
        round_dir = args.run_root / "rounds" / f"round_{round_index:04d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        arrays = stack_records(records)
        atomic_savez(round_dir / f"rank_{rank:02d}.npz", **arrays)
        atomic_write_json(
            round_dir / f"rank_{rank:02d}.json",
            {
                "format": FORMAT,
                "rank": rank,
                "round": round_index,
                "sample_count": len(records),
                "wall_s": collection_wall,
                "flow_wall_s": flow_wall,
                "records": [
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"current", "score_gradient_flow", "gradient_details"}
                    }
                    for row in records
                ],
                "finished_unix_s": time.time(),
            },
        )
        dist.barrier()
        if rank == 0:
            combined = combine_round(round_dir)
            all_records = []
            for worker_rank in range(WORLD_SIZE):
                all_records.extend(load_json(round_dir / f"rank_{worker_rank:02d}.json")["records"])
            collection_summary = summarize_collection(
                combined, all_records, round_index=round_index, round_dir=round_dir
            )
            replay_pool = append_replay_pool(
                load_replay_pool(args.run_root / "replay_pool.npz"), combined
            )
            atomic_savez(args.run_root / "replay_pool.npz", **replay_pool)
            atomic_write_json(
                args.run_root / "replay_pool.json",
                {
                    "format": FORMAT,
                    "capacity": REPLAY_CAPACITY,
                    "size": len(replay_pool["current"]),
                    "source_round": round_index,
                    "sampling": "uniform with replacement",
                    "updated_unix_s": time.time(),
                },
            )
            beta = manifest["strategy"].get("beta")
        else:
            collection_summary = None
            replay_pool = None
            beta = None
        dist.barrier()
        replay_pool = load_replay_pool(args.run_root / "replay_pool.npz")
        if replay_pool is None:
            raise RuntimeError("rank 0 did not publish a replay pool")
        beta_holder = torch.tensor(
            float(beta) if beta is not None else -1.0, dtype=torch.float32, device=device
        )
        dist.broadcast(beta_holder, src=0)
        beta = float(beta_holder.item())
        if beta < 0.0:
            beta, beta_calibration = calibrate_beta(
                model=train_model,
                normalizer=normalizer,
                current=arrays["current"],
                valid=arrays["valid"],
                gradient_flow=arrays["score_gradient_flow"],
                gradient_ok=arrays["gradient_ok"],
                rank=rank,
                device=device,
            )
            beta_holder = torch.tensor(beta, dtype=torch.float32, device=device)
            dist.broadcast(beta_holder, src=0)
            beta = float(beta_holder.item())
            if rank == 0:
                manifest["strategy"]["beta"] = beta
                manifest["strategy"]["beta_calibration_result"] = beta_calibration
                atomic_write_json(args.run_root / "manifest.json", manifest)
        train_summary = train_round(
            model=train_model,
            ema_model=ema_model,
            optimizer=optimizer,
            normalizer=normalizer,
            replay_pool=replay_pool,
            rank=rank,
            device=device,
            beta=beta,
            valid_weighting=manifest["strategy"].get("valid_score_weighting"),
            round_index=round_index,
        )
        dist.barrier()
        if rank == 0:
            next_path = args.run_root / "checkpoints" / f"round_{round_index + 1:04d}.pt"
            save_online_checkpoint(
                next_path,
                model=base_model,
                ema=ema_model,
                optimizer=optimizer,
                normalizer=normalizer,
                checkpoint=checkpoint,
                outer_round=round_index + 1,
                code_commit=manifest["repository"]["commit"],
            )
            atomic_write_json(
                round_dir / "training_summary.json",
                {
                    "format": FORMAT,
                    "stage": "round_trained",
                    "round": round_index,
                    "beta": beta,
                    "training": train_summary,
                    "collection": collection_summary,
                    "next_checkpoint": str(next_path.relative_to(args.run_root)),
                    "finished_unix_s": time.time(),
                },
            )
            atomic_write_json(
                args.run_root / "progress.json",
                {
                    "format": FORMAT,
                    "stage": "round_trained",
                    "completed_round": round_index,
                    "next_round": round_index + 1,
                    "beta": beta,
                    "valid_rate": collection_summary["valid_rate"],
                    "gradient_ok_rate_valid": collection_summary["gradient_ok_rate_valid"],
                    "initial_score_median_all": collection_summary["initial"]["score_median_all"],
                    "flow_train_wall_s": train_summary["wall_s"],
                    "replay_pool_size": train_summary["pool_size"],
                    "flow_updates": train_summary["updates"],
                    "updated_unix_s": time.time(),
                },
            )
        dist.barrier()
        round_index += 1
    dist.destroy_process_group()
    if rank == 0:
        print(json.dumps({"event": "score_gradient_run_stopped", "rounds": round_index}))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Two-GPU first-order score-gradient Flow RL.")
    commands = value.add_subparsers(dest="command", required=True)
    prepare_command = commands.add_parser("prepare")
    prepare_command.add_argument("--run-root", type=Path, required=True)
    prepare_command.add_argument("--q0-checkpoint", type=Path, required=True)
    prepare_command.add_argument("--expected-q0-sha", required=True)
    prepare_command.add_argument("--optimizer-checkpoint", type=Path, required=True)
    prepare_command.add_argument("--expected-optimizer-sha", required=True)
    prepare_command.add_argument("--score-lib", type=Path, required=True)
    prepare_command.add_argument("--score-library-manifest", type=Path, required=True)
    prepare_command.add_argument("--expected-score-lib-sha", required=True)
    prepare_command.add_argument("--expected-commit", required=True)
    prepare_command.add_argument("--reference-manifest", type=Path)
    prepare_command.add_argument("--expected-reference-sha")
    prepare_command.set_defaults(func=prepare)
    run_command = commands.add_parser("run")
    run_command.add_argument("--run-root", type=Path, required=True)
    run_command.add_argument("--max-wall-s", type=float, default=82800.0)
    run_command.add_argument("--reserve-s", type=float, default=3600.0)
    run_command.add_argument("--max-rounds", type=int, default=None)
    run_command.set_defaults(func=run)
    return value


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
