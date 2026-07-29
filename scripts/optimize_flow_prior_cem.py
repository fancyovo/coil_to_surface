from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "gpu_backend" / "python"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from flow_matching.data import CoilNormalizer
from flow_matching.flow import sample_heun
from flow_matching.model import CoilFlowTransformer
from scripts.optimize_native_score_cem import (
    NativeScorePool,
    append_jsonl,
    compact_score_diagnostics,
    file_sha256,
    token_case,
    update_distribution,
    write_json,
)


TOKEN_DIM = 100


def load_flow_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[CoilFlowTransformer, CoilNormalizer, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {"ema", "model_config", "normalizer", "step"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"flow checkpoint is missing keys: {sorted(missing)}")
    model = CoilFlowTransformer(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["ema"])
    model.eval()
    return model, CoilNormalizer.from_dict(checkpoint["normalizer"]), checkpoint


@torch.inference_mode()
def decode_flow_prior(
    model: CoilFlowTransformer,
    normalizer: CoilNormalizer,
    noise: np.ndarray,
    *,
    nfp: int,
    steps: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    noise = np.asarray(noise, dtype=np.float32)
    if noise.ndim != 3 or noise.shape[-1] != TOKEN_DIM:
        raise ValueError(f"noise must have shape (batch, coils, {TOKEN_DIM})")
    if batch_size < 1:
        raise ValueError("flow batch size must be positive")
    key = (int(nfp), int(noise.shape[1]))
    decoded = []
    for start in range(0, len(noise), batch_size):
        stop = min(start + batch_size, len(noise))
        noise_batch = torch.from_numpy(noise[start:stop]).to(device=device)
        nfp_batch = torch.full(
            (stop - start,), int(nfp), dtype=torch.long, device=device
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            normalized = sample_heun(model, noise_batch, nfp_batch, steps=steps)
        decoded.append(
            normalizer.inverse(normalized.float().cpu().numpy(), key)
        )
    return np.concatenate(decoded, axis=0).astype(np.float64, copy=False)


def should_stop_for_wall_budget(
    elapsed_s: float,
    generation_wall_s: list[float],
    max_wall_s: float,
) -> bool:
    if max_wall_s <= 0.0 or not generation_wall_s:
        return False
    recent = generation_wall_s[-min(5, len(generation_wall_s)) :]
    projected_next = 1.15 * float(np.mean(recent))
    return elapsed_s + projected_next >= max_wall_s


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize flow-matching prior noise with diagonal CEM and native score."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--lib",
        type=Path,
        default=REPO_ROOT / "gpu_backend" / "build_native_score" / "libstellarator_gpu.so",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target", choices=("QA", "QH"), default="QH")
    parser.add_argument("--nfp", type=int, default=4)
    parser.add_argument("--n-base-coils", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument("--popsize", type=int, default=160)
    parser.add_argument("--elite", type=int, default=40)
    parser.add_argument("--initial-sigma", type=float, default=1.0)
    parser.add_argument("--min-sigma", type=float, default=0.03)
    parser.add_argument("--max-sigma", type=float, default=2.0)
    parser.add_argument("--smoothing", type=float, default=0.55)
    parser.add_argument("--noise-limit", type=float, default=6.0)
    parser.add_argument("--flow-steps", type=int, default=32)
    parser.add_argument("--flow-batch-size", type=int, default=512)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch-timeout-s", type=float, default=1800.0)
    parser.add_argument("--max-wall-s", type=float, default=32400.0)
    parser.add_argument("--seed", type=int, default=2026073001)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("flow-prior CEM requires CUDA")
    if args.nfp < 1 or args.n_base_coils < 1:
        raise ValueError("nfp and n-base-coils must be positive")
    if args.flow_steps < 1:
        raise ValueError("flow-steps must be positive")
    if not 1 <= args.elite <= args.popsize:
        raise ValueError("elite must be in [1, popsize]")
    if not 0.0 <= args.smoothing < 1.0:
        raise ValueError("smoothing must be in [0, 1)")
    if not 0.0 < args.min_sigma <= args.initial_sigma <= args.max_sigma:
        raise ValueError("sigma bounds must satisfy 0 < min <= initial <= max")
    gpu_ids = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("at least one native-score GPU is required")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = args.out_dir / "candidates.jsonl"
    if candidates_path.exists():
        raise FileExistsError(f"refusing to overwrite existing run {candidates_path}")
    if not args.checkpoint.is_file() or not args.lib.is_file():
        raise FileNotFoundError("checkpoint and native score library must exist")

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    model, normalizer, checkpoint = load_flow_checkpoint(
        args.checkpoint, device=device
    )
    normalizer_key = f"{args.nfp}:{args.n_base_coils}"
    if normalizer_key not in normalizer.current_l1_a:
        raise ValueError(
            f"condition {normalizer_key} is absent from checkpoint normalizer"
        )

    rng = np.random.default_rng(args.seed)
    shape = (args.n_base_coils, TOKEN_DIM)
    mean = np.zeros(shape, dtype=np.float32)
    sigma = np.full(shape, args.initial_sigma, dtype=np.float32)
    checkpoint_sha256 = file_sha256(args.checkpoint)
    manifest = {
        "algorithm": "diagonal_cem_over_flow_prior_noise",
        "target": args.target,
        "nfp": args.nfp,
        "n_base_coils": args.n_base_coils,
        "noise_shape": list(shape),
        "prior": "standard_normal",
        "seed": args.seed,
        "iterations": args.iterations,
        "popsize": args.popsize,
        "elite": args.elite,
        "initial_sigma": args.initial_sigma,
        "min_sigma": args.min_sigma,
        "max_sigma": args.max_sigma,
        "smoothing": args.smoothing,
        "noise_limit": args.noise_limit,
        "flow_steps": args.flow_steps,
        "flow_batch_size": args.flow_batch_size,
        "max_wall_s": args.max_wall_s,
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_state": "ema",
        "model_config": checkpoint["model_config"],
        "normalizer_current_l1_a": normalizer.current_l1_a[normalizer_key],
        "native_lib_path": str(args.lib.resolve()),
        "native_lib_sha256": file_sha256(args.lib),
        "gpu_ids": gpu_ids,
    }
    write_json(args.out_dir / "manifest.json", manifest)

    best_score = float("-inf")
    best_noise: np.ndarray | None = None
    best_case: dict[str, Any] | None = None
    best_result: dict[str, Any] | None = None
    best_generation = 0
    best_candidate = -1
    generation_rows: list[dict[str, Any]] = []
    generation_walls: list[float] = []
    started = time.perf_counter()
    stop_reason = "completed_iterations"

    with NativeScorePool(args.lib, gpu_ids) as pool:
        for generation in range(1, args.iterations + 1):
            elapsed_before = time.perf_counter() - started
            if generation > 1 and should_stop_for_wall_budget(
                elapsed_before, generation_walls, args.max_wall_s
            ):
                stop_reason = "wall_budget"
                break

            generation_started = time.perf_counter()
            eps = rng.standard_normal((args.popsize, *shape), dtype=np.float32)
            population = mean[None] + sigma[None] * eps
            population[0] = mean
            if best_noise is not None and args.popsize > 1:
                population[1] = best_noise
            population = np.clip(
                population, -args.noise_limit, args.noise_limit
            ).astype(np.float32, copy=False)

            decode_started = time.perf_counter()
            tokens = decode_flow_prior(
                model,
                normalizer,
                population,
                nfp=args.nfp,
                steps=args.flow_steps,
                batch_size=args.flow_batch_size,
                device=device,
            )
            torch.cuda.synchronize(device)
            decode_wall_s = time.perf_counter() - decode_started
            if not np.all(np.isfinite(tokens)):
                raise RuntimeError(f"generation {generation} flow output is non-finite")

            cases = [
                token_case(
                    tokens[index],
                    nfp=args.nfp,
                    target=args.target,
                    metadata={
                        "flow_prior_cem_generation": generation,
                        "flow_prior_cem_candidate": index,
                        "flow_prior_cem_seed": args.seed,
                        "flow_checkpoint_step": int(checkpoint["step"]),
                    },
                )
                for index in range(args.popsize)
            ]
            score_started = time.perf_counter()
            evaluated = pool.map(
                cases, target=args.target, timeout_s=args.batch_timeout_s
            )
            score_wall_s = time.perf_counter() - score_started
            scores = np.full(args.popsize, -np.inf, dtype=np.float64)
            statuses: dict[str, int] = {}
            errors = []
            for index, (result, score_elapsed_s, error) in enumerate(evaluated):
                if error is not None or result is None:
                    errors.append({"candidate": index, "error": error})
                    append_jsonl(
                        candidates_path,
                        {
                            "generation": generation,
                            "candidate": index,
                            "score_elapsed_s": score_elapsed_s,
                            "error": error,
                        },
                    )
                    continue
                score = float(result["score"])
                status = str(result["status"])
                if math.isfinite(score):
                    scores[index] = score
                statuses[status] = statuses.get(status, 0) + 1
                append_jsonl(
                    candidates_path,
                    {
                        "generation": generation,
                        "candidate": index,
                        "score": score,
                        "status": status,
                        "score_elapsed_s": score_elapsed_s,
                        "noise": population[index].tolist(),
                        "native_score": compact_score_diagnostics(result),
                    },
                )
                if math.isfinite(score) and score > best_score:
                    best_score = score
                    best_noise = population[index].copy()
                    best_case = cases[index]
                    best_result = result
                    best_generation = generation
                    best_candidate = index

            finite = np.isfinite(scores)
            if np.count_nonzero(finite) < args.elite:
                raise RuntimeError(
                    f"generation {generation} has only "
                    f"{np.count_nonzero(finite)} finite scores"
                )
            elite_indices = np.argsort(scores)[::-1][: args.elite]
            generation_best_index = int(elite_indices[0])
            generation_best_result = evaluated[generation_best_index][0]
            if generation_best_result is None:
                raise RuntimeError("generation best candidate has no score result")
            mean, sigma = update_distribution(
                mean,
                sigma,
                population[elite_indices],
                smoothing=args.smoothing,
                min_sigma=args.min_sigma,
                max_sigma=args.max_sigma,
                latent_limit=args.noise_limit,
            )

            generation_wall_s = time.perf_counter() - generation_started
            generation_walls.append(generation_wall_s)
            elapsed_s = time.perf_counter() - started
            row = {
                "generation": generation,
                "score_mean": float(np.mean(scores[finite])),
                "score_median": float(np.median(scores[finite])),
                "score_max": float(np.max(scores[finite])),
                "best_score": best_score,
                "best_generation": best_generation,
                "best_candidate": best_candidate,
                "generation_best_candidate": generation_best_index,
                "generation_best": compact_score_diagnostics(
                    generation_best_result
                ),
                "sigma_mean": float(np.mean(sigma)),
                "sigma_min": float(np.min(sigma)),
                "sigma_max": float(np.max(sigma)),
                "mean_rms": float(np.sqrt(np.mean(mean * mean))),
                "statuses": statuses,
                "errors": errors,
                "decode_wall_s": decode_wall_s,
                "score_wall_s": score_wall_s,
                "wall_s": generation_wall_s,
                "total_wall_s": elapsed_s,
            }
            generation_rows.append(row)
            write_json(
                args.out_dir / "progress.json",
                {"manifest": manifest, "generations": generation_rows},
            )
            np.savez_compressed(
                args.out_dir / "state_latest.npz",
                mean=mean,
                sigma=sigma,
                best_noise=best_noise,
                generation=np.asarray(generation, dtype=np.int64),
                rng_state=np.asarray(json.dumps(rng.bit_generator.state)),
            )
            print(json.dumps(row, separators=(",", ":")), flush=True)

    if best_case is None or best_result is None or best_noise is None:
        raise RuntimeError("flow-prior CEM did not produce a valid candidate")
    best_case["flow_prior_cem"] = {
        "target": args.target,
        "seed": args.seed,
        "generation": best_generation,
        "candidate": best_candidate,
        "best_score": best_score,
        "noise": best_noise.tolist(),
        "native_score": best_result,
        "manifest": manifest,
    }
    write_json(args.out_dir / "best.json", best_case)
    summary = {
        "manifest": manifest,
        "stop_reason": stop_reason,
        "completed_generations": len(generation_rows),
        "best_score": best_score,
        "best_generation": best_generation,
        "best_candidate": best_candidate,
        "best_status": best_result["status"],
        "best_components": best_result["components"],
        "best_diagnostics": best_result["diagnostics"],
        "generations": generation_rows,
        "total_wall_s": time.perf_counter() - started,
        "best_case": str((args.out_dir / "best.json").resolve()),
    }
    write_json(args.out_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "stop_reason": stop_reason,
                "completed_generations": len(generation_rows),
                "best_score": best_score,
                "best_case": summary["best_case"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
