from __future__ import annotations

import argparse
from collections import Counter
import json
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

from flow_matching.data import CoilNormalizer, file_sha256  # noqa: E402
from flow_matching.optimization import (  # noqa: E402
    CURRENT_NATIVE_SCORE_ABI,
    CURRENT_NATIVE_SCORE_LIBRARY_SHA256,
    QH_OPTIMIZATION_DEFAULTS,
    describe_qh_screening_protocol,
)
from flow_matching.trajectory_dataset import (  # noqa: E402
    COMPONENT_KEYS,
    atomic_savez_compressed,
    atomic_write_json,
    atomic_write_jsonl_gzip,
)
from scripts.optimize_flow_latent import (  # noqa: E402
    score_center,
)
from scripts.flow_runtime import (  # noqa: E402
    decode_noise_rk4,
    load_flow_checkpoint,
    repository_provenance,
    result_score,
    result_valid,
)
from scripts.native_score_runtime import token_case  # noqa: E402


TOKEN_DIM = 100


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decode and globally score random QH flow starts on one GPU."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--nfp", type=int, required=True)
    parser.add_argument("--n-base-coils", type=int, required=True)
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=QH_OPTIMIZATION_DEFAULTS.candidate_count,
    )
    parser.add_argument(
        "--parameter-space",
        choices=("latent", "data"),
        default="latent",
        help=(
            "Draw standard Gaussian Flow latents, or draw independent standard "
            "Gaussians directly in per-coordinate standardized coil space."
        ),
    )
    parser.add_argument(
        "--flow-steps", type=int, default=QH_OPTIMIZATION_DEFAULTS.flow_steps
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iota-degree", type=int, default=3)
    parser.add_argument("--surface-theta-count", type=int, default=128)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.candidate_count < 1 or args.n_base_coils < 1:
        raise ValueError("candidate count and n-base-coils must be positive")
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    torch.cuda.set_device(args.device)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda", args.device)
    rng = np.random.default_rng(args.seed)
    sampled_parameters = rng.standard_normal(
        (args.candidate_count, args.n_base_coils, TOKEN_DIM), dtype=np.float32
    )
    if args.parameter_space == "latent":
        model, normalizer, checkpoint = load_flow_checkpoint(args.checkpoint, device)
        tokens, mapping_wall_s = decode_noise_rk4(
            model,
            normalizer,
            sampled_parameters,
            nfp=args.nfp,
            steps=args.flow_steps,
            device=device,
        )
        optimizer_parameters = sampled_parameters
        clipped_fraction = 0.0
    else:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        normalizer = CoilNormalizer.from_dict(checkpoint["normalizer"])
        started = time.perf_counter()
        tokens = normalizer.inverse(
            sampled_parameters, (args.nfp, args.n_base_coils)
        )
        optimizer_parameters, clipped_fraction = normalizer.transform(
            tokens, (args.nfp, args.n_base_coils)
        )
        mapping_wall_s = time.perf_counter() - started

    rows: list[dict[str, Any]] = []
    score_wall_s = 0.0
    for index, candidate_tokens in enumerate(tokens):
        try:
            result, elapsed = score_center(
                args.lib,
                candidate_tokens,
                nfp=args.nfp,
                score_device=args.device,
                iota_degree=args.iota_degree,
                surface_theta_count=args.surface_theta_count,
                previous_result=None,
            )
            error = None
        except Exception as exc:
            result = {
                "score": 0.0,
                "status": "python_error",
                "components": {key: 0.0 for key in COMPONENT_KEYS},
                "timing": {},
                "diagnostics": {},
            }
            elapsed = 0.0
            error = f"{type(exc).__name__}: {exc}"
        score_wall_s += elapsed
        rows.append(
            {
                "candidate_index": index,
                "score_wall_s": elapsed,
                "native_score": result,
                "error": error,
            }
        )

    valid_indices = [
        index for index, row in enumerate(rows) if result_valid(row["native_score"])
    ]
    selected_index = (
        max(valid_indices, key=lambda index: result_score(rows[index]["native_score"]))
        if valid_indices
        else None
    )
    parameter_key = "latent" if args.parameter_space == "latent" else "normalized_data"
    array_payload = {
        parameter_key: np.asarray(optimizer_parameters, dtype=np.float32),
        "decoded_tokens": np.asarray(tokens, dtype=np.float32),
        "score": np.asarray([result_score(row["native_score"]) for row in rows]),
        "status": np.asarray(
            [row["native_score"].get("status", "missing") for row in rows]
        ),
        "components": np.asarray(
            [
                [
                    float(row["native_score"].get("components", {}).get(key, 0.0))
                    for key in COMPONENT_KEYS
                ]
                for row in rows
            ]
        ),
    }
    if args.parameter_space == "data":
        array_payload["sampled_standard_normal"] = np.asarray(
            sampled_parameters, dtype=np.float32
        )
    arrays_sha = atomic_savez_compressed(
        args.out_dir / "screening_arrays.npz", **array_payload
    )
    results_sha = atomic_write_jsonl_gzip(
        args.out_dir / "screening_native_results.jsonl.gz", rows
    )
    summary = {
        "protocol": describe_qh_screening_protocol(
            candidate_count=args.candidate_count,
            flow_steps=args.flow_steps,
        ),
        "status": "ok" if selected_index is not None else "no_valid_candidate",
        "nfp": args.nfp,
        "n_base_coils": args.n_base_coils,
        "parameter_space": args.parameter_space,
        "candidate_count": args.candidate_count,
        "valid_candidate_count": len(valid_indices),
        "selected_index": selected_index,
        "selected_score": (
            result_score(rows[selected_index]["native_score"])
            if selected_index is not None
            else None
        ),
        "status_counts": dict(
            sorted(Counter(row["native_score"].get("status", "missing") for row in rows).items())
        ),
        "timing": {
            "parameter_mapping_wall_s": mapping_wall_s,
            "decode_wall_s": (
                mapping_wall_s if args.parameter_space == "latent" else 0.0
            ),
            "score_sum_wall_s": score_wall_s,
            "total_wall_s": mapping_wall_s + score_wall_s,
        },
        "provenance": {
            "repository": repository_provenance(REPO_ROOT),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "checkpoint_step": int(checkpoint["step"]),
            "score_library_sha256": file_sha256(args.lib),
            "native_score_abi": CURRENT_NATIVE_SCORE_ABI,
            "validated_default_score_library_sha256": (
                CURRENT_NATIVE_SCORE_LIBRARY_SHA256
            ),
            "flow": (
                {"method": "rk4", "steps": args.flow_steps, "dtype": "fp32"}
                if args.parameter_space == "latent"
                else None
            ),
            "prior": (
                {"kind": "flow_latent_gaussian", "flow_method": "rk4", "steps": args.flow_steps, "dtype": "fp32"}
                if args.parameter_space == "latent"
                else {
                    "kind": "independent_standardized_data_gaussian",
                    "current_projection": "condition-specific L1 and dominant-current sign",
                    "post_projection_clipped_fraction": float(clipped_fraction),
                }
            ),
            "seed": args.seed,
        },
        "files": {
            "screening_arrays.npz": arrays_sha,
            "screening_native_results.jsonl.gz": results_sha,
        },
    }
    if selected_index is not None:
        selected_result = rows[selected_index]["native_score"]
        selected_case = token_case(tokens[selected_index], nfp=args.nfp, target="QH")
        metadata_key = (
            "flow_prior_screening"
            if args.parameter_space == "latent"
            else "data_prior_screening"
        )
        selected_case[metadata_key] = {
            "format": (
                "qh_flow_screened_start_v1"
                if args.parameter_space == "latent"
                else "qh_data_screened_start_v1"
            ),
            (
                "noise"
                if args.parameter_space == "latent"
                else "normalized_coil_tokens"
            ): optimizer_parameters[selected_index].tolist(),
            "candidate_index": selected_index,
            "candidate_count": args.candidate_count,
            "native_score": selected_result,
            "screening_summary": summary,
        }
        atomic_write_json(
            args.out_dir / "selected_start.json", selected_case, allow_nan=True
        )
    atomic_write_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
