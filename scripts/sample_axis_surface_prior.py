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


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "gpu_backend" / "python"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


DIAGNOSTIC_NAMES = (
    "axis_R",
    "axis_Z",
    "axis_residual",
    "axis_topology_trace",
    "axis_topology_det",
    "surface_level",
    "surface_inverse_aspect_ratio",
    "surface_one_period_drift_relative_p95",
    "surface_drift_relative_p95",
    "iota_min",
    "iota_max",
    "qs_global_error",
    "qs_qa_global_error",
    "qs_qp_global_error",
    "score_qh_helicity_advantage",
    "score_qh_helicity_quality",
    "coil_length_mean",
    "coil_curvature_p95",
    "coil_curvature_max",
    "coil_min_intercoil_distance",
    "coil_min_axis_distance",
    "coil_high_mode_energy_fraction",
    "coil_current_abs_max_a",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    diagnostics = dict(result.get("diagnostics") or {})
    return {
        "score": float(result["score"]),
        "status": str(result["status"]),
        "components": {str(name): float(value) for name, value in dict(result.get("components") or {}).items()},
        "diagnostics": {name: _json_value(diagnostics.get(name, float("nan"))) for name in DIAGNOSTIC_NAMES},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and ABI-11-score the axis/surface/contour prior.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--total-count", type=int, default=15600)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--target", choices=("QH",), default="QH")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    if args.total_count <= 0:
        raise ValueError("total-count must be positive")

    from flow_matching.axis_surface_prior import sample_axis_surface_prior, supported_conditions
    from scripts.native_score_runtime import token_case
    from stellarator_gpu import score_coils_native

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / f"shard_{args.shard_index:02d}.jsonl"
    done_path = args.output_dir / f"shard_{args.shard_index:02d}.done.json"
    if done_path.exists():
        raise FileExistsError(done_path)
    existing: set[int] = set()
    if rows_path.exists():
        with rows_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    existing.add(int(json.loads(line)["case_id"]))

    conditions = supported_conditions()
    families = tuple(("near_circular", "balanced", "helical"))
    case_ids = list(range(args.shard_index, args.total_count, args.shard_count))
    started = time.perf_counter()
    scored = 0
    errors = 0
    with rows_path.open("a", encoding="utf-8", buffering=1) as stream:
        for case_id in case_ids:
            if case_id in existing:
                continue
            nfp, n_base_coils = conditions[case_id % len(conditions)]
            family = families[(case_id // len(conditions)) % len(families)]
            generated = sample_axis_surface_prior(
                seed=args.seed + case_id,
                nfp=nfp,
                n_base_coils=n_base_coils,
                family=family,
            )
            case = token_case(
                generated.tokens,
                nfp=nfp,
                target=args.target,
                metadata={"case_id": case_id, **generated.metadata},
            )
            score_started = time.perf_counter()
            error = None
            native = None
            try:
                target_helicity = (1, nfp)
                result = score_coils_native(
                    args.lib,
                    case["raw"]["x"],
                    case["raw"]["y"],
                    case["raw"]["z"],
                    case["raw"]["current"],
                    nfp,
                    device_id=0,
                    target_helicity=target_helicity,
                )
                native = compact_result(result)
            except Exception as exc:
                error = repr(exc)
                errors += 1
            row = {
                "case_id": case_id,
                "nfp": nfp,
                "n_base_coils": n_base_coils,
                "family": family,
                "tokens": generated.tokens.tolist(),
                "reference_axis": generated.reference_axis[::8].tolist(),
                "generator": generated.metadata,
                "native": native,
                "score_wall_s": time.perf_counter() - score_started,
                "error": error,
            }
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=True) + "\n")
            scored += 1
            if scored % args.progress_every == 0:
                elapsed = time.perf_counter() - started
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "shard": args.shard_index,
                            "new_count": scored,
                            "total_for_shard": len(case_ids),
                            "errors": errors,
                            "elapsed_s": elapsed,
                            "mean_s": elapsed / scored,
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    done = {
        "format": "axis_surface_prior_score_shard_v1",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "total_count": args.total_count,
        "case_count": len(case_ids),
        "preexisting_count": len(existing),
        "new_count": scored,
        "errors": errors,
        "elapsed_s": elapsed,
        "score_library": str(args.lib.resolve()),
        "score_library_sha256": file_sha256(args.lib),
        "git_commit": os.environ.get("EXPECTED_COMMIT"),
        "target": args.target,
        "seed": args.seed,
    }
    done_path.write_text(json.dumps(done, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", **done}, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
