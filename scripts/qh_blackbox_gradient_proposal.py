from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_PYTHON = REPO_ROOT / "gpu_backend" / "python"
for path in (REPO_ROOT, GPU_PYTHON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.qh_blackbox_gradient_reference import (
    branch_fingerprint,
    compact_result,
    file_sha256,
    load_rows,
    parse_floats,
    write_json,
)


METHODS = ("g2", "g3")


def parse_validation(value: str) -> tuple[str, Path]:
    try:
        center_id, raw_path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("validation must use CENTER_ID=PATH") from exc
    path = Path(raw_path).expanduser().resolve()
    if not (path / "summary.json").is_file() or not (path / "gradients.npz").is_file():
        raise argparse.ArgumentTypeError(f"invalid latent validation directory: {path}")
    return center_id, path


def append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, separators=(",", ":"), allow_nan=True) + "\n")
        stream.flush()


def prepare(args: argparse.Namespace) -> None:
    from scripts.optimize_flow_prior_zo_adam import decode_noise_rk4, load_flow_checkpoint

    protected = ("manifest.json", "cases.jsonl", "raw_tokens.npy")
    if any((args.output_dir / name).exists() for name in protected):
        raise FileExistsError(f"refusing to overwrite prepared data in {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference_manifest = json.loads(
        (args.reference_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if reference_manifest.get("state") != "complete":
        raise RuntimeError("reference directory is not complete")
    banks = np.load(args.reference_dir / "latent_banks.npz")
    centers = np.asarray(banks["centers"], dtype=np.float32)
    center_by_id = {
        center["center_id"]: (index, center)
        for index, center in enumerate(reference_manifest["centers"])
    }

    validations: dict[str, dict[str, Any]] = {}
    directions: dict[tuple[str, str], np.ndarray] = {}
    for center_id, path in args.validation:
        if center_id in validations:
            raise ValueError(f"duplicate validation for {center_id}")
        if center_id not in center_by_id:
            raise ValueError(f"validation center not in reference: {center_id}")
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        if summary["center_id"] != center_id:
            raise ValueError(f"validation center mismatch for {path}")
        gradient_data = np.load(path / "gradients.npz")
        method_norms = {}
        for method in METHODS:
            gradient = np.asarray(gradient_data[f"{method}_latent"], dtype=np.float64).reshape(-1)
            rms = float(np.sqrt(np.mean(gradient * gradient)))
            if not math.isfinite(rms) or rms <= 0.0:
                raise ValueError(f"invalid {method} gradient RMS for {center_id}: {rms}")
            directions[(center_id, method)] = (gradient / rms).astype(np.float32)
            method_norms[method] = rms
        validations[center_id] = {
            "path": str(path),
            "summary_sha256": file_sha256(path / "summary.json"),
            "gradients_sha256": file_sha256(path / "gradients.npz"),
            "gradient_rms": method_norms,
            "decoded_center_relative_l2": float(summary["decoded_center_relative_l2"]),
        }
    if set(validations) != set(center_by_id):
        missing = sorted(set(center_by_id) - set(validations))
        raise ValueError(f"missing validations for centers: {missing}")

    cases: list[dict[str, Any]] = []
    latent_batches: dict[str, list[np.ndarray]] = {}
    for center_id, (center_index, center) in center_by_id.items():
        center_noise = centers[center_index].reshape(-1)
        for method in METHODS:
            direction = directions[(center_id, method)]
            for step in args.steps:
                for sign in (-1, 1):
                    cases.append(
                        {
                            "case_id": len(cases),
                            "center_id": center_id,
                            "center_index": int(center_index),
                            "nfp": int(center["nfp"]),
                            "n_coils": int(center["n_coils"]),
                            "method": method,
                            "step": float(step),
                            "sign": int(sign),
                        }
                    )
                    latent_batches.setdefault(center_id, []).append(
                        (center_noise + sign * step * direction).reshape(centers.shape[1:])
                    )

    import torch

    model, normalizer, checkpoint = load_flow_checkpoint(args.checkpoint, torch.device(args.device))
    raw_tmp = args.output_dir / "raw_tokens.npy.tmp"
    raw = np.lib.format.open_memmap(
        raw_tmp,
        mode="w+",
        dtype=np.float64,
        shape=(len(cases), centers.shape[1], centers.shape[2]),
    )
    decode_wall_s = 0.0
    case_offset = 0
    per_center_count = len(METHODS) * len(args.steps) * 2
    for center_id, (_, center) in center_by_id.items():
        latent = np.asarray(latent_batches[center_id], dtype=np.float32)
        decoded, wall_s = decode_noise_rk4(
            model,
            normalizer,
            latent,
            nfp=int(center["nfp"]),
            steps=args.rk4_steps,
            device=torch.device(args.device),
        )
        raw[case_offset : case_offset + per_center_count] = decoded
        case_offset += per_center_count
        decode_wall_s += wall_s
    raw.flush()
    del raw
    raw_tmp.replace(args.output_dir / "raw_tokens.npy")
    with (args.output_dir / "cases.jsonl").open("w", encoding="utf-8") as stream:
        for row in cases:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    write_json(
        args.output_dir / "manifest.json",
        {
            "format": "qh_blackbox_gradient_proposal_v1",
            "created_unix_s": time.time(),
            "reference_dir": str(args.reference_dir),
            "reference_manifest_sha256": file_sha256(args.reference_dir / "manifest.json"),
            "score_library": str(args.lib),
            "score_library_sha256": file_sha256(args.lib),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "flow_checkpoint_step": int(checkpoint["step"]),
            "rk4_steps": int(args.rk4_steps),
            "steps": [float(value) for value in args.steps],
            "methods": list(METHODS),
            "case_count": len(cases),
            "decode_wall_s": float(decode_wall_s),
            "validations": validations,
            "state": "prepared",
        },
    )


def score_shard(args: argparse.Namespace) -> None:
    from scripts.optimize_native_score_cem import token_case
    from stellarator_gpu import score_coils_native

    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    if file_sha256(args.lib) != manifest["score_library_sha256"]:
        raise RuntimeError("score library changed after preparation")
    rows = load_rows(args.output_dir / "cases.jsonl")
    raw = np.load(args.output_dir / "raw_tokens.npy", mmap_mode="r")
    output = args.output_dir / f"scores_rank_{args.rank:02d}.jsonl"
    done = {int(row["case_id"]) for row in load_rows(output)} if output.exists() else set()
    selected = [
        row
        for row in rows
        if int(row["case_id"]) % args.world_size == args.rank
        and int(row["case_id"]) not in done
    ]
    started_all = time.perf_counter()
    for ordinal, row in enumerate(selected, start=1):
        case_id = int(row["case_id"])
        case = token_case(raw[case_id], nfp=int(row["nfp"]), target="QH")
        result = None
        error = None
        started = time.perf_counter()
        try:
            coil = case["raw"]
            result = score_coils_native(
                args.lib,
                coil["x"],
                coil["y"],
                coil["z"],
                coil["current"],
                int(row["nfp"]),
                device_id=args.device_id,
                target_helicity=(1, int(row["nfp"])),
            )
        except Exception as exc:
            error = repr(exc)
        append_jsonl(
            output,
            {
                **row,
                "rank": int(args.rank),
                "elapsed_s": float(time.perf_counter() - started),
                "error": error,
                "result": compact_result(result),
            },
        )
        print(f"rank={args.rank} complete={ordinal}/{len(selected)} case_id={case_id}", flush=True)
    write_json(
        args.output_dir / f"scores_rank_{args.rank:02d}.done.json",
        {
            "rank": int(args.rank),
            "new_cases": len(selected),
            "total_done": len(done) + len(selected),
            "wall_s": float(time.perf_counter() - started_all),
        },
    )


def analyze(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    manifest_path = args.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = load_rows(args.output_dir / "cases.jsonl")
    scored: dict[int, dict[str, Any]] = {}
    for path in sorted(args.output_dir.glob("scores_rank_*.jsonl")):
        for row in load_rows(path):
            scored[int(row["case_id"])] = row
    missing = sorted(set(range(len(rows))) - set(scored))
    if missing:
        raise RuntimeError(f"missing {len(missing)} scored proposal cases")

    reference_summary = json.loads(
        (Path(manifest["reference_dir"]) / "summary.json").read_text(encoding="utf-8")
    )
    centers = {
        row["center_id"]: row["reevaluated_center"]
        for row in reference_summary["centers"]
    }
    output_rows = []
    for row in rows:
        evaluated = scored[int(row["case_id"])]
        result = evaluated["result"]
        center = centers[row["center_id"]]
        valid = result is not None and result["status"] == "ok"
        same_branch = valid and tuple(result["branch_fingerprint"]) == tuple(center["branch_fingerprint"])
        output_rows.append(
            {
                **row,
                "center_score": float(center["score"]),
                "status": None if result is None else result["status"],
                "same_branch": bool(same_branch),
                "score": float("nan") if result is None else float(result["score"]),
                "score_delta": float("nan") if result is None else float(result["score"] - center["score"]),
                "elapsed_s": float(evaluated["elapsed_s"]),
                "error": evaluated["error"],
                "components": None if result is None else result["components"],
                "branch_fingerprint": None if result is None else result["branch_fingerprint"],
            }
        )

    paired = []
    for center_id in centers:
        for method in METHODS:
            for step in manifest["steps"]:
                pair = [
                    row
                    for row in output_rows
                    if row["center_id"] == center_id
                    and row["method"] == method
                    and math.isclose(row["step"], step)
                ]
                by_sign = {int(row["sign"]): row for row in pair}
                paired.append(
                    {
                        "center_id": center_id,
                        "method": method,
                        "step": float(step),
                        "minus_status": by_sign[-1]["status"],
                        "plus_status": by_sign[1]["status"],
                        "minus_same_branch": by_sign[-1]["same_branch"],
                        "plus_same_branch": by_sign[1]["same_branch"],
                        "minus_delta": by_sign[-1]["score_delta"],
                        "plus_delta": by_sign[1]["score_delta"],
                        "antithetic_gain": by_sign[1]["score"] - by_sign[-1]["score"],
                    }
                )

    summary = {
        "format": "qh_blackbox_gradient_proposal_summary_v1",
        "case_count": len(output_rows),
        "score_wall_sum_s": float(sum(row["elapsed_s"] for row in output_rows)),
        "score_elapsed_mean_s": float(np.mean([row["elapsed_s"] for row in output_rows])),
        "all_status_ok": all(row["status"] == "ok" for row in output_rows),
        "all_same_branch": all(row["same_branch"] for row in output_rows),
        "pairs": paired,
    }
    write_json(args.output_dir / "proposal_rows.json", output_rows)
    write_json(args.output_dir / "summary.json", summary)
    manifest["state"] = "complete"
    manifest["summary"] = "summary.json"
    write_json(manifest_path, manifest)

    center_ids = list(centers)
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True)
    for axis, center_id in zip(axes.ravel(), center_ids, strict=True):
        for method, marker in (("g2", "o"), ("g3", "s")):
            selected = [row for row in paired if row["center_id"] == center_id and row["method"] == method]
            x = np.asarray([row["step"] for row in selected])
            plus = np.asarray([row["plus_delta"] for row in selected])
            minus = np.asarray([row["minus_delta"] for row in selected])
            axis.plot(x, plus, marker=marker, label=f"{method.upper()} ascent")
            axis.plot(x, minus, marker=marker, linestyle="--", alpha=0.75, label=f"{method.upper()} descent")
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(center_id)
        axis.set_xlabel("latent RMS step")
        axis.set_ylabel("score change")
        axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(args.output_dir / "proposal_score_change.png", dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate true-score proposals along latent G2/G3 gradients.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--validation", action="append", type=parse_validation, default=[])
    parser.add_argument("--steps", type=parse_floats, default=(0.0025, 0.005, 0.01))
    parser.add_argument("--rk4-steps", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.reference_dir = args.reference_dir.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.lib = args.lib.expanduser().resolve()
    modes = sum((args.prepare_only, args.score_only, args.analyze_only))
    if modes != 1:
        parser.error("select exactly one of --prepare-only, --score-only, --analyze-only")
    if args.prepare_only and not args.validation:
        parser.error("--prepare-only requires --validation entries")
    return args


def main() -> None:
    args = parse_args()
    if args.prepare_only:
        prepare(args)
    elif args.score_only:
        score_shard(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
