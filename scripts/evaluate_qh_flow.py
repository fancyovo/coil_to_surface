from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
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
for path in (REPO_ROOT, GPU_PYTHON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from flow_matching.data import CoilNormalizer, GroupStore, load_raw_groups
from flow_matching.flow import sample_heun
from flow_matching.geometry import curve_metrics, geometry_eligible
from flow_matching.model import CoilFlowTransformer
from scripts.optimize_native_score_cem import token_case


def setup() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.cuda.is_available():
        raise RuntimeError("first-generation native evaluation requires CUDA")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"count": 0}
    return {
        "count": len(array),
        "mean": float(np.mean(array)),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def scalar_geometry(metrics: dict[str, torch.Tensor], index: int) -> dict[str, float | bool]:
    output = {}
    for name, values in metrics.items():
        value = values[index].detach().cpu().item()
        output[name] = bool(value) if name == "finite" else float(value)
    return output


@torch.no_grad()
def generate_rank(
    model: CoilFlowTransformer,
    store: GroupStore,
    normalizer: CoilNormalizer,
    bounds: dict[str, tuple[float, float]],
    *,
    count: int,
    seed: int,
    rank: int,
    world_size: int,
    steps: int,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    condition_rng = np.random.default_rng(seed)
    conditions = condition_rng.choice(len(store.keys), size=count, p=store.probabilities)
    candidate_ids = np.arange(count, dtype=np.int64)[rank::world_size]
    rows = []
    for key_index, key in enumerate(store.keys):
        group_ids = candidate_ids[conditions[candidate_ids] == key_index]
        for start in range(0, len(group_ids), batch_size):
            ids = group_ids[start : start + batch_size]
            generator = torch.Generator(device=device).manual_seed(
                seed + rank * 1000003 + key[0] * 101 + key[1] * 1009 + start
            )
            noise = torch.randn((len(ids), key[1], 100), device=device, generator=generator)
            nfp = torch.full((len(ids),), key[0], dtype=torch.long, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                normalized = sample_heun(model, noise, nfp, steps=steps)
            raw = normalizer.inverse(normalized.float().cpu().numpy(), key)
            metrics = curve_metrics(torch.from_numpy(raw).to(device))
            eligible = geometry_eligible(metrics, bounds)
            for local_index, candidate_id in enumerate(ids):
                rows.append(
                    {
                        "candidate_id": int(candidate_id),
                        "nfp": key[0],
                        "n_coils": key[1],
                        "tokens": raw[local_index].tolist(),
                        "geometry": scalar_geometry(metrics, local_index),
                        "geometry_eligible": bool(eligible[local_index].cpu()),
                    }
                )
    rows.sort(key=lambda row: row["candidate_id"])
    return rows


def score_rank(
    rows: list[dict[str, Any]],
    *,
    lib_path: Path,
    device_id: int,
    output_path: Path,
) -> None:
    from stellarator_gpu import score_coils_native

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored = 0
    started = time.perf_counter()
    with output_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            result = None
            error = None
            elapsed = 0.0
            did_score = bool(row["geometry_eligible"])
            if row["geometry_eligible"]:
                case = token_case(
                    np.asarray(row["tokens"], dtype=np.float64),
                    nfp=row["nfp"],
                    target="QH",
                    metadata={"flow_candidate_id": row["candidate_id"]},
                )
                score_started = time.perf_counter()
                try:
                    raw = case["raw"]
                    result = score_coils_native(
                        lib_path,
                        raw["x"],
                        raw["y"],
                        raw["z"],
                        raw["current"],
                        row["nfp"],
                        device_id=device_id,
                        target_helicity=(1, row["nfp"]),
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                elapsed = time.perf_counter() - score_started
                scored += 1
            row["native_score"] = result
            row["score_error"] = error
            row["score_wall_s"] = elapsed
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=True) + "\n")
            stream.flush()
            if did_score and scored % 10 == 0:
                print(
                    json.dumps(
                        {
                            "event": "evaluation_progress",
                            "rank": device_id,
                            "scored": scored,
                            "rows": len(rows),
                            "elapsed_s": time.perf_counter() - started,
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )


def load_rows(output_dir: Path, world_size: int) -> list[dict[str, Any]]:
    rows = []
    for rank in range(world_size):
        with (output_dir / f"rank_{rank:02d}.jsonl").open("r", encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    return sorted(rows, key=lambda row: row["candidate_id"])


def iota_of(native: dict[str, Any]) -> float:
    diagnostics = native["diagnostics"]
    return 0.5 * (float(diagnostics["iota_min"]) + float(diagnostics["iota_max"]))


def analyze(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    good_score: float,
    good_iota: float,
    good_size: float,
) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    group_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scores_all = []
    scores_ok = []
    iotas = []
    qs_errors = []
    sizes = []
    good = []
    for row in rows:
        native = row["native_score"]
        status = "geometry_rejected" if not row["geometry_eligible"] else "error" if native is None else str(native["status"])
        status_counts[status] += 1
        score = float(native["score"]) if native is not None else 0.0
        scores_all.append(score)
        row["analysis_status"] = status
        row["analysis_score"] = score
        group_rows[f"nfp{row['nfp']}_nc{row['n_coils']}"].append(row)
        if status != "ok":
            continue
        diagnostics = native["diagnostics"]
        iota = iota_of(native)
        size = float(diagnostics["surface_inverse_aspect_ratio"])
        valid = float(diagnostics["volume_valid_fraction"])
        qs_error = float(diagnostics["qs_global_error"])
        scores_ok.append(score)
        iotas.append(iota)
        sizes.append(size)
        qs_errors.append(qs_error)
        if score >= good_score and abs(iota) >= good_iota and size >= good_size and valid >= 0.95:
            good.append(row)
    group_summary = {}
    for key, values in sorted(group_rows.items()):
        group_good = [row for row in good if f"nfp{row['nfp']}_nc{row['n_coils']}" == key]
        group_summary[key] = {
            "count": len(values),
            "geometry_eligible_rate": float(np.mean([row["geometry_eligible"] for row in values])),
            "ok_rate": float(np.mean([row["analysis_status"] == "ok" for row in values])),
            "score_mean_all": float(np.mean([row["analysis_score"] for row in values])),
            "good_count": len(group_good),
            "good_rate": len(group_good) / len(values),
        }
    top = sorted(
        [row for row in rows if row["native_score"] is not None],
        key=lambda row: row["analysis_score"],
        reverse=True,
    )[:50]
    top_dir = output_dir / "top_cases"
    top_dir.mkdir(exist_ok=True)
    for rank, row in enumerate(top, start=1):
        case = token_case(
            np.asarray(row["tokens"], dtype=np.float64),
            nfp=row["nfp"],
            target="QH",
            metadata={"flow_candidate_id": row["candidate_id"], "evaluation_rank": rank},
        )
        case["flow_evaluation"] = {
            "geometry": row["geometry"],
            "native_score": row["native_score"],
        }
        (top_dir / f"rank_{rank:03d}_id_{row['candidate_id']:06d}.json").write_text(
            json.dumps(case, indent=2, allow_nan=True) + "\n", encoding="utf-8"
        )
    summary = {
        "count": len(rows),
        "geometry_eligible_count": sum(row["geometry_eligible"] for row in rows),
        "geometry_eligible_rate": float(np.mean([row["geometry_eligible"] for row in rows])),
        "status_counts": dict(status_counts),
        "ok_rate": status_counts["ok"] / len(rows),
        "score_all": distribution(scores_all),
        "score_ok": distribution(scores_ok),
        "abs_iota_ok": distribution([abs(value) for value in iotas]),
        "qs_global_error_ok": distribution(qs_errors),
        "inverse_aspect_ratio_ok": distribution(sizes),
        "good_definition": {
            "score_min": good_score,
            "abs_iota_min": good_iota,
            "inverse_aspect_ratio_min": good_size,
            "volume_valid_fraction_min": 0.95,
        },
        "good_count": len(good),
        "good_rate": len(good) / len(rows),
        "group_summary": group_summary,
        "top_case_dir": str(top_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    plot_summary(rows, summary, output_dir / "evaluation_summary.png")
    return summary


def plot_summary(rows: list[dict[str, Any]], summary: dict[str, Any], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [row for row in rows if row["analysis_status"] == "ok"]
    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    axes[0, 0].hist([row["analysis_score"] for row in rows], bins=40, color="#26667f")
    axes[0, 0].set(title="Unconditional generated score", xlabel="score", ylabel="count")
    if ok:
        axes[0, 1].scatter(
            [abs(iota_of(row["native_score"])) for row in ok],
            [row["analysis_score"] for row in ok],
            s=8,
            alpha=0.45,
        )
        axes[0, 1].set(title="Score vs rotational transform", xlabel="|iota|", ylabel="score")
        axes[1, 0].scatter(
            [row["native_score"]["diagnostics"]["surface_inverse_aspect_ratio"] for row in ok],
            [row["native_score"]["diagnostics"]["qs_global_error"] for row in ok],
            c=[row["analysis_score"] for row in ok],
            s=8,
            alpha=0.55,
        )
        axes[1, 0].set(
            title="QS residual vs surface size",
            xlabel="inverse aspect ratio",
            ylabel="differential QS residual",
            yscale="log",
        )
    groups = list(summary["group_summary"])
    rates = [summary["group_summary"][key]["good_rate"] for key in groups]
    axes[1, 1].bar(np.arange(len(groups)), rates, color="#a04a36")
    axes[1, 1].set_xticks(np.arange(len(groups)), groups, rotation=90, fontsize=7)
    axes[1, 1].set(title="Good-QH fraction by condition", ylabel="fraction", ylim=(0, 1))
    figure.savefig(output, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and natively evaluate first-generation QH flow samples.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8192)
    parser.add_argument("--sample-steps", type=int, default=32)
    parser.add_argument("--sample-batch", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--good-score", type=float, default=66.8)
    parser.add_argument("--good-iota", type=float, default=1.0)
    parser.add_argument("--good-size", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, device = setup()
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    if world_size > 1:
        dist.barrier()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = CoilFlowTransformer(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["ema"])
    model.eval()
    normalizer = CoilNormalizer.from_dict(checkpoint["normalizer"])
    train_raw, _ = load_raw_groups(args.data_dir, "train")
    store = GroupStore(train_raw, normalizer)
    bounds = {
        str(name): (float(values[0]), float(values[1]))
        for name, values in checkpoint["geometry_reference_bounds"].items()
    }
    rows = generate_rank(
        model,
        store,
        normalizer,
        bounds,
        count=args.count,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
        steps=args.sample_steps,
        batch_size=args.sample_batch,
        device=device,
    )
    score_rank(
        rows,
        lib_path=args.lib,
        device_id=local_rank,
        output_path=args.output_dir / f"rank_{rank:02d}.jsonl",
    )
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        summary = analyze(
            load_rows(args.output_dir, world_size),
            output_dir=args.output_dir,
            good_score=args.good_score,
            good_iota=args.good_iota,
            good_size=args.good_size,
        )
        (args.output_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "args": vars(args),
                    "world_size": world_size,
                    "checkpoint_step": checkpoint["step"],
                    "model_config": checkpoint["model_config"],
                },
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, separators=(",", ":")), flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
