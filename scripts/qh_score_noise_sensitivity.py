from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
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

from flow_matching.data import CoilNormalizer, load_raw_groups
from scripts.optimize_native_score_cem import token_case


DEFAULT_IDS = (1446077, 1826200, 2419096)
DEFAULT_SIGMAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def parse_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in value.split(",") if item.strip())
    if any(item <= 0.0 for item in values):
        raise ValueError("noise sigmas must be positive")
    return values


def curve_positions(tokens: np.ndarray, samples: int = 128) -> np.ndarray:
    values = np.asarray(tokens, dtype=np.float64)
    coefficients = values[..., :99].reshape(*values.shape[:-1], 3, 33)
    t = np.arange(samples, dtype=np.float64) / samples
    modes = np.arange(1, 17, dtype=np.float64)
    angles = 2.0 * np.pi * modes[:, None] * t[None]
    position = coefficients[..., 0, None].copy()
    position = position + np.einsum(
        "...cm,mt->...ct", coefficients[..., 1::2], np.sin(angles)
    )
    position = position + np.einsum(
        "...cm,mt->...ct", coefficients[..., 2::2], np.cos(angles)
    )
    return np.moveaxis(position, -2, -1)


def perturbation_metrics(tokens: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    displacement = np.linalg.norm(
        curve_positions(tokens) - curve_positions(reference), axis=-1
    ).reshape(-1)
    coefficient_delta = np.asarray(tokens[..., :99], dtype=np.float64) - np.asarray(
        reference[..., :99], dtype=np.float64
    )
    current_delta = np.asarray(tokens[..., -1], dtype=np.float64) - np.asarray(
        reference[..., -1], dtype=np.float64
    )
    return {
        "position_delta_rms_m": float(np.sqrt(np.mean(displacement**2))),
        "position_delta_p95_m": float(np.percentile(displacement, 95)),
        "position_delta_max_m": float(np.max(displacement)),
        "coefficient_relative_l2": float(
            np.linalg.norm(coefficient_delta)
            / max(np.linalg.norm(reference[..., :99]), 1.0e-30)
        ),
        "current_relative_l2": float(
            np.linalg.norm(current_delta)
            / max(np.linalg.norm(reference[..., -1]), 1.0e-30)
        ),
    }


def find_sources(data_dir: Path, source_ids: tuple[int, ...]) -> dict[int, dict[str, Any]]:
    wanted = set(source_ids)
    found: dict[int, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        groups, _ = load_raw_groups(data_dir, split)
        for key, group in groups.items():
            for source_id in wanted - found.keys():
                matches = np.flatnonzero(group.ids == source_id)
                if len(matches):
                    found[source_id] = {
                        "source_id": source_id,
                        "split": split,
                        "key": key,
                        "tokens": np.asarray(group.tokens[matches[0]], dtype=np.float32),
                    }
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f"source IDs not found in QH flow dataset: {sorted(missing)}")
    return found


def prepare_cases(args: argparse.Namespace) -> None:
    import torch

    source_ids = parse_ints(args.source_ids)
    sigmas = parse_floats(args.sigmas)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    normalizer = CoilNormalizer.from_dict(checkpoint["normalizer"])
    sources = find_sources(args.data_dir, source_ids)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cases = []
    experiment_id = 0
    source_manifest = []
    for source_order, source_id in enumerate(source_ids):
        source = sources[source_id]
        key = tuple(source["key"])
        raw = source["tokens"]
        normalized_batch, clipped_fraction = normalizer.transform(raw[None], key)
        canonical = normalizer.inverse(normalized_batch, key)[0]
        source_manifest.append(
            {
                "source_id": source_id,
                "split": source["split"],
                "nfp": key[0],
                "n_coils": key[1],
                "normalizer_clipped_fraction": clipped_fraction,
                "original_to_canonical": perturbation_metrics(raw, canonical),
            }
        )

        def append_case(
            tokens: np.ndarray,
            *,
            variant: str,
            sigma: float,
            replicate: int,
            actual_noise_rms: float,
            actual_noise_max: float,
        ) -> None:
            nonlocal experiment_id
            cases.append(
                {
                    "experiment_id": experiment_id,
                    "source_id": source_id,
                    "source_order": source_order,
                    "split": source["split"],
                    "nfp": key[0],
                    "n_coils": key[1],
                    "variant": variant,
                    "noise_sigma": sigma,
                    "replicate": replicate,
                    "normalized_noise_rms": actual_noise_rms,
                    "normalized_noise_max": actual_noise_max,
                    "perturbation": perturbation_metrics(tokens, raw),
                    "tokens": np.asarray(tokens, dtype=np.float32).tolist(),
                }
            )
            experiment_id += 1

        append_case(
            raw,
            variant="original",
            sigma=0.0,
            replicate=-1,
            actual_noise_rms=0.0,
            actual_noise_max=0.0,
        )
        append_case(
            canonical,
            variant="canonical",
            sigma=0.0,
            replicate=0,
            actual_noise_rms=0.0,
            actual_noise_max=0.0,
        )
        for sigma_index, sigma in enumerate(sigmas):
            for replicate in range(args.replicates):
                rng = np.random.default_rng(
                    args.seed
                    + source_id * 1000003
                    + sigma_index * 1009
                    + replicate
                )
                actual_noise = rng.standard_normal(raw.shape) * sigma
                actual_noise[..., -1] = 0.0
                perturbed = np.asarray(raw, dtype=np.float64).copy()
                perturbed[..., :99] += actual_noise[..., :99] * normalizer.std[:99]
                append_case(
                    perturbed,
                    variant="shape_noise",
                    sigma=sigma,
                    replicate=replicate,
                    actual_noise_rms=float(np.sqrt(np.mean(actual_noise**2))),
                    actual_noise_max=float(np.max(np.abs(actual_noise))),
                )

    with (args.output_dir / "cases.jsonl").open("w", encoding="utf-8") as stream:
        for row in cases:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    manifest = {
        "source_ids": list(source_ids),
        "sigmas": list(sigmas),
        "replicates": args.replicates,
        "seed": args.seed,
        "case_count": len(cases),
        "checkpoint_step": int(checkpoint["step"]),
        "sources": source_manifest,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def score_partition(args: argparse.Namespace) -> None:
    from stellarator_gpu import score_coils_native

    cases = [
        row
        for row in load_jsonl(args.output_dir / "cases.jsonl")
        if int(row["experiment_id"]) % args.world_size == args.rank
    ]
    started = time.perf_counter()
    output_path = args.output_dir / f"score_rank_{args.rank:02d}.jsonl"
    with output_path.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(cases, start=1):
            case = token_case(
                np.asarray(row["tokens"], dtype=np.float64),
                nfp=int(row["nfp"]),
                target="QH",
                metadata={"noise_experiment_id": int(row["experiment_id"])},
            )
            raw = case["raw"]
            score_started = time.perf_counter()
            result = None
            error = None
            try:
                result = score_coils_native(
                    args.lib,
                    raw["x"],
                    raw["y"],
                    raw["z"],
                    raw["current"],
                    int(row["nfp"]),
                    device_id=args.rank,
                    target_helicity=(1, int(row["nfp"])),
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            output = {
                "experiment_id": int(row["experiment_id"]),
                "score_wall_s": time.perf_counter() - score_started,
                "native_score": result,
                "error": error,
            }
            stream.write(json.dumps(output, separators=(",", ":"), allow_nan=True) + "\n")
            stream.flush()
            if index % 10 == 0:
                print(
                    json.dumps(
                        {
                            "rank": args.rank,
                            "completed": index,
                            "assigned": len(cases),
                            "elapsed_s": time.perf_counter() - started,
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
    (args.output_dir / f"runtime_rank_{args.rank:02d}.json").write_text(
        json.dumps(
            {
                "rank": args.rank,
                "assigned": len(cases),
                "wall_s": time.perf_counter() - started,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(np.mean(array)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def minimum_absolute_iota(diagnostics: dict[str, Any]) -> float:
    lower = float(diagnostics["iota_min"])
    upper = float(diagnostics["iota_max"])
    if lower <= 0.0 <= upper:
        return 0.0
    return min(abs(lower), abs(upper))


def compact_result(case: dict[str, Any], scored: dict[str, Any]) -> dict[str, Any]:
    native = scored.get("native_score")
    row = {
        key: case[key]
        for key in (
            "experiment_id",
            "source_id",
            "nfp",
            "n_coils",
            "variant",
            "noise_sigma",
            "replicate",
            "normalized_noise_rms",
            "normalized_noise_max",
            "perturbation",
        )
    }
    row["score_wall_s"] = float(scored["score_wall_s"])
    row["status"] = "error" if native is None else str(native["status"])
    row["score"] = 0.0 if native is None else float(native["score"])
    if native is not None and native["status"] == "ok":
        diagnostics = native["diagnostics"]
        row["components"] = native["components"]
        row["diagnostics"] = {
            "iota_star": minimum_absolute_iota(diagnostics),
            "surface_inverse_aspect_ratio": float(
                diagnostics["surface_inverse_aspect_ratio"]
            ),
            "surface_long_drift_relative_p95": float(
                diagnostics["surface_drift_relative_p95"]
            ),
            "qh_error_per_helicity": float(diagnostics["qs_global_error"])
            / math.hypot(1.0, int(case["nfp"])),
            "qa_error": float(diagnostics["qs_qa_global_error"]),
            "qp_error": float(diagnostics["qs_qp_global_error"]),
            "helicity_advantage": float(
                diagnostics["score_qh_helicity_advantage"]
            ),
        }
    return row


def summarize_group(rows: list[dict[str, Any]], baseline_score: float) -> dict[str, Any]:
    ok = [row for row in rows if row["status"] == "ok"]
    return {
        "count": len(rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "ok_rate": len(ok) / len(rows),
        "score": distribution([row["score"] for row in rows]),
        "score_fraction_of_original": distribution(
            [row["score"] / max(baseline_score, 1.0e-30) for row in rows]
        ),
        "position_delta_rms_m": distribution(
            [row["perturbation"]["position_delta_rms_m"] for row in rows]
        ),
        "coefficient_relative_l2": distribution(
            [row["perturbation"]["coefficient_relative_l2"] for row in rows]
        ),
        "helicity_advantage": distribution(
            [row["diagnostics"]["helicity_advantage"] for row in ok]
        ),
        "iota_star": distribution(
            [row["diagnostics"]["iota_star"] for row in ok]
        ),
        "surface_inverse_aspect_ratio": distribution(
            [row["diagnostics"]["surface_inverse_aspect_ratio"] for row in ok]
        ),
    }


def plot_summary(rows: list[dict[str, Any]], summary: dict[str, Any], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("Dark2")(
        np.linspace(0.0, 1.0, len(summary["source_ids"]), endpoint=False)
    )
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 8.0))
    for color, source_id in zip(colors, summary["source_ids"], strict=True):
        source_rows = [
            row
            for row in rows
            if row["source_id"] == source_id and row["variant"] != "canonical"
        ]
        grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for row in source_rows:
            grouped[float(row["noise_sigma"])].append(row)
        x = np.asarray(sorted(grouped))
        median = np.asarray([np.median([row["score"] for row in grouped[s]]) for s in x])
        p10 = np.asarray([np.percentile([row["score"] for row in grouped[s]], 10) for s in x])
        p90 = np.asarray([np.percentile([row["score"] for row in grouped[s]], 90) for s in x])
        baseline = float(grouped[0.0][0]["score"])
        axes[0, 0].plot(x, median, "o-", color=color, label=str(source_id))
        axes[0, 0].fill_between(x, p10, p90, color=color, alpha=0.15)
        axes[0, 1].plot(x, median / baseline, "o-", color=color, label=str(source_id))

        ok_rate = np.asarray(
            [np.mean([row["status"] == "ok" for row in grouped[s]]) for s in x]
        )
        advantage = np.asarray(
            [
                np.median(
                    [
                        row["diagnostics"]["helicity_advantage"]
                        for row in grouped[s]
                        if "diagnostics" in row
                    ]
                    or [0.0]
                )
                for s in x
            ]
        )
        axes[1, 0].plot(x, ok_rate, "o-", color=color, label=f"{source_id} ok")
        axes[1, 0].plot(x, advantage, "s--", color=color, alpha=0.75, label=f"{source_id} adv")

    noise_rows = [row for row in rows if row["variant"] == "shape_noise"]
    displacement = np.asarray(
        [row["perturbation"]["position_delta_rms_m"] for row in noise_rows]
    )
    scores = np.asarray([row["score"] for row in noise_rows])
    sigma = np.asarray([row["noise_sigma"] for row in noise_rows])
    scatter = axes[1, 1].scatter(
        displacement, scores, c=np.log10(sigma), cmap="viridis", s=12,
        alpha=0.5, rasterized=True,
    )
    figure.colorbar(scatter, ax=axes[1, 1], label=r"$\log_{10}\sigma$")

    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xscale("symlog", linthresh=5.0e-4)
        axis.set_xlabel("normalized token noise sigma")
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("score v3")
    axes[0, 0].set_title("Score sensitivity (median and P10-P90)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].axhline(1.0, color="#555555", linestyle=":")
    axes[0, 1].set_ylabel("score / original score")
    axes[0, 1].set_title("Relative score retention")
    axes[1, 0].set_ylabel("fraction or helicity advantage")
    axes[1, 0].set_ylim(-0.03, 1.03)
    axes[1, 0].set_title("Native success and QH selectivity")
    axes[1, 0].legend(fontsize=7, ncol=2)
    axes[1, 1].set_xlabel("coil position RMS perturbation [m]")
    axes[1, 1].set_ylabel("score v3")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_title("Physical geometry displacement")
    axes[1, 1].grid(alpha=0.2)
    figure.suptitle("QH score sensitivity around high-quality QUASR coils")
    figure.tight_layout()
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def analyze(args: argparse.Namespace) -> None:
    cases = load_jsonl(args.output_dir / "cases.jsonl")
    scores = []
    for path in sorted(args.output_dir.glob("score_rank_*.jsonl")):
        scores.extend(load_jsonl(path))
    score_by_id = {int(row["experiment_id"]): row for row in scores}
    if len(score_by_id) != len(cases):
        raise RuntimeError(f"expected {len(cases)} scores, found {len(score_by_id)}")
    rows = [compact_result(case, score_by_id[case["experiment_id"]]) for case in cases]
    source_ids = list(parse_ints(args.source_ids))
    per_source = {}
    for source_id in source_ids:
        source_rows = [row for row in rows if row["source_id"] == source_id]
        original = next(row for row in source_rows if row["variant"] == "original")
        canonical = next(row for row in source_rows if row["variant"] == "canonical")
        noise_groups: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for row in source_rows:
            if row["variant"] == "shape_noise":
                noise_groups[float(row["noise_sigma"])].append(row)
        per_source[str(source_id)] = {
            "original": original,
            "canonical": canonical,
            "noise": {
                str(sigma): summarize_group(group, original["score"])
                for sigma, group in sorted(noise_groups.items())
            },
        }
    aggregate = {}
    for sigma in parse_floats(args.sigmas):
        selected = [
            row
            for row in rows
            if row["variant"] == "shape_noise" and row["noise_sigma"] == sigma
        ]
        baselines = {
            source_id: per_source[str(source_id)]["original"]["score"]
            for source_id in source_ids
        }
        selected_with_ratio = []
        for row in selected:
            copied = dict(row)
            copied["score_fraction"] = row["score"] / max(
                baselines[row["source_id"]], 1.0e-30
            )
            selected_with_ratio.append(copied)
        aggregate[str(sigma)] = {
            "count": len(selected),
            "status_counts": dict(Counter(row["status"] for row in selected)),
            "ok_rate": float(
                np.mean([row["status"] == "ok" for row in selected])
            ),
            "score": distribution([row["score"] for row in selected]),
            "score_fraction_of_source_baseline": distribution(
                [row["score_fraction"] for row in selected_with_ratio]
            ),
            "position_delta_rms_m": distribution(
                [row["perturbation"]["position_delta_rms_m"] for row in selected]
            ),
        }
    runtimes = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.output_dir.glob("runtime_rank_*.json"))
    ]
    summary = {
        "source_ids": source_ids,
        "sigmas": list(parse_floats(args.sigmas)),
        "replicates": args.replicates,
        "case_count": len(rows),
        "noise_scope": "shape coefficients only; source currents unchanged",
        "per_source": per_source,
        "aggregate": aggregate,
        "runtime": {
            "rank": runtimes,
            "max_rank_wall_s": max(row["wall_s"] for row in runtimes),
            "sum_score_wall_s": sum(row["score_wall_s"] for row in rows),
        },
        "library_sha256": hashlib.sha256(args.lib.read_bytes()).hexdigest(),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    plot_summary(rows, summary, args.output_dir / "noise_sensitivity.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure score-v3 sensitivity to normalized noise around QH coils.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--source-ids", default=",".join(map(str, DEFAULT_IDS)))
    parser.add_argument("--sigmas", default=",".join(map(str, DEFAULT_SIGMAS)))
    parser.add_argument("--replicates", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.replicates < 1:
        raise ValueError("replicates must be positive")
    if args.prepare_only and args.analyze_only:
        raise ValueError("prepare-only and analyze-only are mutually exclusive")
    if args.prepare_only:
        prepare_cases(args)
    elif args.analyze_only:
        analyze(args)
    else:
        if not 0 <= args.rank < args.world_size:
            raise ValueError("rank must satisfy 0 <= rank < world_size")
        score_partition(args)


if __name__ == "__main__":
    main()
