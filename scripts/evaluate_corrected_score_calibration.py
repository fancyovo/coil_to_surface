from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
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

from flow_matching.data import GroupKey, RawGroup, load_raw_groups
from scripts.collect_qh_iid_score_data import decode_mixed_conditions, load_flow
from scripts.optimize_native_score_cem import NativeScorePool, token_case


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0}
    output: dict[str, float | int] = {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }
    for percentile in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        output[f"p{percentile}"] = float(np.percentile(array, percentile))
    return output


def select_quasr_cases(
    groups: dict[GroupKey, RawGroup], count: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    entries = [
        (key, index)
        for key in sorted(groups)
        for index in range(len(groups[key].tokens))
    ]
    if count > len(entries):
        raise ValueError(f"requested {count} QUASR cases from only {len(entries)} available")
    selected = rng.choice(len(entries), size=count, replace=False)
    rows = []
    for sample_index, entry_index in enumerate(selected):
        key, group_index = entries[int(entry_index)]
        group = groups[key]
        rows.append(
            {
                "kind": "quasr",
                "sample_index": sample_index,
                "source_id": int(group.ids[group_index]),
                "key": key,
                "tokens": np.asarray(group.tokens[group_index], dtype=np.float32),
            }
        )
    return rows


def decode_random_cases(
    checkpoint: Path,
    conditions: list[GroupKey],
    *,
    seed: int,
    flow_steps: int,
    batch_size: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], float]:
    model, normalizer, _ = load_flow(checkpoint, device)
    generator = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    total_decode_s = 0.0
    for start in range(0, len(conditions), batch_size):
        batch_conditions = conditions[start : start + batch_size]
        latents, raw_tokens, decode_s = decode_mixed_conditions(
            model,
            normalizer,
            batch_conditions,
            generator,
            flow_steps=flow_steps,
            device=device,
        )
        total_decode_s += decode_s
        for offset, (key, latent, tokens) in enumerate(
            zip(batch_conditions, latents, raw_tokens, strict=True)
        ):
            rows.append(
                {
                    "kind": "random_flow",
                    "sample_index": start + offset,
                    "source_id": None,
                    "key": key,
                    "latent": np.asarray(latent, dtype=np.float32),
                    "tokens": np.asarray(tokens, dtype=np.float32),
                }
            )
    del model
    torch.cuda.empty_cache()
    return rows, total_decode_s


def load_known_case(label: str, path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload["raw"]
    tokens = np.concatenate(
        [
            np.asarray(raw["x"], dtype=np.float32),
            np.asarray(raw["y"], dtype=np.float32),
            np.asarray(raw["z"], dtype=np.float32),
            np.asarray(raw["current"], dtype=np.float32)[:, None],
        ],
        axis=1,
    )
    return {
        "kind": "known",
        "sample_index": label,
        "source_id": None,
        "key": (int(payload.get("nfp", raw["nfp"])), int(tokens.shape[0])),
        "tokens": tokens,
        "source_path": str(path.resolve()),
    }


def serialize_cases(path: Path, rows: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            payload = {
                "kind": row["kind"],
                "sample_index": row["sample_index"],
                "source_id": row["source_id"],
                "nfp": row["key"][0],
                "n_base_coils": row["key"][1],
                "tokens": row["tokens"].tolist(),
            }
            if "latent" in row:
                payload["latent"] = row["latent"].tolist()
            if "source_path" in row:
                payload["source_path"] = row["source_path"]
            stream.write(json.dumps(payload, separators=(",", ":")) + "\n")


def score_rows(
    rows: list[dict[str, Any]], lib: Path, gpu_ids: list[int], timeout_s: float
) -> tuple[list[dict[str, Any]], float]:
    cases = [
        token_case(
            row["tokens"],
            nfp=row["key"][0],
            target="QH",
            metadata={"calibration_kind": row["kind"]},
        )
        for row in rows
    ]
    started = time.perf_counter()
    with NativeScorePool(lib, gpu_ids) as pool:
        scored = pool.map(cases, target="QH", timeout_s=timeout_s)
    wall_s = time.perf_counter() - started
    output = []
    for row, (result, elapsed_s, error) in zip(rows, scored, strict=True):
        output.append(
            {
                "kind": row["kind"],
                "sample_index": row["sample_index"],
                "source_id": row["source_id"],
                "nfp": row["key"][0],
                "n_base_coils": row["key"][1],
                "native_score": result,
                "score_wall_s": elapsed_s,
                "score_error": error,
            }
        )
    return output, wall_s


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status = Counter(
        "python_error" if row["native_score"] is None else row["native_score"]["status"]
        for row in rows
    )
    scores = [
        0.0 if row["native_score"] is None else float(row["native_score"]["score"])
        for row in rows
    ]
    ok = [row for row in rows if row["native_score"] is not None and row["native_score"]["status"] == "ok"]
    summary: dict[str, Any] = {
        "count": len(rows),
        "status_counts": dict(status),
        "ok_rate": status["ok"] / max(len(rows), 1),
        "score_all": distribution(scores),
        "score_ok": distribution([float(row["native_score"]["score"]) for row in ok]),
        "score_threshold_counts": {
            str(threshold): int(np.sum(np.asarray(scores) >= threshold))
            for threshold in (10, 20, 30, 40, 50, 60, 70, 80)
        },
        "score_call_wall_s": distribution([float(row["score_wall_s"]) for row in rows]),
    }
    if ok:
        diagnostics = [row["native_score"]["diagnostics"] for row in ok]
        summary["ok_diagnostics"] = {
            "target_error_per_helicity": distribution(
                [float(item["qs_target_global_error_per_helicity"]) for item in diagnostics]
            ),
            "qa_error_per_helicity": distribution(
                [float(item["qs_qa_global_error_per_helicity"]) for item in diagnostics]
            ),
            "qp_error_per_helicity": distribution(
                [float(item["qs_qp_global_error_per_helicity"]) for item in diagnostics]
            ),
            "abs_iota": distribution(
                [0.5 * (abs(float(item["iota_min"])) + abs(float(item["iota_max"]))) for item in diagnostics]
            ),
            "inverse_aspect_ratio": distribution(
                [float(item["surface_inverse_aspect_ratio"]) for item in diagnostics]
            ),
        }
    return summary


def plot_results(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = {
        kind: [row for row in rows if row["kind"] == kind]
        for kind in ("quasr", "random_flow")
    }
    colors = {"quasr": "#176b87", "random_flow": "#bc5a45"}
    labels = {"quasr": "QUASR QH", "random_flow": "random flow"}
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for kind, values in grouped.items():
        score = np.asarray(
            [0.0 if row["native_score"] is None else row["native_score"]["score"] for row in values],
            dtype=float,
        )
        axes[0, 0].hist(score, bins=np.linspace(0, 100, 51), alpha=0.58, color=colors[kind], label=labels[kind])
        ordered = np.sort(score)
        axes[0, 1].plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered), color=colors[kind], label=labels[kind])
        ok = [row for row in values if row["native_score"] is not None and row["native_score"]["status"] == "ok"]
        axes[1, 0].scatter(
            [row["native_score"]["diagnostics"]["qs_target_global_error_per_helicity"] for row in ok],
            [row["native_score"]["score"] for row in ok],
            s=9,
            alpha=0.45,
            color=colors[kind],
            label=labels[kind],
        )
    axes[0, 0].set(xlabel="corrected native score", ylabel="count", title="Score distribution")
    axes[0, 1].set(xlabel="corrected native score", ylabel="empirical CDF", title="Score CDF")
    axes[1, 0].set(xlabel="QH differential error per helicity", ylabel="score", xscale="log", title="Score against corrected QH error")
    status_names = sorted(
        {
            "python_error" if row["native_score"] is None else row["native_score"]["status"]
            for row in rows
            if row["kind"] != "known"
        }
    )
    x = np.arange(len(status_names))
    width = 0.38
    for offset, kind in enumerate(("quasr", "random_flow")):
        counts = Counter(
            "python_error" if row["native_score"] is None else row["native_score"]["status"]
            for row in grouped[kind]
        )
        axes[1, 1].bar(x + (offset - 0.5) * width, [counts[name] for name in status_names], width, color=colors[kind], label=labels[kind])
    axes[1, 1].set_xticks(x, status_names, rotation=25, ha="right")
    axes[1, 1].set(ylabel="count", title="Pipeline status")
    for axis in axes.flat:
        axis.legend()
        axis.grid(alpha=0.18)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def parse_known(values: list[str]) -> list[tuple[str, Path]]:
    output = []
    for value in values:
        if "=" not in value:
            raise ValueError("--known-case must be LABEL=PATH")
        label, path = value.split("=", 1)
        output.append((label, Path(path)))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the corrected native QH score on matched QUASR and random-flow samples.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quasr-count", type=int, default=1024)
    parser.add_argument("--random-count", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--flow-steps", type=int, default=256)
    parser.add_argument("--decode-batch", type=int, default=256)
    parser.add_argument("--gpus", default="0,0,1,1,2,2,3,3")
    parser.add_argument("--score-timeout-s", type=float, default=7200.0)
    parser.add_argument("--known-case", action="append", default=[])
    args = parser.parse_args()

    if args.quasr_count <= 0 or args.random_count <= 0:
        raise ValueError("both calibration populations must be nonempty")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    test_groups, data_manifest = load_raw_groups(args.data_dir, "test", verify_hashes=False)
    quasr = select_quasr_cases(test_groups, args.quasr_count, rng)
    matched_conditions = [row["key"] for row in quasr]
    if args.random_count != args.quasr_count:
        indices = rng.choice(len(matched_conditions), size=args.random_count, replace=True)
        random_conditions = [matched_conditions[int(index)] for index in indices]
    else:
        random_conditions = matched_conditions
    random_rows, decode_s = decode_random_cases(
        args.checkpoint,
        random_conditions,
        seed=args.seed + 1,
        flow_steps=args.flow_steps,
        batch_size=args.decode_batch,
        device=torch.device("cuda:0"),
    )
    known = [load_known_case(label, path) for label, path in parse_known(args.known_case)]
    all_cases = quasr + random_rows + known
    serialize_cases(args.output_dir / "prepared_cases.jsonl.gz", all_cases)
    gpu_ids = [int(value) for value in args.gpus.split(",") if value.strip()]
    scored, score_wall_s = score_rows(all_cases, args.lib, gpu_ids, args.score_timeout_s)
    with (args.output_dir / "results.jsonl").open("w", encoding="utf-8") as stream:
        for row in scored:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=True) + "\n")
    groups = {
        kind: summarize_group([row for row in scored if row["kind"] == kind])
        for kind in ("quasr", "random_flow", "known")
    }
    summary = {
        "format": "corrected_native_score_calibration_v1",
        "score_abi": 9,
        "git_commit": __import__("subprocess").run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "library_sha256": file_sha256(args.lib),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "data_manifest_format": data_manifest.get("format"),
        "selection": {
            "split": "test",
            "without_replacement": True,
            "condition_matching": "random flow uses the sampled QUASR (nfp,n_coils) sequence",
            "seed": args.seed,
            "flow_method": "rk4_fp32",
            "flow_steps": args.flow_steps,
        },
        "groups": groups,
        "runtime": {
            "decode_wall_s": decode_s,
            "score_wall_s": score_wall_s,
            "score_throughput_per_s": len(all_cases) / max(score_wall_s, 1.0e-9),
            "total_wall_s": time.perf_counter() - started,
            "score_workers": gpu_ids,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    plot_results(scored, args.output_dir / "score_distribution.png")
    print(json.dumps(summary, separators=(",", ":"), allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
