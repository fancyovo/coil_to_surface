from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


LEARNING_RATES = (0.003, 0.01, 0.03)
BETA1_VALUES = (0.5, 0.7, 0.9)
RANDOM_DIRECTION_COUNTS = (0, 1, 2)


def value_tag(value: float) -> str:
    return format(value, "g").replace(".", "p")


def build_configs() -> list[dict[str, Any]]:
    configs = []
    for learning_rate, beta1, random_directions in itertools.product(
        LEARNING_RATES,
        BETA1_VALUES,
        RANDOM_DIRECTION_COUNTS,
    ):
        configs.append(
            {
                "learning_rate": learning_rate,
                "beta1": beta1,
                "random_directions": random_directions,
                "run_name": (
                    f"lr_{value_tag(learning_rate)}_b1_{value_tag(beta1)}_"
                    f"k_{random_directions}"
                ),
                # Conservative upper-bound score work per step. This is used
                # only to distribute configs across GPUs, not as a result.
                "balance_weight": 2 * (random_directions + 1) + 6.05,
            }
        )
    return configs


def assign_shards(configs: list[dict[str, Any]], shard_count: int) -> list[list[dict[str, Any]]]:
    if shard_count <= 0:
        raise ValueError("shard count must be positive")
    if shard_count == 6 and len(configs) == 27:
        # Three four-run and three five-run jobs are unavoidable. Giving the
        # shorter jobs more K=2 cases keeps the estimated loads within 4.7%.
        direction_patterns = (
            (2, 2, 2, 1),
            (2, 2, 2, 1),
            (2, 2, 1, 1),
            (2, 1, 0, 0, 0),
            (1, 1, 0, 0, 0),
            (1, 1, 0, 0, 0),
        )
        by_direction = {
            count: sorted(
                (
                    config
                    for config in configs
                    if int(config["random_directions"]) == count
                ),
                key=lambda config: str(config["run_name"]),
            )
            for count in RANDOM_DIRECTION_COUNTS
        }
        shards: list[list[dict[str, Any]]] = []
        for pattern in direction_patterns:
            shards.append([by_direction[count].pop(0) for count in pattern])
        if any(by_direction.values()):
            raise RuntimeError("fixed six-GPU shard assignment did not consume the grid")
        return shards
    shards: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    loads = [0.0] * shard_count
    ordered = sorted(
        configs,
        key=lambda config: (-float(config["balance_weight"]), str(config["run_name"])),
    )
    for config in ordered:
        shard_index = min(range(shard_count), key=lambda index: (loads[index], index))
        shards[shard_index].append(config)
        loads[shard_index] += float(config["balance_weight"])
    return shards


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def stream_command(command: list[str], log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one balanced shard of the QH reference-direction sweep."
    )
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=6)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gradient-lib", type=Path, required=True)
    parser.add_argument("--initial-case", type=Path, required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026080601)
    args = parser.parse_args()

    configs = build_configs()
    shards = assign_shards(configs, args.shard_count)
    if not 0 <= args.shard_index < len(shards):
        raise ValueError("shard index is outside the configured range")
    args.output_root.mkdir(parents=True, exist_ok=True)
    shard_configs = shards[args.shard_index]
    shard_manifest_path = args.output_root / f"shard_{args.shard_index:02d}_manifest.json"
    shard_manifest = {
        "format": "qh_reference_direction_sweep_shard_v1",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "iterations": args.iterations,
        "seed": args.seed,
        "gpus": args.gpus,
        "configs": shard_configs,
        "started_unix_s": time.time(),
        "completed_runs": [],
    }
    write_json(shard_manifest_path, shard_manifest)

    for config in shard_configs:
        run_dir = args.output_root / str(config["run_name"])
        summary_path = run_dir / "summary.json"
        state_path = run_dir / "state_latest.npz"
        history_path = run_dir / "history.jsonl"
        manifest_path = run_dir / "manifest.json"
        resume = False
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if int(summary.get("completed_iterations", -1)) >= args.iterations:
                shard_manifest["completed_runs"].append(
                    {"run_name": config["run_name"], "status": "already_complete"}
                )
                write_json(shard_manifest_path, shard_manifest)
                continue
        if any(path.exists() for path in (state_path, history_path, manifest_path)):
            if not all(path.is_file() for path in (state_path, history_path, manifest_path)):
                raise RuntimeError(f"incomplete non-resumable artifacts in {run_dir}")
            resume = True

        command = [
            sys.executable,
            "scripts/optimize_qh_g3_informed_subspace_adam.py",
            "--checkpoint",
            str(args.checkpoint),
            "--gradient-lib",
            str(args.gradient_lib),
            "--initial-case",
            str(args.initial_case),
            "--out-dir",
            str(run_dir),
            "--nfp",
            "4",
            "--iterations",
            str(args.iterations),
            "--rk4-steps",
            "64",
            "--random-directions",
            str(config["random_directions"]),
            "--random-direction-bank-size",
            "2",
            "--perturbation",
            "0.0025",
            "--learning-rate",
            str(config["learning_rate"]),
            "--proposal-mode",
            "adam",
            "--beta1",
            str(config["beta1"]),
            "--beta2",
            "0.999",
            "--seed",
            str(args.seed),
            "--center-rescore-every",
            "20",
            "--candidate-score-chunk-size",
            "1",
            "--plot-every",
            "10",
            "--gpus",
            args.gpus,
        ]
        if resume:
            command.append("--resume")
        stream_command(command, args.output_root / f"{config['run_name']}.driver.log")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(summary["completed_iterations"]) != args.iterations:
            raise RuntimeError(f"run did not reach target iterations: {run_dir}")
        shard_manifest["completed_runs"].append(
            {
                "run_name": config["run_name"],
                "status": "completed",
                "best_score": float(summary["best_score"]),
                "score_calls": int(summary["cumulative_blackbox_score_evaluations"]),
            }
        )
        write_json(shard_manifest_path, shard_manifest)

    shard_manifest["completed_unix_s"] = time.time()
    shard_manifest["status"] = "completed"
    write_json(shard_manifest_path, shard_manifest)


if __name__ == "__main__":
    main()
