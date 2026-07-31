from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StartTarget:
    label: str
    target_score: float
    status: str
    score_min: float
    score_max: float
    direction_seed_offset: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def parse_bound(value: str) -> float:
    if value.lower() in {"inf", "+inf", "none"}:
        return math.inf
    if value.lower() == "-inf":
        return -math.inf
    return float(value)


def parse_target(value: str) -> StartTarget:
    fields = [field.strip() for field in value.split(",")]
    if len(fields) != 6:
        raise argparse.ArgumentTypeError(
            "target must be LABEL,TARGET,STATUS,MIN,MAX,DIRECTION_SEED_OFFSET"
        )
    label, target, status, score_min, score_max, seed_offset = fields
    if not label or status not in {"any", "ok", "rejected"}:
        raise argparse.ArgumentTypeError("target label is required and status must be any, ok, or rejected")
    parsed = StartTarget(
        label=label,
        target_score=float(target),
        status=status,
        score_min=parse_bound(score_min),
        score_max=parse_bound(score_max),
        direction_seed_offset=int(seed_offset),
    )
    if parsed.score_min > parsed.target_score or parsed.target_score > parsed.score_max:
        raise argparse.ArgumentTypeError("target score must lie within [MIN, MAX]")
    return parsed


def target_matches(row: dict[str, Any], target: StartTarget) -> bool:
    score = float(row["score"])
    if not target.score_min <= score <= target.score_max:
        return False
    if target.status == "ok":
        return row["status"] == "ok"
    if target.status == "rejected":
        return row["status"] != "ok"
    return True


def select_targets(
    rows: list[dict[str, Any]], targets: list[StartTarget]
) -> list[tuple[StartTarget, dict[str, Any]]]:
    used_case_ids: set[int] = set()
    selected = []
    for target in targets:
        candidates = [
            row
            for row in rows
            if int(row["case_id"]) not in used_case_ids and target_matches(row, target)
        ]
        if not candidates:
            raise ValueError(f"no unused candidate satisfies target {target}")
        row = min(
            candidates,
            key=lambda item: (
                abs(float(item["score"]) - target.target_score),
                int(item["case_id"]),
            ),
        )
        used_case_ids.add(int(row["case_id"]))
        selected.append((target, row))
    return selected


def summarize_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    score = np.asarray([row["score"] for row in rows], dtype=float)
    status = np.asarray([row["status"] for row in rows], dtype="U32")
    return {
        "count": len(rows),
        "score": {
            "mean": float(np.mean(score)),
            "median": float(np.median(score)),
            "p90": float(np.percentile(score, 90)),
            "p95": float(np.percentile(score, 95)),
            "p99": float(np.percentile(score, 99)),
            "p99_5": float(np.percentile(score, 99.5)),
            "max": float(np.max(score)),
        },
        "score_exceedance_counts": {
            str(threshold): int(np.sum(score >= threshold)) for threshold in (10, 20, 30, 40, 50)
        },
        "status_counts": {str(value): int(np.sum(status == value)) for value in np.unique(status)},
        "status_ok_rate": float(np.mean(status == "ok")),
    }


def plot_distribution(
    rows: list[dict[str, Any]], selected: list[tuple[StartTarget, dict[str, Any]]], output_path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    score = np.asarray([row["score"] for row in rows], dtype=float)
    ok = np.asarray([row["status"] == "ok" for row in rows])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    bins = np.linspace(0.0, max(55.0, float(score.max()) + 1.0), 70)
    axes[0].hist(score[~ok], bins=bins, alpha=0.65, color="#9a4d42", label="rejected")
    axes[0].hist(score[ok], bins=bins, alpha=0.65, color="#237a57", label="status=ok")
    for index, (_, row) in enumerate(selected):
        axes[0].axvline(
            row["score"], color="#111111", alpha=0.4, lw=1.0,
            label="selected starts" if index == 0 else None,
        )
    axes[0].axvline(40.0, color="#555555", ls="--", label="40 / 50 thresholds")
    axes[0].axvline(50.0, color="#222222", ls=":")
    axes[0].set(xlabel="native score", ylabel="count", title="IID random-start score distribution")
    axes[0].legend()

    ordered = np.sort(score)
    survival = 1.0 - np.arange(len(ordered)) / len(ordered)
    axes[1].step(ordered, survival, where="post", color="#486a88")
    axes[1].scatter(
        [row["score"] for _, row in selected],
        [np.mean(score >= row["score"]) for _, row in selected],
        color="#a34137", s=35, zorder=3, label="selected starts",
    )
    axes[1].axvline(40.0, color="#555555", ls="--")
    axes[1].axvline(50.0, color="#222222", ls=":")
    axes[1].set(
        yscale="log", xlabel="native score", ylabel="empirical survival",
        title="Selected coverage of IID upper tail",
    )
    axes[1].legend()
    figure.savefig(output_path, dpi=190)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select pure-IID starts for a native-score Adam sweep.")
    parser.add_argument("--scored-cases", type=Path, required=True)
    parser.add_argument("--random-latents", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target", type=parse_target, action="append", required=True,
        help="LABEL,TARGET,STATUS,MIN,MAX,DIRECTION_SEED_OFFSET; repeat for each start",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = load_jsonl(args.scored_cases)
    with np.load(args.random_latents, allow_pickle=False) as payload:
        latents = np.asarray(payload["latent"], dtype=np.float32)
    if latents.ndim != 3 or latents.shape[1:] != (3, 100):
        raise ValueError(f"unexpected latent shape {latents.shape}")
    if len(rows) != len(latents):
        raise ValueError("scored rows and random latents have different lengths")
    case_ids = [int(row["case_id"]) for row in rows]
    if sorted(case_ids) != list(range(len(latents))):
        raise ValueError("case IDs must map one-to-one to random_latents rows")

    selected = select_targets(rows, args.target)
    panel = []
    for start_id, (target, row) in enumerate(selected):
        case_id = int(row["case_id"])
        start = {
            "format": "qh_iid_score_adam_start_v2",
            "flow_prior_start": {
                "noise": latents[case_id].tolist(),
                "source": "pure_iid_random_score_pool",
                "source_case_id": case_id,
                "recorded_score": float(row["score"]),
                "recorded_status": row["status"],
                "direction_seed_offset": target.direction_seed_offset,
            },
        }
        filename = f"start_{start_id:02d}.json"
        (args.output_dir / filename).write_text(json.dumps(start, indent=2) + "\n", encoding="utf-8")
        panel.append(
            {
                "start_id": start_id,
                "file": filename,
                "label": target.label,
                "stratum": target.label.rsplit("_", 1)[0],
                "target_score": target.target_score,
                "recorded_score": float(row["score"]),
                "recorded_status": row["status"],
                "case_id": case_id,
                "latent_rms": float(row["latent_rms"]),
                "direction_seed_offset": target.direction_seed_offset,
            }
        )
    summary = {
        "format": "qh_iid_score_adam_start_panel_v2",
        "source": {
            "scored_cases": str(args.scored_cases.resolve()),
            "scored_cases_sha256": file_sha256(args.scored_cases),
            "random_latents": str(args.random_latents.resolve()),
            "random_latents_sha256": file_sha256(args.random_latents),
            "selection": "all rows are from the pure IID standard-Gaussian pool",
        },
        "distribution": summarize_distribution(rows),
        "targets": [asdict(target) for target in args.target],
        "starts": panel,
    }
    (args.output_dir / "panel_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    plot_distribution(rows, selected, args.output_dir / "iid_score_distribution_and_starts.png")
    print(json.dumps({"event": "complete", **summary}, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
