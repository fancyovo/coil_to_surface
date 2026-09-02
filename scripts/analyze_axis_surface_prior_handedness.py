from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SIGN_ORDER = ("negative", "crosses_zero", "positive", "missing")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def classify_iota_interval(
    iota_min: Any,
    iota_max: Any,
    *,
    tolerance: float = 1.0e-10,
) -> str:
    lower = finite_float(iota_min)
    upper = finite_float(iota_max)
    if lower is None or upper is None:
        return "missing"
    lower, upper = sorted((lower, upper))
    if upper < -tolerance:
        return "negative"
    if lower > tolerance:
        return "positive"
    return "crosses_zero"


def midpoint(iota_min: Any, iota_max: Any) -> float | None:
    lower = finite_float(iota_min)
    upper = finite_float(iota_max)
    return None if lower is None or upper is None else 0.5 * (lower + upper)


def quantiles(values: Iterable[float | None]) -> dict[str, float | int | None]:
    data = np.asarray([value for value in values if value is not None], dtype=float)
    if not len(data):
        return {"count": 0, "min": None, "p10": None, "median": None, "p90": None, "max": None}
    return {
        "count": int(len(data)),
        "min": float(np.min(data)),
        "p10": float(np.quantile(data, 0.10)),
        "median": float(np.median(data)),
        "p90": float(np.quantile(data, 0.90)),
        "max": float(np.max(data)),
    }


def binomial_two_sided_log10(negative: int, positive: int) -> float | None:
    total = negative + positive
    if total == 0:
        return None
    tail = min(negative, positive)
    logs = [
        math.lgamma(total + 1)
        - math.lgamma(index + 1)
        - math.lgamma(total - index + 1)
        - total * math.log(2.0)
        for index in range(tail + 1)
    ]
    maximum = max(logs)
    log_probability = maximum + math.log(sum(math.exp(value - maximum) for value in logs))
    log_probability = min(0.0, math.log(2.0) + log_probability)
    return log_probability / math.log(10.0)


def sign_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(record["iota_sign"]) for record in records)
    negative = counts["negative"]
    positive = counts["positive"]
    definite = negative + positive
    return {
        "count": len(records),
        "sign_counts": {name: counts[name] for name in SIGN_ORDER},
        "definite_sign_count": definite,
        "negative_fraction_of_definite": negative / definite if definite else None,
        "positive_fraction_of_definite": positive / definite if definite else None,
        "two_sided_binomial_log10_p_vs_equal_signs": binomial_two_sided_log10(negative, positive),
        "iota_midpoint": quantiles(record.get("iota_midpoint") for record in records),
        "score": quantiles(record.get("score") for record in records),
    }


def compact_native_record(row: dict[str, Any]) -> dict[str, Any]:
    native = row.get("native") if isinstance(row.get("native"), dict) else {}
    diagnostics = native.get("diagnostics") if isinstance(native.get("diagnostics"), dict) else {}
    iota_min = finite_float(diagnostics.get("iota_min"))
    iota_max = finite_float(diagnostics.get("iota_max"))
    return {
        "case_id": int(row["case_id"]),
        "nfp": int(row["nfp"]),
        "n_base_coils": int(row["n_base_coils"]),
        "status": str(native.get("status", "missing")),
        "score": finite_float(native.get("score")),
        "iota_min": iota_min,
        "iota_max": iota_max,
        "iota_midpoint": midpoint(iota_min, iota_max),
        "iota_sign": classify_iota_interval(iota_min, iota_max),
    }


def load_source_records(
    source_root: Path,
    selection: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    expected_shards = {
        str(item["file"]): str(item["sha256"])
        for item in selection["input"]["shards"]
    }
    records: list[dict[str, Any]] = []
    provenance: list[dict[str, str]] = []
    for filename, expected_sha in sorted(expected_shards.items()):
        path = source_root / filename
        actual_sha = file_sha256(path)
        if actual_sha != expected_sha:
            raise ValueError(f"source shard hash mismatch for {path}")
        count = 0
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    records.append(compact_native_record(json.loads(line)))
                    count += 1
        provenance.append({"file": str(path.resolve()), "sha256": actual_sha, "row_count": str(count)})
    case_ids = [record["case_id"] for record in records]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate case IDs in source shards")
    if len(records) != int(selection["input"]["row_count"]):
        raise ValueError("source row count does not match the frozen selection manifest")
    return sorted(records, key=lambda record: int(record["case_id"])), provenance


def native_record(native: dict[str, Any]) -> dict[str, Any]:
    diagnostics = native.get("diagnostics") if isinstance(native.get("diagnostics"), dict) else {}
    iota_min = finite_float(diagnostics.get("iota_min"))
    iota_max = finite_float(diagnostics.get("iota_max"))
    return {
        "status": str(native.get("status", "missing")),
        "score": finite_float(native.get("score")),
        "iota_min": iota_min,
        "iota_max": iota_max,
        "iota_midpoint": midpoint(iota_min, iota_max),
        "iota_sign": classify_iota_interval(iota_min, iota_max),
    }


def scalar_sign(value: float | None, tolerance: float = 1.0e-10) -> str:
    if value is None:
        return "missing"
    if value < -tolerance:
        return "negative"
    if value > tolerance:
        return "positive"
    return "crosses_zero"


def load_adam_records(adam_root: Path, selection: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for case in selection["cases"]:
        trajectory_id = str(case["trajectory_id"])
        trajectory_root = adam_root / "trajectories" / trajectory_id
        start = json.loads((trajectory_root / "start.json").read_text(encoding="utf-8"))
        best = json.loads((trajectory_root / "optimization" / "best.json").read_text(encoding="utf-8"))
        history = [
            json.loads(line)
            for line in (trajectory_root / "optimization" / "history.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        start_native = native_record(start["data_prior_screening"]["native_score"])
        optimizer = best["original_space_local_gradient_adam"]
        best_native = native_record(optimizer["native_score"])
        steps = [0] + [int(item["iteration"]) for item in history]
        iotas = [start_native["iota_midpoint"]] + [finite_float(item.get("current_iota")) for item in history]
        finite_iotas = [value for value in iotas if value is not None]
        signs = {scalar_sign(value) for value in finite_iotas}
        if not finite_iotas:
            trajectory_pattern = "missing"
        elif signs == {"negative"}:
            trajectory_pattern = "all_negative"
        elif signs == {"positive"}:
            trajectory_pattern = "all_positive"
        else:
            trajectory_pattern = "crosses_or_touches_zero"
        final_iota = finite_float(history[-1].get("current_iota")) if history else start_native["iota_midpoint"]
        records.append(
            {
                "trajectory_id": trajectory_id,
                "case_id": int(case["case_id"]),
                "nfp": int(case["nfp"]),
                "n_base_coils": int(case["n_base_coils"]),
                "initial_score": start_native["score"],
                "initial_iota": start_native["iota_midpoint"],
                "initial_sign": start_native["iota_sign"],
                "best_score": best_native["score"],
                "best_iteration": int(optimizer["best_iteration"]),
                "best_iota": best_native["iota_midpoint"],
                "best_sign": best_native["iota_sign"],
                "final_score": finite_float(history[-1].get("current_score")) if history else start_native["score"],
                "final_iota": final_iota,
                "final_sign": scalar_sign(final_iota),
                "trajectory_pattern": trajectory_pattern,
                "history_count": len(history),
            }
        )
        curves.append(
            {
                "trajectory_id": trajectory_id,
                "n_base_coils": int(case["n_base_coils"]),
                "steps": steps,
                "iotas": iotas,
            }
        )
    if len(records) != len(selection["cases"]):
        raise ValueError("Adam trajectory count does not match selection manifest")
    return records, curves


def endpoint_sign_summary(records: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    converted = [
        {
            "iota_sign": record[f"{prefix}_sign"],
            "iota_midpoint": record[f"{prefix}_iota"],
            "score": record.get(f"{prefix}_score"),
        }
        for record in records
    ]
    return sign_summary(converted)


def source_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [record for record in records if record["status"] == "ok"]
    threshold_groups = {"all": records, "native_status_ok": ok}
    for threshold in (10, 20, 30, 40):
        threshold_groups[f"score_ge_{threshold}"] = [
            record
            for record in ok
            if record["score"] is not None and record["score"] >= threshold
        ]
    by_nc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_nfp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in ok:
        by_nc[str(record["n_base_coils"])].append(record)
        by_nfp[str(record["nfp"])].append(record)
    return {
        "groups": {name: sign_summary(group) for name, group in threshold_groups.items()},
        "native_status_counts": dict(Counter(record["status"] for record in records)),
        "by_n_base_coils_among_ok": {name: sign_summary(group) for name, group in sorted(by_nc.items())},
        "by_nfp_among_ok": {name: sign_summary(group) for name, group in sorted(by_nfp.items())},
    }


def adam_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(records),
        "initial": endpoint_sign_summary(records, "initial"),
        "best_endpoint": endpoint_sign_summary(records, "best"),
        "final": endpoint_sign_summary(records, "final"),
        "trajectory_pattern_counts": dict(Counter(record["trajectory_pattern"] for record in records)),
        "best_score_ge_50": endpoint_sign_summary(
            [record for record in records if record["best_score"] is not None and record["best_score"] >= 50.0],
            "best",
        ),
        "best_score_ge_60": endpoint_sign_summary(
            [record for record in records if record["best_score"] is not None and record["best_score"] >= 60.0],
            "best",
        ),
    }


def plot_source_handedness(summary: dict[str, Any], output_path: Path) -> None:
    group_keys = ("all", "native_status_ok", "score_ge_10", "score_ge_20", "score_ge_30", "score_ge_40")
    labels = ("all", "status ok", "score >= 10", "score >= 20", "score >= 30", "score >= 40")
    colors = {"negative": "#b34b3c", "crosses_zero": "#6f7782", "positive": "#248277", "missing": "#d5d6d8"}
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.5), constrained_layout=True)
    bottom = np.zeros(len(group_keys), dtype=float)
    for sign in SIGN_ORDER:
        values = []
        for key in group_keys:
            group = summary["groups"][key]
            values.append(group["sign_counts"][sign] / group["count"] if group["count"] else 0.0)
        axes[0].bar(labels, values, bottom=bottom, color=colors[sign], label=sign.replace("_", " "))
        bottom += np.asarray(values)
    axes[0].set(ylabel="fraction", title="Initial native iota sign by score selection", ylim=(0.0, 1.0))
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)

    nc = summary["by_n_base_coils_among_ok"]
    nfp = summary["by_nfp_among_ok"]
    nc_x = np.arange(len(nc), dtype=float)
    nfp_x = np.arange(len(nfp), dtype=float) + len(nc) + 1.0
    axes[1].bar(
        nc_x,
        [value["negative_fraction_of_definite"] for value in nc.values()],
        color="#b34b3c",
        label="by nc",
    )
    axes[1].bar(
        nfp_x,
        [value["negative_fraction_of_definite"] for value in nfp.values()],
        color="#35618d",
        label="by nfp",
    )
    axes[1].axhline(0.5, color="#222222", linestyle="--", linewidth=1.0, label="symmetric 0.5")
    axes[1].set_xticks(
        np.concatenate((nc_x, nfp_x)),
        [f"nc={key}" for key in nc] + [f"nfp={key}" for key in nfp],
        rotation=35,
    )
    axes[1].set(ylabel="negative fraction among definite signs", title="Valid starts by condition", ylim=(0.0, 1.0))
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.2)
    figure.savefig(output_path, dpi=190)
    plt.close(figure)


def plot_adam_iota(records: list[dict[str, Any]], curves: list[dict[str, Any]], output_path: Path) -> None:
    colors = {1: "#35618d", 2: "#248277", 3: "#b27a2d", 4: "#8b4a6f"}
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.5), constrained_layout=True)
    for curve in curves:
        axes[0].plot(
            curve["steps"],
            curve["iotas"],
            color=colors[curve["n_base_coils"]],
            linewidth=0.7,
            alpha=0.20,
        )
    axes[0].axhline(0.0, color="#111111", linestyle="--", linewidth=1.0)
    axes[0].set(xlabel="Adam update", ylabel="native iota midpoint", title="All 120 positive-target Adam trajectories")
    axes[0].grid(alpha=0.2)

    stages = ("initial", "best", "final")
    bottom = np.zeros(len(stages), dtype=float)
    sign_colors = {"negative": "#b34b3c", "crosses_zero": "#6f7782", "positive": "#248277", "missing": "#d5d6d8"}
    for sign in SIGN_ORDER:
        values = []
        for stage in stages:
            counts = Counter(record[f"{stage}_sign"] for record in records)
            values.append(counts[sign] / len(records))
        axes[1].bar(stages, values, bottom=bottom, color=sign_colors[sign], label=sign.replace("_", " "))
        bottom += np.asarray(values)
    axes[1].set(ylabel="fraction", title="Sign persistence through Adam200", ylim=(0.0, 1.0))
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.2)
    figure.savefig(output_path, dpi=190)
    plt.close(figure)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit analytic-prior handedness in source scores and Adam200 trajectories.")
    parser.add_argument("--source-score-root", type=Path, required=True)
    parser.add_argument("--adam-run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    if commit != args.expected_commit:
        raise RuntimeError(f"repository commit {commit} != {args.expected_commit}")
    selection_path = args.adam_run_root / "selection_manifest.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    source_records, source_provenance = load_source_records(args.source_score_root, selection)
    adam_records, curves = load_adam_records(args.adam_run_root, selection)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "format": "axis_surface_prior_handedness_audit_v1",
        "code_commit": commit,
        "frozen_positive_target": "(M,N)=(1,+nfp)",
        "source_score_root": str(args.source_score_root.resolve()),
        "adam_run_root": str(args.adam_run_root.resolve()),
        "selection_manifest": str(selection_path.resolve()),
        "selection_manifest_sha256": file_sha256(selection_path),
        "source_shards": source_provenance,
        "source_population": source_summary(source_records),
        "adam200": adam_summary(adam_records),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "source_population.csv", source_records)
    write_csv(args.output_dir / "adam200_handedness.csv", adam_records)
    plot_source_handedness(summary["source_population"], args.output_dir / "source_handedness.png")
    plot_adam_iota(adam_records, curves, args.output_dir / "adam200_iota.png")
    print(
        json.dumps(
            {
                "event": "axis_surface_prior_handedness_audit_complete",
                "source_count": len(source_records),
                "adam_count": len(adam_records),
                "output_dir": str(args.output_dir),
            },
            separators=(",", ":"),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
