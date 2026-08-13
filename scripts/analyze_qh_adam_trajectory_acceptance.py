from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


COMPONENT_KEYS = (
    "axis",
    "psi",
    "surface",
    "coordinate",
    "volume_qs",
    "iota",
    "coil",
)
SCORE_THRESHOLDS = (50, 60, 70, 80, 85, 90, 92, 94, 95)


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0}
    output: dict[str, float | int | None] = {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }
    for percentile in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        output[f"p{percentile}"] = float(np.percentile(array, percentile))
    return output


def thresholds(values: Iterable[float]) -> dict[str, dict[str, float | int]]:
    array = np.asarray(list(values), dtype=np.float64)
    output = {}
    for value in SCORE_THRESHOLDS:
        count = int(np.sum(array >= value))
        output[str(value)] = {
            "count": count,
            "fraction": count / max(len(array), 1),
        }
    return output


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = 0.5 * (index + end - 1)
        index = end
    return ranks


def correlation(x: Iterable[float], y: Iterable[float]) -> dict[str, float | int]:
    left = np.asarray(list(x), dtype=np.float64)
    right = np.asarray(list(y), dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    if len(left) < 2:
        return {"count": int(len(left)), "pearson": float("nan"), "spearman": float("nan")}
    return {
        "count": int(len(left)),
        "pearson": float(np.corrcoef(left, right)[0, 1]),
        "spearman": float(np.corrcoef(rankdata(left), rankdata(right))[0, 1]),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def native_record(
    *,
    source: str,
    case_id: str,
    nfp: int,
    n_coils: int,
    result: dict[str, Any] | None,
    **extra: Any,
) -> dict[str, Any]:
    result = result or {}
    diagnostics = result.get("diagnostics", {})
    iota_min = float(diagnostics.get("iota_min", float("nan")))
    iota_max = float(diagnostics.get("iota_max", float("nan")))
    record = {
        "source": source,
        "case_id": case_id,
        "nfp": int(nfp),
        "n_base_coils": int(n_coils),
        "score": float(result.get("score", 0.0)),
        "status": str(result.get("status", "python_error")),
        "qh_error": float(
            diagnostics.get("qs_target_global_error_per_helicity", float("nan"))
        ),
        "qa_error": float(
            diagnostics.get("qs_qa_global_error_per_helicity", float("nan"))
        ),
        "qp_error": float(
            diagnostics.get("qs_qp_global_error_per_helicity", float("nan"))
        ),
        "abs_iota": 0.5 * (abs(iota_min) + abs(iota_max)),
        "inverse_aspect_ratio": float(
            diagnostics.get("surface_inverse_aspect_ratio", float("nan"))
        ),
        "surface_level": float(diagnostics.get("surface_level", float("nan"))),
    }
    for key in COMPONENT_KEYS:
        record[f"component_{key}"] = float(
            result.get("components", {}).get(key, float("nan"))
        )
    record.update(extra)
    return record


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in records if row["status"] == "ok"]
    return {
        "count": len(records),
        "status_counts": dict(sorted(Counter(row["status"] for row in records).items())),
        "ok_rate": len(ok) / max(len(records), 1),
        "score_all": distribution(row["score"] for row in records),
        "score_ok": distribution(row["score"] for row in ok),
        "score_thresholds_all": thresholds(row["score"] for row in records),
        "components_ok": {
            key: distribution(row[f"component_{key}"] for row in ok)
            for key in COMPONENT_KEYS
        },
        "physics_ok": {
            key: distribution(row[key] for row in ok)
            for key in (
                "qh_error",
                "qa_error",
                "qp_error",
                "abs_iota",
                "inverse_aspect_ratio",
                "surface_level",
            )
        },
    }


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in records for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def load_rescore_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for shard in sorted(path.glob("shard_[0-9][0-9].jsonl")):
        rows.extend(load_jsonl(shard))
    return rows


def storage_summary(root: Path) -> dict[str, Any]:
    categories: dict[str, int] = defaultdict(int)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name == "training_trace.npz":
            category = "optimizer_training_trace"
        elif "screening" in path.parts:
            category = "screening"
        elif path.name in {"history.jsonl", "center_native_results.jsonl.gz"}:
            category = "optimizer_scores_and_history"
        elif path.name in {"best.json", "final.json", "optimizer_state.pt"}:
            category = "optimizer_checkpoints"
        elif "acceptance_rescore" in path.parts:
            category = "acceptance_rescore"
        else:
            category = "metadata_and_logs"
        categories[category] += path.stat().st_size
    return {
        "total_bytes": int(sum(categories.values())),
        "by_category_bytes": dict(sorted(categories.items())),
    }


def plot_score_overview(
    output: Path,
    groups: dict[str, list[dict[str, Any]]],
    online_best: np.ndarray,
    independent_best: np.ndarray,
    independent_status: list[str],
    selected: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "random_screen": "Random flow candidates",
        "selected_start": "Best of 32 starts",
        "adam_best_global": "Adam best, global rescore",
        "quasr": "QUASR QH",
    }
    colors = {
        "random_screen": "#8a8f98",
        "selected_start": "#db8b2c",
        "adam_best_global": "#b13d3d",
        "quasr": "#18708c",
    }
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    bins = np.linspace(0, 100, 51)
    for name in labels:
        values = np.asarray([row["score"] for row in groups[name]], dtype=float)
        axes[0, 0].hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color=colors[name],
            label=f"{labels[name]} (n={len(values)})",
        )
        ordered = np.sort(values)
        survival = (len(ordered) - np.arange(len(ordered))) / max(len(ordered), 1)
        axes[0, 1].plot(ordered, survival, color=colors[name], linewidth=2, label=labels[name])
    axes[0, 0].set(xlabel="native score", ylabel="density", title="Current ABI-10 score distributions")
    axes[0, 1].set(
        xlabel="native score",
        ylabel="fraction at or above score",
        title="Upper-tail survival",
        yscale="log",
        ylim=(5e-4, 1.05),
    )
    axes[1, 0].scatter(selected, online_best, s=18, alpha=0.55, color="#b13d3d")
    axes[1, 0].plot([45, 100], [45, 100], "--", color="#555555", linewidth=1)
    axes[1, 0].set(
        xlabel="best-of-32 global start score",
        ylabel="online Adam best score",
        title="Optimization gain across 309 trajectories",
        xlim=(45, 100),
        ylim=(45, 100),
    )
    ok = np.asarray([value == "ok" for value in independent_status])
    axes[1, 1].scatter(
        online_best[ok], independent_best[ok], s=18, alpha=0.58, color="#18708c", label="global status=ok"
    )
    if np.any(~ok):
        axes[1, 1].scatter(
            online_best[~ok], independent_best[~ok], s=24, alpha=0.8, marker="x", color="#b13d3d", label="global rejected"
        )
    axes[1, 1].plot([45, 100], [45, 100], "--", color="#555555", linewidth=1)
    axes[1, 1].set(
        xlabel="online continuation best score",
        ylabel="independent global rescore",
        title="History-independent reproducibility",
        xlim=(45, 100),
        ylim=(0, 100),
    )
    for axis in axes.flat:
        axis.grid(alpha=0.18)
        axis.legend(fontsize=8)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_optimization(
    output: Path,
    curves: np.ndarray,
    best_iterations: np.ndarray,
    gains: np.ndarray,
    trajectory_wall: np.ndarray,
    conditions: list[tuple[int, int]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    steps = np.arange(curves.shape[1])
    quantiles = np.percentile(curves, [10, 25, 50, 75, 90], axis=0)
    axes[0, 0].fill_between(steps, quantiles[0], quantiles[4], color="#d8e7ec", label="P10-P90")
    axes[0, 0].fill_between(steps, quantiles[1], quantiles[3], color="#8ebdca", label="P25-P75")
    axes[0, 0].plot(steps, quantiles[2], color="#125f75", linewidth=2.2, label="median")
    axes[0, 0].set(xlabel="Adam step", ylabel="online score", title="Score evolution over all trajectories")

    axes[0, 1].hist(best_iterations, bins=np.arange(-0.5, 201.5, 10), color="#c15b43", alpha=0.85)
    axes[0, 1].set(xlabel="step of online best", ylabel="trajectories", title="Where the best score was reached")

    axes[1, 0].hist(gains, bins=30, color="#d39132", alpha=0.85)
    axes[1, 0].axvline(0, color="#555555", linestyle="--")
    axes[1, 0].set(xlabel="online best minus selected start", ylabel="trajectories", title="Optimization gain")

    n_coils = sorted({value[1] for value in conditions})
    values = [trajectory_wall[[item[1] == n for item in conditions]] / 60.0 for n in n_coils]
    axes[1, 1].boxplot(values, tick_labels=[str(value) for value in n_coils], showfliers=False)
    axes[1, 1].set(xlabel="base-coil count", ylabel="wall minutes per trajectory", title="Cost grows with coil count")
    for axis in axes.flat:
        axis.grid(alpha=0.18)
        if axis is axes[0, 0]:
            axis.legend()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_components(
    output: Path,
    groups: dict[str, list[dict[str, Any]]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ("random_screen", "adam_best_global", "quasr")
    labels = ("Random flow", "Adam best", "QUASR QH")
    colors = ("#969ba3", "#b13d3d", "#18708c")
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    x = np.arange(len(COMPONENT_KEYS))
    width = 0.24
    for offset, (name, label, color) in enumerate(zip(names, labels, colors, strict=True)):
        ok = [row for row in groups[name] if row["status"] == "ok"]
        medians = [np.nanmedian([row[f"component_{key}"] for row in ok]) for key in COMPONENT_KEYS]
        lower = [np.nanpercentile([row[f"component_{key}"] for row in ok], 25) for key in COMPONENT_KEYS]
        upper = [np.nanpercentile([row[f"component_{key}"] for row in ok], 75) for key in COMPONENT_KEYS]
        medians = np.asarray(medians)
        axes[0].bar(x + (offset - 1) * width, medians, width, color=color, label=label)
        axes[0].errorbar(
            x + (offset - 1) * width,
            medians,
            yerr=[medians - lower, np.asarray(upper) - medians],
            fmt="none",
            ecolor="#333333",
            capsize=2,
            linewidth=0.8,
        )
    axes[0].set_xticks(x, COMPONENT_KEYS, rotation=25, ha="right")
    axes[0].set(ylabel="component score (median and IQR)", title="Seven score components, valid samples only", ylim=(0, 103))

    physics_keys = ("qh_error", "abs_iota", "inverse_aspect_ratio")
    positions = []
    values = []
    tick_positions = []
    tick_labels = []
    index = 1
    for key in physics_keys:
        tick_positions.append(index + 1)
        tick_labels.append(key.replace("_", " "))
        for name in names:
            ok_values = np.asarray([row[key] for row in groups[name] if row["status"] == "ok"], dtype=float)
            ok_values = ok_values[np.isfinite(ok_values)]
            if key == "qh_error":
                ok_values = np.log10(np.maximum(ok_values, 1e-12))
            positions.append(index)
            values.append(ok_values)
            index += 1
        index += 1
    boxes = axes[1].boxplot(values, positions=positions, widths=0.72, showfliers=False, patch_artist=True)
    for box, color in zip(boxes["boxes"], colors * len(physics_keys), strict=True):
        box.set_facecolor(color)
        box.set_alpha(0.75)
    axes[1].set_xticks(tick_positions, tick_labels)
    axes[1].set(ylabel="log10(QH error), |iota|, or inverse aspect ratio", title="Physical diagnostics, valid samples only")
    for label, color in zip(labels, colors, strict=True):
        axes[1].plot([], [], color=color, linewidth=8, label=label)
    for axis in axes:
        axis.grid(alpha=0.18)
        axis.legend(fontsize=8)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a completed screened-Adam trajectory corpus.")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("rescore_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    dataset_manifest = json.loads((args.dataset_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((args.dataset_root / "trajectories").glob("*/trajectory_manifest.json"))
    ]
    streams = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((args.dataset_root / "streams").glob("*/progress.json"))
    ]
    rescore_rows = load_rescore_rows(args.rescore_dir)
    expected_rescore_count = 1024 + len(manifests)
    if len(rescore_rows) != expected_rescore_count:
        raise RuntimeError(f"expected {expected_rescore_count} rescore rows, found {len(rescore_rows)}")

    random_records: list[dict[str, Any]] = []
    selected_records: list[dict[str, Any]] = []
    online_best_records: list[dict[str, Any]] = []
    curves = []
    history_rows: list[dict[str, Any]] = []
    condition_counts: Counter[str] = Counter()
    trajectory_rows = []
    for manifest in manifests:
        trajectory_dir = args.dataset_root / "trajectories" / manifest["trajectory_id"]
        nfp = int(manifest["condition"]["nfp"])
        n_coils = int(manifest["condition"]["n_base_coils"])
        condition_counts[manifest["condition"]["group"]] += 1
        screen_rows = load_jsonl(trajectory_dir / "screening" / "screening_native_results.jsonl.gz")
        for index, row in enumerate(screen_rows):
            random_records.append(
                native_record(
                    source="random_screen",
                    case_id=f"{manifest['trajectory_id']}:{index}",
                    nfp=nfp,
                    n_coils=n_coils,
                    result=row["native_score"],
                )
            )
        selected_index = int(manifest["screening"]["selected_index"])
        selected_records.append(
            native_record(
                source="selected_start",
                case_id=manifest["trajectory_id"],
                nfp=nfp,
                n_coils=n_coils,
                result=screen_rows[selected_index]["native_score"],
            )
        )
        best_payload = json.loads((trajectory_dir / "optimization" / "best.json").read_text(encoding="utf-8"))
        best_meta = best_payload["flow_prior_local_full_gradient_adam"]
        online_best_records.append(
            native_record(
                source="adam_best_online",
                case_id=manifest["trajectory_id"],
                nfp=nfp,
                n_coils=n_coils,
                result=best_meta["native_score"],
                best_iteration=int(manifest["optimization"]["best_iteration"]),
            )
        )
        history = load_jsonl(trajectory_dir / "optimization" / "history.jsonl")
        if len(history) != 200:
            raise RuntimeError(f"{manifest['trajectory_id']} has {len(history)} history rows")
        curves.append(
            [float(manifest["optimization"]["initial_score"])]
            + [float(row["current_score"]) for row in history]
        )
        history_rows.extend(history)
        trajectory_rows.append(
            {
                "trajectory_id": manifest["trajectory_id"],
                "nfp": nfp,
                "n_base_coils": n_coils,
                "selected_global_score": float(selected_records[-1]["score"]),
                "initial_online_score": float(manifest["optimization"]["initial_score"]),
                "final_online_score": float(manifest["optimization"]["final_score"]),
                "best_online_score": float(manifest["optimization"]["best_score"]),
                "best_iteration": int(manifest["optimization"]["best_iteration"]),
                "screening_wall_s": float(manifest["timing"]["screening_process_wall_s"]),
                "optimization_wall_s": float(manifest["timing"]["optimization_process_wall_s"]),
                "trajectory_wall_s": float(manifest["timing"]["trajectory_wall_s"]),
            }
        )

    independent_best_records = []
    quasr_records = []
    trajectory_by_id = {row["trajectory_id"]: row for row in trajectory_rows}
    for row in rescore_rows:
        record = native_record(
            source="adam_best_global" if row["kind"] == "adam_best" else "quasr",
            case_id=str(row["case_id"]),
            nfp=int(row["nfp"]),
            n_coils=int(row["n_base_coils"]),
            result=row["native_score"],
            score_wall_s=row["score_wall_s"],
        )
        if row["kind"] == "adam_best":
            record["online_best_score"] = float(row["online_best_score"])
            record["online_best_iteration"] = int(row["online_best_iteration"])
            independent_best_records.append(record)
            trajectory_by_id[row["case_id"]]["best_global_score"] = record["score"]
            trajectory_by_id[row["case_id"]]["best_global_status"] = record["status"]
        else:
            record["source_id"] = int(row["source_id"])
            quasr_records.append(record)
    independent_best_records.sort(key=lambda row: row["case_id"])
    online_best_records.sort(key=lambda row: row["case_id"])
    selected_records.sort(key=lambda row: row["case_id"])
    trajectory_rows.sort(key=lambda row: row["trajectory_id"])

    groups = {
        "random_screen": random_records,
        "selected_start": selected_records,
        "adam_best_online": online_best_records,
        "adam_best_global": independent_best_records,
        "quasr": quasr_records,
    }
    prior = {row["label"]: row for row in dataset_manifest["condition_prior"]["groups"]}
    condition_table = []
    for label, expected in prior.items():
        observed = condition_counts[label]
        condition_table.append(
            {
                "condition": label,
                "nfp": expected["nfp"],
                "n_base_coils": expected["n_coils"],
                "expected_probability": expected["probability"],
                "expected_count": expected["probability"] * len(manifests),
                "observed_count": observed,
                "observed_probability": observed / max(len(manifests), 1),
            }
        )
    tv_distance = 0.5 * sum(
        abs(row["observed_probability"] - row["expected_probability"])
        for row in condition_table
    )

    selected_scores = np.asarray([row["selected_global_score"] for row in trajectory_rows])
    online_scores = np.asarray([row["best_online_score"] for row in trajectory_rows])
    global_scores = np.asarray([row["best_global_score"] for row in trajectory_rows])
    global_status = [row["best_global_status"] for row in trajectory_rows]
    online_gain = online_scores - selected_scores
    global_gain = global_scores - selected_scores
    best_iterations = np.asarray([row["best_iteration"] for row in trajectory_rows])
    trajectory_wall = np.asarray([row["trajectory_wall_s"] for row in trajectory_rows])
    curve_array = np.asarray(curves, dtype=np.float64)

    aggregate_elapsed = max(float(row["elapsed_s"]) for row in streams)
    total_gpu_s = sum(float(row["elapsed_s"]) for row in streams)
    history_timing = {
        key: distribution(float(row.get(key, float("nan"))) for row in history_rows)
        for key in (
            "iteration_wall_s",
            "gradient_wall_s",
            "flow_pipeline_decode_wall_s",
            "proposal_score_wall_s",
            "raw_gradient_rms",
            "update_rms",
        )
    }
    history_timing["local_score_wall_s"] = distribution(
        float(row.get("gradient_pipeline", {}).get("timing_s", {}).get("local_score", float("nan")))
        for row in history_rows
    )

    summary = {
        "format": "qh_adam_trajectory_acceptance_v1",
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_manifest": dataset_manifest,
        "integrity": {
            "completed_trajectories": len(manifests),
            "failure_directories": len(list((args.dataset_root / "failures").glob("*"))),
            "incomplete_directories": len(list((args.dataset_root / "incomplete").glob("*"))),
            "stream_count": len(streams),
            "stream_stop_reasons": dict(sorted(Counter(row["stop_reason"] for row in streams).items())),
            "stream_completed_counts": [int(row["completed_trajectories"]) for row in streams],
        },
        "data_counts": {
            "trajectories": len(manifests),
            "screening_random_candidates": len(random_records),
            "screening_selected_starts": len(selected_records),
            "adam_steps": len(history_rows),
            "stored_gradient_vectors": len(history_rows),
            "stored_random_directions": len(history_rows) * 64,
            "stored_directional_endpoints": len(history_rows) * 128,
            "stored_center_states_including_initial": len(manifests) * 201,
            "independent_best_rescores": len(independent_best_records),
            "independent_quasr_rescores": len(quasr_records),
        },
        "condition_sampling": {
            "observed_counts": dict(sorted(condition_counts.items())),
            "total_variation_from_exact_train_prior": tv_distance,
            "table": condition_table,
        },
        "groups": {name: summarize_records(records) for name, records in groups.items()},
        "optimization": {
            "online_gain_best_minus_selected": distribution(online_gain),
            "independent_gain_best_minus_selected": distribution(global_gain),
            "independent_gain_positive_fraction": float(np.mean(global_gain > 0)),
            "best_iteration": distribution(best_iterations),
            "best_in_last_10_steps_fraction": float(np.mean(best_iterations >= 191)),
            "best_in_last_25_steps_fraction": float(np.mean(best_iterations >= 176)),
            "best_in_last_50_steps_fraction": float(np.mean(best_iterations >= 151)),
            "online_vs_independent_best": correlation(online_scores, global_scores),
            "independent_minus_online": distribution(global_scores - online_scores),
        },
        "timing": {
            "trajectory_wall_s": distribution(trajectory_wall),
            "screening_process_wall_s": distribution(row["screening_wall_s"] for row in trajectory_rows),
            "optimization_process_wall_s": distribution(row["optimization_wall_s"] for row in trajectory_rows),
            "history_step_timing": history_timing,
            "six_stream_elapsed_s": aggregate_elapsed,
            "total_gpu_hours": total_gpu_s / 3600.0,
            "observed_trajectories_per_day_six_gpu": len(manifests) * 86400.0 / aggregate_elapsed,
            "gpu_minutes_per_trajectory": total_gpu_s / 60.0 / len(manifests),
        },
        "storage": storage_summary(args.dataset_root),
    }

    args.output_dir.joinpath("summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    write_records(args.output_dir / "score_records.csv", sum(groups.values(), []))
    write_records(args.output_dir / "trajectory_summary.csv", trajectory_rows)
    write_records(args.output_dir / "condition_counts.csv", condition_table)
    plot_score_overview(
        args.output_dir / "score_distributions.png",
        groups,
        online_scores,
        global_scores,
        global_status,
        selected_scores,
    )
    plot_optimization(
        args.output_dir / "optimization_dynamics.png",
        curve_array,
        best_iterations,
        online_gain,
        trajectory_wall,
        [(row["nfp"], row["n_base_coils"]) for row in trajectory_rows],
    )
    plot_components(args.output_dir / "components_and_physics.png", groups)
    print(json.dumps(summary, separators=(",", ":"), allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
