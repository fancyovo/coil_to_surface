from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COMPONENTS = ("axis", "psi", "surface", "coordinate", "volume_qs", "iota", "coil")
RK4_STEPS = (64, 128, 256)
LEARNING_RATES = (0.003, 0.01, 0.03, 0.05, 0.1)
JOB_METADATA = {
    (64, 0.003): (31855, "COMPLETED", "00:26:29"),
    (64, 0.01): (31860, "FAILED", "00:10:18"),
    (64, 0.03): (31861, "FAILED", "00:04:15"),
    (64, 0.05): (31862, "FAILED", "00:04:12"),
    (64, 0.1): (31856, "FAILED", "00:03:50"),
    (128, 0.003): (31863, "COMPLETED", "00:29:19"),
    (128, 0.01): (31857, "FAILED", "00:11:15"),
    (128, 0.03): (31864, "FAILED", "00:06:13"),
    (128, 0.05): (31865, "FAILED", "00:03:55"),
    (128, 0.1): (31866, "FAILED", "00:03:21"),
    (256, 0.003): (31858, "COMPLETED", "00:37:58"),
    (256, 0.01): (31867, "FAILED", "00:13:27"),
    (256, 0.03): (31868, "FAILED", "00:07:23"),
    (256, 0.05): (31859, "FAILED", "00:05:37"),
    (256, 0.1): (31869, "FAILED", "00:03:20"),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def run_name(steps: int, learning_rate: float) -> str:
    return f"rk4_{steps:03d}_lr_{str(learning_rate).replace('.', 'p')}"


def elapsed_seconds(value: str) -> int:
    hours, minutes, seconds = (int(part) for part in value.split(":"))
    return 3600 * hours + 60 * minutes + seconds


def load_best_score(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if "flow_prior_physical_gradient_adam" in payload:
        return payload["flow_prior_physical_gradient_adam"]["native_score"]
    return payload["flow_prior_standard_adam"]["native_score"]


def postflight_metrics(path: Path) -> tuple[int, int, bool]:
    rows = [line.strip().split(",") for line in path.read_text().splitlines() if line.strip()]
    maximum_utilization = max(int(row[-3].strip()) for row in rows)
    maximum_memory = max(int(row[-2].strip()) for row in rows)
    return maximum_utilization, maximum_memory, bool(rows) and maximum_memory <= 2


def summarize_run(root: Path, steps: int, learning_rate: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    directory = root / run_name(steps, learning_rate)
    history = read_jsonl(directory / "history.jsonl")
    manifest = read_json(directory / "manifest.json")
    job_id, state, elapsed = JOB_METADATA[(steps, learning_rate)]
    best_row = max(history, key=lambda row: float(row["current_score"]))
    final_row = history[-1]
    postflight_utilization, postflight_memory, postflight_clean = postflight_metrics(
        directory / "gpu_postflight.csv"
    )
    iteration_rows = history[1:]
    mean_iteration_wall_s = float(
        np.mean([float(row["iteration_wall_s"]) for row in iteration_rows])
    )
    peak_surface = float(best_row["surface_level"])
    first_surface_drop = next(
        (
            int(row["iteration"])
            for row in history
            if int(row["iteration"]) > int(best_row["iteration"])
            and float(row["surface_level"]) < peak_surface
        ),
        None,
    )
    return (
        {
            "job_id": job_id,
            "state": state,
            "slurm_elapsed": elapsed,
            "slurm_elapsed_s": elapsed_seconds(elapsed),
            "rk4_steps": steps,
            "learning_rate": learning_rate,
            "initial_score": float(history[0]["current_score"]),
            "best_score": float(best_row["current_score"]),
            "best_iteration": int(best_row["iteration"]),
            "last_score": float(final_row["current_score"]),
            "completed_iterations": int(final_row["iteration"]),
            "score_gain_at_best": float(best_row["current_score"] - history[0]["current_score"]),
            "post_peak_change": float(final_row["current_score"] - best_row["current_score"]),
            "best_surface_level": peak_surface,
            "last_surface_level": float(final_row["surface_level"]),
            "best_qh_error": float(best_row["qh_error"]),
            "last_qh_error": float(final_row["qh_error"]),
            "best_iota": float(best_row["iota"]),
            "last_iota": float(final_row["iota"]),
            "first_surface_drop_after_peak": first_surface_drop,
            "mean_iteration_wall_s": mean_iteration_wall_s,
            "total_recorded_wall_s": float(final_row["total_wall_s"]),
            "postflight_utilization_percent": postflight_utilization,
            "postflight_memory_mib": postflight_memory,
            "postflight_clean": postflight_clean,
            "initial_noise_float32_sha256": manifest["initial_noise_float32_sha256"],
            "gradient_lib_sha256": manifest["gradient_lib_sha256"],
            "failure_reason": (
                None
                if state == "COMPLETED"
                else "G2 wrapper raised when the next candidate had a valid non-OK score"
            ),
        },
        history,
    )


def plot_score_curves(
    rows: list[dict[str, Any]],
    histories: dict[tuple[int, float], list[dict[str, Any]]],
    old_history: list[dict[str, Any]],
    old_initial_score: float,
    output: Path,
) -> None:
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(LEARNING_RATES)))
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.3), sharey=True, constrained_layout=True)
    for axis, steps in zip(axes, RK4_STEPS):
        for color, learning_rate in zip(colors, LEARNING_RATES):
            history = histories[(steps, learning_rate)]
            x = [row["iteration"] for row in history]
            y = [row["current_score"] for row in history]
            summary = next(
                row for row in rows
                if row["rk4_steps"] == steps and row["learning_rate"] == learning_rate
            )
            axis.plot(x, y, color=color, lw=1.5, label=f"lr={learning_rate:g}")
            axis.scatter(
                [summary["best_iteration"]], [summary["best_score"]],
                marker="*", s=65, color=color, edgecolor="black", linewidth=0.35, zorder=4,
            )
            if summary["state"] != "COMPLETED":
                axis.scatter([x[-1]], [y[-1]], marker="x", s=34, color=color, zorder=4)
        if steps == 256:
            old_x = [0] + [row["iteration"] for row in old_history]
            old_y = [old_initial_score] + [row["current_score"] for row in old_history]
            axis.plot(old_x, old_y, color="black", lw=2.0, ls="--", label="old K=4 Adam")
        axis.set(title=f"RK4-{steps}", xlabel="Adam step", ylim=(78.5, 94.0))
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("exact native score")
    axes[-1].legend(fontsize=8, loc="lower right")
    figure.suptitle("Physical G2-VJP Adam: complete and failed partial trajectories")
    figure.savefig(output, dpi=190)
    plt.close(figure)


def plot_best_heatmap(rows: list[dict[str, Any]], output: Path) -> None:
    values = np.empty((len(RK4_STEPS), len(LEARNING_RATES)))
    completed = np.empty_like(values, dtype=int)
    states = np.empty_like(values, dtype=object)
    best_steps = np.empty_like(values, dtype=int)
    for i, steps in enumerate(RK4_STEPS):
        for j, learning_rate in enumerate(LEARNING_RATES):
            row = next(
                item for item in rows
                if item["rk4_steps"] == steps and item["learning_rate"] == learning_rate
            )
            values[i, j] = row["best_score"]
            completed[i, j] = row["completed_iterations"]
            states[i, j] = row["state"]
            best_steps[i, j] = row["best_iteration"]
    figure, axis = plt.subplots(figsize=(10.5, 4.6), constrained_layout=True)
    image = axis.imshow(values, cmap="RdYlGn", vmin=88.5, vmax=93.2, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            marker = "C" if states[i, j] == "COMPLETED" else "F"
            axis.text(
                j,
                i,
                f"{values[i, j]:.3f}\nbest@{best_steps[i, j]}\n{marker} {completed[i, j]}/200",
                ha="center",
                va="center",
                fontsize=8,
            )
    axis.set_xticks(range(len(LEARNING_RATES)), [f"{value:g}" for value in LEARNING_RATES])
    axis.set_yticks(range(len(RK4_STEPS)), [str(value) for value in RK4_STEPS])
    axis.set(xlabel="learning rate", ylabel="RK4 steps", title="Best exact score (C=completed, F=failed)")
    figure.colorbar(image, ax=axis, label="best score")
    figure.savefig(output, dpi=190)
    plt.close(figure)


def plot_completed_diagnostics(
    rows: list[dict[str, Any]],
    histories: dict[tuple[int, float], list[dict[str, Any]]],
    old_history: list[dict[str, Any]],
    old_initial_score: float,
    output: Path,
) -> None:
    representative = histories[(256, 0.003)]
    iteration = [row["iteration"] for row in representative]
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for steps in RK4_STEPS:
        history = histories[(steps, 0.003)]
        axes[0, 0].plot(
            [row["iteration"] for row in history],
            [row["current_score"] for row in history],
            label=f"G2 RK4-{steps}",
        )
    axes[0, 0].plot(
        [0] + [row["iteration"] for row in old_history],
        [old_initial_score] + [row["current_score"] for row in old_history],
        color="black",
        ls="--",
        label="old K=4 RK4-256",
    )
    axes[0, 0].set(title="Completed trajectories", ylabel="score")
    axes[0, 0].legend(fontsize=8)
    for name in COMPONENTS:
        axes[0, 1].plot(iteration, [row["components"][name] for row in representative], label=name)
    axes[0, 1].set(title="RK4-256, lr=0.003 components", ylabel="component")
    axes[0, 1].legend(ncol=2, fontsize=8)
    for key, label in (("qh_error", "QH"), ("qa_error", "QA"), ("qp_error", "QP")):
        axes[0, 2].plot(iteration, [row[key] for row in representative], label=label)
    axes[0, 2].set(title="Volume QS diagnostics", yscale="log", ylabel="residual")
    axes[0, 2].legend()
    axes[1, 0].step(iteration, [row["surface_level"] for row in representative], where="post", label="surface level")
    second_axis = axes[1, 0].twinx()
    second_axis.plot(iteration, [row["iota"] for row in representative], color="tab:red", label="iota")
    axes[1, 0].set(title="Selected surface and iota", xlabel="step", ylabel="surface level")
    second_axis.set_ylabel("iota", color="tab:red")
    axes[1, 1].plot(iteration, [max(row["gradient_rms"], 1e-30) for row in representative], label="gradient RMS")
    axes[1, 1].plot(iteration, [max(row["update_rms"], 1e-30) for row in representative], label="update RMS")
    axes[1, 1].set(title="Latent gradient and update", xlabel="step", yscale="log")
    axes[1, 1].legend()
    timing = []
    labels = []
    for steps in RK4_STEPS:
        history = histories[(steps, 0.003)][1:]
        timing.append(
            [
                np.median([row["flow_decode_wall_s"] for row in history]),
                np.median([row["native_provider_wall_s"] for row in history]),
                np.median([row["flow_backward_wall_s"] for row in history]),
            ]
        )
        labels.append(str(steps))
    timing_array = np.asarray(timing)
    bottom = np.zeros(len(RK4_STEPS))
    for index, label in enumerate(("flow forward", "native G2", "flow VJP")):
        axes[1, 2].bar(labels, timing_array[:, index], bottom=bottom, label=label)
        bottom += timing_array[:, index]
    axes[1, 2].set(title="Median accepted-step latency", xlabel="RK4 steps", ylabel="seconds")
    axes[1, 2].legend(fontsize=8)
    for axis in axes.ravel():
        axis.grid(alpha=0.22)
    figure.savefig(output, dpi=190)
    plt.close(figure)


def plot_best_components(
    best_new_score: dict[str, Any],
    old_best_score: dict[str, Any],
    output: Path,
) -> None:
    x = np.arange(len(COMPONENTS))
    width = 0.38
    figure, axis = plt.subplots(figsize=(11.5, 5.2), constrained_layout=True)
    axis.bar(
        x - width / 2,
        [best_new_score["components"][name] for name in COMPONENTS],
        width,
        label=f"G2-VJP best {best_new_score['score']:.3f}",
    )
    axis.bar(
        x + width / 2,
        [old_best_score["components"][name] for name in COMPONENTS],
        width,
        label=f"old K=4 best {old_best_score['score']:.3f}",
    )
    axis.set_xticks(x, COMPONENTS, rotation=20)
    axis.set(ylabel="component score", ylim=(55, 102), title="Best physical-gradient candidate vs old score-93 baseline")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(output, dpi=190)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    args = parser.parse_args()

    summaries: list[dict[str, Any]] = []
    histories: dict[tuple[int, float], list[dict[str, Any]]] = {}
    for steps in RK4_STEPS:
        for learning_rate in LEARNING_RATES:
            summary, history = summarize_run(args.sweep_root, steps, learning_rate)
            summaries.append(summary)
            histories[(steps, learning_rate)] = history

    old_summary = read_json(args.old_root / "summary.json")
    old_history = read_jsonl(args.old_root / "history.jsonl")
    old_best_score = load_best_score(args.old_root / "best.json")
    best_new = max(summaries, key=lambda row: row["best_score"])
    best_new_score = load_best_score(
        args.sweep_root / run_name(best_new["rk4_steps"], best_new["learning_rate"]) / "best.json"
    )
    completed = [row for row in summaries if row["state"] == "COMPLETED"]
    output_summary = {
        "format": "qh_physical_gradient_adam_sweep_acceptance_v1",
        "runs": summaries,
        "completed_count": len(completed),
        "failed_count": len(summaries) - len(completed),
        "all_postflights_clean": all(row["postflight_clean"] for row in summaries),
        "unique_initial_noise_hashes": sorted({row["initial_noise_float32_sha256"] for row in summaries}),
        "unique_gradient_library_hashes": sorted({row["gradient_lib_sha256"] for row in summaries}),
        "best_new_run": best_new,
        "old_baseline": {
            "initial_score": old_summary["initial_score"],
            "best_score": old_summary["best_score"],
            "best_iteration": old_summary["best_iteration"],
            "final_score": old_summary["final_score"],
            "mean_iteration_wall_s": old_summary["mean_iteration_wall_s"],
            "total_wall_s": old_summary["total_wall_s"],
        },
        "completed_speedups_vs_old": {
            str(row["rk4_steps"]): old_summary["mean_iteration_wall_s"] / row["mean_iteration_wall_s"]
            for row in completed
        },
        "best_score_gap_to_old": float(best_new["best_score"] - old_summary["best_score"]),
        "best_component_comparison": {
            "new": best_new_score["components"],
            "old": old_best_score["components"],
        },
    }
    (args.sweep_root / "acceptance_summary.json").write_text(
        json.dumps(output_summary, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    with (args.sweep_root / "acceptance_runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    plot_score_curves(
        summaries,
        histories,
        old_history,
        float(old_summary["initial_score"]),
        args.sweep_root / "score_curves.png",
    )
    plot_best_heatmap(summaries, args.sweep_root / "best_score_heatmap.png")
    plot_completed_diagnostics(
        summaries,
        histories,
        old_history,
        float(old_summary["initial_score"]),
        args.sweep_root / "completed_diagnostics.png",
    )
    plot_best_components(
        best_new_score,
        old_best_score,
        args.sweep_root / "best_components.png",
    )
    print(json.dumps(output_summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
