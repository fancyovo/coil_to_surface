from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


COMPONENTS = ("axis", "psi", "surface", "coordinate", "volume_qs", "iota", "coil")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def rms(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array)))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 0.0 else float("nan")


def run_key(path: Path) -> tuple[int, float]:
    manifest = read_json(path / "manifest.json")
    return int(manifest["rk4_steps"]), float(manifest["learning_rate"])


def load_run(path: Path) -> dict[str, Any]:
    rk4_steps, learning_rate = run_key(path)
    history = read_jsonl(path / "history.jsonl")
    states = [read_json(path / "trajectory" / f"step_{index:04d}.json") for index in range(len(history))]
    noise = np.asarray([state["noise"] for state in states], dtype=np.float64)
    gradient = np.asarray([state["latent_gradient"] for state in states], dtype=np.float64)
    gradient_rms = np.asarray([rms(item) for item in gradient], dtype=np.float64)
    score = np.asarray([row["current_score"] for row in history], dtype=np.float64)
    surface = np.asarray([row["surface_level"] for row in history], dtype=np.float64)
    components = {
        name: np.asarray([row["components"][name] for row in history], dtype=np.float64)
        for name in COMPONENTS
    }
    delta_noise = noise[1:] - noise[:-1]
    score_delta = score[1:] - score[:-1]
    predicted_delta = np.einsum("ij,ij->i", gradient[:-1].reshape(len(delta_noise), -1), delta_noise.reshape(len(delta_noise), -1))
    update_cosine = np.asarray(
        [cosine(gradient[index], delta_noise[index]) for index in range(len(delta_noise))]
    )
    gradient_cosine = np.asarray(
        [cosine(gradient[index], gradient[index + 1]) for index in range(len(gradient) - 1)]
    )
    step_rms = np.asarray([rms(item) for item in delta_noise], dtype=np.float64)
    arc = np.concatenate(([0.0], np.cumsum(step_rms)))
    best_index = int(np.argmax(score))
    later_drop = np.flatnonzero(
        (np.arange(len(surface)) > best_index) & (surface < surface[best_index] - 1.0e-12)
    )
    first_surface_drop = int(later_drop[0]) if later_drop.size else len(score)
    return {
        "path": path,
        "rk4_steps": rk4_steps,
        "learning_rate": learning_rate,
        "history": history,
        "noise": noise,
        "gradient": gradient,
        "gradient_rms": gradient_rms,
        "score": score,
        "surface": surface,
        "components": components,
        "score_delta": score_delta,
        "predicted_delta": predicted_delta,
        "update_cosine": update_cosine,
        "gradient_cosine": gradient_cosine,
        "step_rms": step_rms,
        "arc": arc,
        "best_index": best_index,
        "first_surface_drop": first_surface_drop,
    }


def interval_statistics(run: dict[str, Any], begin: int, end: int) -> dict[str, Any]:
    actual = run["score_delta"][begin:end]
    predicted = run["predicted_delta"][begin:end]
    update_cosine = run["update_cosine"][begin:end]
    gradient_cosine = run["gradient_cosine"][begin:end]
    gradient_rms = run["gradient_rms"][begin : end + 1]
    return {
        "transition_begin": int(begin),
        "transition_end_exclusive": int(end),
        "count": int(len(actual)),
        "actual_score_change": float(np.sum(actual)),
        "predicted_score_change": float(np.sum(predicted)),
        "actual_negative_count": int(np.count_nonzero(actual < 0.0)),
        "g_dot_update_positive_count": int(np.count_nonzero(predicted > 0.0)),
        "opposed_count": int(np.count_nonzero((predicted > 0.0) & (actual < 0.0))),
        "median_actual_delta": float(np.median(actual)) if len(actual) else float("nan"),
        "median_g_dot_update": float(np.median(predicted)) if len(predicted) else float("nan"),
        "median_gradient_update_cosine": float(np.median(update_cosine)) if len(update_cosine) else float("nan"),
        "median_consecutive_gradient_cosine": float(np.median(gradient_cosine)) if len(gradient_cosine) else float("nan"),
        "min_consecutive_gradient_cosine": float(np.min(gradient_cosine)) if len(gradient_cosine) else float("nan"),
        "gradient_rms_min": float(np.min(gradient_rms)) if len(gradient_rms) else float("nan"),
        "gradient_rms_median": float(np.median(gradient_rms)) if len(gradient_rms) else float("nan"),
        "gradient_rms_max": float(np.max(gradient_rms)) if len(gradient_rms) else float("nan"),
    }


def path_alignment(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    max_tau = min(
        reference["learning_rate"] * (len(reference["score"]) - 1),
        candidate["learning_rate"] * (len(candidate["score"]) - 1),
    )
    tau = np.linspace(0.0, max_tau, 256)
    reference_tau = reference["learning_rate"] * np.arange(len(reference["score"]))
    candidate_tau = candidate["learning_rate"] * np.arange(len(candidate["score"]))
    reference_score = np.interp(tau, reference_tau, reference["score"])
    candidate_score = np.interp(tau, candidate_tau, candidate["score"])
    score_rmse = rms(reference_score - candidate_score)

    ref_flat = reference["noise"].reshape(len(reference["noise"]), -1)
    cand_flat = candidate["noise"].reshape(len(candidate["noise"]), -1)
    ref_interp = np.stack([np.interp(tau, reference_tau, ref_flat[:, column]) for column in range(ref_flat.shape[1])], axis=1)
    cand_interp = np.stack([np.interp(tau, candidate_tau, cand_flat[:, column]) for column in range(cand_flat.shape[1])], axis=1)
    latent_rms = np.sqrt(np.mean((ref_interp - cand_interp) ** 2, axis=1))
    ref_displacement = np.sqrt(np.mean((ref_interp - ref_interp[0]) ** 2, axis=1))
    scale = max(float(np.max(ref_displacement)), 1.0e-30)
    return {
        "reference_lr": reference["learning_rate"],
        "candidate_lr": candidate["learning_rate"],
        "max_eta_step": float(max_tau),
        "score_rmse": score_rmse,
        "latent_path_rms_median": float(np.median(latent_rms)),
        "latent_path_rms_max": float(np.max(latent_rms)),
        "latent_path_max_relative_to_reference_displacement": float(np.max(latent_rms) / scale),
    }


def prebranch_path_alignment(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    max_tau = min(
        reference["learning_rate"] * max(reference["first_surface_drop"] - 1, 0),
        candidate["learning_rate"] * max(candidate["first_surface_drop"] - 1, 0),
    )
    tau = np.linspace(0.0, max_tau, 256)
    reference_tau = reference["learning_rate"] * np.arange(len(reference["score"]))
    candidate_tau = candidate["learning_rate"] * np.arange(len(candidate["score"]))
    reference_score = np.interp(tau, reference_tau, reference["score"])
    candidate_score = np.interp(tau, candidate_tau, candidate["score"])
    return {
        "prebranch_max_eta_step": float(max_tau),
        "prebranch_score_rmse": rms(reference_score - candidate_score),
        "reference_peak_eta_step": float(reference["learning_rate"] * reference["best_index"]),
        "candidate_peak_eta_step": float(candidate["learning_rate"] * candidate["best_index"]),
        "reference_surface_drop_eta_step": float(reference["learning_rate"] * reference["first_surface_drop"]),
        "candidate_surface_drop_eta_step": float(candidate["learning_rate"] * candidate["first_surface_drop"]),
    }


def component_change(run: dict[str, Any], begin: int, end: int) -> dict[str, float]:
    return {
        name: float(run["components"][name][end] - run["components"][name][begin])
        for name in COMPONENTS
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(runs: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    colors = {0.003: "#3b528b", 0.01: "#21918c", 0.03: "#5ec962"}
    for rk4_steps in (64, 128, 256):
        axis = axes[0, 0] if rk4_steps == 64 else axes[0, 1] if rk4_steps == 128 else axes[1, 0]
        for run in runs:
            lr = run["learning_rate"]
            if run["rk4_steps"] != rk4_steps or lr not in colors:
                continue
            eta_step = lr * np.arange(len(run["score"]))
            axis.plot(eta_step, run["score"], color=colors[lr], label=f"lr={lr:g}")
            best = run["best_index"]
            axis.scatter(lr * best, run["score"][best], color=colors[lr], marker="*", s=75, zorder=3)
        axis.set(title=f"RK4-{rk4_steps}: score vs learning-rate time", xlabel=r"$\eta k$", ylabel="exact score")
        axis.grid(alpha=0.25)
        axis.legend()

    axis = axes[1, 1]
    run = next(item for item in runs if item["rk4_steps"] == 64 and item["learning_rate"] == 0.003)
    transitions = np.arange(1, len(run["score"]))
    axis.plot(transitions, run["score_delta"], label=r"exact $\Delta S$", color="#d1495b")
    axis.plot(transitions, run["predicted_delta"], label=r"$g_{G2}\cdot\Delta z$", color="#167c80")
    axis.axvline(run["best_index"], color="black", linestyle="--", linewidth=1.2, label="score peak")
    axis.axvline(run["first_surface_drop"], color="#e09f3e", linestyle=":", linewidth=1.8, label="first surface drop")
    axis.set(title="RK4-64, lr=0.003: predicted and realized one-step change", xlabel="transition ending at step", ylabel="score units")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_components(runs: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    complete = [run for run in runs if run["learning_rate"] == 0.003]
    x = np.arange(len(COMPONENTS))
    width = 0.24
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    for offset, run in enumerate(complete):
        best = run["best_index"]
        before_drop = run["first_surface_drop"] - 1
        changes = component_change(run, best, before_drop)
        axes[0].bar(x + (offset - 1) * width, [changes[name] for name in COMPONENTS], width=width, label=f"RK4-{run['rk4_steps']}")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set(title="Component change: score peak to just before branch drop", ylabel="component score change", xticks=x, xticklabels=COMPONENTS)
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    for run in complete:
        steps = np.arange(1, len(run["score"]))
        axes[1].plot(steps, run["gradient_cosine"], label=f"RK4-{run['rk4_steps']}")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set(title="Consecutive G2 latent-gradient cosine", xlabel="transition ending at step", ylabel="cosine", ylim=(-1.02, 1.02))
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_postpeak_bias(runs: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    complete = [run for run in runs if run["learning_rate"] == 0.003]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True, sharey=True)
    for axis, run in zip(axes, complete):
        begin = run["best_index"]
        end = run["first_surface_drop"] - 1
        steps = np.arange(begin + 1, end + 1)
        actual = np.cumsum(run["score_delta"][begin:end])
        predicted = np.cumsum(run["predicted_delta"][begin:end])
        axis.plot(steps, actual, color="#d1495b", linewidth=2.0, label=r"cumulative exact $\Delta S$")
        axis.plot(steps, predicted, color="#167c80", linewidth=2.0, label=r"cumulative $g_{G2}\cdot\Delta z$")
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set(title=f"RK4-{run['rk4_steps']}", xlabel="step before first surface-level change")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("change from score peak")
    axes[0].legend()
    figure.suptitle("Systematic post-peak mismatch on the unchanged surface branch", fontsize=14)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose deterministic bias in the saved physical-gradient Adam sweep.")
    parser.add_argument("--sweep-root", type=Path, required=True)
    args = parser.parse_args()

    run_paths = sorted(path for path in args.sweep_root.iterdir() if path.is_dir() and (path / "manifest.json").exists())
    runs = [load_run(path) for path in run_paths]
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for run in runs:
        best = run["best_index"]
        drop = run["first_surface_drop"]
        post_end = min(drop - 1, len(run["score"]) - 1)
        summary = {
            "rk4_steps": run["rk4_steps"],
            "learning_rate": run["learning_rate"],
            "recorded_steps": len(run["score"]) - 1,
            "best_step": best,
            "best_score": float(run["score"][best]),
            "first_surface_drop": drop if drop < len(run["score"]) else None,
            "peak_to_pre_drop_score_change": float(run["score"][post_end] - run["score"][best]) if post_end >= best else float("nan"),
            "peak_to_pre_drop_components": component_change(run, best, post_end) if post_end >= best else {},
            "all_transitions": interval_statistics(run, 0, len(run["score_delta"])),
            "post_peak_same_surface": interval_statistics(run, best, max(best, drop - 1)),
        }
        summaries.append(summary)
        for index, (actual, predicted, update_cosine, gradient_cosine) in enumerate(
            zip(run["score_delta"], run["predicted_delta"], run["update_cosine"], run["gradient_cosine"]), start=1
        ):
            rows.append({
                "rk4_steps": run["rk4_steps"],
                "learning_rate": run["learning_rate"],
                "step": index,
                "score": run["score"][index],
                "score_delta": actual,
                "g_dot_update": predicted,
                "gradient_update_cosine": update_cosine,
                "consecutive_gradient_cosine": gradient_cosine,
                "step_rms": run["step_rms"][index - 1],
                "arc_length": run["arc"][index],
                "surface_level": run["surface"][index],
            })

    alignments: list[dict[str, Any]] = []
    for rk4_steps in (64, 128, 256):
        family = sorted((run for run in runs if run["rk4_steps"] == rk4_steps), key=lambda run: run["learning_rate"])
        reference = next(run for run in family if run["learning_rate"] == 0.003)
        for candidate in family:
            if candidate["learning_rate"] in (0.01, 0.03):
                alignments.append({
                    "rk4_steps": rk4_steps,
                    **path_alignment(reference, candidate),
                    **prebranch_path_alignment(reference, candidate),
                })

    output = {
        "format": "qh_physical_gradient_bias_analysis_v1",
        "sweep_root": str(args.sweep_root.resolve()),
        "interpretation": "g_dot_update uses the G2 latent gradient saved at step k and the actual Adam displacement z[k+1]-z[k]; score_delta is the independently recomputed ABI-9 score change.",
        "runs": summaries,
        "eta_time_path_alignments": alignments,
    }
    with (args.sweep_root / "bias_analysis.json").open("w", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    write_csv(args.sweep_root / "bias_transitions.csv", rows)
    plot_summary(runs, args.sweep_root / "bias_time_alignment.png")
    plot_components(runs, args.sweep_root / "bias_components_and_gradient.png")
    plot_postpeak_bias(runs, args.sweep_root / "bias_postpeak_zoom.png")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
