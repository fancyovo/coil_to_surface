from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stellarator_eval import EvalConfig
from stellarator_eval.field import FieldInput
from stellarator_eval.pipeline import evaluate_field_input
from stellarator_eval.quasr import (
    build_quasr_metadata_index,
    choose_quasr_eval_params,
    load_quasr_field_input,
    load_quasr_metadata,
)
from stellarator_eval.score import ScoreConfig, evaluate_quality_score
from stellarator_eval.serialization import jsonable, write_json


QUASR_ROOT_ENV = "QUASR_ROOT"
QUASR_METADATA_ENV = "QUASR_METADATA"


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return None if not value else Path(value).expanduser()


def _parse_csv_numbers(text: str, typ=float) -> list[Any]:
    return [typ(x) for x in text.replace(",", " ").split() if x.strip()]


def _load_metadata_from_batches(paths: list[Path]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for path in paths:
        batch = json.loads(path.read_text(encoding="utf-8"))
        for run in batch.get("runs") or []:
            device_id = run.get("device_id")
            if device_id is None:
                device_id = ((run.get("quasr_info") or {}).get("device_id"))
            metadata = run.get("metadata")
            if device_id is not None and metadata:
                out[int(device_id)] = metadata
    return out


def _perturb_field_input(base: FieldInput, *, noise_level: float, rng: np.random.Generator) -> FieldInput:
    def perturb(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        return arr * (1.0 + rng.normal(0.0, noise_level, size=arr.shape))

    return FieldInput(
        coeffs_x=perturb(base.coeffs_x),
        coeffs_y=perturb(base.coeffs_y),
        coeffs_z=perturb(base.coeffs_z),
        currents=np.asarray(base.currents, dtype=float).copy(),
        nfp=int(base.nfp),
        name=f"{base.name}_noise_{noise_level:.3g}",
    )


def _make_config(args: argparse.Namespace, eval_params: dict[str, Any]) -> EvalConfig:
    cfg = EvalConfig()
    cfg.current_unit = "A"
    cfg.omp_threads = args.omp_threads
    cfg.axis.gpu_device = args.gpu_device
    cfg.psi.gpu_device = args.gpu_device
    cfg.scan.gpu_device = args.gpu_device
    cfg.boozer.gpu_device = args.gpu_device
    cfg.psi.a = float(eval_params["a"])
    cfg.boozer.initial_iota = float(eval_params["initial_iota"])
    cfg.psi.n_r = args.psi_n_r
    cfg.psi.n_z = args.psi_n_z
    cfg.psi.n_phi = args.psi_n_phi
    cfg.psi.linear_solver = args.psi_linear_solver
    cfg.psi.normal_eq_precision = args.psi_normal_eq_precision
    cfg.boozer.surface_order = args.surface_order
    cfg.boozer.qs_sdim = args.qs_sdim
    cfg.scan.max_boozer_candidates = args.max_boozer_candidates
    if args.levels:
        cfg.scan.levels = tuple(_parse_csv_numbers(args.levels, float))
    return cfg


def _target_from_metadata(metadata: dict[str, Any] | None) -> str | None:
    if metadata is None or metadata.get("helicity") is None:
        return None
    return "QA" if int(metadata["helicity"]) == 0 else "QH"


def _flatten_score(score: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"score": score["score"], "status": score["status"]}
    for key, value in score["components"].items():
        row[f"component_{key}"] = value
    details = score["details"]
    for key in (
        "selected_qs_error",
        "selected_qs_target",
        "screen_best_psi_level",
        "screen_min_rel_distance_p95",
        "coil_length_mean",
        "coil_curvature_p95",
        "coil_curvature_max",
        "coil_min_intercoil_distance",
        "coil_min_axis_distance",
        "coil_high_mode_energy_fraction",
        "coil_current_abs_max_a",
    ):
        if key in details:
            row[key] = details[key]
    return row


def _flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    axis = result.get("axis") or {}
    fit = ((result.get("psi") or {}).get("fit_info") or {})
    best = result.get("best_surface") or {}
    timing = result.get("timing") or {}
    row = {
        "axis_has_axis": axis.get("has_axis"),
        "axis_best_residual": axis.get("best_residual"),
        "axis_topology_class": axis.get("topology_class"),
        "psi_validation_angle_p95": fit.get("validation_angle_p95"),
        "psi_validation_angle_l2": fit.get("validation_angle_l2"),
        "best_surface_iota": best.get("iota"),
        "best_surface_volume": best.get("volume"),
        "best_surface_G": best.get("G"),
        "best_surface_newton_residual": best.get("newton_residual_norm"),
        "total_time_s": result.get("total_time_s"),
    }
    for key in (
        "axis_s",
        "axis_trace_s",
        "axis_topology_s",
        "psi_fit_s",
        "surface_screen_s",
        "surface_screen_fieldline_trace_s",
        "surface_extract_1d_newton_s",
        "boozer_ls_s",
        "boozer_newton_s",
        "boozer_qs_s",
        "boozer_candidates_s",
    ):
        if key in timing:
            row[f"timing_{key}"] = timing[key]
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot_noise_curves(rows: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    for device_id in sorted({int(row["ID"]) for row in rows}):
        sub = [row for row in rows if int(row["ID"]) == device_id]
        levels = sorted({float(row["noise_level"]) for row in sub})
        means = []
        lows = []
        highs = []
        for level in levels:
            vals = np.asarray([float(row["score"]) for row in sub if float(row["noise_level"]) == level], dtype=float)
            means.append(float(np.mean(vals)))
            lows.append(float(np.min(vals)))
            highs.append(float(np.max(vals)))
        ax.plot(levels, means, marker="o", linewidth=1.8, label=str(device_id))
        ax.fill_between(levels, lows, highs, alpha=0.12)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("noise_level")
    ax.set_ylabel("score")
    ax.set_ylim(0, 100)
    ax.set_title("Score under multiplicative Fourier-coefficient noise")
    ax.grid(alpha=0.25)
    ax.legend(title="ID", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "noise_score_curves.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=150)
    component_names = ["axis", "psi", "surface", "boozer", "physics", "coil"]
    for component in component_names:
        levels = sorted({float(row["noise_level"]) for row in rows})
        means = []
        for level in levels:
            vals = [
                float(row[f"component_{component}"])
                for row in rows
                if float(row["noise_level"]) == level and row.get(f"component_{component}") is not None
            ]
            means.append(float(np.mean(vals)) if vals else np.nan)
        ax.plot(levels, means, marker=".", linewidth=1.4, label=component)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("noise_level")
    ax.set_ylabel("component score")
    ax.set_ylim(0, 100)
    ax.set_title("Mean component scores under noise")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "noise_component_curves.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate selected QUASR devices after multiplicative coefficient noise.")
    parser.add_argument("--quasr-root", type=Path, default=_env_path(QUASR_ROOT_ENV), required=False)
    parser.add_argument("--metadata", type=Path, default=_env_path(QUASR_METADATA_ENV), required=False)
    parser.add_argument("--metadata-from-batch", action="append", type=Path, default=[])
    parser.add_argument("--ids", required=True, help="Comma- or space-separated QUASR IDs.")
    parser.add_argument("--levels", default="0,0.001,0.003,0.01,0.03,0.1,0.2,0.3")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/score_experiment/noise"))
    parser.add_argument("--default-a", type=float, default=0.05)
    parser.add_argument("--a-minor-fraction", type=float, default=0.9)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--omp-threads", type=int, default=1)
    parser.add_argument("--psi-n-r", type=int, default=80)
    parser.add_argument("--psi-n-z", type=int, default=80)
    parser.add_argument("--psi-n-phi", type=int, default=80)
    parser.add_argument("--psi-linear-solver", choices=["qr", "normal_eq"], default="qr")
    parser.add_argument("--psi-normal-eq-precision", choices=["fp32", "fp64"], default="fp32")
    parser.add_argument("--surface-order", type=int, default=6)
    parser.add_argument("--qs-sdim", type=int, default=16)
    parser.add_argument("--max-boozer-candidates", type=int, default=3)
    parser.add_argument("--screen-levels", dest="levels_override", default=None)
    args = parser.parse_args()
    if args.quasr_root is None:
        raise ValueError(f"QUASR root is required; pass --quasr-root or set {QUASR_ROOT_ENV}")

    metadata_index: dict[int, dict[str, Any]] = {}
    if args.metadata is not None:
        metadata_index = build_quasr_metadata_index(load_quasr_metadata(args.metadata))
    if args.metadata_from_batch:
        metadata_index.update(_load_metadata_from_batches(args.metadata_from_batch))

    ids = _parse_csv_numbers(args.ids, int)
    noise_levels = _parse_csv_numbers(args.levels, float)
    if args.levels_override is not None:
        args.levels = args.levels_override
    else:
        args.levels = None
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    cfg_score = ScoreConfig()
    for device_id in ids:
        base, info = load_quasr_field_input(args.quasr_root, int(device_id))
        metadata = metadata_index.get(int(device_id))
        adapt = choose_quasr_eval_params(
            metadata,
            default_a=args.default_a,
            a_minor_fraction=args.a_minor_fraction,
        )
        cfg = _make_config(args, adapt)
        target = _target_from_metadata(metadata)
        for noise_level in noise_levels:
            nrep = 1 if float(noise_level) == 0.0 else args.repeats
            for rep in range(nrep):
                field_input = base if float(noise_level) == 0.0 else _perturb_field_input(base, noise_level=float(noise_level), rng=rng)
                run_dir = out_dir / f"id_{int(device_id):07d}" / f"noise_{float(noise_level):.3g}".replace(".", "p") / f"rep_{rep:02d}"
                result = evaluate_field_input(field_input, config=cfg, output_dir=run_dir)
                score = evaluate_quality_score(
                    result,
                    field_input=field_input,
                    current_unit="A",
                    metadata=metadata,
                    target=target,
                    config=cfg_score,
                )
                row = {
                    "ID": int(device_id),
                    "noise_level": float(noise_level),
                    "repeat": int(rep),
                    "target": target,
                    "nfp": int(info["nfp"]),
                }
                row.update(_flatten_score(score))
                row.update(_flatten_result(result))
                rows.append(row)
                write_json(run_dir / "quality_score.json", score)
                print(
                    f"ID={int(device_id):7d} noise={float(noise_level):.3g} rep={rep:02d} "
                    f"status={score['status']:>10s} score={score['score']:.2f} total={result.get('total_time_s', 0.0):.2f}s"
                )
    _write_csv(out_dir / "noise_score_rows.csv", rows)
    write_json(out_dir / "noise_score_rows.json", rows)
    _plot_noise_curves(rows, out_dir)

    by_level: dict[str, Any] = {}
    for level in sorted({float(row["noise_level"]) for row in rows}):
        vals = np.asarray([float(row["score"]) for row in rows if float(row["noise_level"]) == level], dtype=float)
        by_level[f"{level:.6g}"] = {
            "count": int(vals.size),
            "mean": float(np.mean(vals)),
            "min": float(np.min(vals)),
            "median": float(np.median(vals)),
            "max": float(np.max(vals)),
        }
    write_json(out_dir / "noise_summary.json", {"rows": len(rows), "by_level": by_level})


if __name__ == "__main__":
    main()
