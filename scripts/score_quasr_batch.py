from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stellarator_eval.score import ScoreConfig, evaluate_quality_score


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _flatten(prefix: str, data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            out[f"{prefix}_{key}"] = value
    return out


def _target_from_batch_name(path: Path) -> str | None:
    text = str(path).upper()
    if "QH" in text:
        return "QH"
    if "QA" in text:
        return "QA"
    return None


def _ref_value(value: Any) -> str | None:
    if isinstance(value, dict) and value.get("$type") == "ref":
        return str(value.get("value"))
    return None


def _array_data(value: Any) -> np.ndarray:
    if isinstance(value, dict) and value.get("@class") == "array":
        return np.asarray(value.get("data"), dtype=float)
    return np.asarray(value, dtype=float)


def _quasr_serial_path(quasr_root: Path, device_id: int) -> Path:
    f_id = int(device_id) // 1000
    return quasr_root / "simsopt_serials" / f"{f_id:04d}" / f"serial{int(device_id):07d}.json"


def _resolve_current(objs: dict[str, Any], ref: str) -> float:
    obj = objs[ref]
    cls = obj.get("@class")
    if cls == "Current":
        return float(obj.get("current"))
    if cls == "ScaledCurrent":
        inner = _ref_value(obj.get("current_to_scale"))
        if inner is None:
            raise ValueError(f"ScaledCurrent {ref} has no current_to_scale ref")
        return float(obj.get("scale")) * _resolve_current(objs, inner)
    raise ValueError(f"unsupported current object {ref}: {cls}")


def _load_field_input_from_serial_json(quasr_root: Path, device_id: int) -> Any:
    serial_path = _quasr_serial_path(quasr_root, int(device_id))
    data = json.loads(serial_path.read_text(encoding="utf-8"))
    objs = data["simsopt_objs"]
    nfp = None
    for obj in objs.values():
        if obj.get("@class") == "SurfaceXYZTensorFourier":
            nfp = int(obj["nfp"])
            break
    if nfp is None:
        raise ValueError(f"{serial_path} has no SurfaceXYZTensorFourier nfp")
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    currents: list[float] = []
    for obj in objs.values():
        if obj.get("@class") != "Coil":
            continue
        curve_ref = _ref_value(obj.get("curve"))
        current_ref = _ref_value(obj.get("current"))
        if curve_ref is None or current_ref is None:
            continue
        curve = objs.get(curve_ref)
        if not curve or curve.get("@class") != "CurveXYZFourier":
            continue
        dofs_ref = _ref_value(curve.get("dofs"))
        if dofs_ref is None:
            continue
        dofs_obj = objs[dofs_ref]
        dofs = _array_data(dofs_obj["x"])
        order = int(curve["order"])
        n_coeff = 2 * order + 1
        if dofs.size != 3 * n_coeff:
            raise ValueError(f"{serial_path}: CurveXYZFourier dof size {dofs.size} incompatible with order={order}")
        xs.append(dofs[:n_coeff])
        ys.append(dofs[n_coeff : 2 * n_coeff])
        zs.append(dofs[2 * n_coeff :])
        currents.append(_resolve_current(objs, current_ref))
    if not xs:
        raise ValueError(f"{serial_path} has no direct CurveXYZFourier coils")
    return SimpleNamespace(
        coeffs_x=np.asarray(xs, dtype=float),
        coeffs_y=np.asarray(ys, dtype=float),
        coeffs_z=np.asarray(zs, dtype=float),
        currents=np.asarray(currents, dtype=float),
        nfp=int(nfp),
        name=f"quasr_{int(device_id):07d}",
        n_base_coils=len(xs),
        order=(np.asarray(xs).shape[1] - 1) // 2,
    )


def _load_field_input(quasr_root: Path | None, device_id: Any) -> tuple[Any | None, str]:
    if quasr_root is None or device_id is None:
        return None, "none"
    try:
        return _load_field_input_from_serial_json(quasr_root, int(device_id)), "loaded_json"
    except Exception as json_exc:
        json_status = repr(json_exc)
    try:
        from stellarator_eval.quasr import load_quasr_field_input

        field_input, _ = load_quasr_field_input(quasr_root, int(device_id))
        return field_input, "loaded_simsopt"
    except Exception as exc:
        return None, f"json={json_status}; simsopt={exc!r}"


def _row_from_run(
    batch_path: Path,
    run: dict[str, Any],
    index: int,
    cfg: ScoreConfig,
    target: str | None,
    *,
    quasr_root: Path | None,
    field_cache: dict[int, tuple[Any | None, str]],
) -> dict[str, Any]:
    result = run.get("result") or {}
    metadata = run.get("metadata")
    device_id = run.get("device_id")
    if device_id is None:
        device_id = ((run.get("quasr_info") or {}).get("device_id"))
    field_input = None
    coil_input_status = "none"
    if device_id is not None:
        did = int(device_id)
        if did not in field_cache:
            field_cache[did] = _load_field_input(quasr_root, did)
        field_input, coil_input_status = field_cache[did]
    score = evaluate_quality_score(
        result,
        field_input=field_input,
        current_unit="A",
        metadata=metadata,
        target=target,
        config=cfg,
    )
    row: dict[str, Any] = {
        "source_batch": batch_path.parent.name,
        "source_index": index,
        "ID": device_id,
        "status": score["status"],
        "score": score["score"],
        "coil_input_status": coil_input_status,
    }
    row.update(_flatten("component", score["components"]))
    details = score["details"]
    for key in (
        "selected_qs_error",
        "selected_qs_target",
        "screen_best_psi_level",
        "screen_min_rel_distance_p95",
        "axis_residual_score",
        "psi_angle_p95_score",
        "coil_missing",
        "coil_length_mean",
        "coil_length_max",
        "coil_curvature_p95",
        "coil_curvature_max",
        "coil_min_intercoil_distance",
        "coil_min_axis_distance",
        "coil_current_abs_mean_a",
        "coil_current_abs_max_a",
        "coil_current_cv",
        "coil_high_mode_energy_fraction",
    ):
        if key in details:
            row[key] = details[key]
    axis = result.get("axis") or {}
    row["axis_best_residual"] = axis.get("best_residual")
    row["axis_topology_class"] = axis.get("topology_class")
    fit = ((result.get("psi") or {}).get("fit_info") or {})
    row["psi_validation_angle_p95"] = fit.get("validation_angle_p95")
    row["psi_validation_angle_l2"] = fit.get("validation_angle_l2")
    best = result.get("best_surface") or {}
    for key in ("iota", "volume", "G", "newton_residual_norm", "initial_boozer_residual_norm"):
        if key in best:
            row[f"best_surface_{key}"] = best.get(key)
    timing = result.get("timing") or {}
    row["total_time_s"] = result.get("total_time_s")
    for key in ("axis_s", "psi_fit_s", "surface_screen_s", "boozer_candidates_s"):
        if key in timing:
            row[f"timing_{key}"] = timing.get(key)
    if metadata:
        for key in ("helicity", "nfp", "mean_iota", "minor_radius", "aspect_ratio", "qs_error"):
            if key in metadata:
                row[f"meta_{key}"] = metadata[key]
    return row


def _plot_histograms(rows: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    scores = np.asarray([_finite(row.get("score")) for row in rows], dtype=float)
    scores = scores[np.isfinite(scores)]
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=150)
    bins = np.linspace(0.0, 100.0, 31)
    ax.hist(scores, bins=bins, color="#386cb0", alpha=0.82, edgecolor="white")
    ax.set_xlabel("score")
    ax.set_ylabel("count")
    ax.set_title("QUASR score distribution")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "score_histogram.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=150)
    colors = {"surface": "#1b9e77", "no_surface": "#d95f02", "no_axis": "#7570b3"}
    for status in ("surface", "no_surface", "no_axis"):
        vals = np.asarray([_finite(row.get("score")) for row in rows if row.get("status") == status], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            ax.hist(vals, bins=bins, alpha=0.62, edgecolor="white", label=status, color=colors.get(status))
    ax.set_xlabel("score")
    ax.set_ylabel("count")
    ax.set_title("QUASR score by status")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "score_histogram_by_status.png")
    plt.close(fig)

    component_names = ["axis", "psi", "surface", "boozer", "physics", "coil"]
    data = [
        [float(row[f"component_{name}"]) for row in rows if row.get(f"component_{name}") is not None]
        for name in component_names
    ]
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    ax.boxplot(data, tick_labels=component_names, showfliers=False)
    ax.set_ylim(0, 100)
    ax.set_ylabel("component score")
    ax.set_title("Score component distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "component_boxplot.png")
    plt.close(fig)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = np.asarray([_finite(row.get("score")) for row in rows], dtype=float)
    scores = scores[np.isfinite(scores)]
    by_status: dict[str, Any] = {}
    for status in sorted({str(row.get("status")) for row in rows}):
        vals = np.asarray([_finite(row.get("score")) for row in rows if row.get("status") == status], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            by_status[status] = {
                "count": int(vals.size),
                "mean": float(np.mean(vals)),
                "p10": float(np.percentile(vals, 10)),
                "median": float(np.median(vals)),
                "p90": float(np.percentile(vals, 90)),
            }
    return {
        "count": len(rows),
        "status_counts": dict(Counter(str(row.get("status")) for row in rows)),
        "score": {
            "mean": float(np.mean(scores)) if scores.size else None,
            "p10": float(np.percentile(scores, 10)) if scores.size else None,
            "median": float(np.median(scores)) if scores.size else None,
            "p90": float(np.percentile(scores, 90)) if scores.size else None,
            "min": float(np.min(scores)) if scores.size else None,
            "max": float(np.max(scores)) if scores.size else None,
        },
        "by_status": by_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Add continuous quality scores to existing QUASR batch summaries.")
    parser.add_argument("--batch-summary", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/score_experiment/quasr_batch_scores"))
    parser.add_argument("--quasr-root", type=Path, default=None, help="Optional QUASR root. If set, coil engineering metrics are included.")
    parser.add_argument("--target", choices=["auto", "QA", "QH", "QP", "min"], default="auto")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    cfg = ScoreConfig()
    rows: list[dict[str, Any]] = []
    field_cache: dict[int, tuple[Any | None, str]] = {}
    for batch_path in args.batch_summary:
        batch = _load_json(batch_path)
        batch_target = _target_from_batch_name(batch_path) if args.target == "auto" else None if args.target == "min" else args.target
        for i, run in enumerate(batch.get("runs") or []):
            rows.append(
                _row_from_run(
                    batch_path,
                    run,
                    i,
                    cfg,
                    batch_target,
                    quasr_root=args.quasr_root,
                    field_cache=field_cache,
                )
            )
    rows.sort(key=lambda row: (str(row.get("source_batch")), int(row.get("ID") or -1)))
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "score_rows.csv", rows)
    summary = _summary(rows)
    summary["score_config"] = cfg.__dict__
    _write_json(out_dir / "score_summary.json", summary)
    if not args.no_plots:
        _plot_histograms(rows, out_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
