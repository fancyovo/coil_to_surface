from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def number(row: dict, key: str, default: float = float("nan")) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def statistics(values) -> dict:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "p05": float(np.percentile(array, 5)),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    for index in np.flatnonzero(counts > 1):
        members = inverse == index
        ranks[members] = np.mean(ranks[members])
    return ranks


def spearman(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(keep) < 3:
        return float("nan")
    return float(np.corrcoef(rank(x[keep]), rank(y[keep]))[0, 1])


def top_audit(rows: list[dict], score_key: str, count: int = 20) -> dict:
    top = sorted(rows, key=lambda row: number(row, score_key), reverse=True)[:count]
    return {
        "count": len(top),
        "case_ids": [int(row["case_id"]) for row in top],
        "score": statistics(number(row, score_key) for row in top),
        "absolute_iota": statistics(abs(number(row, "iota")) for row in top),
        "native_qs_global_error": statistics(number(row, "qs_global_error") for row in top),
        "metadata_qs_error": statistics(number(row, "metadata_qs_error") for row in top),
        "surface_inverse_aspect_ratio": statistics(
            number(row, "inverse_aspect_ratio") for row in top
        ),
        "below_unit_iota_count": sum(abs(number(row, "iota")) < 1.0 for row in top),
    }


def extract_native_score(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "native_score" in payload:
        return payload["native_score"]
    return payload["cem"]["native_score"]


def quasr_metadata_summary(path: Path) -> dict:
    rows = load_csv(path)
    qh = [row for row in rows if row.get("helicity", "").strip() == "1"]
    qh_nfp3 = [row for row in qh if int(row["nfp"]) == 3]

    def group(rows_: list[dict]) -> dict:
        ordered = sorted(rows_, key=lambda row: number(row, "qs_error"))
        best_tenth = ordered[: max(1, len(ordered) // 10)]
        return {
            "count": len(rows_),
            "absolute_mean_iota": statistics(abs(number(row, "mean_iota")) for row in rows_),
            "below_unit_iota_count": sum(abs(number(row, "mean_iota")) < 1.0 for row in rows_),
            "base_coil_counts": dict(sorted(Counter(int(row["n_base_coils"]) for row in rows_).items())),
            "best_metadata_qs_tenth_count": len(best_tenth),
            "best_metadata_qs_tenth_base_coil_counts": dict(
                sorted(Counter(int(row["n_base_coils"]) for row in best_tenth).items())
            ),
            "best_metadata_qs_tenth_absolute_mean_iota": statistics(
                abs(number(row, "mean_iota")) for row in best_tenth
            ),
        }

    return {"all_qh": group(qh), "qh_nfp3": group(qh_nfp3)}


def plot_comparison(rows: list[dict], output: Path) -> None:
    old_score = np.asarray([number(row, "old_score") for row in rows])
    new_score = np.asarray([number(row, "new_score") for row in rows])
    iota = np.asarray([abs(number(row, "iota")) for row in rows])
    qs = np.asarray([number(row, "qs_global_error") for row in rows])
    size = np.asarray([number(row, "inverse_aspect_ratio") for row in rows])

    figure, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))
    scatter = axes[0, 0].scatter(
        old_score, new_score, c=iota, cmap="viridis", s=22, alpha=0.72, edgecolors="none"
    )
    limits = [min(np.min(old_score), np.min(new_score)), max(np.max(old_score), np.max(new_score))]
    axes[0, 0].plot(limits, limits, color="#555555", linestyle=":", linewidth=1.2)
    axes[0, 0].set(xlabel="Old score", ylabel="Score v2")
    figure.colorbar(scatter, ax=axes[0, 0], label=r"Native $|\iota|$")

    axes[0, 1].scatter(iota, old_score, s=18, alpha=0.45, label="old", color="#6f7f8f")
    axes[0, 1].scatter(iota, new_score, s=18, alpha=0.55, label="v2", color="#b43b2f")
    axes[0, 1].axvline(1.0, color="#555555", linestyle=":", linewidth=1.2)
    axes[0, 1].set(xlabel=r"Native $|\iota|$", ylabel="Score")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].scatter(qs, old_score, s=18, alpha=0.45, label="old", color="#6f7f8f")
    axes[1, 0].scatter(qs, new_score, s=18, alpha=0.55, label="v2", color="#b43b2f")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set(xlabel="Native differential QH error", ylabel="Score")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].scatter(size, old_score, s=18, alpha=0.45, label="old", color="#6f7f8f")
    axes[1, 1].scatter(size, new_score, s=18, alpha=0.55, label="v2", color="#b43b2f")
    axes[1, 1].axvline(0.03, color="#555555", linestyle=":", linewidth=1.2)
    axes[1, 1].set(xlabel="Selected surface inverse aspect ratio", ylabel="Score")
    axes[1, 1].legend(frameon=False)

    for axis in axes.flat:
        axis.grid(alpha=0.18)
    figure.suptitle("QH score v2 anti-degeneracy audit on matched QUASR samples")
    figure.tight_layout()
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-rows", type=Path, required=True)
    parser.add_argument("--new-rows", type=Path, required=True)
    parser.add_argument("--degenerate-old", type=Path, required=True)
    parser.add_argument("--degenerate-new", type=Path, required=True)
    parser.add_argument("--quasr-metadata", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    old_by_id = {int(row["case_id"]): row for row in load_csv(args.old_rows)}
    new_by_id = {int(row["case_id"]): row for row in load_csv(args.new_rows)}
    common_ids = sorted(set(old_by_id) & set(new_by_id))
    rows = []
    for case_id in common_ids:
        old = old_by_id[case_id]
        new = new_by_id[case_id]
        if old["status"] != "ok" or new["status"] != "ok" or int(new["helicity"]) != 1:
            continue
        row = dict(new)
        row["old_score"] = old["score"]
        row["new_score"] = new["score"]
        rows.append(row)
    if not rows:
        raise RuntimeError("no matched successful QH rows")

    old_scores = [number(row, "old_score") for row in rows]
    new_scores = [number(row, "new_score") for row in rows]
    quality = [-math.log10(max(number(row, "metadata_qs_error"), 1.0e-300)) for row in rows]
    native_quality = [-math.log10(max(number(row, "qs_global_error"), 1.0e-300)) for row in rows]
    iota = [abs(number(row, "iota")) for row in rows]
    size = [number(row, "inverse_aspect_ratio") for row in rows]
    old_degenerate = extract_native_score(args.degenerate_old)
    new_degenerate = extract_native_score(args.degenerate_new)
    new_degenerate_score = float(new_degenerate["score"])
    summary = {
        "matched_successful_qh_count": len(rows),
        "score": {"old": statistics(old_scores), "v2": statistics(new_scores)},
        "spearman": {
            "old_score_vs_metadata_qs_quality": spearman(old_scores, quality),
            "v2_score_vs_metadata_qs_quality": spearman(new_scores, quality),
            "old_score_vs_native_qs_quality": spearman(old_scores, native_quality),
            "v2_score_vs_native_qs_quality": spearman(new_scores, native_quality),
            "old_score_vs_absolute_iota": spearman(old_scores, iota),
            "v2_score_vs_absolute_iota": spearman(new_scores, iota),
            "old_score_vs_surface_size": spearman(old_scores, size),
            "v2_score_vs_surface_size": spearman(new_scores, size),
        },
        "absolute_iota": statistics(iota),
        "below_unit_iota": {
            "count": sum(value < 1.0 for value in iota),
            "old_score": statistics(score for score, value in zip(old_scores, iota) if value < 1.0),
            "v2_score": statistics(score for score, value in zip(new_scores, iota) if value < 1.0),
        },
        "top20": {"old": top_audit(rows, "old_score"), "v2": top_audit(rows, "new_score")},
        "degenerate_case": {
            "old_score": float(old_degenerate["score"]),
            "v2_score": new_degenerate_score,
            "v2_quasr_percentile": float(100.0 * np.mean(np.asarray(new_scores) <= new_degenerate_score)),
            "v2_components": new_degenerate["components"],
            "v2_iota": float(new_degenerate["diagnostics"]["iota_min"]),
            "v2_qs_global_error": float(new_degenerate["diagnostics"]["qs_global_error"]),
        },
    }
    if args.quasr_metadata:
        summary["quasr_metadata"] = quasr_metadata_summary(args.quasr_metadata)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "score_v2_anticheat.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    plot_comparison(rows, args.output_dir / "score_v2_anticheat.png")


if __name__ == "__main__":
    main()
