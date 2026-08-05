from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def correlation(left: np.ndarray, right: np.ndarray, *, ranked: bool = False) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(finite) < 3:
        return float("nan")
    x = left[finite]
    y = right[finite]
    if ranked:
        x = rankdata(x)
        y = rankdata(y)
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def percentile(values: np.ndarray, fraction: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, fraction)) if finite.size else float("nan")


def top_overlap(legacy: np.ndarray, candidate: np.ndarray, count: int) -> float:
    count = min(max(count, 1), legacy.size)
    old_top = set(np.argsort(legacy)[-count:])
    new_top = set(np.argsort(candidate)[-count:])
    return len(old_top & new_top) / count


def load_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("worker_*.jsonl")):
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    variants = sorted({name for row in rows for name in row["variants"]})
    legacy_score = np.asarray([row["legacy"]["score"] for row in rows], dtype=np.float64)
    legacy_qh = np.asarray(
        [row["legacy"]["diagnostics"]["qs_target_global_error_per_helicity"] for row in rows],
        dtype=np.float64,
    )
    legacy_time = np.asarray([row["legacy"]["timing"]["total_s"] for row in rows])
    summary = {
        "count": len(rows),
        "legacy_status_counts": {},
        "legacy_score": {
            "median": percentile(legacy_score, 0.5),
            "p90": percentile(legacy_score, 0.9),
            "max": float(np.max(legacy_score)),
        },
        "legacy_total_s": {
            "median": percentile(legacy_time, 0.5),
            "p95": percentile(legacy_time, 0.95),
        },
        "variants": {},
    }
    for row in rows:
        status = row["legacy"]["status"]
        summary["legacy_status_counts"][status] = summary["legacy_status_counts"].get(status, 0) + 1
    for name in variants:
        candidate = np.asarray([row["variants"][name]["score"] for row in rows])
        candidate_qh = np.asarray([
            row["variants"][name]["diagnostics"]["qs_target_global_error_per_helicity"]
            for row in rows
        ])
        candidate_time = np.asarray([row["variants"][name]["timing"]["total_s"] for row in rows])
        surface_time = np.asarray([
            row["variants"][name]["timing"]["surface_screen_s"] for row in rows
        ])
        statuses: dict[str, int] = {}
        axis_errors = []
        for row in rows:
            result = row["variants"][name]
            statuses[result["status"]] = statuses.get(result["status"], 0) + 1
            axis_errors.append(
                np.hypot(
                    result["diagnostics"]["axis_R"] - row["legacy"]["diagnostics"]["axis_R"],
                    result["diagnostics"]["axis_Z"] - row["legacy"]["diagnostics"]["axis_Z"],
                )
            )
        high80 = legacy_score >= 80.0
        high90 = legacy_score >= 90.0
        variant_summary = {
            "status_counts": statuses,
            "score_pearson": correlation(legacy_score, candidate),
            "score_spearman": correlation(legacy_score, candidate, ranked=True),
            "score_delta_median": percentile(candidate - legacy_score, 0.5),
            "score_delta_p05": percentile(candidate - legacy_score, 0.05),
            "score_delta_p95": percentile(candidate - legacy_score, 0.95),
            "top20pct_overlap": top_overlap(legacy_score, candidate, max(1, len(rows) // 5)),
            "top10pct_overlap": top_overlap(legacy_score, candidate, max(1, len(rows) // 10)),
            "high80_count": int(np.count_nonzero(high80)),
            "high80_spearman": correlation(legacy_score[high80], candidate[high80], ranked=True),
            "high90_count": int(np.count_nonzero(high90)),
            "high90_spearman": correlation(legacy_score[high90], candidate[high90], ranked=True),
            "qh_log10_spearman": correlation(
                np.log10(legacy_qh), np.log10(candidate_qh), ranked=True
            ),
            "total_s_median": percentile(candidate_time, 0.5),
            "total_s_p95": percentile(candidate_time, 0.95),
            "surface_s_median": percentile(surface_time, 0.5),
            "speedup_median_ratio": percentile(legacy_time, 0.5) /
                percentile(candidate_time, 0.5),
            "axis_error_max": float(np.nanmax(axis_errors)),
        }
        summary["variants"][name] = variant_summary
    return summary


def plot(rows: list[dict], summary: dict, output: Path) -> None:
    names = sorted(summary["variants"])
    legacy = np.asarray([row["legacy"]["score"] for row in rows])
    figure, axes = plt.subplots(2, len(names), figsize=(4.2 * len(names), 7.2), squeeze=False)
    for column, name in enumerate(names):
        candidate = np.asarray([row["variants"][name]["score"] for row in rows])
        timing = np.asarray([row["variants"][name]["timing"]["total_s"] for row in rows])
        high = legacy >= 80.0
        axes[0, column].scatter(legacy[~high], candidate[~high], s=12, alpha=0.55, color="#557a95")
        axes[0, column].scatter(legacy[high], candidate[high], s=18, alpha=0.8, color="#c84b31")
        limits = [min(legacy.min(), candidate.min()), max(legacy.max(), candidate.max())]
        axes[0, column].plot(limits, limits, color="black", lw=0.8, ls="--")
        axes[0, column].set_title(
            f"{name}\nSpearman={summary['variants'][name]['score_spearman']:.3f}"
        )
        axes[0, column].set(xlabel="legacy score", ylabel="continuous score")
        axes[1, column].scatter(legacy, timing, s=13, alpha=0.65, color="#2d6a4f")
        axes[1, column].set(
            xlabel="legacy score",
            ylabel="continuous total time (s)",
            title=f"median {np.median(timing):.2f} s",
        )
        axes[1, column].grid(alpha=0.2)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.run_root)
    if not rows:
        raise ValueError("no benchmark rows found")
    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    plot(rows, summary, args.output_dir / "ranking_and_timing.png")
    print(json.dumps(summary, allow_nan=True))


if __name__ == "__main__":
    main()
