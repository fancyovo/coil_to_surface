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


def rescore_with_surface(result: dict, surface_component: float) -> float:
    diagnostics = result["diagnostics"]
    before = float(diagnostics["score_before_qh_iota_gate"])
    old_surface = float(result["components"]["surface"])
    replaced_before = before + 0.10 * (surface_component - old_surface)
    return replaced_before * float(diagnostics["score_qh_total_iota_factor"]) * float(
        diagnostics["score_qh_total_helicity_factor"]
    )


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
        subset = [row for row in rows if name in row["variants"]]
        old_score = np.asarray([row["legacy"]["score"] for row in subset], dtype=np.float64)
        old_qh = np.asarray([
            row["legacy"]["diagnostics"]["qs_target_global_error_per_helicity"]
            for row in subset
        ])
        old_time = np.asarray([row["legacy"]["timing"]["total_s"] for row in subset])
        candidate = np.asarray([row["variants"][name]["score"] for row in subset])
        candidate_qh = np.asarray([
            row["variants"][name]["diagnostics"]["qs_target_global_error_per_helicity"]
            for row in subset
        ])
        candidate_time = np.asarray([row["variants"][name]["timing"]["total_s"] for row in subset])
        surface_time = np.asarray([
            row["variants"][name]["timing"]["surface_screen_s"] for row in subset
        ])
        statuses: dict[str, int] = {}
        axis_errors = []
        for row in subset:
            result = row["variants"][name]
            statuses[result["status"]] = statuses.get(result["status"], 0) + 1
            axis_errors.append(
                np.hypot(
                    result["diagnostics"]["axis_R"] - row["legacy"]["diagnostics"]["axis_R"],
                    result["diagnostics"]["axis_Z"] - row["legacy"]["diagnostics"]["axis_Z"],
                )
            )
        high80 = old_score >= 80.0
        high90 = old_score >= 90.0
        component_summary = {}
        for component in subset[0]["legacy"]["components"]:
            old_component = np.asarray([
                row["legacy"]["components"][component] for row in subset
            ])
            new_component = np.asarray([
                row["variants"][name]["components"][component] for row in subset
            ])
            component_summary[component] = {
                "spearman": correlation(old_component, new_component, ranked=True),
                "delta_median": percentile(new_component - old_component, 0.5),
                "delta_p05": percentile(new_component - old_component, 0.05),
                "delta_p95": percentile(new_component - old_component, 0.95),
                "high90_spearman": correlation(
                    old_component[high90], new_component[high90], ranked=True
                ),
            }
        old_level = np.asarray([
            row["legacy"]["diagnostics"]["surface_level"] for row in subset
        ])
        new_level = np.asarray([
            row["variants"][name]["diagnostics"]["surface_level"] for row in subset
        ])
        alternative_surface = {
            "size_only": [],
            "size_confidence": [],
            "size_level": [],
            "selected_65size_35level": [],
            "valid_constant": [],
        }
        for row in subset:
            result = row["variants"][name]
            diagnostics = result["diagnostics"]
            size = 100.0 * float(diagnostics["score_surface_size"])
            confidence = 100.0 * float(diagnostics["surface_confidence_mean"])
            normalized_level = np.clip(
                float(diagnostics["surface_level"]) / 0.81, 0.0, 1.0
            )
            level_score = 100.0 * normalized_level * normalized_level * (
                3.0 - 2.0 * normalized_level
            )
            definitions = {
                "size_only": size,
                "size_confidence": 0.85 * size + 0.15 * confidence,
                "size_level": 0.75 * size + 0.25 * level_score,
                "selected_65size_35level": 0.65 * size + 0.35 * level_score,
                "valid_constant": 100.0,
            }
            for definition, surface_component in definitions.items():
                alternative_surface[definition].append(
                    rescore_with_surface(result, surface_component)
                )
        surface_ablation = {}
        for definition, values in alternative_surface.items():
            rescored = np.asarray(values)
            surface_ablation[definition] = {
                "score_spearman": correlation(old_score, rescored, ranked=True),
                "high80_spearman": correlation(
                    old_score[high80], rescored[high80], ranked=True
                ),
                "high90_spearman": correlation(
                    old_score[high90], rescored[high90], ranked=True
                ),
                "top20pct_overlap": top_overlap(
                    old_score, rescored, max(1, len(subset) // 5)
                ),
                "top10pct_overlap": top_overlap(
                    old_score, rescored, max(1, len(subset) // 10)
                ),
            }
        variant_summary = {
            "count": len(subset),
            "status_counts": statuses,
            "score_pearson": correlation(old_score, candidate),
            "score_spearman": correlation(old_score, candidate, ranked=True),
            "score_delta_median": percentile(candidate - old_score, 0.5),
            "score_delta_p05": percentile(candidate - old_score, 0.05),
            "score_delta_p95": percentile(candidate - old_score, 0.95),
            "top20pct_overlap": top_overlap(old_score, candidate, max(1, len(subset) // 5)),
            "top10pct_overlap": top_overlap(old_score, candidate, max(1, len(subset) // 10)),
            "high80_count": int(np.count_nonzero(high80)),
            "high80_spearman": correlation(old_score[high80], candidate[high80], ranked=True),
            "high90_count": int(np.count_nonzero(high90)),
            "high90_spearman": correlation(old_score[high90], candidate[high90], ranked=True),
            "qh_log10_spearman": correlation(
                np.log10(old_qh), np.log10(candidate_qh), ranked=True
            ),
            "total_s_median": percentile(candidate_time, 0.5),
            "total_s_p95": percentile(candidate_time, 0.95),
            "surface_s_median": percentile(surface_time, 0.5),
            "speedup_median_ratio": percentile(old_time, 0.5) /
                percentile(candidate_time, 0.5),
            "axis_error_max": float(np.nanmax(axis_errors)),
            "surface_level_spearman": correlation(old_level, new_level, ranked=True),
            "surface_level_delta_median": percentile(new_level - old_level, 0.5),
            "components": component_summary,
            "surface_ablation": surface_ablation,
        }
        summary["variants"][name] = variant_summary
    return summary


def plot(rows: list[dict], summary: dict, output: Path) -> None:
    names = sorted(summary["variants"])
    figure, axes = plt.subplots(2, len(names), figsize=(4.2 * len(names), 7.2), squeeze=False)
    for column, name in enumerate(names):
        subset = [row for row in rows if name in row["variants"]]
        legacy = np.asarray([row["legacy"]["score"] for row in subset])
        candidate = np.asarray([row["variants"][name]["score"] for row in subset])
        timing = np.asarray([row["variants"][name]["timing"]["total_s"] for row in subset])
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
