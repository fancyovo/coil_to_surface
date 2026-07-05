from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import median

import numpy as np


DEFAULT_LEVELS = (1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 3e-1)
ZH_FONT_CANDIDATES = (
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Microsoft YaHei",
    "SimHei",
    "WenQuanYi Zen Hei",
    "PingFang SC",
    "Sarasa Gothic SC",
    "Arial Unicode MS",
)


def parse_levels(text: str) -> list[float]:
    return [float(x) for x in text.replace(",", " ").split() if x.strip()]


def load_payload(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def perturb_case(payload: dict, key: str, level: float, seed: int, components: tuple[str, ...]) -> dict:
    out = deepcopy(payload)
    entry = out[key]
    rng = np.random.default_rng(seed)
    for comp in components:
        arr = np.asarray(entry[comp], dtype=np.float64)
        eps = rng.normal(loc=0.0, scale=level, size=arr.shape)
        entry[comp] = (arr + arr * eps).tolist()
    base_name = str(entry.get("name", key))
    entry["name"] = f"{base_name}_pert_l{level:.3g}_s{seed}"
    return out


def classify_result(summary: dict | None, returncode: int, timed_out: bool) -> str:
    if timed_out:
        return "timeout"
    if returncode != 0 or summary is None:
        return "crash"
    axis = summary.get("axis", {})
    if not axis.get("has_axis", False):
        return "no_axis"
    if summary.get("best_surface") is None:
        return "no_surface"
    return "surface"


def flatten_trial_record(level: float, seed: int, status: str, elapsed_s: float, returncode: int, summary: dict | None) -> dict:
    rec: dict[str, object] = {
        "level": level,
        "seed": seed,
        "status": status,
        "wall_time_s": elapsed_s,
        "returncode": returncode,
    }
    if summary is None:
        return rec

    axis = summary.get("axis", {})
    best = summary.get("best_surface") or {}
    timing = summary.get("timing", {})
    rec.update(
        {
            "total_time_s": summary.get("total_time_s"),
            "has_axis": axis.get("has_axis"),
            "axis_residual": axis.get("best_residual"),
            "axis_generation": axis.get("generation"),
            "best_surface_exists": bool(summary.get("best_surface")),
            "best_surface_psi": best.get("psi_level"),
            "best_surface_iota": best.get("iota"),
            "best_surface_volume": best.get("volume"),
            "best_surface_G": best.get("G"),
            "best_surface_qs_QA": (best.get("qs_errors") or {}).get("QA"),
            "surface_screen_ok_count": sum(1 for x in summary.get("surface_screen", {}).get("levels", []) if x.get("ok")),
            "surface_candidate_count": len(summary.get("surface_candidates", [])),
            "warnings_count": len(summary.get("warnings", [])),
        }
    )
    for key, value in timing.items():
        rec[f"timing_{key}"] = value
    return rec


def summarize_group(records: list[dict]) -> dict:
    status_counter = Counter(str(r["status"]) for r in records)
    values = defaultdict(list)
    for r in records:
        for key, value in r.items():
            if key in {"level", "seed", "status", "returncode"}:
                continue
            if isinstance(value, (int, float)) and np.isfinite(value):
                values[key].append(float(value))
    out = {
        "count": len(records),
        "status_counts": dict(status_counter),
    }
    for key, arr in values.items():
        out[f"{key}_median"] = median(arr)
        out[f"{key}_max"] = max(arr)
        out[f"{key}_min"] = min(arr)
    return out


def configure_plot_fonts() -> bool:
    try:
        import matplotlib
        from matplotlib import font_manager
    except Exception:
        return False
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ZH_FONT_CANDIDATES:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans", "sans-serif"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "sans-serif"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return False


def maybe_make_plots(output_dir: Path, trials: list[dict], aggregate_rows: list[dict]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    plot_paths: list[str] = []
    levels = [float(x["level"]) for x in aggregate_rows]
    if not levels:
        return plot_paths
    use_zh = configure_plot_fonts()

    status_order = ["surface", "no_surface", "no_axis", "crash", "timeout"]
    colors = {
        "surface": "#2ca02c",
        "no_surface": "#ff7f0e",
        "no_axis": "#d62728",
        "crash": "#9467bd",
        "timeout": "#8c564b",
    }

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for status in status_order:
        xs = [float(r["level"]) for r in trials if r["status"] == status]
        ys = [float(r.get("total_time_s", r["wall_time_s"])) for r in trials if r["status"] == status]
        if xs:
            ax.scatter(xs, ys, s=28, alpha=0.8, label=status, color=colors[status])
    med_total = [float(row.get("total_time_s_median", np.nan)) for row in aggregate_rows]
    med_axis = [float(row.get("timing_axis_s_median", np.nan)) for row in aggregate_rows]
    med_psi = [float(row.get("timing_psi_fit_s_median", np.nan)) for row in aggregate_rows]
    med_screen = [float(row.get("timing_surface_screen_s_median", np.nan)) for row in aggregate_rows]
    med_boozer = [float(row.get("timing_boozer_candidates_s_median", np.nan)) for row in aggregate_rows]
    ax.plot(levels, med_total, color="black", linewidth=1.5, label="median total")
    ax.plot(levels, med_axis, color="#1f77b4", linewidth=1.1, label="median axis")
    ax.plot(levels, med_psi, color="#17becf", linewidth=1.1, label="median psi fit")
    ax.plot(levels, med_screen, color="#bcbd22", linewidth=1.1, label="median psi0 screen")
    ax.plot(levels, med_boozer, color="#7f7f7f", linewidth=1.1, label="median Boozer")
    ax.set_xscale("log")
    ax.set_xlabel("perturbation level")
    ax.set_ylabel("time [s]")
    ax.set_title("01 扰动鲁棒性：时间与状态" if use_zh else "01 perturbation robustness: time and status")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    time_plot = output_dir / "time_vs_level.png"
    fig.savefig(time_plot, dpi=160)
    plt.close(fig)
    plot_paths.append(time_plot.name)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bottoms = np.zeros(len(levels))
    counts = np.array([float(row["count"]) for row in aggregate_rows], dtype=float)
    for status in status_order:
        vals = np.array([float(row.get("status_counts", {}).get(status, 0)) for row in aggregate_rows], dtype=float)
        frac = np.divide(vals, counts, out=np.zeros_like(vals), where=counts > 0)
        ax.bar(levels, frac, bottom=bottoms, width=np.array(levels) * 0.18, label=status, color=colors[status], align="center")
        bottoms += frac
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("perturbation level")
    ax.set_ylabel("fraction")
    ax.set_title("01 扰动鲁棒性：结果分类占比" if use_zh else "01 perturbation robustness: outcome fractions")
    ax.grid(True, which="both", axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    status_plot = output_dir / "status_fraction_vs_level.png"
    fig.savefig(status_plot, dpi=160)
    plt.close(fig)
    plot_paths.append(status_plot.name)
    return plot_paths


def write_report(
    output_dir: Path,
    *,
    case_file: Path,
    key: str,
    levels: list[float],
    seeds: list[int],
    components: tuple[str, ...],
    timeout_s: float,
    aggregate_rows: list[dict],
    plot_paths: list[str],
) -> None:
    lines: list[str] = []
    lines.append("# 01 线圈扰动鲁棒性报告")
    lines.append("")
    lines.append("## 实验设置")
    lines.append("")
    lines.append(f"- 基准样例: `{case_file}`，key=`{key}`")
    lines.append(f"- 扰动分量: `{', '.join(components)}`")
    lines.append("- 扰动方式: 对每个几何系数独立施加乘性高斯噪声，即 `c -> c + c * ε`，其中 `ε ~ N(0, level^2)`")
    lines.append(f"- level: `{', '.join(f'{x:.3g}' for x in levels)}`")
    lines.append(f"- seeds: `{', '.join(str(x) for x in seeds)}`")
    lines.append(f"- 单次评估超时: `{timeout_s:.1f} s`")
    lines.append("- 主流程使用当前默认 GPU 配置；额外指定 `initial_iota=-2.9` 以对齐 01 的工作点。")
    lines.append("")
    if plot_paths:
        lines.append("## 图")
        lines.append("")
        for name in plot_paths:
            lines.append(f"![{name}]({name})")
            lines.append("")
    lines.append("## 聚合结果")
    lines.append("")
    lines.append("| level | count | surface | no_surface | no_axis | crash | timeout | total median [s] | axis median [s] | psi median [s] | screen median [s] | boozer median [s] |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in aggregate_rows:
        sc = row.get("status_counts", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{row['level']:.3g}",
                    str(int(row["count"])),
                    str(int(sc.get("surface", 0))),
                    str(int(sc.get("no_surface", 0))),
                    str(int(sc.get("no_axis", 0))),
                    str(int(sc.get("crash", 0))),
                    str(int(sc.get("timeout", 0))),
                    f"{row.get('total_time_s_median', float('nan')):.3f}",
                    f"{row.get('timing_axis_s_median', float('nan')):.3f}",
                    f"{row.get('timing_psi_fit_s_median', float('nan')):.3f}",
                    f"{row.get('timing_surface_screen_s_median', float('nan')):.3f}",
                    f"{row.get('timing_boozer_candidates_s_median', float('nan')):.3f}",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## 说明")
    lines.append("")
    lines.append("- `surface`: 找到磁轴，至少一个 `psi0` 通过筛选，并有一个 Boozer 候选成功收敛。")
    lines.append("- `no_surface`: 找到磁轴，但没有可用磁面。")
    lines.append("- `no_axis`: 找轴阶段没有达到容差，流程提前结束。")
    lines.append("- `crash/timeout`: 进程异常退出或超时。")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark robustness of the evaluator under multiplicative perturbations of 01.json.")
    p.add_argument("--case-file", type=Path, default=Path("examples/01.json"))
    p.add_argument("--key", default="raw")
    p.add_argument("--output-dir", type=Path, default=Path("runs/robustness_01_perturb"))
    p.add_argument("--levels", default=",".join(str(x) for x in DEFAULT_LEVELS))
    p.add_argument("--seeds", type=int, default=4, help="Number of seeds per level, starting from 0.")
    p.add_argument("--timeout-s", type=float, default=40.0)
    p.add_argument("--components", default="x,y,z", help="Comma-separated subset of x,y,z to perturb.")
    p.add_argument("--reuse-existing-trials", action="store_true", help="Reuse output_dir/trials.json and only rebuild aggregate files, plots, and report.")
    p.add_argument("--a", type=float, default=0.05)
    p.add_argument("--initial-iota", type=float, default=-2.9)
    p.add_argument("--qs-sdim", type=int, default=16)
    p.add_argument("--python-bin", default=sys.executable)
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    case_file = (repo_root / args.case_file).resolve() if not args.case_file.is_absolute() else args.case_file
    output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = output_dir / "cases"
    runs_dir = output_dir / "runs"
    cases_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    levels = parse_levels(args.levels)
    seeds = list(range(args.seeds))
    components = tuple(x.strip() for x in args.components.split(",") if x.strip())

    trials_path = output_dir / "trials.json"
    if args.reuse_existing_trials:
        if not trials_path.exists():
            raise FileNotFoundError(f"{trials_path} does not exist")
        trials = json.loads(trials_path.read_text(encoding="utf-8"))
        levels = sorted({float(rec["level"]) for rec in trials})
        seeds = sorted({int(rec["seed"]) for rec in trials})
    else:
        payload = load_payload(case_file)
        trials = []
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")

        for level in levels:
            for seed in seeds:
                tag = f"l{level:.3g}_s{seed}"
                case_path = cases_dir / f"{tag}.json"
                run_dir = runs_dir / tag
                run_dir.mkdir(parents=True, exist_ok=True)
                perturbed = perturb_case(payload, args.key, level, seed, components)
                with case_path.open("w", encoding="utf-8") as f:
                    json.dump(perturbed, f, ensure_ascii=False, indent=2)

                cmd = [
                    args.python_bin,
                    "-m",
                    "stellarator_eval.cli",
                    "--case-file",
                    str(case_path),
                    "--key",
                    args.key,
                    "--output-dir",
                    str(run_dir),
                    "--omp-threads",
                    "1",
                    "--a",
                    str(args.a),
                    "--initial-iota",
                    str(args.initial_iota),
                    "--qs-sdim",
                    str(args.qs_sdim),
                ]
                t0 = time.perf_counter()
                timed_out = False
                returncode = -999
                summary = None
                stdout_text = ""
                stderr_text = ""
                try:
                    proc = subprocess.run(
                        cmd,
                        cwd=repo_root,
                        env=env,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        timeout=args.timeout_s,
                        check=False,
                    )
                    returncode = proc.returncode
                    stdout_text = proc.stdout
                    stderr_text = proc.stderr
                except subprocess.TimeoutExpired as exc:
                    timed_out = True
                    stdout_text = exc.stdout or ""
                    stderr_text = exc.stderr or ""
                elapsed_s = time.perf_counter() - t0

                summary_path = run_dir / "summary.json"
                if summary_path.exists():
                    try:
                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    except Exception:
                        summary = None

                status = classify_result(summary, returncode, timed_out)
                record = flatten_trial_record(level, seed, status, elapsed_s, returncode, summary)
                record["stdout_tail"] = stdout_text[-2000:]
                record["stderr_tail"] = stderr_text[-2000:]
                trials.append(record)

                print(
                    f"[{tag}] status={status:>10s} wall={elapsed_s:6.2f}s "
                    f"axis={record.get('axis_residual', float('nan'))!s} total={record.get('total_time_s', float('nan'))!s}"
                )

    aggregate_rows = []
    grouped: dict[float, list[dict]] = defaultdict(list)
    for rec in trials:
        grouped[float(rec["level"])].append(rec)
    for level in levels:
        row = summarize_group(grouped[level])
        row["level"] = level
        aggregate_rows.append(row)

    (output_dir / "trials.json").write_text(json.dumps(trials, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "aggregate.json").write_text(json.dumps(aggregate_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "trials.csv").open("w", newline="", encoding="utf-8") as f:
        keys = sorted({k for rec in trials for k in rec.keys()})
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(trials)

    plot_paths = maybe_make_plots(output_dir, trials, aggregate_rows)
    write_report(
        output_dir,
        case_file=case_file,
        key=args.key,
        levels=levels,
        seeds=seeds,
        components=components,
        timeout_s=args.timeout_s,
        aggregate_rows=aggregate_rows,
        plot_paths=plot_paths,
    )
    print(f"wrote: {output_dir}")


if __name__ == "__main__":
    main()
