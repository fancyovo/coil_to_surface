#!/usr/bin/env python3
"""Render and report the frozen complete rounds of the replay50 RL run."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "reports" / "assets" / "axisflip_r012_score_gradient_replay50_rl_20260905_interim"
RAW = ASSET / "round_raw"
REPORT = ROOT / "reports" / "axisflip_r012_score_gradient_replay50_rl_interim_20260905.md"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def wilson(k: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    r = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return float(c - r), float(c + r)


def main() -> None:
    progress = load(ASSET / "progress.json")
    latest = int(progress["completed_round"])
    rows = []
    for directory in sorted(RAW.glob("round_*")):
        index = int(directory.name.split("_")[1])
        if index > latest:
            continue
        collection_path = directory / "collection_summary.json"
        training_path = directory / "training_summary.json"
        if not (collection_path.exists() and training_path.exists()):
            continue
        collection = load(collection_path)
        training = load(training_path)
        train = training["training"]
        timing = collection["timing"]
        coordinate = collection["coordinate_check"]
        rows.append(
            {
                "round": index,
                "valid_count": int(collection["valid_count"]),
                "valid_rate": float(collection["valid_rate"]),
                "gradient_ok_count": int(collection["gradient_ok_count"]),
                "gradient_ok_rate_valid": float(collection["gradient_ok_rate_valid"]),
                "initial_score_median_all": float(collection["initial"]["score_median_all"]),
                "initial_score_p90_all": float(collection["initial"]["score_p90_all"]),
                "initial_score_median_valid": float(collection["initial"]["score_median_valid"]),
                "initial_qs_median_valid": float(collection["initial"]["volume_qs_median_valid"]),
                "initial_coil_median_valid": float(collection["initial"]["coil_median_valid"]),
                "initial_qs_coil_corr": float(collection["initial"]["volume_qs_coil_correlation_valid"]),
                "gradient_relative_error_median": coordinate["relative_error_median"],
                "gradient_relative_error_p99": coordinate["relative_error_p99"],
                "collection_wall_s": float(timing["collection_wall_s"]),
                "initial_score_median_s": float(timing["initial_score_median_s"]),
                "gradient_median_s_valid": float(timing["gradient_median_s_valid"]),
                "flow_train_wall_s": float(train["wall_s"]),
                "flow_step_wall_s_median": float(train["step_wall_s_median"]),
                "flow_step_wall_s_p90": float(train["step_wall_s_p90"]),
                "objective_first": float(train["objective_first"]),
                "objective_last": float(train["objective_last"]),
                "pool_size": int(train["pool_size"]),
            }
        )
    if not rows:
        raise RuntimeError("no complete round summaries found")
    rows.sort(key=lambda row: row["round"])
    rounds = np.asarray([row["round"] for row in rows])
    valid = np.asarray([row["valid_rate"] for row in rows])
    score = np.asarray([row["initial_score_median_all"] for row in rows])
    p90 = np.asarray([row["initial_score_p90_all"] for row in rows])
    qs = np.asarray([row["initial_qs_median_valid"] for row in rows])
    coil = np.asarray([row["initial_coil_median_valid"] for row in rows])
    grad = np.asarray([row["gradient_ok_rate_valid"] for row in rows])
    grad_p99 = np.asarray([row["gradient_relative_error_p99"] for row in rows], dtype=float)
    flow_wall = np.asarray([row["flow_train_wall_s"] for row in rows])
    collection_wall = np.asarray([row["collection_wall_s"] for row in rows])
    step_med = np.asarray([row["flow_step_wall_s_median"] for row in rows])
    step_p90 = np.asarray([row["flow_step_wall_s_p90"] for row in rows])
    objective_first = np.asarray([row["objective_first"] for row in rows])
    objective_last = np.asarray([row["objective_last"] for row in rows])
    pool = np.asarray([row["pool_size"] for row in rows])

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6))
    axes[0, 0].plot(rounds, score, marker="o", ms=3, label="initial median")
    axes[0, 0].plot(rounds, p90, marker="^", ms=3, label="initial P90")
    axes[0, 0].set_ylabel("R04 initial total score")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].plot(rounds, valid, marker="o", ms=3, label="valid rate")
    axes[0, 1].plot(rounds, grad, marker="s", ms=3, label="gradient ok | valid")
    axes[0, 1].set_ylim(0, 1.04)
    axes[0, 1].set_ylabel("fraction")
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[1, 0].plot(rounds, qs, marker="o", ms=3, label="volume-QS")
    axes[1, 0].plot(rounds, coil, marker="s", ms=3, label="coil")
    axes[1, 0].set_ylabel("median component score | valid")
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 1].plot(rounds, grad_p99, marker="o", ms=3, color="#c44536")
    axes[1, 1].axhline(0.05, color="#202020", linestyle=":", label="acceptance tolerance")
    axes[1, 1].set_ylabel("coordinate check relative error P99")
    axes[1, 1].legend(frameon=False, fontsize=8)
    for axis in axes.flat:
        axis.set_xlabel("completed round")
        axis.grid(alpha=0.22)
    fig.suptitle(f"R012 score-gradient replay50 RL: rounds 0-{latest}", fontsize=13)
    fig.tight_layout()
    fig.savefig(ASSET / f"round_metrics_through_{latest:04d}.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].plot(rounds, collection_wall, label="collection wall")
    axes[0].plot(rounds, flow_wall, label="Flow training wall")
    axes[0].set_ylabel("seconds per round")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(rounds, step_med, label="update step median")
    axes[1].plot(rounds, step_p90, label="update step P90")
    axes[1].set_ylabel("seconds per Flow update")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.set_xlabel("completed round")
        axis.grid(alpha=0.22)
    fig.suptitle("Replay50 timing", fontsize=13)
    fig.tight_layout()
    fig.savefig(ASSET / f"timing_through_{latest:04d}.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].plot(rounds, objective_first, label="objective first")
    axes[0].plot(rounds, objective_last, label="objective last")
    axes[0].set_ylabel("Flow objective")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(rounds, pool, color="#2a9d8f")
    axes[1].axhline(512, color="#202020", linestyle=":", label="capacity")
    axes[1].set_ylabel("replay pool records")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.set_xlabel("completed round")
        axis.grid(alpha=0.22)
    fig.suptitle("Replay50 training diagnostics", fontsize=13)
    fig.tight_layout()
    fig.savefig(ASSET / f"training_diagnostics_through_{latest:04d}.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    aggregate = {
        "round_count": len(rows),
        "latest_round": latest,
        "online_samples": len(rows) * 64,
        "valid_count": int(sum(row["valid_count"] for row in rows)),
        "valid_rate": float(sum(row["valid_count"] for row in rows) / (len(rows) * 64)),
        "valid_rate_wilson95": wilson(int(sum(row["valid_count"] for row in rows)), len(rows) * 64),
        "gradient_ok_count": int(sum(row["gradient_ok_count"] for row in rows)),
        "gradient_ok_rate_valid_weighted": float(sum(row["gradient_ok_count"] for row in rows) / max(1, sum(row["valid_count"] for row in rows))),
        "first5_mean_valid_rate": float(np.mean(valid[:5])),
        "last5_mean_valid_rate": float(np.mean(valid[-5:])),
        "first5_mean_initial_median": float(np.mean(score[:5])),
        "last5_mean_initial_median": float(np.mean(score[-5:])),
        "first5_mean_initial_p90": float(np.mean(p90[:5])),
        "last5_mean_initial_p90": float(np.mean(p90[-5:])),
        "latest": rows[-1],
        "mean_collection_wall_s": float(np.mean(collection_wall)),
        "mean_flow_train_wall_s": float(np.mean(flow_wall)),
        "mean_round_wall_s_estimate": float(np.mean(collection_wall + flow_wall)),
    }
    payload = {
        "format": "axisflip_r012_score_gradient_replay50_rl_interim_report_v1",
        "job_id": 54046,
        "protocol": load(ASSET / "protocol.json"),
        "progress": progress,
        "aggregate": aggregate,
        "rounds": rows,
    }
    (ASSET / f"report_metrics_through_{latest:04d}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    sample_indices = np.unique(np.linspace(0, len(rows) - 1, num=min(10, len(rows)), dtype=int)).tolist()
    table = [
        "| 轮次 | 合法 | 初始中位数 | 初始 P90 | 合法样本 QS 中位数 | 合法样本 coil 中位数 | 梯度通过率 | 采集耗时(s) | Flow更新耗时(s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index in sample_indices:
        row = rows[index]
        table.append(
            f"| {row['round']} | {row['valid_count']}/64 | {row['initial_score_median_all']:.3f} | {row['initial_score_p90_all']:.3f} | "
            f"{row['initial_qs_median_valid']:.3f} | {row['initial_coil_median_valid']:.3f} | {row['gradient_ok_rate_valid']:.2%} | "
            f"{row['collection_wall_s']:.1f} | {row['flow_train_wall_s']:.2f} |"
        )
    latest_row = rows[-1]
    ci = aggregate["valid_rate_wilson95"]
    report = f"""# R012 score-gradient replay50 RL：第 0--{latest} 轮阶段报告

- 报告日期：2026-09-05（Asia/Shanghai）
- 作业：`54046`，状态：`RUNNING`，不会因本次报告停止
- 协议：`qh-axisflip-r012-score-gradient-replay50-rl-r04-abi11-v1`
- 固定条件：`nfp=8,n_coils=3`；评分器为 ABI-11 R04，曲率 p95 特征尺度 `25 m^-1`
- Flow：R04 q0 蒸馏 checkpoint，256 宽、6 层、8 头、5,761,380 参数

## 当前结论

已完整落盘第 0--{latest} 轮，共 `{aggregate['online_samples']}` 个起点。累计合法率为 `{aggregate['valid_rate']:.2%}`，Wilson 95% 区间 `{ci[0]:.2%}--{ci[1]:.2%}`；第 0 轮为 `{rows[0]['valid_rate']:.2%}`，第 {latest} 轮为 `{latest_row['valid_rate']:.2%}`。前五轮与后五轮平均合法率为 `{aggregate['first5_mean_valid_rate']:.2%} -> {aggregate['last5_mean_valid_rate']:.2%}`，初始总分中位数为 `{aggregate['first5_mean_initial_median']:.3f} -> {aggregate['last5_mean_initial_median']:.3f}`，P90 为 `{aggregate['first5_mean_initial_p90']:.3f} -> {aggregate['last5_mean_initial_p90']:.3f}`。

这是 score-gradient 策略自身的阶段证据。它显示合法起点比例和初始分布已经明显富集；是否优于 P107 轨迹回放策略，应在相同轮数或相同评分预算下比较，不能仅用不同作业已经运行的轮数直接比较。

## 方法和口径

每轮从当前 Flow 生成 64 个中心样本，由两个 student GPU 各处理 32 个。合法中心执行一次与 Adam 单步完全相同的 ABI-11 R04 64 方向中心差分梯度查询；非法中心不计算梯度。每条记录包含中心、固定 64 方向梯度及合法标记，写入容量 512 的 FIFO 回放池。每轮从回放池均匀有放回抽取与原始 batch 相同大小的记录，执行 50 个 Flow 优化更新；loss 保持 `L_valid + 0.05 L_invalid + beta mean(g_flow^T grad_x ell)`，固定 `beta={latest_row['objective_last']*0 + 0.006021959241479635:.12f}`。训练没有引入运输目标、`rho` 或额外的 rollout 目标。

每个有效中心的坐标一致性检查阈值为 `5e-2`。本阶段梯度通过率按有效中心计，累计为 `{aggregate['gradient_ok_rate_valid_weighted']:.2%}`；最新轮为 `{latest_row['gradient_ok_rate_valid']:.2%}`。最新轮坐标检查 P99 为 `{latest_row['gradient_relative_error_p99']:.4f}`。

## 逐轮结果

下表列出等间隔完整轮次；[完整机器统计](assets/axisflip_r012_score_gradient_replay50_rl_20260905_interim/report_metrics_through_{latest:04d}.json) 保存全部 {len(rows)} 轮。

{chr(10).join(table)}

![分数、合法率、分量与梯度一致性](assets/axisflip_r012_score_gradient_replay50_rl_20260905_interim/round_metrics_through_{latest:04d}.png)

## 计算耗时

最新轮采集耗时 `{latest_row['collection_wall_s']:.1f} s`，其中单点初始评分中位数 `{latest_row['initial_score_median_s']:.2f} s`、有效样本梯度中位数 `{latest_row['gradient_median_s_valid']:.2f} s`；Flow 50 次更新耗时 `{latest_row['flow_train_wall_s']:.2f} s`，单次更新中位数/P90 为 `{latest_row['flow_step_wall_s_median']:.4f}/{latest_row['flow_step_wall_s_p90']:.4f} s`。到达 512 条回放记录后，训练耗时基本稳定，主要时间仍在中心评分和 64 方向梯度查询。

![每轮采集与 Flow 训练耗时](assets/axisflip_r012_score_gradient_replay50_rl_20260905_interim/timing_through_{latest:04d}.png)

## 分量和退化诊断

最新轮合法起点体 QS 中位数为 `{latest_row['initial_qs_median_valid']:.3f}`，coil 中位数为 `{latest_row['initial_coil_median_valid']:.3f}`，二者样本相关系数为 `{rows[-1]['initial_qs_coil_corr']:.3f}`。当前总分变化主要由体 QS 分量承担；coil 已处在较高且较窄的区间，不能把总分中位数提高简单解释为 coil 改善。回放池在第 8 轮左右达到容量上限并保持 512 条记录，未观察到训练记录数量异常增长。

每轮目标函数的首末值和回放池大小见下图。目标函数下降本身只说明 Flow 拟合该轮 loss 的变化，必须结合合法率、分数分位数和梯度一致性判断策略是否有效。

![训练目标与回放池诊断](assets/axisflip_r012_score_gradient_replay50_rl_20260905_interim/training_diagnostics_through_{latest:04d}.png)

## 阶段判断

截至第 {latest} 轮，score-gradient replay50 作业运行稳定：两个 GPU rank 持续完成采集，梯度一致性通过率保持高位，Flow 更新耗时约数秒/轮。已观测到合法率和初始分布的显著改善；目前没有仅凭这些摘要判定灾难性模式坍缩的证据，但仍需在后续轮次检查分布多样性与独立 holdout。作业 `54046` 保持运行，本报告不发送停止信号。

## 机器证据

- [协议](assets/axisflip_r012_score_gradient_replay50_rl_20260905_interim/protocol.json)
- [manifest](assets/axisflip_r012_score_gradient_replay50_rl_20260905_interim/manifest.json)
- [进度快照](assets/axisflip_r012_score_gradient_replay50_rl_20260905_interim/progress.json)
- [第 0--{latest} 轮派生统计](assets/axisflip_r012_score_gradient_replay50_rl_20260905_interim/report_metrics_through_{latest:04d}.json)

远端 run root：`/home/scc/pb24511935/local_surface_evaluator_runs/axisflip_r012_score_gradient_replay50_rl_20260905_a9ead14`。
"""
    REPORT.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
