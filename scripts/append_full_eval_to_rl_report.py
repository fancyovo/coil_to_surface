"""Append one completed physical evaluation to the Students RL acceptance report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


MARKER = "<!-- students-weighted-full-evaluation -->"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def append(report: Path, evaluation: Path, selection: Path) -> None:
    summary = load(evaluation)
    selected = load(selection)
    if summary.get("status") != "completed":
        raise ValueError("evaluation summary is not completed")
    if MARKER in report.read_text(encoding="utf-8"):
        raise FileExistsError("report already contains this evaluation marker")
    source = summary["source_psi_selection"]
    surface = summary["surface_selection"]
    downstream = summary["downstream"]
    surface_row = surface["selected"]
    downstream_surface = downstream.get("surface", {})
    desc = downstream.get("desc", {})
    artifact_dir = evaluation.parent.name
    selection_dir = selection.parent.name
    lines = [
        "", MARKER, "", "## Students 加权 RL 当前最高样本完整物理评估", "",
        f"本节评估截至作业 `{selected['selection_boundary_next_round'] - 1}` 轮已完成数据中 native 初始总分最高的合法位型：",
        f"`{selected['source_sample_id']}`（来源 round `{selected['source_round']}`、rank `{selected['source_rank']}`、样本序号 `{selected['source_sample_index']}`）。",
        f"初始总分为 `{selected['source_score']:.8f}`，分量为 `{json.dumps(selected['source_components'], ensure_ascii=False, sort_keys=True)}`。",
        f"选择记录：`assets/{selection_dir}/selection.json`；完整评估摘要：`assets/{artifact_dir}/evaluation_summary.json`。", "",
        "| 阶段 | 结果 |", "|---|---|",
        f"| source-psi | `{source.get('selected', source).get('a_m', source.get('selected_a_m', 'see summary'))}`，完整候选见 `assets/{artifact_dir}/source_psi_selection.json` |",
        f"| 最大连续标准面 | `s={surface_row.get('target_s')}`，体积 `{surface_row.get('final_abs_volume_m3')}` m3，质量 `{surface_row.get('acceptance_checks', {}).get('evaluation_quality', surface_row.get('surface_provenance', {}).get('valid'))}` |",
        f"| 面参数 | iota `{downstream_surface.get('iota')}`，G `{downstream_surface.get('G')}`，rho `{downstream_surface.get('rho')}` |",
        f"| DESC | 状态 `{downstream.get('status')}`，结果见 `assets/{artifact_dir}/full/desc/` |",
        "", "固定流程产物包括 source-psi、alpha+nu、标准 Simsopt LS/Newton、外侧失败边界、Poincare、Boozer、HTML 和 DESC。完整机器结果以 `evaluation_summary.json` 及其引用的原始 JSON 为准。", "",
    ]
    with report.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    append(args.report, args.evaluation, args.selection)
