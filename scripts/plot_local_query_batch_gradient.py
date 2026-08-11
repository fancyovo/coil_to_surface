from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads((args.asset_dir / "full600_summary.json").read_text(encoding="utf-8"))
    comparison = summary["local_score"]["gradient_comparison"]
    local = np.asarray(comparison["local_directional"], dtype=np.float64)
    exact = np.asarray(comparison["exact_directional"], dtype=np.float64)

    figure, axis = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    axis.scatter(exact, local, s=18, alpha=0.7, edgecolors="none")
    limit = max(float(np.max(np.abs(exact))), float(np.max(np.abs(local))))
    axis.plot([-limit, limit], [-limit, limit], color="black", linewidth=1, linestyle="--")
    cosine = comparison["coordinate_and_surface_selection_omitted_exact_cosine"]
    axis.set(
        xlabel="Exact fixed-surface directional derivative",
        ylabel="Batched local directional derivative",
        title=f"300-D gradient agreement (cosine = {cosine:.5f})",
    )
    axis.grid(alpha=0.25)
    figure.savefig(args.asset_dir / "full300_gradient_agreement.png", dpi=190)
    plt.close(figure)

    rows = [
        json.loads(line)
        for line in (args.asset_dir / "smoke_history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    first = rows[0]["gradient_pipeline"]["timing_s"]
    stage_names = [
        "endpoint flow",
        "field create",
        "center capture",
        "axis refine",
        "axis samples",
        "psi",
        "surface/flux/alpha/QS",
        "formal proposal",
    ]
    stage_values = [
        rows[0]["endpoint_decode_wall_s"],
        first["field_create"],
        first["center_capture"],
        first["axis_refine"],
        first["axis_samples"],
        first["psi"],
        first["local_score"],
        rows[0]["proposal_decode_wall_s"] + rows[0]["proposal_score_wall_s"],
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].barh(stage_names[::-1], stage_values[::-1], color="#2878B5")
    axes[0].set(xlabel="seconds", title="One full-gradient Adam step")
    axes[0].grid(axis="x", alpha=0.25)
    iterations = [0] + [row["iteration"] for row in rows]
    scores = [json.loads((args.asset_dir / "smoke_summary.json").read_text())["initial_score"]]
    scores.extend(row["current_score"] for row in rows)
    axes[1].plot(iterations, scores, marker="o", color="#C82423")
    axes[1].set(xlabel="Adam step", ylabel="formal score", title="Three-step smoke test")
    axes[1].grid(alpha=0.25)
    figure.savefig(args.asset_dir / "full_gradient_timing_and_smoke.png", dpi=190)
    plt.close(figure)


if __name__ == "__main__":
    main()
