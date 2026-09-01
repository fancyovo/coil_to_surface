from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flow_matching.axis_surface_prior_v2 import prototype_presets, sample_shaped_prior_prototype
from scripts.analyze_axis_surface_prior import physical_curves


PALETTE = ("#1f4e79", "#b04a3a", "#3f7d20", "#7b4f9d")


def _row(prototype: Any, case_id: int) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "nfp": prototype.metadata["nfp"],
        "n_base_coils": prototype.metadata["n_base_coils"],
        "family": prototype.metadata["preset"],
        "tokens": prototype.tokens.tolist(),
    }


def write_html(prototype: Any, output_path: Path, case_id: int) -> None:
    import plotly.graph_objects as go

    traces = []
    inner = prototype.inner_reference_surface
    traces.append(
        go.Surface(
            x=inner[:, :, 0],
            y=inner[:, :, 1],
            z=inner[:, :, 2],
            surfacecolor=np.tile(np.linspace(0.0, 1.0, inner.shape[0])[:, None], (1, inner.shape[1])),
            colorscale=((0.0, "#276678"), (0.5, "#8ec6c5"), (1.0, "#276678")),
            opacity=0.63,
            showscale=False,
            hoverinfo="skip",
            name="geometric inner reference surface",
        )
    )
    winding = prototype.winding_surface
    for index in range(0, winding.shape[0], max(1, winding.shape[0] // 32)):
        curve = np.vstack((winding[index], winding[index, :1]))
        traces.append(
            go.Scatter3d(
                x=curve[:, 0], y=curve[:, 1], z=curve[:, 2], mode="lines",
                line={"color": "#888888", "width": 1}, opacity=0.32,
                hoverinfo="skip", showlegend=False,
            )
        )
    for index in range(0, winding.shape[1], max(1, winding.shape[1] // 18)):
        curve = np.vstack((winding[:, index], winding[:1, index]))
        traces.append(
            go.Scatter3d(
                x=curve[:, 0], y=curve[:, 1], z=curve[:, 2], mode="lines",
                line={"color": "#888888", "width": 1}, opacity=0.32,
                hoverinfo="skip", showlegend=False,
            )
        )
    axis = np.vstack((prototype.reference_axis, prototype.reference_axis[:1]))
    traces.append(
        go.Scatter3d(
            x=axis[:, 0], y=axis[:, 1], z=axis[:, 2], mode="lines",
            line={"color": "#111111", "width": 7}, name="construction reference axis",
            hoverinfo="name",
        )
    )
    for coil, base_index in physical_curves(_row(prototype, case_id), samples=256):
        closed = np.vstack((coil, coil[:1]))
        traces.append(
            go.Scatter3d(
                x=closed[:, 0], y=closed[:, 1], z=closed[:, 2], mode="lines",
                line={"color": PALETTE[base_index % len(PALETTE)], "width": 5},
                hoverinfo="skip", showlegend=False,
            )
        )
    metadata = prototype.metadata
    title = (
        f"{metadata['preset']} | nfp={metadata['nfp']}, nc={metadata['n_base_coils']} | "
        f"axis ΔR={metadata['axis_R_peak_to_peak_m']:.3f} m, ΔZ={metadata['axis_Z_peak_to_peak_m']:.3f} m"
    )
    figure = go.Figure(traces)
    figure.update_layout(
        title=title,
        template="plotly_white",
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
        scene={
            "aspectmode": "data",
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "camera": {"eye": {"x": 1.55, "y": -1.65, "z": 1.15}},
        },
    )
    figure.write_html(output_path, include_plotlyjs=True, full_html=True)


def render_contact_sheet(
    prototypes: list[Any], output_path: Path, *, title: str = "Shaped analytic-prior v2 prototypes"
) -> None:
    figure = plt.figure(figsize=(15.5, 5.4))
    for panel, prototype in enumerate(prototypes, start=1):
        axis = figure.add_subplot(1, len(prototypes), panel, projection="3d")
        inner = prototype.inner_reference_surface
        axis.plot_surface(
            inner[::3, ::3, 0], inner[::3, ::3, 1], inner[::3, ::3, 2],
            color="#4f9da6", alpha=0.43, linewidth=0, shade=True,
        )
        winding = prototype.winding_surface
        axis.plot_wireframe(
            winding[::8, ::6, 0], winding[::8, ::6, 1], winding[::8, ::6, 2],
            color="#777777", linewidth=0.32, alpha=0.35,
        )
        reference_axis = prototype.reference_axis
        reference_axis = np.vstack((reference_axis, reference_axis[:1]))
        axis.plot(reference_axis[:, 0], reference_axis[:, 1], reference_axis[:, 2], color="#111111", linewidth=2.2)
        for coil, base_index in physical_curves(_row(prototype, panel), samples=192):
            coil = np.vstack((coil, coil[:1]))
            axis.plot(coil[:, 0], coil[:, 1], coil[:, 2], color=PALETTE[base_index], linewidth=1.05, alpha=0.93)
        meta = prototype.metadata
        axis.set_title(
            f"{meta['preset'].replace('_', ' ')}\n"
            f"axis ΔR={meta['axis_R_peak_to_peak_m']:.2f} m, ΔZ={meta['axis_Z_peak_to_peak_m']:.2f} m",
            fontsize=11,
        )
        axis.set_box_aspect((1.0, 1.0, 0.58))
        axis.set_axis_off()
        axis.view_init(27, -54)
    figure.suptitle(
        f"{title}: black axis, colored coils, gray winding surface, teal geometric inner reference surface",
        fontsize=13,
        y=0.98,
    )
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.84, wspace=0.01)
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render geometry-only v2 prior prototypes for user approval.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nfp", type=int, default=4)
    parser.add_argument("--n-base-coils", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prototypes = []
    index = []
    for offset, preset in enumerate(prototype_presets()):
        prototype = sample_shaped_prior_prototype(
            seed=args.seed + offset,
            nfp=args.nfp,
            n_base_coils=args.n_base_coils,
            preset=preset,
        )
        prototypes.append(prototype)
        filename = f"prototype_{offset + 1}_{preset}.html"
        write_html(prototype, args.output_dir / filename, offset + 1)
        index.append({"preset": preset, "html": filename, "metadata": prototype.metadata})
    render_contact_sheet(prototypes, args.output_dir / "prototype_contact_sheet.png")
    (args.output_dir / "prototype_index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "rendered", "output_dir": str(args.output_dir), "count": len(prototypes)}))


if __name__ == "__main__":
    main()
