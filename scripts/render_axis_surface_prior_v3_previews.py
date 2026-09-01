from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flow_matching.axis_surface_prior import evaluate_fourier, supported_conditions
from flow_matching.axis_surface_prior_v2 import sample_shaped_prior_prototype
from scripts.analyze_axis_surface_prior import write_coil_html
from scripts.render_axis_surface_prior_v2_prototypes import render_contact_sheet


PREVIEW_CASE_IDS = (4, 10, 21, 22)


def curve_diagnostics(tokens: np.ndarray, samples: int = 1024) -> dict[str, float]:
    curves = evaluate_fourier(tokens, samples=samples)
    delta = 2.0 * np.pi / samples
    curvature_values = []
    torsion_values = []
    for curve in curves:
        first = (np.roll(curve, -1, axis=0) - np.roll(curve, 1, axis=0)) / (2.0 * delta)
        second = (np.roll(curve, -1, axis=0) - 2.0 * curve + np.roll(curve, 1, axis=0)) / delta**2
        third = (np.roll(second, -1, axis=0) - np.roll(second, 1, axis=0)) / (2.0 * delta)
        cross = np.cross(first, second)
        cross_norm = np.linalg.norm(cross, axis=1)
        speed = np.linalg.norm(first, axis=1)
        curvature_values.extend((cross_norm / np.maximum(speed**3, 1.0e-12)).tolist())
        torsion_values.extend(
            (np.abs(np.einsum("ij,ij->i", cross, third)) / np.maximum(cross_norm**2, 1.0e-12)).tolist()
        )
    return {
        "curvature_p95_per_m": float(np.percentile(curvature_values, 95)),
        "curvature_max_per_m": float(np.max(curvature_values)),
        "abs_torsion_p95_per_m": float(np.percentile(torsion_values, 95)),
        "abs_torsion_max_per_m": float(np.max(torsion_values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render unscored compact-flexible v3 coil previews.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["AXIS_PRIOR_HTML_BACKEND"] = "three"
    records = []
    prototypes = []
    for case_id in PREVIEW_CASE_IDS:
        nfp, n_base_coils = supported_conditions()[case_id % len(supported_conditions())]
        prototype = sample_shaped_prior_prototype(
            seed=args.seed + case_id,
            nfp=nfp,
            n_base_coils=n_base_coils,
            preset="compact_flexible",
            sample_role="registered_scoring",
        )
        row = {
            "case_id": case_id,
            "nfp": nfp,
            "n_base_coils": n_base_coils,
            "family": "compact_flexible",
            "tokens": prototype.tokens.tolist(),
            "native": None,
        }
        record = {
            "case_id": case_id,
            "seed": args.seed + case_id,
            "nfp": nfp,
            "n_base_coils": n_base_coils,
            "generator": prototype.metadata,
            "coil_geometry": curve_diagnostics(prototype.tokens),
        }
        filename = f"coils_compact_flexible_case_{case_id:05d}.html"
        write_coil_html(row, args.output_dir / filename, "compact flexible v3 preview")
        record["html"] = filename
        records.append(record)
        prototypes.append(prototype)
    render_contact_sheet(
        prototypes,
        args.output_dir / "preview_contact_sheet.png",
        title="Compact flexible analytic-prior v3 previews",
    )
    (args.output_dir / "preview_index.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "rendered", "count": len(records), "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
