from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stellarator_eval.config import EvalConfig
from stellarator_eval.field import load_case_file
from stellarator_eval.quasr import (
    build_quasr_metadata_index,
    choose_quasr_eval_params,
    load_quasr_field_input,
    load_quasr_metadata,
)
from stellarator_eval.volume_pipeline import evaluate_coils_to_volume_qs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path)
    parser.add_argument("--key", default="raw")
    parser.add_argument("--quasr-root", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--id", type=int)
    parser.add_argument("--helicity", type=int, choices=(0, 1))
    parser.add_argument("--current-unit", default="A")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--points", type=int, default=100000)
    parser.add_argument("--alpha-fit-points", type=int, default=30000)
    parser.add_argument("--alpha-order", type=int, default=12)
    parser.add_argument("--precision", choices=("fp32", "fp64"), default="fp32")
    parser.add_argument("--grid-xy", type=int, default=144)
    parser.add_argument("--grid-phi", type=int, default=96)
    args = parser.parse_args()

    metadata_row = None
    if args.id is not None:
        if args.quasr_root is None:
            raise ValueError("--id requires --quasr-root")
        field_input, _ = load_quasr_field_input(args.quasr_root, args.id)
        if args.metadata is not None:
            metadata_row = build_quasr_metadata_index(load_quasr_metadata(args.metadata)).get(args.id)
        params = choose_quasr_eval_params(metadata_row)
        psi_a = float(params["a"])
        helicity = int(args.helicity if args.helicity is not None else (metadata_row or {}).get("helicity", 1))
    else:
        if args.case is None:
            raise ValueError("pass --case or --id with --quasr-root")
        field_input = load_case_file(args.case, args.key)
        psi_a = EvalConfig().psi.a
        helicity = int(args.helicity if args.helicity is not None else 1)

    base = EvalConfig(current_unit=args.current_unit, omp_threads=16)
    volume = replace(
        base.volume_qs,
        point_count=args.points,
        alpha_fit_point_count=args.alpha_fit_points,
        alpha_radial_order=args.alpha_order,
        alpha_poloidal_order=args.alpha_order,
        alpha_toroidal_order=args.alpha_order,
        precision=args.precision,
        grid_xy=args.grid_xy,
        grid_phi=args.grid_phi,
    )
    config = replace(base, psi=replace(base.psi, a=psi_a), volume_qs=volume)
    target = (1, 0 if helicity == 0 else field_input.nfp)
    result = evaluate_coils_to_volume_qs(
        field_input,
        config,
        target_helicity=target,
        output_dir=args.output_dir,
    )
    compact = {
        "status": result["status"],
        "reason": result.get("reason"),
        "timing": result["timing"],
        "s_edge": (result.get("volume_qs") or {}).get("s_edge"),
        "iota": ((result.get("volume_qs") or {}).get("alpha") or {}).get("iota_coefficients"),
        "target_metric": (((result.get("volume_qs") or {}).get("metrics") or {}).get("target") or {}).get(
            "f_C_over_B3_rms"
        ),
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
