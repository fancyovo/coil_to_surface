from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_PYTHON = REPO_ROOT / "gpu_backend" / "python"
if str(GPU_PYTHON) not in sys.path:
    sys.path.insert(0, str(GPU_PYTHON))

from stellarator_gpu import score_coils_native

from prepare_score_eval_profile_cases import PRODUCTION_SCORE, load_case


VARIANTS = {
    "baseline": {},
    "psi_grid72": {"psi_n_r": 72, "psi_n_z": 72, "psi_n_phi": 72},
    "psi_grid64": {"psi_n_r": 64, "psi_n_z": 64, "psi_n_phi": 64},
    "psi_grid56": {"psi_n_r": 56, "psi_n_z": 56, "psi_n_phi": 56},
    "psi_grid48": {"psi_n_r": 48, "psi_n_z": 48, "psi_n_phi": 48},
    "axis_hint_no_fp64": {"axis_hint_require_continuation": 2},
    "axis_hint_fp64_offset1e3": {},
    "axis_hint_no_fp64_offset1e3": {"axis_hint_require_continuation": 2},
    "iota_constant": {"iota_degree": 0},
    "iota_cubic": {"iota_degree": 3},
    "psi_normal_eq_fp32": {"psi_solver_mode": 1, "psi_precision_mode": 2},
    "axis720": {"axis_trace_steps": 720, "axis_sample_count": 180},
    "surface96": {"surface_theta_count": 96},
    "surface64": {"surface_theta_count": 64},
    "combo_conservative": {
        "psi_n_r": 64,
        "psi_n_z": 64,
        "psi_n_phi": 64,
        "axis_trace_steps": 720,
        "axis_sample_count": 180,
        "surface_theta_count": 96,
    },
    "combo_aggressive": {
        "psi_n_r": 56,
        "psi_n_z": 56,
        "psi_n_phi": 56,
        "axis_trace_steps": 640,
        "axis_sample_count": 160,
        "surface_theta_count": 64,
        "surface_trace_steps": 300,
    },
}

HINT_OFFSET_RADIUS = {
    "axis_hint_fp64_offset1e3": 1.0e-3,
    "axis_hint_no_fp64_offset1e3": 1.0e-3,
}


def evaluate(lib: Path, case: dict, device: int, variant: str) -> dict:
    x, y, z, currents, nfp = load_case(Path(case["case_path"]))
    offset_radius = HINT_OFFSET_RADIUS.get(variant, 0.0)
    offset_angle = 2.0 * math.pi * ((int(case["case_id"]) * 2654435761) % 10000) / 10000.0
    hint_R = float(case["axis_R"]) + offset_radius * math.cos(offset_angle)
    hint_Z = float(case["axis_Z"]) + offset_radius * math.sin(offset_angle)
    overrides = {
        **PRODUCTION_SCORE,
        "axis_hint_enabled": 1,
        "axis_hint_require_continuation": 1,
        "axis_hint_R": hint_R,
        "axis_hint_Z": hint_Z,
        **VARIANTS[variant],
    }
    started = time.perf_counter()
    result = score_coils_native(
        lib,
        x,
        y,
        z,
        currents,
        nfp,
        device_id=device,
        target_helicity=(1, nfp),
        config_overrides=overrides,
    )
    wall_s = time.perf_counter() - started
    diagnostics = result["diagnostics"]
    if result["status"] == "ok" and int(diagnostics["axis_used_hint"]) != 1:
        raise RuntimeError(
            f"case {case['case_id']} variant {variant} did not use the supplied axis hint: "
            f"status={result['status']}"
        )
    return {
        "case_id": int(case["case_id"]),
        "nfp": int(case["nfp"]),
        "n_base_coils": int(case["n_base_coils"]),
        "metadata_qs_error": float(case["metadata_qs_error"]),
        "selection_score": float(case["selection_score"]),
        "variant": variant,
        "hint_offset_radius": offset_radius,
        "caller_wall_s": wall_s,
        "config_overrides": overrides,
        "result": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark native score variants using strict supplied-axis continuation only."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=tuple(VARIANTS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--warmups", type=int, default=1)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = manifest["cases"][: args.case_limit]
    if not cases:
        raise ValueError("manifest contains no benchmark cases")
    for _ in range(args.warmups):
        evaluate(args.lib, cases[0], args.device, "baseline")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for repeat in range(args.repeats):
            for case_index, case in enumerate(cases):
                shift = (repeat + case_index) % len(args.variants)
                ordered = args.variants[shift:] + args.variants[:shift]
                for variant in ordered:
                    record = evaluate(args.lib, case, args.device, variant)
                    record["repeat"] = repeat
                    stream.write(json.dumps(record, allow_nan=True, separators=(",", ":")) + "\n")
                    stream.flush()
    print(json.dumps({"cases": len(cases), "variants": args.variants, "repeats": args.repeats}))


if __name__ == "__main__":
    main()
