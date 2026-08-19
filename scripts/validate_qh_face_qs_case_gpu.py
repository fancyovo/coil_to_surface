from __future__ import annotations

import argparse
import atexit
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.desc_psi_volume_initial_guess_experiment import load_psi_model  # noqa: E402
from scripts.diagnose_alpha_boozer_residual import GpuBOnlyFieldAdapter  # noqa: E402
from scripts.guarded_boozer_from_alpha_nu import evaluate_state, surface_from_dofs  # noqa: E402
from stellarator_eval.config import PsiFitConfig  # noqa: E402
from stellarator_eval.field import load_case_file  # noqa: E402
from stellarator_eval.psi import _make_gpu_field  # noqa: E402


TWOPI = 2.0 * np.pi


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def area_weighted_helical_error(surface, field, helicity_m: int, helicity_n: int, *, sdim: int = 16, n_alpha: int = 64) -> float:
    sampled = surface_from_dofs(
        surface.get_dofs(),
        nfp=surface.nfp,
        order=surface.mpol,
        size=2 * sdim,
    )
    xyz = sampled.gamma()
    field.set_points(xyz.reshape((-1, 3)))
    modb = np.linalg.norm(np.asarray(field.B(), dtype=float).reshape(xyz.shape), axis=2)
    ds = np.linalg.norm(np.asarray(sampled.normal(), dtype=float), axis=2)
    theta = sampled.quadpoints_theta[None, :] * TWOPI
    zeta = sampled.quadpoints_phi[:, None] * sampled.nfp * TWOPI
    alpha = np.mod(helicity_m * theta - helicity_n * zeta, TWOPI)
    bins = np.linspace(0.0, TWOPI, n_alpha + 1)
    projected = np.zeros_like(modb)
    for index in range(n_alpha):
        mask = (alpha >= bins[index]) & (alpha < bins[index + 1])
        if np.any(mask):
            projected[mask] = np.sum(modb[mask] * ds[mask]) / np.sum(ds[mask])
    denominator = float(np.mean(ds * projected**2))
    return float(np.mean(ds * (modb - projected) ** 2) / max(denominator, 1e-30))


def effective_geometry(surface) -> dict[str, float]:
    xyz = np.asarray(surface.gamma(), dtype=float)
    major_radius = float(np.mean(np.sqrt(xyz[:, :, 0] ** 2 + xyz[:, :, 1] ** 2)))
    from simsopt.geo import Volume

    volume = abs(float(Volume(surface).J()))
    minor_radius = math.sqrt(volume / max(2.0 * np.pi**2 * major_radius, 1e-30))
    return {
        "volume_m3": volume,
        "major_radius_m": major_radius,
        "effective_minor_radius_m": minor_radius,
        "inverse_aspect_ratio": minor_radius / max(major_radius, 1e-30),
    }


def validate_case(case_dir: Path, *, gpu_lib: Path, device: int) -> dict[str, Any]:
    root = case_dir / "face_qs"
    output_path = root / "validation_summary.json"
    if output_path.is_file():
        existing = read_json(output_path)
        if existing.get("status") == "ok":
            return existing
        raise FileExistsError(f"failed validation already exists for {case_dir.name}")
    prepared = read_json(root / "prepare_summary.json")
    metadata = read_json(case_dir / "metadata.json")
    result: dict[str, Any] = {"case_id": metadata["case_id"], "status": "failed", "surfaces": []}
    started = time.perf_counter()
    gpu_field = None
    try:
        if prepared.get("status") != "ok":
            raise RuntimeError("GPU preparation did not pass")
        field_input = load_case_file(case_dir / "case.json", "raw")
        config = PsiFitConfig(
            backend="gpu",
            gpu_lib_path=str(gpu_lib.resolve()),
            gpu_device=device,
            gpu_segments_per_coil=256,
        )
        gpu_field = _make_gpu_field(field_input, field_input.nfp, config, "A")
        atexit.register(gpu_field.close)
        field = GpuBOnlyFieldAdapter(gpu_field, precision="fp32")
        model = load_psi_model(Path(prepared["source_psi_dir"]) / "psi_model.npz")
        for target in prepared["surfaces"]:
            surface_started = time.perf_counter()
            solve_dir = root / "cpu_solve" / target["name"]
            solve_summary_path = solve_dir / "summary.json"
            row: dict[str, Any] = {"name": target["name"], "s_level": float(target["s_level"]), "status": "failed"}
            if not solve_summary_path.is_file():
                row["error"] = "missing CPU solve summary"
                result["surfaces"].append(row)
                continue
            solve_summary = read_json(solve_summary_path)
            row["cpu_solve"] = solve_summary
            output_surface = solve_summary.get("output_surface")
            if output_surface is None or not Path(output_surface).is_file():
                row["error"] = "CPU solve did not produce a candidate surface"
                result["surfaces"].append(row)
                continue
            with np.load(target["surface_npz"]) as initial_saved:
                initial_dofs = np.asarray(initial_saved["dofs"], dtype=float)
                initial_iota = float(initial_saved["iota"])
                initial_G = float(initial_saved["G"])
                nfp = int(initial_saved["nfp"])
                order = int(initial_saved["order"])
            with np.load(output_surface) as final_saved:
                final_dofs = np.asarray(final_saved["dofs"], dtype=float)
                final_iota = float(final_saved["iota"])
                final_G = float(final_saved["G"])
            reference_surface = surface_from_dofs(initial_dofs, nfp=nfp, order=order, size=97, offset=0.371)
            reference_xyz = reference_surface.gamma()
            initial = evaluate_state(
                initial_dofs,
                field=field,
                model=model,
                nfp=nfp,
                order=order,
                iota=initial_iota,
                G=initial_G,
                target_s=float(target["s_level"]),
                reference_xyz=reference_xyz,
                grid_sizes=[2 * order + 1, 49, 97],
            )
            final = evaluate_state(
                final_dofs,
                field=field,
                model=model,
                nfp=nfp,
                order=order,
                iota=final_iota,
                G=final_G,
                target_s=float(target["s_level"]),
                reference_xyz=reference_xyz,
                grid_sizes=[2 * order + 1, 49, 97],
            )
            final_surface = surface_from_dofs(final_dofs, nfp=nfp, order=order, size=32)
            dense = final["grids"][-1]
            checks = {
                "solver_converged": bool(solve_summary.get("solver_converged", False)),
                "dense_relative_l2": float(dense["relative_l2"]) <= 1e-4,
                "dense_normal_field_p95": float(dense["normal_B_sine_p95"]) <= 1e-4,
                "toroidal_winding": float(final["geometry"]["geometric_toroidal_winding"]["min"]) > 0.0,
                "normal_nonzero": float(final["geometry"]["normal_norm"]["min"]) > 1e-12,
            }
            qs = {
                "QA_1_0": area_weighted_helical_error(final_surface, field, 1, 0),
                "QH_1_1": area_weighted_helical_error(final_surface, field, 1, 1),
                "QP_0_1": area_weighted_helical_error(final_surface, field, 0, 1),
            }
            row.update(
                {
                    "status": "accepted" if all(checks.values()) else "validation_rejected",
                    "accepted": all(checks.values()),
                    "acceptance_checks": checks,
                    "initial": initial,
                    "final": final,
                    "initial_iota": initial_iota,
                    "final_iota": final_iota,
                    "iota_error": final_iota - initial_iota,
                    "surface_qs_error": qs,
                    "geometry": effective_geometry(final_surface),
                    "validation_wall_s": time.perf_counter() - surface_started,
                }
            )
            result["surfaces"].append(row)
        result["status"] = "ok"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        if gpu_field is not None:
            gpu_field.close()
            atexit.unregister(gpu_field.close)
    result["total_wall_s"] = time.perf_counter() - started
    write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate all solved surfaces for one trajectory center on GPU.")
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--gpu-lib", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    result = validate_case(args.case_dir, gpu_lib=args.gpu_lib, device=args.device)
    print(json.dumps({"case_id": result["case_id"], "status": result["status"], "wall_s": result["total_wall_s"]}), flush=True)
    if result["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
