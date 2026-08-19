from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_GPU_SYMBOLS = (
    "sgpu_trace_axis_samples",
    "sgpu_fit_psi_fullgpu",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_logged(command: list[str], log_path: Path) -> float:
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd="/",
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(f"command exited {completed.returncode}; see {log_path}")
    return elapsed


def validate_gpu_library(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    library = ctypes.CDLL(str(path.resolve()))
    missing = [symbol for symbol in REQUIRED_GPU_SYMBOLS if not hasattr(library, symbol)]
    if missing:
        raise RuntimeError(f"GPU library is missing required ABI symbols: {', '.join(missing)}")


def prepare_case(case_dir: Path, *, gpu_lib: Path, device: int) -> dict[str, Any]:
    output = case_dir / "face_qs"
    summary_path = output / "prepare_summary.json"
    if summary_path.is_file():
        existing = read_json(summary_path)
        if existing.get("status") == "ok":
            return existing
        raise FileExistsError(f"failed preparation already exists for {case_dir.name}")
    output.mkdir(parents=True, exist_ok=False)
    metadata = read_json(case_dir / "metadata.json")
    case_file = case_dir / "case.json"
    source_dir = output / "source_psi"
    alpha_dir = output / "alpha"
    alpha_nu_dir = output / "alpha_nu"
    targets = metadata["surface_targets"]
    levels = sorted({float(target["s_level"]) for target in targets})
    rho_values = sorted({float(target["rho_in_alpha_fit"]) for target in targets})
    timings: dict[str, float] = {}
    started = time.perf_counter()
    result: dict[str, Any] = {
        "format": metadata["format"],
        "case_id": metadata["case_id"],
        "status": "failed",
        "gpu_device": int(device),
        "gpu_library": str(gpu_lib.resolve()),
    }
    try:
        source_command = [
            sys.executable,
            "-m",
            "stellarator_eval.cli",
            "--case-file",
            str(case_file),
            "--key",
            "raw",
            "--output-dir",
            str(source_dir),
            "--current-unit",
            "A",
            "--a",
            str(metadata["source_a_m"]),
            "--levels",
            ",".join(f"{value:.12g}" for value in levels),
            "--max-boozer-candidates",
            "0",
            "--psi-n-r",
            "48",
            "--psi-n-z",
            "48",
            "--psi-n-phi",
            "48",
            "--psi-backend",
            "fullgpu",
            "--psi-linear-solver",
            "qr",
            "--psi-normal-eq-precision",
            "fp32",
            "--psi-gpu-lib",
            str(gpu_lib),
            "--psi-gpu-device",
            str(device),
            "--axis-gpu-lib",
            str(gpu_lib),
            "--axis-gpu-device",
            str(device),
            "--screen-gpu-lib",
            str(gpu_lib),
            "--screen-gpu-device",
            str(device),
            "--screen-gpu-verify-precision",
            "none",
            "--surface-gpu-lib",
            str(gpu_lib),
            "--surface-gpu-device",
            str(device),
        ]
        timings["source_psi_s"] = run_logged(source_command, output / "source_psi.log")
        if not (source_dir / "psi_model.npz").is_file():
            raise RuntimeError("source psi stage did not produce psi_model.npz")

        alpha_command = [
            sys.executable,
            str(ROOT / "scripts" / "alpha_clebsch_ls_experiment.py"),
            "--run-dir",
            str(source_dir),
            "--case-file",
            str(case_file),
            "--s-edge",
            str(metadata["alpha_s_edge"]),
            "--out-dir",
            str(alpha_dir),
            "--orders",
            "12:12:16",
            "--iota-degree",
            "3",
            "--train-points",
            "120000",
            "--validation-points",
            "60000",
            "--ray-candidate-oversampling",
            "1.25",
            "--minimum-candidate-valid-fraction",
            "0.0",
            "--sampling-backend",
            "gpu-ray",
            "--precision",
            "fp32",
            "--gpu-lib",
            str(gpu_lib),
            "--device",
            "cuda",
            "--skip-fieldline-plot",
        ]
        timings["alpha_s"] = run_logged(alpha_command, output / "alpha.log")

        nu_command = [
            sys.executable,
            str(ROOT / "scripts" / "diagnose_alpha_toroidal_correction.py"),
            "--run-dir",
            str(source_dir),
            "--case-file",
            str(case_file),
            "--alpha-dir",
            str(alpha_dir),
            "--alpha-fit",
            "alpha_fit_L12_M12_N16.npz",
            "--output-dir",
            str(alpha_nu_dir),
            "--s-edge",
            str(metadata["alpha_s_edge"]),
            "--rho-values",
            ",".join(f"{value:.12g}" for value in rho_values),
            "--nu-orders",
            "12",
            "--surface-order",
            "12",
            "--gpu-lib",
            str(gpu_lib),
            "--gpu-device",
            str(device),
            "--field-precision",
            "fp32",
            "--save-surfaces",
        ]
        timings["alpha_nu_s"] = run_logged(nu_command, output / "alpha_nu.log")

        nu_summary = read_json(alpha_nu_dir / "summary.json")
        saved_surfaces = nu_summary.get("saved_surfaces", [])
        surfaces = []
        for target in targets:
            rho = float(target["rho_in_alpha_fit"])
            if not saved_surfaces:
                raise RuntimeError("alpha+nu summary contains no saved surfaces")
            saved_surface = min(saved_surfaces, key=lambda row: abs(float(row["rho"]) - rho))
            if abs(float(saved_surface["rho"]) - rho) > 1e-8:
                raise RuntimeError(f"saved alpha+nu rho does not match target {rho}")
            path = Path(saved_surface["alpha_nu"])
            if not path.is_file():
                raise FileNotFoundError(path)
            surfaces.append({**target, "surface_npz": str(path.resolve())})
        result.update(
            {
                "status": "ok",
                "source_psi_dir": str(source_dir.resolve()),
                "alpha_dir": str(alpha_dir.resolve()),
                "alpha_nu_dir": str(alpha_nu_dir.resolve()),
                "surfaces": surfaces,
                "timing_s": timings,
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        result["timing_s"] = timings
    result["total_wall_s"] = time.perf_counter() - started
    write_json(summary_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run source-psi and alpha+nu preparation for one GPU shard.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--gpu-lib", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index")
    validate_gpu_library(args.gpu_lib)
    records = read_json(args.experiment_root / "cases.json")
    selected = [row for index, row in enumerate(records) if index % args.shard_count == args.shard_index]
    if args.limit > 0:
        selected = selected[: args.limit]
    counts = {"ok": 0, "failed": 0}
    for item_index, row in enumerate(selected, start=1):
        case_dir = args.experiment_root / "cases" / row["case_id"]
        result = prepare_case(case_dir, gpu_lib=args.gpu_lib, device=args.device)
        counts[result["status"]] += 1
        print(
            json.dumps(
                {
                    "event": "prepared",
                    "case_id": row["case_id"],
                    "status": result["status"],
                    "index": item_index,
                    "count": len(selected),
                    "wall_s": result["total_wall_s"],
                }
            ),
            flush=True,
        )
    write_json(
        args.experiment_root / f"prepare_shard_{args.shard_index:02d}.json",
        {"shard_index": args.shard_index, "shard_count": args.shard_count, "selected": len(selected), "counts": counts},
    )


if __name__ == "__main__":
    main()
