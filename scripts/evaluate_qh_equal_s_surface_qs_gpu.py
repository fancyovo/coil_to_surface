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

from stellarator_eval.volume_qs import (  # noqa: E402
    _cartesian_gradient,
    _surface_radius_on_rays,
    evaluate_psi_tensor_numpy,
    load_psi_model,
)


TWOPI = 2.0 * np.pi


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def polynomial_derivative(coefficients: np.ndarray, value: float) -> float:
    return float(sum(power * coefficient * value ** (power - 1) for power, coefficient in enumerate(coefficients, 1)))


def periodic_surface_area_density(R: np.ndarray, Z: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Return |dX/dtheta x dX/dphi| on a uniform one-period cylindrical grid."""
    if R.shape != Z.shape or R.shape != phi.shape or R.ndim != 2:
        raise ValueError("R, Z, and phi must be equally shaped two-dimensional grids")
    n_phi, n_theta = R.shape
    dtheta = TWOPI / n_theta
    dphi = TWOPI / (n_phi * _period_count_from_phi(phi))
    R_theta = (np.roll(R, -1, axis=1) - np.roll(R, 1, axis=1)) / (2.0 * dtheta)
    Z_theta = (np.roll(Z, -1, axis=1) - np.roll(Z, 1, axis=1)) / (2.0 * dtheta)
    R_phi = (np.roll(R, -1, axis=0) - np.roll(R, 1, axis=0)) / (2.0 * dphi)
    Z_phi = (np.roll(Z, -1, axis=0) - np.roll(Z, 1, axis=0)) / (2.0 * dphi)
    cosine = np.cos(phi)
    sine = np.sin(phi)
    tangent_theta = np.stack((R_theta * cosine, R_theta * sine, Z_theta), axis=-1)
    tangent_phi = np.stack(
        (R_phi * cosine - R * sine, R_phi * sine + R * cosine, Z_phi),
        axis=-1,
    )
    return np.linalg.norm(np.cross(tangent_theta, tangent_phi), axis=-1)


def _period_count_from_phi(phi: np.ndarray) -> int:
    values = np.unwrap(np.asarray(phi[:, 0], dtype=float))
    if len(values) < 2:
        raise ValueError("the phi grid needs at least two points")
    step = float(np.median(np.diff(values)))
    period = step * len(values)
    nfp = int(round(TWOPI / period))
    if nfp < 1 or not math.isclose(period * nfp, TWOPI, rel_tol=1e-7, abs_tol=1e-10):
        raise ValueError("phi grid does not span one integer field period")
    return nfp


def sample_equal_s_surface(model, s_level: float, *, n_phi: int, n_theta: int) -> dict[str, np.ndarray | float]:
    phi_values = (np.arange(n_phi, dtype=float) + 0.371) * TWOPI / (model.nfp * n_phi)
    theta_values = (np.arange(n_theta, dtype=float) + 0.613) * TWOPI / n_theta
    phi, theta = np.meshgrid(phi_values, theta_values, indexing="ij")
    radius, root_residual = _surface_radius_on_rays(
        model,
        float(s_level),
        theta,
        phi,
        max_radius=model.a,
    )
    axis_R, axis_Z, _, _ = model.axis_at(phi.ravel())
    R = axis_R.reshape(phi.shape) + radius.reshape(phi.shape) * np.cos(theta)
    Z = axis_Z.reshape(phi.shape) + radius.reshape(phi.shape) * np.sin(theta)
    s, grad_R, grad_Z, grad_phi_coordinate = evaluate_psi_tensor_numpy(model, R, Z, phi)
    grad_s = _cartesian_gradient(
        grad_R.ravel(),
        grad_phi_coordinate.ravel(),
        grad_Z.ravel(),
        phi.ravel(),
        R.ravel(),
    )
    xyz = np.column_stack((R.ravel() * np.cos(phi.ravel()), R.ravel() * np.sin(phi.ravel()), Z.ravel()))
    area_density = periodic_surface_area_density(R, Z, phi)
    return {
        "xyz": xyz,
        "grad_s": grad_s,
        "area_weight": area_density.ravel(),
        "s": s.ravel(),
        "s_level": float(s_level),
        "root_residual": np.asarray(root_residual, dtype=float).ravel(),
        "radius": np.asarray(radius, dtype=float).ravel(),
    }


def weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.sum(weights * values * values) / max(np.sum(weights), 1e-300)))


def weighted_effective_fraction(weights: np.ndarray) -> float:
    return float(np.sum(weights) ** 2 / max(len(weights) * np.sum(weights * weights), 1e-300))


def differential_qh_surface_metrics(
    points: dict[str, np.ndarray | float],
    B: np.ndarray,
    grad_B: np.ndarray,
    *,
    flux_derivative: float,
    iota: float,
    G: float,
    nfp: int,
) -> dict[str, float | int]:
    B = np.asarray(B, dtype=float)
    grad_B = np.asarray(grad_B, dtype=float)
    grad_s = np.asarray(points["grad_s"], dtype=float)
    weights = np.asarray(points["area_weight"], dtype=float)
    magnitude = np.linalg.norm(B, axis=1)
    grad_magnitude = np.einsum("nij,ni->nj", grad_B, B) / np.maximum(magnitude[:, None], 1e-30)
    grad_psi = float(flux_derivative) * grad_s
    A = np.sum(np.cross(B, grad_psi) * grad_magnitude, axis=1)
    C = np.sum(B * grad_magnitude, axis=1)
    residual = ((float(iota) - int(nfp)) * A - float(G) * C) / np.maximum(magnitude**3, 1e-30)
    helicity_norm = math.hypot(1.0, float(nfp))
    normal_sine = np.abs(np.sum(B * grad_s, axis=1)) / np.maximum(
        magnitude * np.linalg.norm(grad_s, axis=1), 1e-30
    )
    return {
        "point_count": int(len(B)),
        "raw_area_rms": weighted_rms(residual, weights),
        "per_helicity_area_rms": weighted_rms(residual / helicity_norm, weights),
        "raw_unweighted_rms": float(np.sqrt(np.mean(residual * residual))),
        "area_weight_effective_fraction": weighted_effective_fraction(weights),
        "normal_B_sine_area_rms": weighted_rms(normal_sine, weights),
        "normal_B_sine_p95": float(np.percentile(normal_sine, 95)),
        "s_residual_rms": float(np.sqrt(np.mean((np.asarray(points["s"]) - float(points["s_level"])) ** 2))),
        "root_residual_max": float(np.max(points["root_residual"])),
        "radius_min_m": float(np.min(points["radius"])),
        "radius_max_m": float(np.max(points["radius"])),
    }


def evaluate_case(
    case_dir: Path,
    *,
    gpu_lib: Path,
    device: int,
    n_phi: int,
    n_theta: int,
    output_name: str,
    overwrite: bool,
) -> dict[str, Any]:
    output_path = case_dir / "face_qs" / output_name
    if output_path.is_file() and not overwrite:
        return read_json(output_path)
    metadata = read_json(case_dir / "metadata.json")
    prepared = read_json(case_dir / "face_qs" / "prepare_summary.json")
    result: dict[str, Any] = {
        "format": "qh_equal_s_differential_qs_v1",
        "case_id": metadata["case_id"],
        "status": "failed",
        "grid": {"n_phi_per_period": int(n_phi), "n_theta": int(n_theta)},
        "surfaces": [],
    }
    started = time.perf_counter()
    gpu_field = None
    try:
        from stellarator_eval.config import PsiFitConfig
        from stellarator_eval.field import load_case_file
        from stellarator_eval.psi import _make_gpu_field

        if prepared.get("status") != "ok":
            result["status"] = "prepare_skipped"
            result["error"] = "GPU preparation did not pass"
            return result
        field_input = load_case_file(case_dir / "case.json", "raw")
        config = PsiFitConfig(
            backend="gpu",
            gpu_lib_path=str(gpu_lib.resolve()),
            gpu_device=device,
            gpu_segments_per_coil=256,
        )
        gpu_field = _make_gpu_field(field_input, field_input.nfp, config, "A")
        atexit.register(gpu_field.close)
        model = load_psi_model(Path(prepared["source_psi_dir"]) / "psi_model.npz")
        alpha_summary = read_json(Path(prepared["alpha_dir"]) / "summary.json")
        calibration_coefficients = np.asarray(alpha_summary["calibration_polynomial_coeffs"], dtype=float)
        for target in prepared["surfaces"]:
            surface_started = time.perf_counter()
            s_level = float(target["s_level"])
            points = sample_equal_s_surface(model, s_level, n_phi=n_phi, n_theta=n_theta)
            with np.load(target["surface_npz"]) as saved:
                iota = float(saved["iota"])
                G = float(saved["G"])
            field_started = time.perf_counter()
            B, grad_B = gpu_field.eval_B_grad(points["xyz"], precision="fp32")
            field_wall_s = time.perf_counter() - field_started
            metrics = differential_qh_surface_metrics(
                points,
                B,
                grad_B,
                flux_derivative=polynomial_derivative(calibration_coefficients, s_level),
                iota=iota,
                G=G,
                nfp=field_input.nfp,
            )
            result["surfaces"].append(
                {
                    "name": target["name"],
                    "s_level": s_level,
                    "rho_in_alpha_fit": float(target["rho_in_alpha_fit"]),
                    "iota": iota,
                    "G": G,
                    "flux_derivative": polynomial_derivative(calibration_coefficients, s_level),
                    "metrics": metrics,
                    "field_wall_s": field_wall_s,
                    "wall_s": time.perf_counter() - surface_started,
                }
            )
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
    parser = argparse.ArgumentParser(description="Evaluate area-weighted differential QH on strict fitted equal-s surfaces.")
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--gpu-lib", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--n-phi", type=int, default=96)
    parser.add_argument("--n-theta", type=int, default=96)
    parser.add_argument("--output-name", default="equal_s_qs_summary.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = evaluate_case(
        args.case_dir,
        gpu_lib=args.gpu_lib,
        device=args.device,
        n_phi=args.n_phi,
        n_theta=args.n_theta,
        output_name=args.output_name,
        overwrite=args.overwrite,
    )
    print(json.dumps({"case_id": result["case_id"], "status": result["status"], "wall_s": result["total_wall_s"]}), flush=True)
    if result["status"] not in {"ok", "prepare_skipped"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
