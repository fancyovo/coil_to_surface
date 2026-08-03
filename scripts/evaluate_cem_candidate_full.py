from __future__ import annotations

import argparse
import base64
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import types
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_float_list(value: str) -> list[float]:
    values = [float(item) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise ValueError("expected a non-empty comma-separated list of positive values")
    return values


def select_largest_surface(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [
        row
        for row in rows
        if row.get("best_surface") is not None
        and np.isfinite(float(row["best_surface"].get("volume", float("nan"))))
    ]
    return max(valid, key=lambda row: float(row["best_surface"]["volume"])) if valid else None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=True), encoding="utf-8")


def install_headless_tkinter_stub() -> None:
    try:
        import tkinter  # noqa: F401
    except ModuleNotFoundError:
        module = types.ModuleType("tkinter")

        class HeadlessTk:
            def winfo_fpixels(self, value: str) -> float:
                del value
                return 72.0

        module.Tk = HeadlessTk
        module._tkinter = types.SimpleNamespace(TclError=RuntimeError)
        sys.modules["tkinter"] = module


def preflight_desc_environment() -> dict[str, Any]:
    install_headless_tkinter_stub()
    import jax
    import desc.plotting  # noqa: F401

    jax_devices = list(jax.devices())
    devices = [str(device) for device in jax_devices]
    return {
        "jax_backend": str(jax.default_backend()),
        "jax_devices": devices,
        "gpu_available": any(
            str(getattr(device, "platform", "")).lower() == "gpu"
            for device in jax_devices
        ),
    }


def surface_path(run_dir: Path, summary: dict[str, Any]) -> Path:
    level = float(summary["best_surface"]["psi_level"])
    return run_dir / f"level_{level:.6g}".replace(".", "p") / "boozer_surface.npz"


def rotate_z(points: np.ndarray, angle: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    out = points.copy()
    out[..., 0] = cosine * points[..., 0] - sine * points[..., 1]
    out[..., 1] = sine * points[..., 0] + cosine * points[..., 1]
    return out


def build_full_period_surface_mesh(
    xyz: np.ndarray,
    colors: np.ndarray,
    nfp: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz, dtype=float)
    colors = np.asarray(colors, dtype=float)
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or colors.shape != xyz.shape:
        raise ValueError("xyz and colors must have matching (nphi, ntheta, 3) shapes")
    if nfp <= 0:
        raise ValueError("nfp must be positive")

    full_xyz = np.concatenate(
        [rotate_z(xyz, 2.0 * np.pi * period / nfp) for period in range(nfp)],
        axis=0,
    )
    full_colors = np.concatenate([colors] * nfp, axis=0)
    nphi, ntheta, _ = full_xyz.shape
    triangles: list[int] = []
    for i in range(nphi):
        i_next = (i + 1) % nphi
        for j in range(ntheta):
            j_next = (j + 1) % ntheta
            a = i * ntheta + j
            b = i_next * ntheta + j
            c = i_next * ntheta + j_next
            d = i * ntheta + j_next
            triangles.extend((a, b, d, b, c, d))
    return full_xyz, full_colors, np.asarray(triangles, dtype=np.uint32)


def write_image_html(image_path: Path, output_path: Path, title: str) -> None:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    output_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title>"
        "<style>html,body{margin:0;background:#f5f5f3;color:#171717;font-family:Georgia,serif}"
        "main{min-height:100vh;display:grid;place-items:center;padding:24px;box-sizing:border-box}"
        "img{display:block;max-width:100%;max-height:calc(100vh - 48px);box-shadow:0 8px 30px #0002}"
        "</style></head><body><main>"
        f"<img src='data:image/png;base64,{encoded}' alt='{title}'>"
        "</main></body></html>",
        encoding="utf-8",
    )


def save_periodic_colored_contours(
    *,
    values: np.ndarray,
    output_path: Path,
    xlabel: str,
    ylabel: str,
    title: str,
    colorbar_label: str,
    ncontours: int = 32,
) -> None:
    """Save colored contour lines for data periodic in both coordinates."""
    import matplotlib.pyplot as plt

    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("values must have a two-dimensional periodic grid")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must be finite")
    if ncontours < 2:
        raise ValueError("ncontours must be at least 2")

    closed = np.pad(values, ((0, 1), (0, 1)), mode="wrap")
    x = np.linspace(0.0, 2.0 * np.pi, values.shape[0] + 1)
    y = np.linspace(0.0, 2.0 * np.pi, values.shape[1] + 1)
    value_min = float(np.min(values))
    value_max = float(np.max(values))
    if value_max == value_min:
        scale = max(abs(value_min), 1.0)
        value_min -= 1.0e-12 * scale
        value_max += 1.0e-12 * scale
    levels = np.linspace(value_min, value_max, ncontours)

    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    contours = axis.contour(
        x,
        y,
        closed.T,
        levels=levels,
        cmap="turbo",
        linewidths=1.0,
    )
    axis.set_xlim(x[0], x[-1])
    axis.set_ylim(y[0], y[-1])
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    figure.colorbar(contours, ax=axis, label=colorbar_label)
    figure.tight_layout()
    figure.savefig(output_path, dpi=190)
    plt.close(figure)


def write_three_geometry_html(
    *,
    output_path: Path,
    xyz: np.ndarray,
    colors: np.ndarray,
    coils: list[np.ndarray],
    nfp: int,
    b_min: float,
    b_max: float,
) -> None:
    full_xyz, full_colors, triangles = build_full_period_surface_mesh(xyz, colors, nfp)

    payload = {
        "positions": np.asarray(full_xyz, dtype=np.float32).reshape(-1).tolist(),
        "colors": np.asarray(full_colors, dtype=np.float32).reshape(-1).tolist(),
        "indices": triangles.tolist(),
        "coils": [np.asarray(coil, dtype=np.float32).reshape(-1).tolist() for coil in coils],
        "nfp": nfp,
        "bMin": b_min,
        "bMax": b_max,
    }
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    template = """<!doctype html>
<html><head><meta charset="utf-8"><title>Optimized coils and Boozer surface</title>
<style>
html,body,#view{width:100%;height:100%;margin:0;overflow:hidden;background:#f4f4f0}
#label{position:fixed;left:16px;top:14px;padding:9px 11px;background:#fffffff0;border:1px solid #2223;
font:14px/1.35 Georgia,serif;color:#171717;z-index:2}
</style>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.164.1/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.164.1/examples/jsm/"}}</script>
</head><body><div id="view"></div><div id="label">Full device<br>|B|: __BMIN__ to __BMAX__ T</div>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
const data=__DATA__;
const root=document.getElementById('view');
const scene=new THREE.Scene(); scene.background=new THREE.Color(0xf4f4f0);
const camera=new THREE.PerspectiveCamera(40,innerWidth/innerHeight,0.001,100);
const renderer=new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2)); renderer.setSize(innerWidth,innerHeight);
renderer.outputColorSpace=THREE.SRGBColorSpace; root.appendChild(renderer.domElement);
const geometry=new THREE.BufferGeometry();
geometry.setAttribute('position',new THREE.Float32BufferAttribute(data.positions,3));
geometry.setAttribute('color',new THREE.Float32BufferAttribute(data.colors,3));
geometry.setIndex(data.indices); geometry.computeVertexNormals();
const material=new THREE.MeshStandardMaterial({vertexColors:true,side:THREE.DoubleSide,roughness:0.72,metalness:0.02,transparent:true,opacity:0.88});
scene.add(new THREE.Mesh(geometry,material));
const coilMaterial=new THREE.LineBasicMaterial({color:0x111111});
for(const values of data.coils){const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(values,3));scene.add(new THREE.LineLoop(g,coilMaterial));}
scene.add(new THREE.HemisphereLight(0xffffff,0x777777,2.2));
const key=new THREE.DirectionalLight(0xffffff,2.2);key.position.set(2,-3,4);scene.add(key);
const bounds=new THREE.Box3().setFromObject(scene);const center=bounds.getCenter(new THREE.Vector3());const size=bounds.getSize(new THREE.Vector3()).length();
camera.position.set(center.x+0.95*size,center.y-1.25*size,center.z+0.8*size);camera.near=size/1000;camera.far=size*20;camera.updateProjectionMatrix();
const controls=new OrbitControls(camera,renderer.domElement);controls.target.copy(center);controls.enableDamping=true;
function draw(){controls.update();renderer.render(scene,camera);requestAnimationFrame(draw)} draw();
addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight)});
</script></body></html>"""
    output_path.write_text(
        template.replace("__DATA__", data)
        .replace("__BMIN__", f"{b_min:.5g}")
        .replace("__BMAX__", f"{b_max:.5g}"),
        encoding="utf-8",
    )


def render_boozer_and_geometry(
    *,
    case_file: Path,
    surface_npz: Path,
    output_dir: Path,
    current_unit: str,
    surface_order: int,
    poincare_nfieldlines: int = 8,
    poincare_tmax: float = 2.0e7,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from scripts.desc_psi_volume_initial_guess_experiment import make_xyz_surface
    from scripts.plot_poincare_validation import plot_poincare_from_dofs
    from stellarator_eval.field import build_field, load_case_file

    output_dir.mkdir(parents=True, exist_ok=True)
    field_input = load_case_file(case_file, "raw")
    built = build_field(field_input, current_unit=current_unit)
    surface, surface_meta = make_xyz_surface(
        surface_npz,
        nfp=field_input.nfp,
        order=surface_order,
        stellsym=True,
        nphi=96,
        ntheta=192,
    )
    xyz = np.asarray(surface.gamma(), dtype=float)
    built.field.set_points(xyz.reshape(-1, 3))
    b_abs = np.linalg.norm(np.asarray(built.field.B(), dtype=float), axis=1).reshape(xyz.shape[:2])
    save_periodic_colored_contours(
        values=b_abs,
        output_path=output_dir / "boozer_b.png",
        xlabel=r"field-period angle $N_{\rm FP}\phi$",
        ylabel=r"Boozer $\theta$",
        title=r"Colored $|B|$ contours on largest Boozer-solvable surface",
        colorbar_label=r"$|B|$ [T]",
    )

    write_image_html(output_dir / "boozer_b.png", output_dir / "boozer_b.html", "Boozer |B|")

    b_min = float(np.min(b_abs))
    b_max = float(np.max(b_abs))
    color_scale = plt.get_cmap("turbo")((b_abs - b_min) / max(b_max - b_min, 1.0e-30))[..., :3]
    coil_points = [np.asarray(coil.curve.gamma(), dtype=float) for coil in built.field.coils]
    write_three_geometry_html(
        output_path=output_dir / "coils_surface.html",
        xyz=xyz,
        colors=color_scale,
        coils=coil_points,
        nfp=field_input.nfp,
        b_min=b_min,
        b_max=b_max,
    )

    figure = plt.figure(figsize=(8.0, 7.2))
    axis_3d = figure.add_subplot(111, projection="3d")
    stride_phi = max(1, xyz.shape[0] // 36)
    stride_theta = max(1, xyz.shape[1] // 72)
    for period in range(field_input.nfp):
        gamma = rotate_z(xyz, 2.0 * np.pi * period / field_input.nfp)
        axis_3d.plot_surface(
            gamma[..., 0],
            gamma[..., 1],
            gamma[..., 2],
            facecolors=color_scale,
            rstride=stride_phi,
            cstride=stride_theta,
            linewidth=0,
            antialiased=False,
            shade=False,
            alpha=0.82,
        )
    for gamma in coil_points:
        axis_3d.plot(gamma[:, 0], gamma[:, 1], gamma[:, 2], color="#161616", linewidth=1.4)
    all_points = np.concatenate([xyz.reshape(-1, 3), *coil_points], axis=0)
    center = 0.5 * (all_points.min(axis=0) + all_points.max(axis=0))
    extent = float(np.max(np.ptp(all_points, axis=0))) * 0.55
    axis_3d.set_xlim(center[0] - extent, center[0] + extent)
    axis_3d.set_ylim(center[1] - extent, center[1] + extent)
    axis_3d.set_zlim(center[2] - extent, center[2] + extent)
    axis_3d.set_box_aspect((1, 1, 1))
    axis_3d.set_axis_off()
    axis_3d.view_init(elev=27, azim=-48)
    figure.tight_layout(pad=0)
    figure.savefig(output_dir / "coils_surface.png", dpi=190, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)

    surface_data = np.load(surface_npz)
    coil_dofs = np.concatenate(
        [field_input.coeffs_x, field_input.coeffs_y, field_input.coeffs_z], axis=1
    )
    poincare = plot_poincare_from_dofs(
        surface_data["dofs"],
        coil_dofs,
        field_input.currents,
        field_input.nfp,
        mpol=surface_order,
        ntor=surface_order,
        coil_order=field_input.order,
        nfieldlines=poincare_nfieldlines,
        tmax_fl=poincare_tmax,
        save_path=output_dir / "poincare.png",
        marker_size=5,
        dpi=180,
    )
    return {
        "surface_meta": surface_meta,
        "b_abs_min": b_min,
        "b_abs_mean": float(np.mean(b_abs)),
        "b_abs_max": b_max,
        "poincare": poincare,
        "boozer_b_png": str(output_dir / "boozer_b.png"),
        "boozer_b_html": str(output_dir / "boozer_b.html"),
        "coils_surface_html": str(output_dir / "coils_surface.html"),
        "coils_surface_png": str(output_dir / "coils_surface.png"),
    }


def save_desc_plot(name: str, output_dir: Path, plotter, *args, **kwargs) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    started = time.perf_counter()
    try:
        figure, _ = plotter(*args, **kwargs)
        path = output_dir / f"{name}.png"
        figure.savefig(path, dpi=190, bbox_inches="tight")
        plt.close(figure)
        return {"success": True, "path": str(path), "time_s": time.perf_counter() - started}
    except Exception as exc:
        plt.close("all")
        return {
            "success": False,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "time_s": time.perf_counter() - started,
        }


def run_desc_boundary_solve(
    *,
    case_file: Path,
    surface_npz: Path,
    output_dir: Path,
    current_unit: str,
    surface_order: int,
    resolution: int,
    maxiter: int,
    ftol: float,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    install_headless_tkinter_stub()

    from desc.geometry import FourierRZToroidalSurface
    from desc.plotting import (
        plot_1d,
        plot_boozer_modes,
        plot_boozer_surface,
        plot_boundary,
        plot_qs_error,
    )
    from simsopt.geo import ToroidalFlux

    from scripts.desc_psi_volume_initial_guess_experiment import (
        build_equilibrium,
        force_stats,
        make_xyz_surface,
        write_vmec_input_from_surface,
    )
    from stellarator_eval.field import build_field, load_case_file

    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}
    try:
        field_input = load_case_file(case_file, "raw")
        built = build_field(field_input, current_unit=current_unit)
        surface, _ = make_xyz_surface(
            surface_npz,
            nfp=field_input.nfp,
            order=surface_order,
            stellsym=True,
            nphi=96,
            ntheta=144,
        )
        toroidal_flux = float(ToroidalFlux(surface, built.field).J())
        result["toroidal_flux"] = toroidal_flux
        input_path = output_dir / "input.check"
        result["input"] = write_vmec_input_from_surface(surface, toroidal_flux, input_path)
        desc_surface = FourierRZToroidalSurface.from_input_file(str(input_path))
        eq = build_equilibrium(
            desc_surface,
            toroidal_flux,
            resolution,
            resolution,
            resolution,
            constructor_ensure_nested=True,
        )
        result["nested_initial"] = bool(eq.is_nested())
        result.update(force_stats(eq, "initial"))
        result["plots_initial"] = {
            "boundary": save_desc_plot("boundary_initial", output_dir, plot_boundary, eq)
        }
        started = time.perf_counter()
        try:
            solve_result = eq.solve(maxiter=maxiter, ftol=ftol, verbose=1)
            result["solve_call_success"] = True
            result["solve_return"] = str(solve_result)[:4000]
            optimizer = solve_result[1] if isinstance(solve_result, tuple) and len(solve_result) > 1 else None
            if optimizer is not None:
                result["optimizer_success"] = bool(getattr(optimizer, "success", False))
                result["optimizer_message"] = str(getattr(optimizer, "message", ""))
                for name in ("cost", "nit", "nfev", "njev", "optimality"):
                    if hasattr(optimizer, name):
                        value = getattr(optimizer, name)
                        result[f"optimizer_{name}"] = float(value)
        except Exception as exc:
            result["solve_call_success"] = False
            result["solve_error"] = repr(exc)
            result["solve_traceback"] = traceback.format_exc()
        result["solve_time_s"] = time.perf_counter() - started
        result["nested_final"] = bool(eq.is_nested())
        result.update(force_stats(eq, "final"))
        try:
            eq.save(str(output_dir / "equilibrium.h5"))
            result["equilibrium"] = str(output_dir / "equilibrium.h5")
        except Exception as exc:
            result["equilibrium_save_error"] = repr(exc)

        if result.get("solve_call_success"):
            result["plots_final"] = {
                "boundary": save_desc_plot("boundary", output_dir, plot_boundary, eq),
                "boozer_modes": save_desc_plot(
                    "boozer_modes", output_dir, plot_boozer_modes, eq, rho=10, num_modes=12, log=True
                ),
                "boozer_B": save_desc_plot(
                    "boozer_B", output_dir, plot_boozer_surface, eq, rho=1.0, fill=False, ncontours=32
                ),
                "qs_QA": save_desc_plot(
                    "qs_QA", output_dir, plot_qs_error, eq, helicity=(1, 0), log=True
                ),
                "qs_QH": save_desc_plot(
                    "qs_QH", output_dir, plot_qs_error, eq, helicity=(1, field_input.nfp), log=True
                ),
                "qs_QP": save_desc_plot(
                    "qs_QP", output_dir, plot_qs_error, eq, helicity=(0, 1), log=True
                ),
                "iota": save_desc_plot("iota", output_dir, plot_1d, eq, "iota"),
            }
    except Exception as exc:
        result["setup_error"] = repr(exc)
        result["setup_traceback"] = traceback.format_exc()
    write_json(output_dir / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the stable Boozer/DESC evaluation for a CEM candidate.")
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", choices=("QA", "QH"), required=True)
    parser.add_argument("--a-values", default="0.05,0.08,0.12,0.16,0.20")
    parser.add_argument(
        "--levels",
        default="0.001,0.002,0.004,0.008,0.02,0.04,0.08,0.12,0.16,0.20,0.24,0.30,0.36,0.49,0.64,0.81",
    )
    parser.add_argument("--gpu-lib", default="gpu_backend/build_mixed/libstellarator_gpu.so")
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--desc-resolution", type=int, default=8)
    parser.add_argument("--desc-maxiter", type=int, default=50)
    parser.add_argument("--desc-ftol", type=float, default=1.0e-8)
    args = parser.parse_args()

    from stellarator_eval.config import EvalConfig
    from stellarator_eval.pipeline import evaluate_case_file

    args.case_file = (
        (REPO_ROOT / args.case_file).resolve()
        if not args.case_file.is_absolute()
        else args.case_file.resolve()
    )
    args.output_dir = (
        (REPO_ROOT / args.output_dir).resolve()
        if not args.output_dir.is_absolute()
        else args.output_dir.resolve()
    )
    gpu_lib = Path(args.gpu_lib)
    args.gpu_lib = str(
        (REPO_ROOT / gpu_lib).resolve()
        if not gpu_lib.is_absolute()
        else gpu_lib.resolve()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    a_values = parse_float_list(args.a_values)
    levels = parse_float_list(args.levels)
    rows = []
    total_started = time.perf_counter()
    for a_value in a_values:
        os.chdir(REPO_ROOT)
        run_dir = args.output_dir / f"a_{a_value:.4g}".replace(".", "p")
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            row = {
                "a": a_value,
                "run_dir": str(run_dir),
                "best_surface": summary.get("best_surface"),
                "quality_score": summary.get("quality_score"),
                "warnings": summary.get("warnings"),
                "total_time_s": 0.0,
                "reused": True,
            }
            rows.append(row)
            write_json(args.output_dir / "sweep_progress.json", {"rows": rows})
            print(json.dumps(row, separators=(",", ":"), allow_nan=True), flush=True)
            continue
        base = EvalConfig(current_unit="A")
        config = replace(
            base,
            axis=replace(base.axis, gpu_lib_path=args.gpu_lib, gpu_device=args.gpu_device),
            psi=replace(base.psi, a=a_value, gpu_lib_path=args.gpu_lib, gpu_device=args.gpu_device),
            scan=replace(
                base.scan,
                levels=levels,
                max_boozer_candidates=6,
                gpu_lib_path=args.gpu_lib,
                gpu_device=args.gpu_device,
            ),
            boozer=replace(base.boozer, gpu_lib_path=args.gpu_lib, gpu_device=args.gpu_device),
        )
        started = time.perf_counter()
        try:
            summary = evaluate_case_file(
                args.case_file,
                key="raw",
                config=config,
                output_dir=run_dir,
                target=args.target,
            )
            row = {
                "a": a_value,
                "run_dir": str(run_dir),
                "best_surface": summary.get("best_surface"),
                "quality_score": summary.get("quality_score"),
                "warnings": summary.get("warnings"),
                "total_time_s": time.perf_counter() - started,
            }
        except Exception as exc:
            row = {
                "a": a_value,
                "run_dir": str(run_dir),
                "best_surface": None,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "total_time_s": time.perf_counter() - started,
            }
        finally:
            os.chdir(REPO_ROOT)
        rows.append(row)
        write_json(args.output_dir / "sweep_progress.json", {"rows": rows})
        print(json.dumps(row, separators=(",", ":"), allow_nan=True), flush=True)

    best = select_largest_surface(rows)
    result: dict[str, Any] = {
        "case_file": str(args.case_file),
        "target": args.target,
        "rows": rows,
        "best": best,
        "total_sweep_time_s": time.perf_counter() - total_started,
    }
    if best is None:
        result["status"] = "no_boozer_surface"
        write_json(args.output_dir / "full_summary.json", result)
        raise SystemExit("no Boozer LS/Newton surface succeeded")

    best_run_dir = Path(best["run_dir"])
    best_summary = json.loads((best_run_dir / "summary.json").read_text(encoding="utf-8"))
    source_surface = surface_path(best_run_dir, best_summary)
    saved_surface = args.output_dir / "best_boozer_surface.npz"
    shutil.copy2(source_surface, saved_surface)
    result["best_surface_npz"] = str(saved_surface)
    result["visualization"] = render_boozer_and_geometry(
        case_file=args.case_file,
        surface_npz=saved_surface,
        output_dir=args.output_dir / "assets",
        current_unit="A",
        surface_order=int(best_summary["config"]["boozer"]["surface_order"]),
    )
    result["desc"] = run_desc_boundary_solve(
        case_file=args.case_file,
        surface_npz=saved_surface,
        output_dir=args.output_dir / "desc",
        current_unit="A",
        surface_order=int(best_summary["config"]["boozer"]["surface_order"]),
        resolution=args.desc_resolution,
        maxiter=args.desc_maxiter,
        ftol=args.desc_ftol,
    )
    result["status"] = "completed"
    result["total_time_s"] = time.perf_counter() - total_started
    write_json(args.output_dir / "full_summary.json", result)
    print(json.dumps({"status": result["status"], "output": str(args.output_dir)}), flush=True)


if __name__ == "__main__":
    main()
