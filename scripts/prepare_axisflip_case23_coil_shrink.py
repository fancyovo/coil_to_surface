from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "gpu_backend" / "python"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from flow_matching.data import CoilNormalizer, file_sha256
from scripts.axisflip_coil_scale import (
    coil_score_decomposition,
    effective_radius_m,
    evaluate_curves,
    full_coil_curves,
    load_full_axis_points,
    rotate_z,
    scale_about_axis,
    scale_about_coil_axis_anchor,
    tokens_from_raw,
)
from scripts.native_score_runtime import token_case, write_json
from scripts.optimize_flow_latent import score_config
from scripts.prepare_axis_surface_prior_adam200 import exact_standardized_start
from scripts.sample_axis_surface_prior import compact_result


POINTWISE_PROTOCOL_ID = "qh-axisflip-v4-case23-axis-centered-coil-shrink-adam200-64d-abi11-v1"
COIL_ANCHOR_PROTOCOL_ID = "qh-axisflip-v4-case23-coil-anchor-shrink-adam200-64d-abi11-v1"
PROTOCOL_BY_ANCHOR_MODE = {
    "pointwise-nearest-axis": POINTWISE_PROTOCOL_ID,
    "per-coil-axis-anchor": COIL_ANCHOR_PROTOCOL_ID,
}


def parse_scales(value: str) -> list[float]:
    scales = [float(item) for item in value.split(",") if item.strip()]
    if not scales or any(not 0.0 < item <= 1.0 for item in scales):
        raise ValueError("scales must lie in (0,1]")
    if len(scales) != len(set(scales)):
        raise ValueError("scales must be unique")
    return sorted(scales, reverse=True)


def scale_tag(scale: float) -> str:
    return f"scale_{scale:.3f}".replace(".", "p")


def full_surface_points(mesh_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with np.load(mesh_path) as saved:
        one_period = np.asarray(saved["xyz_one_period"], dtype=np.float64)
        nfp = int(saved["nfp"])
    full = np.concatenate(
        [rotate_z(one_period, 2.0 * math.pi * period / nfp) for period in range(nfp)],
        axis=0,
    )
    return one_period, full, nfp


def periodic_envelope(indices: np.ndarray, distances: np.ndarray, count: int) -> np.ndarray:
    envelope = np.full(count, -np.inf, dtype=np.float64)
    np.maximum.at(envelope, indices, distances)
    valid = np.flatnonzero(np.isfinite(envelope))
    if len(valid) < 2:
        raise RuntimeError("surface samples do not cover the magnetic axis")
    query = np.arange(count)
    extended_x = np.concatenate((valid - count, valid, valid + count))
    extended_y = np.tile(envelope[valid], 3)
    return np.interp(query, extended_x, extended_y)


def geometry_metrics(
    tokens: np.ndarray,
    *,
    nfp: int,
    axis_points: np.ndarray,
    axis_tree: cKDTree,
    surface_tree: cKDTree,
    surface_envelope: np.ndarray,
) -> dict[str, float]:
    curves = full_coil_curves(tokens, nfp, samples=512)
    points = np.concatenate(curves, axis=0)
    axis_distance, axis_index = axis_tree.query(points, workers=1)
    surface_distance, _ = surface_tree.query(points, workers=1)
    tube_margin = axis_distance - surface_envelope[axis_index]
    return {
        "effective_radius_m": effective_radius_m(tokens),
        "reference_axis_distance_min_m": float(np.min(axis_distance)),
        "reference_axis_distance_p05_m": float(np.quantile(axis_distance, 0.05)),
        "reference_axis_distance_median_m": float(np.median(axis_distance)),
        "reference_axis_distance_mean_m": float(np.mean(axis_distance)),
        "reference_surface_nearest_distance_m": float(np.min(surface_distance)),
        "reference_surface_distance_p05_m": float(np.quantile(surface_distance, 0.05)),
        "reference_surface_tube_margin_min_m": float(np.min(tube_margin)),
        "reference_surface_tube_margin_p05_m": float(np.quantile(tube_margin, 0.05)),
        "reference_surface_tube_margin_median_m": float(np.median(tube_margin)),
    }


def mesh_for_html(one_period: np.ndarray, nfp: int) -> tuple[np.ndarray, np.ndarray]:
    phi_stride = max(1, one_period.shape[0] // 48)
    theta_stride = max(1, one_period.shape[1] // 72)
    reduced = one_period[::phi_stride, ::theta_stride]
    full = np.concatenate(
        [rotate_z(reduced, 2.0 * math.pi * period / nfp) for period in range(nfp)],
        axis=0,
    )
    nphi, ntheta, _ = full.shape
    triangles = []
    for i in range(nphi):
        i_next = (i + 1) % nphi
        for j in range(ntheta):
            j_next = (j + 1) % ntheta
            a = i * ntheta + j
            b = i_next * ntheta + j
            c = i_next * ntheta + j_next
            d = i * ntheta + j_next
            triangles.extend((a, b, d, b, c, d))
    return full, np.asarray(triangles, dtype=np.int32)


def write_geometry_html(
    path: Path,
    *,
    one_period_surface: np.ndarray,
    tokens: np.ndarray,
    axis_points: np.ndarray,
    nfp: int,
    scale: float,
    label: str | None = None,
) -> None:
    surface, triangles = mesh_for_html(one_period_surface, nfp)
    payload = {
        "surface": surface.astype(np.float32).reshape(-1).tolist(),
        "triangles": triangles.tolist(),
        "coils": [curve.astype(np.float32).reshape(-1).tolist() for curve in full_coil_curves(tokens, nfp, samples=384)],
        "axis": axis_points.astype(np.float32).reshape(-1).tolist(),
        "scale": scale,
        "nfp": nfp,
    }
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    display_label = label or f"coil scale {scale:.3f}"
    template = """<!doctype html><html><head><meta charset="utf-8"><title>Axis-centered coil shrink</title>
<style>html,body,#view{width:100%;height:100%;margin:0;overflow:hidden;background:#f5f5f2}#label{position:fixed;left:16px;top:14px;padding:9px 11px;background:#fffffff0;border:1px solid #2223;font:14px/1.35 Arial,sans-serif;color:#171717;z-index:2}</style>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.164.1/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.164.1/examples/jsm/"}}</script></head><body><div id="view"></div><div id="label">axisflip_case_0000023<br>__LABEL__<br>reference surface s=0.81</div>
<script type="module">import * as THREE from 'three';import {OrbitControls} from 'three/addons/controls/OrbitControls.js';const data=__DATA__;const scene=new THREE.Scene();scene.background=new THREE.Color(0xf5f5f2);const camera=new THREE.PerspectiveCamera(40,innerWidth/innerHeight,.001,100);const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setSize(innerWidth,innerHeight);document.getElementById('view').appendChild(renderer.domElement);const sg=new THREE.BufferGeometry();sg.setAttribute('position',new THREE.Float32BufferAttribute(data.surface,3));sg.setIndex(data.triangles);sg.computeVertexNormals();scene.add(new THREE.Mesh(sg,new THREE.MeshStandardMaterial({color:0x8cc5d6,side:THREE.DoubleSide,transparent:true,opacity:.5,roughness:.8})));const colors=[0x9f2b2b,0x205f9a,0x2f7d4b,0x9a6a20];for(let i=0;i<data.coils.length;i++){const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(data.coils[i],3));scene.add(new THREE.LineLoop(g,new THREE.LineBasicMaterial({color:colors[Math.floor(i/(2*data.nfp))%colors.length]})));}const ag=new THREE.BufferGeometry();ag.setAttribute('position',new THREE.Float32BufferAttribute(data.axis,3));scene.add(new THREE.LineLoop(ag,new THREE.LineBasicMaterial({color:0x222222})));scene.add(new THREE.HemisphereLight(0xffffff,0x777777,2.2));const key=new THREE.DirectionalLight(0xffffff,2);key.position.set(2,-3,4);scene.add(key);const bounds=new THREE.Box3().setFromObject(scene),center=bounds.getCenter(new THREE.Vector3()),size=bounds.getSize(new THREE.Vector3()).length();camera.position.set(center.x+size,center.y-1.2*size,center.z+.75*size);camera.near=size/1000;camera.far=size*20;camera.updateProjectionMatrix();const controls=new OrbitControls(camera,renderer.domElement);controls.target.copy(center);controls.enableDamping=true;function draw(){controls.update();renderer.render(scene,camera);requestAnimationFrame(draw)}draw();addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight)});</script></body></html>"""
    path.write_text(
        template.replace("__DATA__", encoded).replace("__LABEL__", display_label),
        encoding="utf-8",
    )


def write_geometry_png(
    path: Path,
    *,
    one_period_surface: np.ndarray,
    tokens: np.ndarray,
    axis_points: np.ndarray,
    nfp: int,
    scale: float,
    label: str | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(8.0, 7.2))
    axis = figure.add_subplot(111, projection="3d")
    reduced = one_period_surface[:: max(1, one_period_surface.shape[0] // 32), :: max(1, one_period_surface.shape[1] // 48)]
    for period in range(nfp):
        surface = rotate_z(reduced, 2.0 * math.pi * period / nfp)
        axis.plot_surface(surface[..., 0], surface[..., 1], surface[..., 2], color="#8cc5d6", alpha=0.32, linewidth=0, shade=True)
    colors = ("#9f2b2b", "#205f9a", "#2f7d4b", "#9a6a20")
    curves = full_coil_curves(tokens, nfp, samples=384)
    for index, curve in enumerate(curves):
        base_index = index // (2 * nfp)
        axis.plot(curve[:, 0], curve[:, 1], curve[:, 2], color=colors[base_index % len(colors)], linewidth=1.0)
    axis.plot(axis_points[:, 0], axis_points[:, 1], axis_points[:, 2], color="#202020", linewidth=1.0)
    all_points = np.concatenate((axis_points, one_period_surface.reshape(-1, 3), *curves), axis=0)
    center = 0.5 * (all_points.min(axis=0) + all_points.max(axis=0))
    extent = 0.54 * float(np.max(np.ptp(all_points, axis=0)))
    axis.set_xlim(center[0] - extent, center[0] + extent)
    axis.set_ylim(center[1] - extent, center[1] + extent)
    axis.set_zlim(center[2] - extent, center[2] + extent)
    axis.set_box_aspect((1, 1, 1))
    axis.set_axis_off()
    axis.view_init(elev=27, azim=-48)
    title = label or f"Coil scale {scale:.3f}"
    axis.set_title(f"{title}; original accepted surface s=0.81")
    figure.tight_layout(pad=0.2)
    figure.savefig(path, dpi=190, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Shrink axisflip case 23 coils around its verified magnetic axis.")
    parser.add_argument("--source-best", type=Path, required=True)
    parser.add_argument("--source-start", type=Path, required=True)
    parser.add_argument("--axis-data", type=Path, required=True)
    parser.add_argument("--surface-mesh", type=Path, required=True)
    parser.add_argument("--surface-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--score-lib", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-score-lib-sha", required=True)
    parser.add_argument("--expected-checkpoint-sha", required=True)
    parser.add_argument("--expected-source-best-sha", required=True)
    parser.add_argument("--expected-surface-sha", required=True)
    parser.add_argument(
        "--anchor-mode",
        choices=tuple(PROTOCOL_BY_ANCHOR_MODE),
        default="pointwise-nearest-axis",
    )
    parser.add_argument("--scales", type=parse_scales, default=parse_scales("1.0,0.8,0.65,0.55,0.50,0.45,0.40,0.35,0.30,0.25,0.20"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    if commit != args.expected_commit:
        raise RuntimeError(f"repository commit {commit} != expected {args.expected_commit}")
    dirty = subprocess.check_output(["git", "status", "--short", "--untracked-files=no"], cwd=REPO_ROOT, text=True)
    if dirty.strip():
        raise RuntimeError("tracked experiment worktree is dirty")
    expected_hashes = (
        (args.score_lib, args.expected_score_lib_sha),
        (args.checkpoint, args.expected_checkpoint_sha),
        (args.source_best, args.expected_source_best_sha),
        (args.surface_source, args.expected_surface_sha),
    )
    for path, expected in expected_hashes:
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"hash mismatch for {path}: {actual} != {expected}")
    protocol_id = PROTOCOL_BY_ANCHOR_MODE[args.anchor_mode]
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != protocol_id:
        raise RuntimeError("protocol ID mismatch")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    candidates_dir = args.output_dir / "candidates"
    visuals_dir = args.output_dir / "visuals"
    candidates_dir.mkdir()
    visuals_dir.mkdir()

    source_best = json.loads(args.source_best.read_text(encoding="utf-8"))
    source_start = json.loads(args.source_start.read_text(encoding="utf-8"))
    nfp = int(source_best["nfp"])
    source_tokens = tokens_from_raw(source_best["raw"])
    nc = len(source_tokens)
    if (nfp, nc) != (6, 4):
        raise RuntimeError(f"frozen source condition changed: {(nfp, nc)}")
    with np.load(args.axis_data) as axis_data:
        axis_points = load_full_axis_points(axis_data)
    one_period_surface, surface_points, surface_nfp = full_surface_points(args.surface_mesh)
    if surface_nfp != nfp:
        raise RuntimeError("surface and source nfp differ")
    surface_flat = surface_points.reshape(-1, 3)
    axis_tree = cKDTree(axis_points)
    surface_tree = cKDTree(surface_flat)
    surface_axis_distance, surface_axis_index = axis_tree.query(surface_flat, workers=1)
    surface_envelope = periodic_envelope(surface_axis_index, surface_axis_distance, len(axis_points))

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    normalizer = CoilNormalizer.from_dict(checkpoint["normalizer"])
    source_l1 = float(source_start["data_prior_screening"]["current_l1_a"])
    from stellarator_gpu import score_coils_native

    rows = []
    for scale in args.scales:
        if args.anchor_mode == "pointwise-nearest-axis":
            scaled, fit = scale_about_axis(
                source_tokens, axis_points, scale, samples=1024
            )
        else:
            scaled, fit = scale_about_coil_axis_anchor(
                source_tokens, axis_points, scale, samples=1024
            )
        fit["anchor_mode"] = args.anchor_mode
        parameters, current_l1, represented, roundtrip = exact_standardized_start(
            scaled, normalizer, condition=(nfp, nc)
        )
        if abs(current_l1 - source_l1) > max(1.0e-2, 1.0e-7 * source_l1):
            raise RuntimeError("coil-current L1 changed during geometry scaling")
        case = token_case(represented, nfp=nfp, target="QH")
        native = compact_result(
            score_coils_native(
                args.score_lib,
                case["raw"]["x"],
                case["raw"]["y"],
                case["raw"]["z"],
                case["raw"]["current"],
                nfp,
                device_id=0,
                target_helicity=(1, nfp),
                config_overrides=score_config(iota_degree=3, surface_theta_count=128, axis_hint=None),
            )
        )
        geometry = geometry_metrics(
            represented,
            nfp=nfp,
            axis_points=axis_points,
            axis_tree=axis_tree,
            surface_tree=surface_tree,
            surface_envelope=surface_envelope,
        )
        decomposition = coil_score_decomposition(native) if native.get("status") == "ok" else None
        tag = scale_tag(scale)
        case["data_prior_screening"] = {
            "format": "axisflip_case23_axis_centered_shrink_start_v1",
            "protocol_id": protocol_id,
            "normalized_coil_tokens": parameters.tolist(),
            "current_l1_a": current_l1,
            "native_score": native,
            "native_score_target_helicity": [1, nfp],
            "source_best_sha256": args.expected_source_best_sha,
            "reference_surface_sha256": args.expected_surface_sha,
            "scale": scale,
            "fit": fit,
            "roundtrip": roundtrip,
            "geometry": geometry,
        }
        case_path = candidates_dir / f"{tag}.json"
        write_json(case_path, case)
        write_geometry_png(
            visuals_dir / f"{tag}.png",
            one_period_surface=one_period_surface,
            tokens=represented,
            axis_points=axis_points,
            nfp=nfp,
            scale=scale,
        )
        write_geometry_html(
            visuals_dir / f"{tag}.html",
            one_period_surface=one_period_surface,
            tokens=represented,
            axis_points=axis_points,
            nfp=nfp,
            scale=scale,
        )
        row = {
            "tag": tag,
            "scale": scale,
            "status": native.get("status"),
            "score": native.get("score"),
            "components": native.get("components"),
            "diagnostics": native.get("diagnostics"),
            "coil_decomposition": decomposition,
            "geometry": geometry,
            "fit": fit,
            "roundtrip": roundtrip,
            "case": str(case_path.resolve()),
            "case_sha256": file_sha256(case_path),
            "visual_png": str((visuals_dir / f"{tag}.png").resolve()),
            "visual_html": str((visuals_dir / f"{tag}.html").resolve()),
        }
        rows.append(row)
        print(json.dumps({"event": "candidate", "scale": scale, "status": row["status"], "score": row["score"], **geometry}, separators=(",", ":")), flush=True)

    manifest = {
        "format": "axisflip_case23_axis_centered_shrink_scan_v1",
        "protocol_id": protocol_id,
        "status": "complete",
        "code_commit": commit,
        "source": {
            "sample_id": "axisflip_case_0000023",
            "adam2000_best_iteration": int(source_best["original_space_local_gradient_adam"]["best_iteration"]),
            "best_path": str(args.source_best.resolve()),
            "best_sha256": args.expected_source_best_sha,
            "axis_data": str(args.axis_data.resolve()),
            "axis_points_sha256": sha256_array(axis_points),
            "surface": str(args.surface_source.resolve()),
            "surface_sha256": args.expected_surface_sha,
            "surface_role": "fixed geometric reference from the original field",
        },
        "score_library_sha256": args.expected_score_lib_sha,
        "checkpoint_sha256": args.expected_checkpoint_sha,
        "transformation": {
            "anchor_mode": args.anchor_mode,
            "definition": (
                "p_scaled=a_nearest(point)+scale*(p-a_nearest(point))"
                if args.anchor_mode == "pointwise-nearest-axis"
                else "p_scaled=a_nearest(coil_centroid)+scale*(p-a_nearest(coil_centroid))"
            ),
            "axis": "full verified magnetic axis from the a=0.08 source fit",
            "refit": "least-squares order-16 Fourier coefficients at fixed coil parameter",
            "currents": "unchanged",
        },
        "surface_reference": {
            "axis_radius_min_m": float(np.min(surface_axis_distance)),
            "axis_radius_median_m": float(np.median(surface_axis_distance)),
            "axis_radius_p95_m": float(np.quantile(surface_axis_distance, 0.95)),
            "axis_radius_max_m": float(np.max(surface_axis_distance)),
        },
        "candidates": rows,
    }
    write_json(args.output_dir / "scan_summary.json", manifest)
    print(json.dumps({"event": "complete", "candidate_count": len(rows), "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
