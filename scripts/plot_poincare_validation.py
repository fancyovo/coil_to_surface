from __future__ import annotations

"""Poincare validation plot for a saved Boozer candidate surface.

This script is the maintained version of the temporary `tmp.py` diagnostic:
it traces field lines from inside a candidate surface and overlays the surface
cross section on several toroidal cuts.  It is intentionally kept outside the
main scoring path because it is a visual validation step for detailed reports.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _load_case(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if "raw" in data:
        inp = data["raw"]
        x = np.atleast_2d(np.asarray(inp["x"], dtype=float))
        y = np.atleast_2d(np.asarray(inp["y"], dtype=float))
        z = np.atleast_2d(np.asarray(inp["z"], dtype=float))
        if not (x.shape == y.shape == z.shape):
            raise ValueError(f"raw x/y/z coefficient shapes differ: {x.shape}, {y.shape}, {z.shape}")
        coeff_count = int(x.shape[1])
        currents = np.asarray(inp["current"], dtype=float).ravel()
        if currents.size != x.shape[0]:
            raise ValueError(f"raw case has {x.shape[0]} coils but {currents.size} currents")
        nfp = int(data.get("nfp", inp.get("nfp")))
        coil_dofs = np.concatenate([x, y, z], axis=1)
        return coil_dofs, currents, nfp, coeff_count

    inp = data["input"]
    coeff_count = int(inp.get("coeff_count", 33))
    packed = np.asarray(inp["packed_values"], dtype=float).ravel()
    block = 3 * coeff_count + 1
    nfp = int(round(float(packed[-1])))
    coil_part = packed[:-1]
    if coil_part.size % block != 0:
        raise ValueError(f"packed coil vector length {coil_part.size} is not a multiple of {block}")
    n_base = coil_part.size // block
    coil_dofs = []
    currents = []
    for i in range(n_base):
        row = coil_part[i * block : (i + 1) * block]
        x = row[:coeff_count]
        y = row[coeff_count : 2 * coeff_count]
        z = row[2 * coeff_count : 3 * coeff_count]
        coil_dofs.append(np.concatenate([x, y, z]))
        currents.append(float(row[-1]))
    return np.asarray(coil_dofs, dtype=float), np.asarray(currents, dtype=float), nfp, coeff_count


def plot_poincare_from_dofs(
    surf_dofs,
    coil_dofs,
    currents,
    nfp,
    *,
    mpol=6,
    ntor=6,
    coil_order=16,
    coil_quadpoints=300,
    nfieldlines=20,
    radial_width=None,
    edge_fraction=0.95,
    stop_fraction=1.5,
    tmax_fl=2.0e8,
    phis=None,
    save_path=None,
    tol=1e-11,
    plot_surface=True,
    marker_size=10,
    dpi=150,
    psi_model=None,
    psi_target=None,
):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **_: x

    from simsopt.field import (
        BiotSavart,
        Current,
        MaxRStoppingCriterion,
        MaxZStoppingCriterion,
        MinRStoppingCriterion,
        MinZStoppingCriterion,
        compute_fieldlines,
        coils_via_symmetries,
    )
    from simsopt.geo import CurveXYZFourier, SurfaceXYZTensorFourier

    surface = SurfaceXYZTensorFourier(
        mpol=int(mpol),
        ntor=int(ntor),
        nfp=int(nfp),
        stellsym=True,
        quadpoints_phi=np.linspace(0.0, 1.0 / int(nfp), 96, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 256, endpoint=False),
    )
    surface.set_dofs(np.asarray(surf_dofs, dtype=float).ravel())

    coil_dofs = np.atleast_2d(np.asarray(coil_dofs, dtype=float))
    curves_half = []
    for row in coil_dofs:
        curve = CurveXYZFourier(int(coil_quadpoints), int(coil_order))
        curve.set_dofs(row.ravel())
        curves_half.append(curve)

    currents = np.atleast_1d(np.asarray(currents, dtype=float)).ravel()
    current_mean = float(np.mean(np.abs(currents)))
    trace_currents = currents / current_mean if current_mean > 0 else currents
    print(f"Poincare currents normalized by mean|I|={current_mean:.6g}")
    current_objs = [Current(float(val)) for val in trace_currents]
    for current in current_objs:
        current.fix_all()

    coils = coils_via_symmetries(curves_half, current_objs, int(nfp), True)
    bs = BiotSavart(coils)

    gamma0 = surface.gamma()[0]
    r0 = np.linalg.norm(gamma0[:, :2], axis=1)
    z0 = gamma0[:, 2]
    seed_r = float(0.5 * (np.min(r0) + np.max(r0)))
    seed_z = float(np.mean(z0))

    if phis is None:
        phis = [(i / 4) * (2 * np.pi / int(nfp)) for i in range(4)]
    else:
        phis = list(np.asarray(phis, dtype=float).ravel())

    r_outer_phi0 = float(np.max(np.linalg.norm(surface.gamma()[0, :, :2], axis=1)))
    gamma = surface.gamma()
    r_all = np.linalg.norm(gamma[:, :, :2], axis=2)
    z_all = gamma[:, :, 2]
    r_center = 0.5 * (float(np.min(r_all)) + float(np.max(r_all)))
    z_center = 0.5 * (float(np.min(z_all)) + float(np.max(z_all)))
    r_half_width = 0.5 * (float(np.max(r_all)) - float(np.min(r_all))) * float(stop_fraction)
    z_half_width = 0.5 * (float(np.max(z_all)) - float(np.min(z_all))) * float(stop_fraction)
    stopping_criteria = [
        MinRStoppingCriterion(max(r_center - r_half_width, 0.0)),
        MaxRStoppingCriterion(r_center + r_half_width),
        MinZStoppingCriterion(z_center - z_half_width),
        MaxZStoppingCriterion(z_center + z_half_width),
    ]
    print(
        f"Stopping box: R=[{max(r_center - r_half_width, 0.0):.6g}, {r_center + r_half_width:.6g}], "
        f"Z=[{z_center - z_half_width:.6g}, {z_center + z_half_width:.6g}]"
    )

    r_stop = float(seed_r + float(edge_fraction) * (r_outer_phi0 - seed_r))
    if radial_width is None:
        r_end = r_stop
    else:
        r_end = min(seed_r + float(radial_width), r_stop)
    if r_end <= seed_r:
        r_end = r_stop
    R0 = np.linspace(seed_r, r_end, int(nfieldlines))
    Z0 = np.full(int(nfieldlines), seed_z)
    print(f"R0 range = [{R0[0]:.6g}, {R0[-1]:.6g}], boundary R(phi=0) approx {r_outer_phi0:.6g}")

    print("Beginning field line tracing")
    fieldlines_phi_hits = []
    for r_start, z_start in tqdm(list(zip(R0, Z0)), desc="Tracing fieldlines", unit="line"):
        _, hits_one = compute_fieldlines(
            bs,
            [r_start],
            [z_start],
            tmax=float(tmax_fl),
            tol=float(tol),
            phis=phis,
            stopping_criteria=stopping_criteria,
        )
        fieldlines_phi_hits.append(hits_one[0])

    hit_counts = [
        0 if np.asarray(hits).ndim != 2 else int(np.sum(np.asarray(hits)[:, 1] >= 0))
        for hits in fieldlines_phi_hits
    ]
    print("Poincare hits per line =", hit_counts)

    psi_drift = None
    if psi_model is not None:
        from stellarator_eval.volume_qs import evaluate_psi_tensor_numpy

        line_rows = []
        all_errors = []
        for line_index, (r_start, z_start, hits) in enumerate(
            zip(R0, Z0, fieldlines_phi_hits)
        ):
            start_s = float(
                evaluate_psi_tensor_numpy(psi_model, r_start, z_start, 0.0)[0]
            )
            hits = np.asarray(hits)
            if hits.ndim != 2 or hits.shape[0] == 0:
                continue
            valid = hits[hits[:, 1] >= 0]
            if valid.size == 0:
                continue
            hit_r = np.hypot(valid[:, 2], valid[:, 3])
            hit_phi = np.mod(np.arctan2(valid[:, 3], valid[:, 2]), 2.0 * np.pi)
            hit_s = evaluate_psi_tensor_numpy(
                psi_model, hit_r, valid[:, 4], hit_phi
            )[0]
            error = np.abs(hit_s - start_s)
            all_errors.append(error)
            scale = max(abs(start_s), 1e-12)
            line_rows.append(
                {
                    "line": int(line_index),
                    "start_s": start_s,
                    "hit_count": int(error.size),
                    "absolute_drift_p95": float(np.percentile(error, 95.0)),
                    "absolute_drift_max": float(np.max(error)),
                    "relative_drift_p95": float(np.percentile(error, 95.0) / scale),
                    "relative_drift_max": float(np.max(error) / scale),
                }
            )
        concatenated = np.concatenate(all_errors) if all_errors else np.empty(0)
        psi_drift = {
            "target_s": None if psi_target is None else float(psi_target),
            "lines": line_rows,
            "all_absolute_drift_p95": (
                None if concatenated.size == 0 else float(np.percentile(concatenated, 95.0))
            ),
            "all_absolute_drift_max": (
                None if concatenated.size == 0 else float(np.max(concatenated))
            ),
        }
        print("Psi drift =", json.dumps(psi_drift, ensure_ascii=True))

    nrowcol = int(np.ceil(np.sqrt(len(phis))))
    fig, axs = plt.subplots(nrowcol, nrowcol, figsize=(8, 5))
    axs = np.asarray(axs).reshape((nrowcol, nrowcol))
    line_colors = plt.get_cmap("viridis")(np.linspace(0.0, 1.0, len(fieldlines_phi_hits)))
    for i, phi in enumerate(phis):
        ax = axs[i // nrowcol, i % nrowcol]
        ax.set_aspect("equal")
        ax.set_title(f"$\\phi = {phi / np.pi:.2f}\\pi$ ", loc="left", y=0.0)
        if i // nrowcol == nrowcol - 1:
            ax.set_xlabel("$R$")
        if i % nrowcol == 0:
            ax.set_ylabel("$Z$")
        ax.grid(True, linewidth=0.5)
        for line_index, hits in enumerate(fieldlines_phi_hits):
            hits = np.asarray(hits)
            if hits.ndim != 2 or hits.shape[0] == 0:
                continue
            data = hits[np.where(hits[:, 1] == i)[0], :]
            if data.size == 0:
                continue
            r = np.sqrt(data[:, 2] ** 2 + data[:, 3] ** 2)
            ax.scatter(r, data[:, 4], s=marker_size, linewidths=0, color=line_colors[line_index])
        if plot_surface:
            # `compute_fieldlines` uses physical toroidal angle in radians,
            # while `Surface.cross_section` expects the angle normalized by 2*pi.
            cross_section = surface.cross_section(phi=phi / (2 * np.pi))
            r_interp = np.sqrt(cross_section[:, 0] ** 2 + cross_section[:, 1] ** 2)
            z_interp = cross_section[:, 2]
            ax.plot(
                np.r_[r_interp, r_interp[0]],
                np.r_[z_interp, z_interp[0]],
                linewidth=1,
                c="k",
            )
    for i in range(len(phis), nrowcol * nrowcol):
        axs[i // nrowcol, i % nrowcol].axis("off")
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=int(dpi))
        print(f"Saved Poincare plot to {save_path}")
    plt.close(fig)
    return {
        "hit_counts": hit_counts,
        "seed_r_min": float(R0[0]),
        "seed_r_max": float(R0[-1]),
        "surface_r_outer_phi0": r_outer_phi0,
        "edge_fraction": float(edge_fraction),
        "tmax_fl": float(tmax_fl),
        "tol": float(tol),
        "psi_drift": psi_drift,
        "save_path": None if save_path is None else str(save_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Poincare validation from a saved Boozer surface and packed coil JSON.")
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument("--surface-npz", type=Path, required=True)
    parser.add_argument("--psi-model", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mpol", type=int, default=6)
    parser.add_argument("--ntor", type=int, default=6)
    parser.add_argument("--coil-order", type=int, default=16)
    parser.add_argument("--coil-quadpoints", type=int, default=300)
    parser.add_argument("--nfieldlines", type=int, default=20)
    parser.add_argument("--radial-width", type=float, default=None)
    parser.add_argument("--edge-fraction", type=float, default=0.95)
    parser.add_argument("--stop-fraction", type=float, default=1.5)
    parser.add_argument("--tmax-fl", type=float, default=2.0e8)
    parser.add_argument("--tol", type=float, default=1e-11)
    parser.add_argument("--marker-size", type=float, default=10.0)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    coil_dofs, currents, nfp, _ = _load_case(args.case_file)
    surf = np.load(args.surface_npz)
    psi_model = None
    if args.psi_model is not None:
        from stellarator_eval.volume_qs import load_psi_model

        psi_model = load_psi_model(args.psi_model)
    result = plot_poincare_from_dofs(
        surf["dofs"],
        coil_dofs,
        currents,
        nfp,
        mpol=args.mpol,
        ntor=args.ntor,
        coil_order=args.coil_order,
        coil_quadpoints=args.coil_quadpoints,
        nfieldlines=args.nfieldlines,
        radial_width=args.radial_width,
        edge_fraction=args.edge_fraction,
        stop_fraction=args.stop_fraction,
        tmax_fl=args.tmax_fl,
        save_path=args.output,
        tol=args.tol,
        marker_size=args.marker_size,
        dpi=args.dpi,
        psi_model=psi_model,
        psi_target=float(surf["s_level"]) if "s_level" in surf.files else None,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
