from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_cem_candidate_full import (
    install_headless_tkinter_stub,
    save_desc_plot,
    save_periodic_colored_contours,
    write_image_html,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replot saved direct and DESC Boozer |B| data as colored contour lines."
    )
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument("--surface-npz", type=Path, required=True)
    parser.add_argument("--equilibrium", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--current-unit", default="A")
    parser.add_argument("--surface-order", type=int, default=None)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg", force=True)
    install_headless_tkinter_stub()
    from desc.io import load
    from desc.plotting import plot_boozer_surface

    from scripts.desc_psi_volume_initial_guess_experiment import make_xyz_surface
    from stellarator_eval.field import build_field, load_case_file

    args.output_dir.mkdir(parents=True, exist_ok=True)
    direct_dir = args.output_dir / "assets"
    desc_dir = args.output_dir / "desc"
    direct_dir.mkdir(parents=True, exist_ok=True)
    desc_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.surface_npz) as saved:
        if "order" in saved:
            surface_order = int(saved["order"])
        elif args.surface_order is not None:
            surface_order = args.surface_order
        else:
            raise ValueError("surface NPZ has no order metadata; pass --surface-order")

    field_input = load_case_file(args.case_file, "raw")
    built = build_field(field_input, current_unit=args.current_unit)
    surface, _ = make_xyz_surface(
        args.surface_npz,
        nfp=field_input.nfp,
        order=surface_order,
        stellsym=True,
        nphi=96,
        ntheta=192,
    )
    xyz = np.asarray(surface.gamma(), dtype=float)
    built.field.set_points(xyz.reshape(-1, 3))
    b_abs = np.linalg.norm(np.asarray(built.field.B(), dtype=float), axis=1).reshape(
        xyz.shape[:2]
    )
    direct_png = direct_dir / "boozer_b.png"
    save_periodic_colored_contours(
        values=b_abs,
        output_path=direct_png,
        xlabel=r"field-period angle $N_{\rm FP}\phi$",
        ylabel=r"Boozer $\theta$",
        title=r"Colored $|B|$ contours on largest Boozer-solvable surface",
        colorbar_label=r"$|B|$ [T]",
    )
    write_image_html(direct_png, direct_dir / "boozer_b.html", "Boozer |B| contours")

    equilibrium = load(str(args.equilibrium))
    result = save_desc_plot(
        "boozer_B",
        desc_dir,
        plot_boozer_surface,
        equilibrium,
        rho=1.0,
        fill=False,
        ncontours=32,
    )
    if not result["success"]:
        raise RuntimeError(result)

    print(
        f"direct={direct_png} min={np.min(b_abs):.9g} max={np.max(b_abs):.9g}\n"
        f"desc={result['path']}"
    )


if __name__ == "__main__":
    main()
