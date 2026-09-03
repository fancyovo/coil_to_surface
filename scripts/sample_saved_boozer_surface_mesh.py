from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample one saved Boozer surface over one field period.")
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nphi", type=int, default=192)
    parser.add_argument("--ntheta", type=int, default=256)
    args = parser.parse_args()
    if args.nphi < 8 or args.ntheta < 16:
        raise ValueError("surface sampling grid is too small")

    from simsopt.geo import SurfaceXYZTensorFourier

    with np.load(args.surface) as saved:
        nfp = int(saved["nfp"])
        order = int(saved["order"])
        stellsym = bool(saved["stellsym"])
        dofs = np.asarray(saved["dofs"], dtype=np.float64)
    surface = SurfaceXYZTensorFourier(
        mpol=order,
        ntor=order,
        nfp=nfp,
        stellsym=stellsym,
        quadpoints_phi=np.linspace(0.0, 1.0 / nfp, args.nphi, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, args.ntheta, endpoint=False),
    )
    surface.set_dofs(dofs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        xyz_one_period=np.asarray(surface.gamma(), dtype=np.float64),
        nfp=np.asarray(nfp, dtype=np.int64),
        order=np.asarray(order, dtype=np.int64),
        stellsym=np.asarray(stellsym, dtype=np.bool_),
        source=np.asarray(str(args.surface.resolve())),
    )
    print(f"sampled {args.output} with shape {surface.gamma().shape}")


if __name__ == "__main__":
    main()

