from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import CurveXYZFourier


@dataclass
class FieldInput:
    coeffs_x: np.ndarray
    coeffs_y: np.ndarray
    coeffs_z: np.ndarray
    currents: np.ndarray
    nfp: int
    name: str = ""

    @property
    def n_base_coils(self) -> int:
        return int(self.coeffs_x.shape[0])

    @property
    def order(self) -> int:
        return int((self.coeffs_x.shape[1] - 1) // 2)


@dataclass
class BuiltField:
    field: BiotSavart
    base_curves: list[CurveXYZFourier]
    nfp: int
    n_base_coils: int
    n_total_coils: int
    coil_r0: float


def load_case_file(path: str | Path, key: str = "raw") -> FieldInput:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if key not in data:
        raise KeyError(f"{path} does not contain key {key!r}")
    d = data[key]
    return FieldInput(
        coeffs_x=np.asarray(d["x"], dtype=float),
        coeffs_y=np.asarray(d["y"], dtype=float),
        coeffs_z=np.asarray(d["z"], dtype=float),
        currents=np.asarray(d["current"], dtype=float),
        nfp=int(data.get("nfp", d.get("nfp"))),
        name=str(d.get("name", key)),
    )


def input_from_flat_vector(values: Any, nfp: int, coeff_count: int = 33, current_unit: str = "MA") -> FieldInput:
    arr = np.asarray(values, dtype=float).ravel()
    block = 3 * coeff_count + 1
    if arr.size % block != 0:
        raise ValueError(f"flat vector length {arr.size} is not a multiple of {block}")
    n_base = arr.size // block
    x = np.empty((n_base, coeff_count))
    y = np.empty((n_base, coeff_count))
    z = np.empty((n_base, coeff_count))
    cur = np.empty(n_base)
    for i in range(n_base):
        b = arr[i * block : (i + 1) * block]
        x[i] = b[:coeff_count]
        y[i] = b[coeff_count : 2 * coeff_count]
        z[i] = b[2 * coeff_count : 3 * coeff_count]
        cur[i] = b[-1]
    return FieldInput(x, y, z, cur, int(nfp), name=f"flat_{n_base}_coils")


def input_from_packed_vector(values: Any, coeff_count: int = 33) -> FieldInput:
    arr = np.asarray(values, dtype=float).ravel()
    block = 3 * coeff_count + 1
    if arr.size < block + 1 or (arr.size - 1) % block != 0:
        raise ValueError(
            f"packed vector length must be n_base_coils * {block} + 1, "
            f"got {arr.size}"
        )
    nfp_float = float(arr[-1])
    nfp = int(round(nfp_float))
    if not np.isclose(nfp_float, nfp) or nfp <= 0:
        raise ValueError(f"last packed vector entry must be a positive integer nfp, got {nfp_float}")
    return input_from_flat_vector(arr[:-1], nfp=nfp, coeff_count=coeff_count)


def normalize_currents(currents: np.ndarray, current_unit: str) -> np.ndarray:
    unit = current_unit.lower()
    if unit in {"ma", "megaamp", "megaamps"}:
        return np.asarray(currents, dtype=float) * 1e6
    if unit in {"a", "amp", "amps"}:
        return np.asarray(currents, dtype=float)
    raise ValueError(f"unknown current_unit={current_unit!r}; use 'MA' or 'A'")


def build_field(field_input: FieldInput, current_unit: str = "MA") -> BuiltField:
    x = np.atleast_2d(np.asarray(field_input.coeffs_x, dtype=float))
    y = np.atleast_2d(np.asarray(field_input.coeffs_y, dtype=float))
    z = np.atleast_2d(np.asarray(field_input.coeffs_z, dtype=float))
    if not (x.shape == y.shape == z.shape):
        raise ValueError(f"x/y/z coefficient arrays must have the same shape, got {x.shape}, {y.shape}, {z.shape}")
    currents = normalize_currents(np.asarray(field_input.currents, dtype=float), current_unit)
    if currents.size != x.shape[0]:
        raise ValueError(f"got {x.shape[0]} base coils but {currents.size} currents")
    order = (x.shape[1] - 1) // 2
    if 2 * order + 1 != x.shape[1]:
        raise ValueError("coefficient count must be odd: 2*order+1")

    curves: list[CurveXYZFourier] = []
    r0_values = []
    for i in range(x.shape[0]):
        curve = CurveXYZFourier(quadpoints=max(50, 15 * order), order=order)
        curve.set_dofs(np.concatenate([x[i], y[i], z[i]]))
        curves.append(curve)
        gamma = curve.gamma()
        r0_values.append(float(np.mean(np.sqrt(gamma[:, 0] ** 2 + gamma[:, 1] ** 2))))
    coils = coils_via_symmetries(curves, [Current(c) for c in currents], nfp=field_input.nfp, stellsym=True)
    return BuiltField(
        field=BiotSavart(coils),
        base_curves=curves,
        nfp=int(field_input.nfp),
        n_base_coils=len(curves),
        n_total_coils=len(coils),
        coil_r0=float(np.mean(r0_values)),
    )
