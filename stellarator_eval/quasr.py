from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .field import FieldInput


def _py_scalar(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def _normalize_meta_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        value = _py_scalar(value)
        if isinstance(value, float):
            out[key] = float(value)
            continue
        if isinstance(value, int):
            out[key] = int(value)
            continue
        out[key] = value
    if "ID" in out:
        out["ID"] = int(float(out["ID"]))
    if "nfp" in out:
        out["nfp"] = int(float(out["nfp"]))
    if "nc_per_hp" in out:
        out["nc_per_hp"] = int(float(out["nc_per_hp"]))
    if "Nsurfaces" in out:
        out["Nsurfaces"] = int(float(out["Nsurfaces"]))
    if "helicity" in out:
        out["helicity"] = int(float(out["helicity"]))
    return out


def load_quasr_metadata(metadata_path: str | Path) -> list[dict[str, Any]]:
    path = Path(metadata_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8") as f:
            return [_normalize_meta_row(row) for row in csv.DictReader(f)]
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [_normalize_meta_row(dict(row)) for row in data]
        raise ValueError(f"{path} must contain a JSON list")
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(_normalize_meta_row(json.loads(line)))
        return rows
    if suffix == ".pkl":
        try:
            with path.open("rb") as f:
                obj = pickle.load(f)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"loading {path} requires optional dependencies such as pandas; "
                "please convert the metadata to CSV/JSON first"
            ) from exc
        if hasattr(obj, "to_dict"):
            rows = obj.to_dict(orient="records")
            return [_normalize_meta_row(dict(row)) for row in rows]
        raise ValueError(f"{path} pickle payload does not look like a pandas DataFrame")
    raise ValueError(f"unsupported metadata format: {path.suffix}")


def build_quasr_metadata_index(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["ID"]): row for row in rows}


def quasr_serial_path(quasr_root: str | Path, device_id: int) -> Path:
    root = Path(quasr_root)
    f_id = int(device_id) // 1000
    return root / "simsopt_serials" / f"{f_id:04d}" / f"serial{int(device_id):07d}.json"


def load_quasr_field_input(quasr_root: str | Path, device_id: int) -> tuple[FieldInput, dict[str, Any]]:
    from simsopt._core import load

    serial_path = quasr_serial_path(quasr_root, device_id)
    surfaces, coils = load(str(serial_path))
    if not isinstance(surfaces, list) or not surfaces:
        raise ValueError(f"{serial_path} did not return a nonempty surface list")
    base_coils = [coil for coil in coils if type(coil.curve).__name__ == "CurveXYZFourier"]
    if not base_coils:
        raise ValueError(f"{serial_path} did not contain any direct CurveXYZFourier base coils")

    order = int(base_coils[0].curve.order)
    n_coeff = 2 * order + 1
    coeff_blocks = []
    currents = []
    for coil in base_coils:
        x = np.asarray(coil.curve.x, dtype=float)
        if x.size != 3 * n_coeff:
            raise ValueError(f"{serial_path}: expected {3 * n_coeff} curve dofs, got {x.size}")
        coeff_blocks.append(x)
        currents.append(float(coil.current.get_value()))
    arr = np.asarray(coeff_blocks, dtype=float)
    field_input = FieldInput(
        coeffs_x=arr[:, :n_coeff],
        coeffs_y=arr[:, n_coeff : 2 * n_coeff],
        coeffs_z=arr[:, 2 * n_coeff : 3 * n_coeff],
        currents=np.asarray(currents, dtype=float),
        nfp=int(surfaces[0].nfp),
        name=f"quasr_{int(device_id):07d}",
    )
    info = {
        "device_id": int(device_id),
        "serial_path": str(serial_path),
        "surface_count": len(surfaces),
        "surface_type": type(surfaces[0]).__name__,
        "nfp": int(surfaces[0].nfp),
        "stellsym": bool(surfaces[0].stellsym),
        "nc_per_hp": len(base_coils),
        "n_total_coils": len(coils),
        "curve_order": order,
        "n_coeff": n_coeff,
        "current_unit": "A",
    }
    return field_input, info


def choose_quasr_eval_params(
    metadata_row: dict[str, Any] | None,
    *,
    default_a: float = 0.05,
    a_minor_fraction: float = 0.9,
    explicit_a: float | None = None,
    explicit_initial_iota: float | None = None,
) -> dict[str, Any]:
    row = metadata_row or {}
    minor_radius = row.get("minor_radius")
    if explicit_a is not None:
        a = float(explicit_a)
    elif minor_radius is not None:
        a = min(float(default_a), float(a_minor_fraction) * float(minor_radius))
    else:
        a = float(default_a)
    if a <= 0.0:
        raise ValueError(f"nonpositive evaluation radius a={a}")

    if explicit_initial_iota is not None:
        initial_iota = float(explicit_initial_iota)
        iota_source = "explicit"
    elif row.get("mean_iota") is not None:
        initial_iota = float(row["mean_iota"])
        iota_source = "metadata_mean_iota"
    else:
        initial_iota = -2.0
        iota_source = "fallback_default"

    return {
        "a": a,
        "minor_radius": None if minor_radius is None else float(minor_radius),
        "minor_radius_used_fraction": None if minor_radius is None else a / float(minor_radius),
        "initial_iota": initial_iota,
        "initial_iota_source": iota_source,
    }


def quasr_failure_case_payload(
    field_input: FieldInput,
    *,
    device_id: int,
    metadata_row: dict[str, Any] | None = None,
    source_root: str | Path | None = None,
) -> dict[str, Any]:
    raw = {
        "name": f"quasr_{int(device_id):07d}_raw",
        "x": np.asarray(field_input.coeffs_x, dtype=float).tolist(),
        "y": np.asarray(field_input.coeffs_y, dtype=float).tolist(),
        "z": np.asarray(field_input.coeffs_z, dtype=float).tolist(),
        "current": np.asarray(field_input.currents, dtype=float).tolist(),
        "device_id": int(device_id),
        "current_unit": "A",
    }
    if source_root is not None:
        raw["quasr_root"] = str(source_root)
    if metadata_row is not None:
        raw["metadata"] = metadata_row
    return {
        "raw": raw,
        "nfp": int(field_input.nfp),
    }
