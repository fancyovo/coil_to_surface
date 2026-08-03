from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


TOKEN_DIM = 100
CURVE_ORDER = 16
COEFF_COUNT = 2 * CURVE_ORDER + 1


@dataclass(frozen=True)
class ParsedCoils:
    tokens: np.ndarray
    nfp: int
    curve_order: int


def stable_split(device_id: int) -> int:
    """Return 0/1/2 for a deterministic 90/5/5 train/validation/test split."""
    digest = hashlib.blake2b(
        str(int(device_id)).encode("ascii"), digest_size=8, person=b"qh-flow-v1"
    ).digest()
    bucket = int.from_bytes(digest, "little") % 100
    return 0 if bucket < 90 else 1 if bucket < 95 else 2


def _object_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    objects = payload.get("simsopt_objs")
    if not isinstance(objects, dict):
        raise ValueError("SIMSON payload has no simsopt_objs mapping")
    index: dict[str, dict[str, Any]] = {}
    for key, value in objects.items():
        if not isinstance(value, dict):
            continue
        index[str(key)] = value
        if value.get("@name") is not None:
            index[str(value["@name"])] = value
    return index


def _resolve_ref(value: Any, objects: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if isinstance(value, dict) and value.get("$type") == "ref":
        name = str(value.get("value"))
        if name not in objects:
            raise ValueError(f"unresolved SIMSON reference {name!r}")
        return objects[name]
    if isinstance(value, dict):
        return value
    raise ValueError(f"expected SIMSON object or reference, got {type(value).__name__}")


def _array_data(value: Any) -> np.ndarray:
    if isinstance(value, dict) and isinstance(value.get("data"), list):
        return np.asarray(value["data"], dtype=np.float64)
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float64)
    raise ValueError("SIMSON array has no data list")


def _resolve_current(value: Any, objects: dict[str, dict[str, Any]]) -> float:
    node = _resolve_ref(value, objects)
    class_name = node.get("@class")
    if class_name == "Current":
        current = node.get("current")
        if isinstance(current, (int, float)):
            return float(current)
        dofs = _resolve_ref(node.get("dofs"), objects)
        values = _array_data(dofs.get("x"))
        if values.size != 1:
            raise ValueError(f"Current DOFs has {values.size} entries")
        return float(values[0])
    if class_name == "ScaledCurrent":
        return float(node["scale"]) * _resolve_current(node["current_to_scale"], objects)
    raise ValueError(f"unsupported current class {class_name!r}")


def parse_simson_coils(payload: dict[str, Any]) -> ParsedCoils:
    if payload.get("@class") != "SIMSON":
        raise ValueError(f"expected SIMSON payload, got {payload.get('@class')!r}")
    graph = payload.get("graph")
    if not isinstance(graph, list) or len(graph) < 2 or not graph[0] or not graph[1]:
        raise ValueError("SIMSON graph must contain nonempty surface and coil lists")
    objects = _object_index(payload)
    surface = _resolve_ref(graph[0][0], objects)
    nfp = int(surface["nfp"])

    tokens = []
    curve_orders = set()
    for coil_ref in graph[1]:
        coil = _resolve_ref(coil_ref, objects)
        if coil.get("@class") != "Coil":
            continue
        curve = _resolve_ref(coil.get("curve"), objects)
        if curve.get("@class") != "CurveXYZFourier":
            continue
        order = int(curve.get("order", -1))
        if order < 0 or order > CURVE_ORDER:
            raise ValueError(f"CurveXYZFourier order {order} is outside [0, {CURVE_ORDER}]")
        curve_orders.add(order)
        dofs = _resolve_ref(curve.get("dofs"), objects)
        coefficients = _array_data(dofs.get("x"))
        source_coeff_count = 2 * order + 1
        if coefficients.size != 3 * source_coeff_count:
            raise ValueError(
                f"expected {3 * source_coeff_count} curve coefficients, got {coefficients.size}"
            )
        padded = np.zeros(3 * COEFF_COUNT, dtype=np.float64)
        for coordinate in range(3):
            source_start = coordinate * source_coeff_count
            target_start = coordinate * COEFF_COUNT
            padded[target_start : target_start + source_coeff_count] = coefficients[
                source_start : source_start + source_coeff_count
            ]
        current_a = _resolve_current(coil.get("current"), objects)
        token = np.concatenate([padded, np.asarray([current_a])])
        if token.size != TOKEN_DIM or not np.all(np.isfinite(token)):
            raise ValueError("coil token is nonfinite or has the wrong dimension")
        tokens.append(token)
    if not tokens:
        raise ValueError("SIMSON graph contains no direct CurveXYZFourier base coils")
    if len(curve_orders) != 1:
        raise ValueError(f"base coils use mixed Fourier orders: {sorted(curve_orders)}")
    return ParsedCoils(
        tokens=np.asarray(tokens, dtype=np.float32),
        nfp=nfp,
        curve_order=curve_orders.pop(),
    )


def load_simson_coils(path: str | Path) -> ParsedCoils:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return parse_simson_coils(payload)
