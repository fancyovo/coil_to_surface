from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np


class GpuError(RuntimeError):
    pass


class CoilFieldGpu:
    def __init__(
        self,
        lib_path: str | Path,
        coeffs_x,
        coeffs_y,
        coeffs_z,
        currents_a,
        nfp: int,
        segments_per_coil: int = 256,
        device_id: int = 0,
    ):
        self.lib = ctypes.CDLL(str(lib_path))
        self._bind()
        self.handle = ctypes.c_void_p()
        self.coeffs_x = np.ascontiguousarray(np.atleast_2d(coeffs_x), dtype=np.float64)
        self.coeffs_y = np.ascontiguousarray(np.atleast_2d(coeffs_y), dtype=np.float64)
        self.coeffs_z = np.ascontiguousarray(np.atleast_2d(coeffs_z), dtype=np.float64)
        self.currents_a = np.ascontiguousarray(currents_a, dtype=np.float64).ravel()
        if not (self.coeffs_x.shape == self.coeffs_y.shape == self.coeffs_z.shape):
            raise ValueError("coeffs_x/y/z must have the same shape")
        if self.currents_a.size != self.coeffs_x.shape[0]:
            raise ValueError("currents size must equal n_base_coils")
        code = self.lib.sgpu_create_field(
            self.coeffs_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self.coeffs_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self.coeffs_z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self.currents_a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(self.coeffs_x.shape[0]),
            ctypes.c_int(self.coeffs_x.shape[1]),
            ctypes.c_int(nfp),
            ctypes.c_int(segments_per_coil),
            ctypes.c_int(device_id),
            ctypes.byref(self.handle),
        )
        self._check(code)
        self.nfp = int(nfp)
        self.segments_per_coil = int(segments_per_coil)
        self.device_id = int(device_id)

    def _bind(self):
        self.lib.sgpu_create_field.restype = ctypes.c_int
        self.lib.sgpu_create_field.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.sgpu_destroy_field.restype = None
        self.lib.sgpu_destroy_field.argtypes = [ctypes.c_void_p]
        self.lib.sgpu_segment_count.restype = ctypes.c_int
        self.lib.sgpu_segment_count.argtypes = [ctypes.c_void_p]
        self.lib.sgpu_eval_B.restype = ctypes.c_int
        self.lib.sgpu_eval_B.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
        ]
        self.lib.sgpu_trace_period.restype = ctypes.c_int
        self.lib.sgpu_trace_period.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.sgpu_last_error.restype = ctypes.c_char_p

    def _check(self, code: int):
        if code:
            msg = self.lib.sgpu_last_error()
            raise GpuError(msg.decode("utf-8", "replace") if msg else "unknown GPU backend error")

    @property
    def segment_count(self) -> int:
        return int(self.lib.sgpu_segment_count(self.handle))

    def close(self):
        if getattr(self, "handle", None):
            self.lib.sgpu_destroy_field(self.handle)
            self.handle = ctypes.c_void_p()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def eval_B(self, xyz):
        pts = np.ascontiguousarray(xyz, dtype=np.float64).reshape(-1, 3)
        out = np.empty_like(pts)
        code = self.lib.sgpu_eval_B(
            self.handle,
            pts.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(len(pts)),
        )
        self._check(code)
        return out

    def trace_period(self, R0, Z0, steps: int, nfp: int | None = None):
        R0 = np.ascontiguousarray(R0, dtype=np.float64).ravel()
        Z0 = np.ascontiguousarray(Z0, dtype=np.float64).ravel()
        if R0.shape != Z0.shape:
            raise ValueError("R0/Z0 shape mismatch")
        R1 = np.empty_like(R0)
        Z1 = np.empty_like(Z0)
        code = self.lib.sgpu_trace_period(
            self.handle,
            R0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            Z0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            R1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            Z1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(R0.size),
            ctypes.c_int(self.nfp if nfp is None else nfp),
            ctypes.c_int(steps),
        )
        self._check(code)
        return R1, Z1


def load_case(path: str | Path, key: str = "raw"):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    d = data[key]
    return (
        np.asarray(d["x"], dtype=np.float64),
        np.asarray(d["y"], dtype=np.float64),
        np.asarray(d["z"], dtype=np.float64),
        np.asarray(d["current"], dtype=np.float64) * 1e6,
        int(data["nfp"]),
    )


def eval_fourier_block(c, t):
    c = np.asarray(c, dtype=np.float64)
    order = (c.size - 1) // 2
    t = np.asarray(t, dtype=np.float64)
    x = np.full_like(t, c[0], dtype=np.float64)
    dxdt = np.zeros_like(t, dtype=np.float64)
    for m in range(1, order + 1):
        s = np.sin(2.0 * np.pi * m * t)
        co = np.cos(2.0 * np.pi * m * t)
        sin_c = c[2 * m - 1]
        cos_c = c[2 * m]
        x += sin_c * s + cos_c * co
        dxdt += 2.0 * np.pi * m * (sin_c * co - cos_c * s)
    return x, dxdt


def make_segments_cpu(coeffs_x, coeffs_y, coeffs_z, currents_a, nfp: int, segments_per_coil: int):
    xs = []
    wdl = []
    t = (np.arange(segments_per_coil, dtype=np.float64) + 0.5) / segments_per_coil
    for bx, by, bz, cur in zip(coeffs_x, coeffs_y, coeffs_z, currents_a):
        px, vx = eval_fourier_block(bx, t)
        py, vy = eval_fourier_block(by, t)
        pz, vz = eval_fourier_block(bz, t)
        vx = vx / segments_per_coil
        vy = vy / segments_per_coil
        vz = vz / segments_per_coil
        for k in range(nfp):
            ang = 2.0 * np.pi * k / nfp
            ca, sa = np.cos(ang), np.sin(ang)
            rx = ca * px - sa * py
            ry = sa * px + ca * py
            rvx = ca * vx - sa * vy
            rvy = sa * vx + ca * vy
            xs.append(np.column_stack([rx, ry, pz]))
            wdl.append(cur * np.column_stack([rvx, rvy, vz]))
            mx, my, mz = px, -py, -pz
            mvx, mvy, mvz = vx, -vy, -vz
            rx = ca * mx - sa * my
            ry = sa * mx + ca * my
            rvx = ca * mvx - sa * mvy
            rvy = sa * mvx + ca * mvy
            xs.append(np.column_stack([rx, ry, mz]))
            wdl.append((-cur) * np.column_stack([rvx, rvy, mvz]))
    return np.vstack(xs), np.vstack(wdl)


def eval_B_segments_cpu(points, seg_pos, seg_wdl):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    out = np.zeros_like(points)
    for i, p in enumerate(points):
        r = p[None, :] - seg_pos
        r2 = np.sum(r * r, axis=1)
        invr3 = 1.0 / np.maximum(r2, 1e-300) ** 1.5
        out[i] = 1e-7 * np.sum(np.cross(seg_wdl, r) * invr3[:, None], axis=0)
    return out
