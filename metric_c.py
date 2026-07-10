import numpy as np

if __name__ == "__main__" and __package__ is None:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "metrics"

from simsopt.geo import (
    CurveXYZFourier,
    CurveLength,
    CurveCurveDistance,
    CurveSurfaceDistance,
    LpCurveCurvature,
    LpCurveTorsion,
    LinkingNumber,
)
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.objectives import SquaredFlux

from .metric_s import dofs2surface

COIL_ORDER = 16
COIL_QUADPOINTS = 300


# =========================
# 工具函数
# =========================


def dofs2coil(dofs, order=None, nfp=None, currents=None):
    """dofs (n_coils, dof_dim) → list of CurveXYZFourier。

    order 必须与 dofs 的 Fourier 阶数一致 (dof_dim = 3*(2*order+1)); None → COIL_ORDER(16)。

    nfp/currents 参数保留为兼容旧调用；本函数始终返回半周期曲线。
    需要完整线圈时使用 dofs2coils。
    """
    if order is None:
        order = COIL_ORDER
    dofs = np.atleast_2d(dofs)
    curves_half = []
    for row in dofs:
        c = CurveXYZFourier(COIL_QUADPOINTS, order)
        c.x = row
        curves_half.append(c)
    return curves_half


def dofs2coils(dofs, nfp, currents, order=None):
    """半周期 coil dofs + currents → 完整 simsopt Coil 列表。"""
    curves_half = dofs2coil(dofs, order=order)
    cur_objs = [Current(float(cv)) for cv in np.atleast_1d(np.asarray(currents, dtype=float)).ravel()]
    for co in cur_objs:
        co.fix_all()
    return coils_via_symmetries(curves_half, cur_objs, int(nfp), True)


# =====================================================
# metric
# =====================================================


def J_coil_length(dofs_coil, order=None):
    """sum of coil lengths."""
    coils = dofs2coil(dofs_coil, order=order)
    return float(sum(CurveLength(c).J() for c in coils))


def J_coil_curvature(dofs_coil, p=2, order=None):
    """max Lp curvature across all coils."""
    coils = dofs2coil(dofs_coil, order=order)
    vals = [float(LpCurveCurvature(c, p=p).J()) for c in coils]
    return max(vals)


def J_coil_torsion(dofs_coil, p=2, order=None):
    """max Lp torsion across all coils."""
    coils = dofs2coil(dofs_coil, order=order)
    vals = [float(LpCurveTorsion(c, p=p).J()) for c in coils]
    return max(vals)


def J_coil_linking(dofs_coil, order=None, downsample=1, nfp=None, currents=None):
    """Gauss linking number — detects coil interlocking.

    若传 nfp+currents: 在 dofs2coils 给出的完整线圈上计算。
    否则在半周期曲线上计算。
    """
    if nfp is not None and currents is not None:
        coils = dofs2coils(dofs_coil, nfp, currents, order=order)
        curves = [c.curve for c in coils]
    else:
        curves = dofs2coil(dofs_coil, order=order)
    ln = LinkingNumber(curves, downsample=downsample)
    return float(ln.J())


def J_coil_distance(dofs_coil, threshold=0.1, order=None):
    """coil-coil distance penalty."""
    coils = dofs2coil(dofs_coil, order=order)
    ccd = CurveCurveDistance(coils, minimum_distance=float(threshold))
    return float(ccd.J())


def J_surface_coil_distance(dofs_surface, dofs_coil, nfp, threshold=0.1, order=None,
                            mpol=None, ntor=None):
    """surface-coil distance penalty."""
    surface = dofs2surface(dofs_surface, nfp, mpol=mpol, ntor=ntor)
    coils = dofs2coil(dofs_coil, order=order)
    csd = CurveSurfaceDistance(coils, surface, minimum_distance=float(threshold))
    return float(csd.J())


def J_bdotn(dofs_surface, dofs_coil, nfp, currents, definition="local",
            order=None):
    """squared normal flux (B·n)² / |B|² over surface, locally normalized.

    Args:
        currents: 每根线圈的电流值 (与 dofs_coil 行数一致)。**必传**。
        definition: ``"local"`` (默认) / ``"quadratic flux"`` / ``"normalized"``。
        order: 线圈 Fourier 阶数 (QUASR=16); None 时用 dofs2coil 默认。
    """
    surface = dofs2surface(dofs_surface, nfp)
    coils = dofs2coils(dofs_coil, nfp, currents, order=order)
    bs = BiotSavart(coils)
    sf = SquaredFlux(surface=surface, field=bs, definition=definition)
    return float(sf.J())


def J_total_c(dofs_surface, dofs_coil, nfp, currents, weights=None, order=None):
    """combined coil-side metric returning a dict.

    Args:
        currents: 每根线圈电流值 (与 dofs_coil 行数一致)。**必传**。
        order: 线圈 Fourier 阶数 (QUASR=16); None 时用 dofs2coil 默认。

    Returns dict with keys: length, curvature, torsion, linking,
    coil_distance, surface_distance, bdotn, total.
    """
    if weights is None:
        weights = {}

    w_len = weights.get("length", 1.0)
    w_curv = weights.get("curvature", 1.0)
    w_tors = weights.get("torsion", 1.0)
    w_link = weights.get("linking", 1.0)
    w_cdist = weights.get("coil_distance", 1.0)
    w_sdist = weights.get("surface_distance", 1.0)
    w_bdotn = weights.get("bdotn", 1.0)

    v_len = J_coil_length(dofs_coil, order=order)
    v_curv = J_coil_curvature(dofs_coil, order=order)
    v_tors = J_coil_torsion(dofs_coil, order=order)
    v_link = J_coil_linking(dofs_coil, order=order, nfp=nfp, currents=currents)
    v_cdist = J_coil_distance(dofs_coil, order=order)
    v_sdist = J_surface_coil_distance(dofs_surface, dofs_coil, nfp, order=order)
    v_bdotn = J_bdotn(dofs_surface, dofs_coil, nfp, currents=currents, order=order)

    total = (
        w_len * v_len
        + w_curv * v_curv
        + w_tors * v_tors
        + w_link * v_link
        + w_cdist * v_cdist
        + w_sdist * v_sdist
        + w_bdotn * v_bdotn
    )

    return {
        "length": v_len,
        "curvature": v_curv,
        "torsion": v_tors,
        "linking": v_link,
        "coil_distance": v_cdist,
        "surface_distance": v_sdist,
        "bdotn": v_bdotn,
        "total": total,
    }


if __name__ == "__main__":
    from .load_data import load_case, change_xyzcurve_modes, df

    ID = 625292
    surfaces, coils, volume, mean_iota, nc_per_hp, helicity, nfp = load_case(ID)
    print('order:',coils[0].curve.order)
    dofs_surface = surfaces[-1].x
    dofs_coil = np.array([change_xyzcurve_modes(c.curve, target_order=COIL_ORDER)
                          for c in coils[:int(nc_per_hp)]])
    currents = np.array([c.current.full_x[0] for c in coils[:int(nc_per_hp)]], dtype=float)

    print(f"nfp: {nfp}, n_coils: {len(coils)}")
    print(f"surface dof dim: {len(dofs_surface)}, coil dof dim: {dofs_coil.shape[1]}")
    print(f"half-period coils: {len(dofs_coil)}, expanded coils: {len(dofs_coil) * int(nfp) * 2}")
    print()

    print("coil length:          ", J_coil_length(dofs_coil))
    print("coil curvature (L2):  ", J_coil_curvature(dofs_coil, p=2))
    print("coil torsion (L2):    ", J_coil_torsion(dofs_coil, p=2))
    print("coil-coil distance:   ", J_coil_distance(dofs_coil))
    print("surface-coil distance:", J_surface_coil_distance(dofs_surface, dofs_coil, nfp))
    print("B·n squared flux:     ", J_bdotn(dofs_surface, dofs_coil, nfp, currents=currents))
    print()

    result = J_total_c(dofs_surface, dofs_coil, nfp, currents=currents)
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\n--- weighted example (coil distance × 0.1, bdotn × 10) ---")
    result_w = J_total_c(dofs_surface, dofs_coil, nfp, currents=currents,
                         weights={"coil_distance": 0.1, "bdotn": 10.0})
    for k, v in result_w.items():
        print(f"  {k}: {v}")
