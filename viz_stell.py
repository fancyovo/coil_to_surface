"""Visualisation utilities for stellarator DOF diagnostics.

Provides `plot_dof_overview` — a single figure with surface (3D mode heatmaps),
coil (coil × DOF heatmaps), and current values.

Power scaling uses the ACTUAL Fourier mode numbers derived from simsopt source,
NOT the raw grid indices.  See SurfaceXYZTensorFourier docstring for the
basis-function convention.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib import cm
import os
from tqdm import tqdm

if __name__ == "__main__" and __package__ is None:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "metrics"

from .metric_s import dofs2surface, surf2desc
import desc.plotting as desc_plot
from simsopt.geo import plotting, SurfaceRZFourier, SurfaceXYZTensorFourier, CurveXYZFourier
from simsopt.field import (
    BiotSavart,
    Coil,
    Current,
    MaxRStoppingCriterion,
    MaxZStoppingCriterion,
    MinRStoppingCriterion,
    MinZStoppingCriterion,
    coils_via_symmetries,
    compute_fieldlines,
)

def plot_dofs(surf_dofs, mpol, ntor, nfp, coil_dofs=None, coil_order=16,
              save_path=None, title=None, show=True, expand_symmetries=True):
    """Plot surface + coils from flat DOF arrays using simsopt.geo.plotting.

    Args:
        surf_dofs: flat 1D numpy array of surface DOFs (SurfaceXYZTensorFourier format)
        mpol, ntor: poloidal/toroidal mode numbers
        nfp: number of field periods
        coil_dofs: (n_half_coils, dof_dim) numpy array of HALF-PERIOD coil DOFs, or None
        coil_order: Fourier order for coil curves
        expand_symmetries: if True, expand half-period coils to full set via coils_via_symmetries
        save_path: if given, save figure to this path
        title: optional plot title
        show: whether to call plt.show()
    """
    from simsopt.field import coils_via_symmetries

    surface = SurfaceXYZTensorFourier(nfp=nfp, stellsym=True, mpol=mpol, ntor=ntor)
    surface.set_dofs(np.asarray(surf_dofs, dtype=float).ravel())

    items = [surface]

    if coil_dofs is not None:
        coil_dofs = np.atleast_2d(coil_dofs)
        curves = []
        cur = Current(1.0)
        cur.fix_all()
        for row in coil_dofs:
            curve = CurveXYZFourier(300, coil_order)
            curve.set_dofs(np.asarray(row, dtype=float).ravel())
            curves.append(curve)

        if expand_symmetries:
            coils = coils_via_symmetries(curves, [cur]*len(curves), nfp, True)
        else:
            coils = [Coil(c, cur) for c in curves]

        items.extend(coils)

    if title:
        plt.figure()

    plotting.plot(items, show=False)

    if title:
        plt.title(title)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    if show:
        plt.show()
    return items



def plot_dof_overview(surf_dofs, mpol, ntor, coil_dofs, coil_order,
                      cur_vals, nfp=3, title=None, save_path=None):
    """Single-figure overview of all DOF components with actual mode-number scaling.

    Surface scaling: 10^(m_act + nfp·n_act) where m_act, n_act are the
    real Fourier mode numbers (cos/sin terms mapped to their true mode index).

    Subplots:  Surface X/Y/Z | Coil DOFs | Current values
    """



    def _surface_dof_to_grid(dofs, mpol, ntor):
        """Unpack flat surface DOFs into (3, 2*mpol+1, 2*ntor+1) arrays.

        Follows normalize_dofs() in load_data.py — same iteration order as the
        simsopt flat DOF vector.  Grid position (i,j) maps to:
        i=0..mpol     → cos(iθ),        actual poloidal mode = i
        i=mpol+1..2m  → sin((i-m)θ),     actual poloidal mode = i-mpol
        j=0..ntor     → cos(j·nfp·φ),    actual toroidal mode = j
        j=ntor+1..2n  → sin((j-n)·nfp·φ), actual toroidal mode = j-ntor
        """
        grid_m = 2 * mpol + 1
        grid_n = 2 * ntor + 1
        A = np.full((3, grid_m, grid_n), np.nan)
        p = 0
        for d in range(3):
            for m in range(grid_m):
                for n in range(grid_n):
                    skip = ((n <= ntor and m > mpol) or (n > ntor and m <= mpol)) if d == 0 else \
                        ((n <= ntor and m <= mpol) or (n > ntor and m > mpol))
                    if not skip and p < len(dofs):
                        A[d, m, n] = dofs[p]
                        p += 1
        return A


    def _actual_mode(i, j, mpol, ntor, nfp):
        """Return the actual Fourier mode numbers for grid position (i, j).

        Based on SurfaceXYZTensorFourier basis functions:
        w_i(θ):  i<=mpol → cos(iθ),   i>mpol → sin((i-mpol)θ)
        v_j(φ):  j<=ntor → cos(j·nfp·φ), j>ntor → sin((j-ntor)·nfp·φ)

        Returns (m, n) — the actual poloidal and toroidal mode indices.
        """
        m = i if i <= mpol else i - mpol
        n = j if j <= ntor else j - ntor
        return m, n


    def _scale_surface(A, mpol, ntor, nfp):
        """Apply 10^(m + n) power scaling using actual mode numbers."""
        grid_m = 2 * mpol + 1
        grid_n = 2 * ntor + 1
        scaled = np.full_like(A, np.nan)
        for i in range(grid_m):
            for j in range(grid_n):
                m_act, n_act = _actual_mode(i, j, mpol, ntor, nfp)
                factor = np.e ** (m_act + n_act)
                for d in range(3):
                    if not np.isnan(A[d, i, j]):
                        scaled[d, i, j] = A[d, i, j] * factor
        return scaled


    def _scale_coil(coil_dofs, n_half, order):
        """Apply 10^m power scaling, where m is the actual Fourier mode number.

        CurveXYZFourier DOF ordering (per component block of 2*order+1):
        index 0          → mode 0 (constant)
        index 1,2        → mode 1  (sin, cos)
        index 3,4        → mode 2  (sin, cos)
        ...
        index 2k-1, 2k   → mode k  (sin, cos)
        Three blocks: X, then Y, then Z.
        """
        scaled = coil_dofs.copy().astype(float)
        block_size = 2 * order + 1
        for k in range(n_half):
            for i in range(scaled.shape[1]):
                local_i = i % block_size
                mode = 0 if local_i == 0 else (local_i + 1) // 2
                scaled[k, i] *= np.e ** mode
        return scaled


    surf_dofs = np.atleast_1d(np.asarray(surf_dofs, dtype=float)).ravel()
    coil_dofs = np.atleast_2d(np.asarray(coil_dofs, dtype=float))
    cur_vals = np.atleast_1d(np.asarray(cur_vals, dtype=float)).ravel()
    n_half = coil_dofs.shape[0]

    # --- Surface ---
    A = _surface_dof_to_grid(surf_dofs, mpol, ntor)
    A_scaled = _scale_surface(A, mpol, ntor, nfp)
    vmax = np.nanmax(np.abs(A_scaled))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax) if vmax > 0 else None

    # --- Coil ---
    coil_scaled = _scale_coil(coil_dofs, n_half, coil_order)
    cvmax = np.max(np.abs(coil_scaled))
    cnorm = TwoSlopeNorm(vmin=-cvmax, vcenter=0, vmax=cvmax) if cvmax > 0 else None

    # --- Layout ---
    n_cols = max(3, n_half)
    fig = plt.figure(figsize=(max(16, 5 * n_cols), 9))

    for d, lab in enumerate(['X', 'Y', 'Z']):
        ax = fig.add_subplot(3, n_cols, d + 1)
        im = ax.imshow(A_scaled[d], aspect='auto', origin='lower',
                       cmap='RdBu_r', norm=norm)
        ax.set_title(f'{lab}  ×10^(m+n)')
        ax.set_xlabel('grid col j'); ax.set_ylabel('grid row i')
        plt.colorbar(im, ax=ax, fraction=0.046)
    for d in range(3, n_cols):
        fig.add_subplot(3, n_cols, d + 1).axis('off')

    for k in range(n_half):
        ax = fig.add_subplot(3, n_cols, n_cols + k + 1)
        im = ax.imshow(coil_scaled[k].reshape(1, -1), aspect='auto', origin='lower',
                       cmap='RdBu_r', norm=cnorm)
        ax.set_title(f'Coil {k+1}  ×10^m')
        ax.set_xlabel('DOF index i'); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    for k in range(n_half, n_cols):
        fig.add_subplot(3, n_cols, n_cols + k + 1).axis('off')

    cmap_cur = plt.cm.RdBu_r
    c_abs_max = max(abs(cur_vals).max(), 1e-10)
    for k in range(n_half):
        ax = fig.add_subplot(3, n_cols, 2 * n_cols + k + 1)
        val = cur_vals[k]
        color = cmap_cur(0.5 + 0.5 * val / c_abs_max)
        ax.bar(0, val, color=color, width=0.6)
        ax.set_xticks([]); ax.set_ylabel('Current')
        ax.set_title(f'Coil {k+1}')
        ha = 'center'; va = 'bottom' if val >= 0 else 'top'
        ax.text(0, val, f'{val:.3e}', ha=ha, va=va, fontsize=10)

    main_title = title or f'DOF Overview  nfp={nfp}  mpol={mpol}  ntor={ntor}'
    fig.suptitle(main_title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_boozer_contour(dofs, nfp, rho=1, fill=False, ncontours=30, title=None,
                        mpol=None, ntor=None, L=None, M=None, N=None):
    """Plot |B| contour on Boozer surface from pre-normalised surface DOFs.

    Pipeline: DOFs → SIMSOPT surface → DESC(Equilibrium from VMEC input) → solve → plot.
    Follows the same DESC usage pattern as metric_s.py: surf2desc builds the Equilibrium
    from a VMEC input file, no VMEC run needed.  DOFs are assumed already normalised
    (use load_case() + normalize_dofs() from metrics.load_data).

    Parameters
    ----------
    dofs : ndarray
        Pre-normalised flat surface DOF vector.
    nfp : int
        Number of field periods.
    rho : float
        Flux surface label (0–1). Default 1 (LCFS).
    fill : bool
        Filled contours if True, line contours if False.
    ncontours : int
        Number of contour levels.
    title : str
        Plot title.
    mpol, ntor : int
        Surface resolution.
    L, M, N : int
        DESC spectral resolution. Defaults from metric_s.

    Returns
    -------
    eq : desc.equilibrium.Equilibrium
        Solved DESC equilibrium.
    """
    surface = dofs2surface(dofs, nfp, mpol=mpol, ntor=ntor)
    eq = surf2desc(surface, L=L, M=M, N=N)
    eq.solve(ftol=1e-8, maxiter=50, verbose=0)
    desc_plot.plot_boozer_surface(eq, rho=rho, fill=fill, ncontours=ncontours)
    if title:
        plt.title(title)
    plt.show()
    return eq


def plot_boozer_modes(dofs, nfp, num_modes=8, rho=10, log=True,
                            mpol=None, ntor=None, L=None, M=None, N=None,
                            title=None, save_path=None):
    """Plot Boozer |B| Fourier spectrum from pre-normalised surface DOFs.

    Pipeline: DOFs → SIMSOPT surface → DESC(Equilibrium from VMEC input) → solve →
    plot Boozer mode amplitudes |B_mn|.

    Parameters
    ----------
    dofs : ndarray
        Pre-normalised flat surface DOF vector.
    nfp : int
        Number of field periods.
    num_modes : int
        Number of dominant modes to display. Default 8.
    rho : int or array
        Number of flux surfaces to evaluate (evenly spaced in (0,1]), or explicit
        rho values. Default 10 surfaces.
    log : bool
        Log-scale y-axis.
    mpol, ntor : int
        Surface resolution. Defaults from metric_s.
    L, M, N : int
        DESC spectral resolution. Defaults from metric_s.
    title : str
        Plot title.
    save_path : str or Path
        If given, save figure to this path instead of showing.

    Returns
    -------
    eq : desc.equilibrium.Equilibrium
        Solved DESC equilibrium.
    """
    surface = dofs2surface(dofs, nfp, mpol=mpol, ntor=ntor)
    eq = surf2desc(surface, L=L, M=M, N=N)
    eq.solve(ftol=1e-8, maxiter=50, verbose=0)

    desc_plot.plot_boozer_modes(eq, log=log, num_modes=num_modes, rho=rho)

    if title:
        plt.title(title)
    if save_path:
        plt.gcf().savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close('all')
    else:
        plt.show()
    return eq


def plot_histograms(df, data_cols, condition_col=None, n_bins=10, log=False):
    """按条件列分组，对每个数据列绘制频率分布直方图（红蓝色阶按条件列的值着色）。

    画出 len(data_cols) × 1 张图，每张图上叠加 condition_col 各分段区间的直方图。

    Args:
        df: pandas DataFrame.
        data_cols: 要观察的数据列名列表。
        condition_col: 用于分段着色的条件列名。若为 None 则不分组，单色绘制。
        n_bins: 条件列的分段数（或直方图的 bins 数，当 condition_col=None 时）。
        log: 是否对 x 轴取对数。
    """
    import pandas as pd
    from pandas.api.types import is_numeric_dtype

    cmap = plt.get_cmap("coolwarm")
    n_data = len(data_cols)

    if condition_col is not None and condition_col in df.columns:
        # 等宽分段
        cond_vals = df[condition_col].dropna()
        edges = np.linspace(cond_vals.min(), cond_vals.max(), n_bins + 1)
        ranges = list(zip(edges[:-1], edges[1:]))
        mids = np.array([(lo + hi) / 2 for lo, hi in ranges])
        norm = Normalize(vmin=mids.min(), vmax=mids.max())
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
    else:
        ranges = [(None, None)]
        mids = [0]
        norm = None
        sm = None

    fig, axes = plt.subplots(n_data, 1, figsize=(8, 4 * n_data), squeeze=False)
    axes = axes[:, 0]

    for idx, col in enumerate(data_cols):
        ax = axes[idx]
        if col not in df.columns or not is_numeric_dtype(df[col]):
            ax.set_title(f"{col} (skipped — not numeric)")
            continue

        for i, (lo, hi) in enumerate(ranges):
            if lo is None:
                sub = df[[col]].dropna()
            else:
                sub = df[(df[condition_col] >= lo) & (df[condition_col] < hi)][[col]].dropna()
            if sub.empty:
                continue

            vals = sub[col].values
            if log:
                vals = vals[vals > 0]
                if len(vals) == 0:
                    continue
                vals = np.log10(vals)

            color = sm.to_rgba(mids[i]) if sm is not None else "steelblue"
            label = f"{condition_col} [{lo:.3g}, {hi:.3g})" if lo is not None else col
            ax.hist(vals, bins=30, alpha=0.6, color=color, label=label, edgecolor="white")

        xlabel = f"log10({col})" if log else col
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        ax.set_title(col)
        if condition_col is not None:
            ax.legend(fontsize=7, loc="upper right")

    if sm is not None:
        cbar = fig.colorbar(sm, ax=axes.tolist())
        cbar.set_label(condition_col)

    plt.tight_layout()
    plt.show()


def plot_scatter_matrix(df, x_cols, condition_col=None, y_cols=None,
                        c_col=None, log_x=False, log_y=False, s=5):
    """DataFrame 散点图矩阵。

    - 若未指定 y_cols（第四维）：绘制 x_cols 中所有两两组合（去掉对角线自相关）。
    - 若指定了 y_cols：绘制 x_cols（横轴）× y_cols（纵轴）的网格图。
    - c_col 可选，用红蓝色阶对每个点着色。
    - condition_col 可选，按该列的分段区间叠加不同颜色（用于不变量的条件下展示分布）。

    Args:
        df: pandas DataFrame.
        x_cols: 横坐标列名列表。
        condition_col: 条件列名，按等宽分段上色。None 则不分组。
        y_cols: 纵坐标列名列表。None 则 x_cols 两两组合。
        c_col: 第三维列名，用红蓝色阶着色（覆盖 condition_col 的着色）。
        log_x, log_y: 是否对 x / y 轴取对数。
        s: 散点大小。
    """
    import pandas as pd
    from pandas.api.types import is_numeric_dtype
    from itertools import combinations

    # 确定列对
    if y_cols is None:
        pairs = list(combinations(x_cols, 2))
        y_cols_eff = x_cols
    else:
        pairs = [(x, y) for x in x_cols for y in y_cols]
        y_cols_eff = y_cols

    n_rows = len(x_cols)
    n_cols = len(y_cols_eff)

    # 条件分段
    has_cond = condition_col is not None and condition_col in df.columns
    if has_cond:
        cond_vals = df[condition_col].dropna()
        edges = np.linspace(cond_vals.min(), cond_vals.max(), 8)
        ranges = list(zip(edges[:-1], edges[1:]))
        mids = np.array([(lo + hi) / 2 for lo, hi in ranges])
        cmap_cond = plt.get_cmap("coolwarm")
        norm_cond = Normalize(vmin=mids.min(), vmax=mids.max())
        sm_cond = cm.ScalarMappable(cmap=cmap_cond, norm=norm_cond)
        sm_cond.set_array([])
        colors_iter = [sm_cond.to_rgba(m) for m in mids]
        labels_iter = [f"{condition_col} [{lo:.3g},{hi:.3g})" for lo, hi in ranges]
    else:
        ranges = [(None, None)]
        colors_iter = ["steelblue"]
        labels_iter = ["all"]

    has_c = c_col is not None and c_col in df.columns

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 3.5 * n_rows),
                             squeeze=False)

    for i, xcol in enumerate(x_cols):
        for j, ycol in enumerate(y_cols_eff):
            ax = axes[i][j]
            if xcol == ycol:
                ax.set_visible(False)
                continue

            # 如果是指定了 y_cols 之外的组合（如 pair 不在 pairs 里），也要处理
            for (px, py) in pairs:
                if px == xcol and py == ycol:
                    break
            else:
                ax.set_visible(False)
                continue

            if not is_numeric_dtype(df[xcol]) or not is_numeric_dtype(df[ycol]):
                ax.set_title(f"skip non-numeric", fontsize=8)
                continue

            for ri, (lo, hi) in enumerate(ranges):
                if lo is None:
                    sub = df[[xcol, ycol] + ([c_col] if has_c else [])].dropna().copy()
                else:
                    sub = df[(df[condition_col] >= lo) & (df[condition_col] < hi)]
                    sub = sub[[xcol, ycol] + ([c_col] if has_c else [])].dropna().copy()
                if sub.empty:
                    continue

                x = sub[xcol].values.astype(float)
                y = sub[ycol].values.astype(float)

                if log_x:
                    mask = x > 0; x = x[mask]; y = y[mask]
                    x = np.log10(x)
                if log_y:
                    mask = y > 0; x = x[mask]; y = y[mask]
                    y = np.log10(y)
                if len(x) == 0:
                    continue

                if has_c:
                    c_vals = sub[c_col].values.astype(float)[:len(x)]
                    c_norm = Normalize(vmin=np.percentile(c_vals, 2), vmax=np.percentile(c_vals, 98))
                    ax.scatter(x, y, c=c_vals, cmap="coolwarm", norm=c_norm,
                              alpha=0.6, s=s, edgecolors="none")
                else:
                    color = colors_iter[ri % len(colors_iter)]
                    ax.scatter(x, y, color=color, alpha=0.6, s=s, edgecolors="none")

            xlabel = f"log10({xcol})" if log_x else xcol
            ylabel = f"log10({ycol})" if log_y else ycol
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(f"{xcol} vs {ycol}", fontsize=9)
            ax.grid(True, alpha=0.3)

    # 隐藏空子图
    for i in range(n_rows):
        for j in range(n_cols):
            if not axes[i][j].has_data() and axes[i][j].get_visible():
                axes[i][j].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_poincare_from_dofs(
    surf_dofs,
    coil_dofs,
    currents,
    nfp,
    mpol=10,
    ntor=10,
    coil_order=16,
    coil_quadpoints=300,
    nfieldlines=20,
    radial_width=None,
    edge_fraction=1.1,
    stop_fraction=1.5,
    tmax_fl=2.0e8,
    phis=None,
    save_path=None,
    tol=1e-5,
    plot_surface=True,
    plot=True,
    marker_size=10,
):
    """Trace field lines from surface/coil DOFs and save a Poincare plot.

    Currents are normalized by their mean absolute value before tracing. This
    keeps relative current signs/ratios while making ``tmax_fl`` the tracing
    length control independent of the generated current norm.
    """
    surface = SurfaceXYZTensorFourier(
        mpol=int(mpol),
        ntor=int(ntor),
        nfp=int(nfp),
        stellsym=True,
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
    print(f"R0 range = [{R0[0]:.6g}, {R0[-1]:.6g}], boundary R(phi=0) ≈ {r_outer_phi0:.6g}")

    print("Beginning field line tracing")
    fieldlines_phi_hits = []
    for r_start, z_start in tqdm(
        list(zip(R0, Z0)),
        desc="Tracing fieldlines",
        unit="line",
    ):
        tys_one, hits_one = compute_fieldlines(
            bs,
            [r_start],
            [z_start],
            tmax=float(tmax_fl),
            tol=float(tol),
            phis=phis,
            stopping_criteria=stopping_criteria,
        )
        fieldlines_phi_hits.append(hits_one[0])
    hit_counts = [0 if np.asarray(hits).ndim != 2 else int(np.sum(np.asarray(hits)[:, 1] >= 0))
                  for hits in fieldlines_phi_hits]
    print("Poincare hits per line =", hit_counts)

    surf_arg = surface if plot_surface else None
    nrowcol = int(np.ceil(np.sqrt(len(phis))))
    fig, axs = plt.subplots(nrowcol, nrowcol, figsize=(8, 5))
    axs = np.asarray(axs).reshape((nrowcol, nrowcol))
    for i, phi in enumerate(phis):
        ax = axs[i // nrowcol, i % nrowcol]
        ax.set_aspect("equal")
        ax.set_title(f"$\\phi = {phi / np.pi:.2f}\\pi$ ", loc="left", y=0.0)
        if i // nrowcol == nrowcol - 1:
            ax.set_xlabel("$r$")
        if i % nrowcol == 0:
            ax.set_ylabel("$z$")
        ax.grid(True, linewidth=0.5)
        for hits in fieldlines_phi_hits:
            hits = np.asarray(hits)
            if hits.ndim != 2 or hits.shape[0] == 0:
                continue
            data = hits[np.where(hits[:, 1] == i)[0], :]
            if data.size == 0:
                continue
            r = np.sqrt(data[:, 2] ** 2 + data[:, 3] ** 2)
            ax.scatter(r, data[:, 4], s=marker_size, linewidths=0)
        if surf_arg is not None:
            cross_section = surf_arg.cross_section(phi=phi)
            r_interp = np.sqrt(cross_section[:, 0] ** 2 + cross_section[:, 1] ** 2)
            z_interp = cross_section[:, 2]
            ax.plot(r_interp, z_interp, linewidth=1, c="k")
    for i in range(len(phis), nrowcol * nrowcol):
        axs[i // nrowcol, i % nrowcol].axis("off")
    plt.tight_layout()
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved Poincare plot to {save_path}")
    if plot:
        plt.show()
    else:
        plt.close(fig)
    return fig


def poincare_boundary(
    surf_dofs,
    coil_dofs,
    currents,
    nfp,
    mpol=10,
    ntor=10,
    coil_order=16,
    coil_quadpoints=300,
    nfieldlines=5,
    inner_fraction=0.9,
    outer_fraction=1.1,
    stop_fraction=1.5,
    tmax_fl=2.0e8,
    phis=None,
    save_path=None,
    tol=1e-5,
    plot_surface=True,
    plot=True,
    marker_size=10,
):
    """Poincare plot for field lines initialized from 0.9x to 1.1x boundary radius.

    Currents are normalized by their mean absolute value before tracing. This
    keeps relative current signs/ratios while making ``tmax_fl`` the tracing
    length control independent of the generated current norm.
    """
    surface = SurfaceXYZTensorFourier(
        mpol=int(mpol),
        ntor=int(ntor),
        nfp=int(nfp),
        stellsym=True,
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
    r_outer_phi0 = float(np.max(r0))
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

    fractions = np.linspace(float(inner_fraction), float(outer_fraction), int(nfieldlines))
    R0 = seed_r + fractions * (r_outer_phi0 - seed_r)
    Z0 = np.full(int(nfieldlines), seed_z)
    print("Boundary fractions =", fractions)
    print(f"R0 range = [{R0[0]:.6g}, {R0[-1]:.6g}], boundary R(phi=0) ≈ {r_outer_phi0:.6g}")

    if phis is None:
        phis = [(i / 4) * (2 * np.pi / int(nfp)) for i in range(4)]
    else:
        phis = list(np.asarray(phis, dtype=float).ravel())

    print("Beginning field line tracing")
    fieldlines_phi_hits = []
    for r_start, z_start in tqdm(
        list(zip(R0, Z0)),
        desc="Tracing boundary fieldlines",
        unit="line",
    ):
        tys_one, hits_one = compute_fieldlines(
            bs,
            [r_start],
            [z_start],
            tmax=float(tmax_fl),
            tol=float(tol),
            phis=phis,
            stopping_criteria=stopping_criteria,
        )
        fieldlines_phi_hits.append(hits_one[0])
    hit_counts = [0 if np.asarray(hits).ndim != 2 else int(np.sum(np.asarray(hits)[:, 1] >= 0))
                  for hits in fieldlines_phi_hits]
    print("Poincare hits per line =", hit_counts)

    surf_arg = surface if plot_surface else None
    nrowcol = int(np.ceil(np.sqrt(len(phis))))
    fig, axs = plt.subplots(nrowcol, nrowcol, figsize=(8, 5))
    axs = np.asarray(axs).reshape((nrowcol, nrowcol))
    for i, phi in enumerate(phis):
        ax = axs[i // nrowcol, i % nrowcol]
        ax.set_aspect("equal")
        ax.set_title(f"$\\phi = {phi / np.pi:.2f}\\pi$ ", loc="left", y=0.0)
        if i // nrowcol == nrowcol - 1:
            ax.set_xlabel("$r$")
        if i % nrowcol == 0:
            ax.set_ylabel("$z$")
        ax.grid(True, linewidth=0.5)
        for hits in fieldlines_phi_hits:
            hits = np.asarray(hits)
            if hits.ndim != 2 or hits.shape[0] == 0:
                continue
            data = hits[np.where(hits[:, 1] == i)[0], :]
            if data.size == 0:
                continue
            r = np.sqrt(data[:, 2] ** 2 + data[:, 3] ** 2)
            ax.scatter(r, data[:, 4], s=marker_size, linewidths=0)
        if surf_arg is not None:
            cross_section = surf_arg.cross_section(phi=phi)
            r_interp = np.sqrt(cross_section[:, 0] ** 2 + cross_section[:, 1] ** 2)
            z_interp = cross_section[:, 2]
            ax.plot(r_interp, z_interp, linewidth=1, c="k")
    for i in range(len(phis), nrowcol * nrowcol):
        axs[i // nrowcol, i % nrowcol].axis("off")
    plt.tight_layout()
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved Poincare plot to {save_path}")
    if plot:
        plt.show()
    else:
        plt.close(fig)
    return fig


if __name__ == "__main__":
    from .load_data import load_case, change_xyzcurve_modes, df

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ID = 625292
    surfaces, coils, volume, mean_iota, nc_per_hp, helicity, nfp = load_case(ID)
    print("order:", coils[0].curve.order)
    dofs_surface = surfaces[-1].x
    dofs_coil = np.array([change_xyzcurve_modes(c.curve, target_order=16)
                          for c in coils[:int(nc_per_hp)]])
    currents = np.array([c.current.full_x[0] for c in coils[:int(nc_per_hp)]], dtype=float)

    print(f"ID: {ID}")
    print(f"nfp: {nfp}, helicity: {helicity}, nc_per_hp: {nc_per_hp}")
    print(f"volume: {volume}, mean_iota: {mean_iota}")
    print(f"surface dof dim: {len(dofs_surface)}, coil dofs shape: {dofs_coil.shape}")
    print(f"currents: {currents}")

    result = poincare_boundary(
        dofs_surface,
        dofs_coil,
        currents,
        nfp=int(nfp),
        mpol=10,
        ntor=10,
        coil_order=16,
        nfieldlines=5,
    )
    print(result)
