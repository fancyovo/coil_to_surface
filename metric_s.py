import numpy as np
import tempfile
import os
import shutil


if __name__ == "__main__" and __package__ is None:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "metrics"

from simsopt.geo import SurfaceXYZTensorFourier, SurfaceRZFourier
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual

# vmecpp / desc 改为**惰性 import** (各自函数内): 它们 (尤其 desc 拉 jax) import 慢 ~30s,
# 而 VMEC2000 路径 (dofs2surface/surf2vmec/J_QS, eval_one_physical 走这条) 不需要它们。
# 并发子进程时若顶层 import, 8×jax import 风暴会让每个 VMEC 子进程超时。

MPOL_XYZ = 10
NTOR_XYZ = 10
MPOL_RZ = 6
NTOR_RZ = 6
STELLSYM = True

QS_SURFACES = np.linspace(0.1, 1, 10, endpoint=False)
DESC_QS_SURFACES = np.linspace(0.1, 1, 10, endpoint=False)

DESC_L = 8
DESC_M = 8
DESC_N = 8


def dofs2surface(dofs, nfp, mpol=None, ntor=None):
    surface = SurfaceXYZTensorFourier(
        mpol=MPOL_XYZ if mpol is None else int(mpol),
        ntor=NTOR_XYZ if ntor is None else int(ntor),
        nfp=int(nfp),
        stellsym=STELLSYM,
    )
    dofs = np.asarray(dofs, dtype=float).ravel()
    surface.set_dofs(dofs)
    return surface


# =========================
# simsopt Vmec (Fortran)
# =========================


def surf2vmec(surface):
    vmec = Vmec(verbose=0)
    vmec.boundary = surface
    vmec.indata.nzeta = 14
    vmec.indata.ntheta = 16
    vmec.indata.ns_array[:] = 0
    vmec.indata.ns_array[0] = 16
    vmec.indata.ns_array[1] = 50
    vmec.indata.ftol_array[:] = 0
    vmec.indata.ftol_array[0] = 1.0e-16
    vmec.indata.ftol_array[1] = 1.0e-11
    vmec.indata.niter_array[:] = 0
    vmec.indata.niter_array[0] = 1000
    vmec.indata.niter_array[1] = 5000
    vmec.indata.nfp = surface.nfp
    vmec.indata.phiedge = 1.0
    vmec.need_to_run_code = True
    return vmec


def J_QS(dofs, nfp, helicity, mpol=None, ntor=None):
    hm, hn = 1, helicity
    surface = dofs2surface(dofs, nfp, mpol=mpol, ntor=ntor)
    vmec = surf2vmec(surface); vmec.run()
    qs = QuasisymmetryRatioResidual(vmec, surfaces=QS_SURFACES, helicity_m=hm, helicity_n=hn)
    return float(np.sqrt(qs.total()))


def J_iota(dofs, nfp, mpol=None, ntor=None):
    surface = dofs2surface(dofs, nfp, mpol=mpol, ntor=ntor)
    vmec = surf2vmec(surface); vmec.run()
    return float(np.abs(vmec.mean_iota()))


def J_A(dofs, nfp, mpol=None, ntor=None):
    return float(dofs2surface(dofs, nfp, mpol=mpol, ntor=ntor).aspect_ratio())


def J_total_s(dofs, nfp, helicity, mpol=None, ntor=None):
    hm, hn = 1, helicity
    surface = dofs2surface(dofs, nfp, mpol=mpol, ntor=ntor)
    vmec = surf2vmec(surface); vmec.run()
    qs = QuasisymmetryRatioResidual(vmec, surfaces=QS_SURFACES, helicity_m=hm, helicity_n=hn)
    return float(np.sqrt(qs.total())), float(np.abs(vmec.mean_iota())), float(surface.aspect_ratio())


# =========================
# vmecpp — VmecppVmec (vmecpp.simsopt_compat.Vmec) via Fortran INDATA
# =========================


def surf2vmecpp(surface):
    from vmecpp.simsopt_compat import Vmec as VmecppVmec   # 惰性
    tmpdir = tempfile.mkdtemp(prefix="vmecpp_")
    input_path = os.path.join(tmpdir, "input.temp")
    vmec = surf2vmec(surface)
    vmec.write_input(input_path)
    vmecpp = VmecppVmec(input_path, verbose=False)
    vmecpp._tmpdir = tmpdir
    return vmecpp


def _vmecpp_cleanup(vmecpp):
    d = getattr(vmecpp, "_tmpdir", None)
    if d:
        shutil.rmtree(d, ignore_errors=True)


def J_QS_vmecpp(dofs, nfp, helicity):
    hm, hn = 1, helicity
    surface = dofs2surface(dofs, nfp)
    vmec = surf2vmecpp(surface); vmec.run()
    qs = QuasisymmetryRatioResidual(vmec, surfaces=QS_SURFACES, helicity_m=hm, helicity_n=hn)
    r = float(np.sqrt(qs.total()))
    _vmecpp_cleanup(vmec)
    return r


def J_iota_vmecpp(dofs, nfp):
    surface = dofs2surface(dofs, nfp)
    vmec = surf2vmecpp(surface); vmec.run()
    r = float(np.abs(vmec.mean_iota()))
    _vmecpp_cleanup(vmec)
    return r


def J_A_vmecpp(dofs, nfp):
    return float(dofs2surface(dofs, nfp).aspect_ratio())


def J_total_vmecpp(dofs, nfp, helicity):
    hm, hn = 1, helicity
    surface = dofs2surface(dofs, nfp)
    vmec = surf2vmecpp(surface); vmec.run()
    qs = QuasisymmetryRatioResidual(vmec, surfaces=QS_SURFACES, helicity_m=hm, helicity_n=hn)
    r = (
        float(np.sqrt(qs.total())),
        float(np.abs(vmec.mean_iota())),
        float(surface.aspect_ratio()),
    )
    _vmecpp_cleanup(vmec)
    return r

# =========================
# DESC — zero current, default solver
# =========================


def surf2desc(surface, L=None, M=None, N=None):
    from desc.equilibrium import Equilibrium                # 惰性 (拉 jax)
    from desc.geometry import FourierRZToroidalSurface
    from desc.profiles import PowerSeriesProfile
    vmec = surf2vmec(surface)
    tmpdir = tempfile.mkdtemp(prefix="desc_")
    input_path = os.path.join(tmpdir, "input.temp")
    vmec.write_input(input_path)
    surf = FourierRZToroidalSurface.from_input_file(input_path)
    shutil.rmtree(tmpdir, ignore_errors=True)

    if L is None: L = DESC_L
    if M is None: M = DESC_M
    if N is None: N = DESC_N

    return Equilibrium(
        L=L, M=M, N=N,
        surface=surf,
        pressure=PowerSeriesProfile([0, 0, 0]),
        current=PowerSeriesProfile([0]),
        Psi=vmec.indata.phiedge,
    )


def J_QS_desc(dofs, nfp, helicity, L=None, M=None, N=None):
    from desc.objectives import QuasisymmetryTwoTerm        # 惰性
    from desc.grid import LinearGrid
    surface = dofs2surface(dofs, nfp)
    eq = surf2desc(surface, L=L, M=M, N=N)
    eq.solve(ftol=1e-8, maxiter=50, verbose=0)
    qs_values = []
    for s in DESC_QS_SURFACES:
        grid_s = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym, rho=np.array([s]))
        qs_obj = QuasisymmetryTwoTerm(eq, helicity=(1, helicity * eq.NFP), grid=grid_s)
        qs_obj.build()
        qs_values.append(float(qs_obj.compute_scalar(eq.params_dict)))
    return float(np.sqrt(np.mean(np.array(qs_values) ** 2)))


def J_iota_desc(dofs, nfp, L=None, M=None, N=None):
    from desc.grid import LinearGrid                        # 惰性
    surface = dofs2surface(dofs, nfp)
    eq = surf2desc(surface, L=L, M=M, N=N)
    eq.solve(ftol=1e-8, maxiter=50, verbose=0)
    grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym, rho=np.array([1.0]))
    return float(eq.compute("iota", grid=grid)["iota"][-1])


def J_A_desc(dofs, nfp, L=None, M=None, N=None):
    from desc.grid import LinearGrid                        # 惰性
    surface = dofs2surface(dofs, nfp)
    eq = surf2desc(surface, L=L, M=M, N=N)
    eq.solve(ftol=1e-8, maxiter=50, verbose=0)
    grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym, rho=np.array([1.0]))
    R0 = float(eq.compute("R0", grid=grid)["R0"])
    a = float(eq.compute("a", grid=grid)["a"])
    return R0 / a


def J_total_desc(dofs, nfp, helicity, L=None, M=None, N=None):
    from desc.grid import LinearGrid                        # 惰性
    from desc.objectives import QuasisymmetryTwoTerm 
    surface = dofs2surface(dofs, nfp)
    eq = surf2desc(surface, L=L, M=M, N=N)
    eq.solve(ftol=1e-8, maxiter=50, verbose=0)
    grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym, rho=np.array([1.0]))
    R0 = float(eq.compute("R0", grid=grid)["R0"])
    a = float(eq.compute("a", grid=grid)["a"])
    qs_values = []
    for s in DESC_QS_SURFACES:
        grid_s = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym, rho=np.array([s]))
        qs_obj = QuasisymmetryTwoTerm(eq, helicity=(1, helicity * eq.NFP), grid=grid_s)
        qs_obj.build()
        qs_values.append(float(qs_obj.compute_scalar(eq.params_dict)))
    qs = float(np.sqrt(np.mean(np.array(qs_values) ** 2)))
    iota = float(eq.compute("iota", grid=grid)["iota"][-1])
    return qs, iota, R0 / a


# =========================
# test — batch 10 random QUASR surfaces (timing + all three metrics)
# =========================

if __name__ == "__main__":

    from load_data import load_case
    import pickle
    import time

    # with open("//Users/icar1a/Desktop/code/projects/QUASR_08072024/QUASR_08072024.pkl" , "rb") as f:
    #     df = pickle.load(f)

    ID = 625292 
    t0 = time.time()
    surfaces, coils, volume, mean_iota, nc_per_hp, helicity, nfp = load_case(ID)
    t1 = time.time() - t0
    print(f"Loaded case {ID} in {t1:.1f}s")

    t0 = time.time()    
    result_s = J_total_desc(surfaces[-1].x, surfaces[-1].nfp, helicity=helicity)
    t1 = time.time() - t0
    print(f"VMEC (desc): {t1:.1f}s")
    print(f"  QS residual: {result_s[0]:.3e}")
    print(f"  iota: {result_s[1]:.3e}")
    print(f"  aspect ratio: {result_s[2]:.3f}")
