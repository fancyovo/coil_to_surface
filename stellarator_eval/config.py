from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence


@dataclass
class AxisGAConfig:
    method: str = "fixed_point"
    backend: str = "gpu"
    grid: int = 16
    keep: int = 16
    span: float = 0.5
    z_center: float = 0.0
    rk4_steps: int = 800
    max_generations: int = 32
    tol: float = 1e-7
    axis_trace_steps: int = 240
    gpu_lib_path: str = "gpu_backend/build_mixed/libstellarator_gpu.so"
    gpu_segments_per_coil: int = 256
    gpu_device: int = 0
    gpu_trace_precision: str = "mixed64"
    gpu_verify_precision: str = "fp64"
    gpu_threads_per_line: int = 256
    staged: bool = False
    switch_tol: float = 1e-6
    coarse_precision: str = "fp32"
    fine_precision: str = "mixed64"
    fine_grid: int = 8
    fine_keep: int = 8
    fine_max_generations: int = 96
    fine_span_min: float = 1e-8
    fixed_point_grid: int = 48
    fixed_point_max_candidates: int = 16
    fixed_point_local_min_candidates: int = 64
    fixed_point_newton_iters: int = 6
    fixed_point_fallback_grid: int = 96
    fixed_point_fallback_max_candidates: int = 96
    fixed_point_fallback_newton_iters: int = 8
    fixed_point_verify_top: int = 8
    fixed_point_r_floor: float = 1e-4
    fixed_point_fd_rel: float = 2e-4
    fixed_point_fd_abs: float = 2e-6
    fixed_point_topology_filter: bool = True
    fixed_point_require_elliptic: bool = True
    fixed_point_topology_fd_rel: float = 2e-4
    fixed_point_topology_fd_abs: float = 2e-6
    fixed_point_topology_margin: float = 2e-2
    fixed_point_prefer_round_elliptic: bool = True


@dataclass
class PsiFitConfig:
    backend: str = "fullgpu"
    linear_solver: str = "qr"
    normal_eq_backend: str = "auto"
    normal_eq_precision: str = "fp32"
    a: float = 0.05
    rho_min: float = 0.002
    poly_degree: int = 10
    m_tor: int = 12
    ridge: float = 1e-6
    n_r: int = 80
    n_z: int = 80
    n_phi: int = 80
    batch_size: int = 20000
    validation_points: int = 4000
    gpu_lib_path: str = "gpu_backend/build_mixed/libstellarator_gpu.so"
    gpu_segments_per_coil: int = 256
    gpu_device: int = 0


@dataclass
class SurfaceScanConfig:
    levels: Sequence[float] = field(
        default_factory=lambda: (0.001, 0.002, 0.004, 0.008, 0.012, 0.02, 0.04, 0.08, 0.12, 0.16)
    )
    n_alpha: int = 256
    trace_steps: int = 800
    max_radius_scale: float = 1.0
    drift_rel_tol: float = 0.30
    drift_abs_tol: float = 5e-4
    max_boozer_candidates: int = 3
    curve_newton_tol: float = 1e-12
    curve_newton_maxiter: int = 20
    trace_backend: str = "gpu"
    gpu_lib_path: str = "gpu_backend/build_mixed/libstellarator_gpu.so"
    gpu_segments_per_coil: int = 256
    gpu_device: int = 0
    gpu_trace_precision: str = "mixed64"
    gpu_verify_precision: str = "fp64"
    gpu_threads_per_line: int = 256
    gpu_verify_candidates: int = 3


@dataclass
class BoozerConfig:
    surface_order: int = 6
    stellsym: bool = True
    initial_iota: float = -2.0
    auto_initial_iota: bool = True
    auto_initial_iota_default_only: bool = True
    auto_initial_iota_default_value: float = -2.0
    constraint_weight: float = 1.0
    ls_maxiter: int = 100
    ls_tol: float = 1e-10
    newton_maxiter: int = 30
    newton_tol: float = 1e-12
    qs_sdim: int = 16
    surface_extract_backend: str = "gpu"
    gpu_lib_path: str = "gpu_backend/build_mixed/libstellarator_gpu.so"
    gpu_device: int = 0


@dataclass
class DiagnosticsConfig:
    export_axis_heatmap: bool = False
    axis_heatmap_grid: int = 256
    axis_heatmap_filename: str = "axis_residual_heatmap.png"
    export_psi_slices: bool = False
    psi_slice_grid: int = 241
    psi_slice_phi_count: int = 17
    psi_slice_filename: str = "psi_slices.png"
    plot_dpi: int = 170


@dataclass
class EvalConfig:
    axis: AxisGAConfig = field(default_factory=AxisGAConfig)
    psi: PsiFitConfig = field(default_factory=PsiFitConfig)
    scan: SurfaceScanConfig = field(default_factory=SurfaceScanConfig)
    boozer: BoozerConfig = field(default_factory=BoozerConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    current_unit: str = "MA"
    omp_threads: int = 1

    def to_dict(self) -> dict:
        return asdict(self)
