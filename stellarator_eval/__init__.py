import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from .config import AxisGAConfig, EvalConfig, PsiFitConfig, SurfaceScanConfig, BoozerConfig, DiagnosticsConfig


def evaluate_case_file(*args, **kwargs):
    from .pipeline import evaluate_case_file as _evaluate_case_file

    return _evaluate_case_file(*args, **kwargs)


def evaluate_coils(*args, **kwargs):
    from .pipeline import evaluate_coils as _evaluate_coils

    return _evaluate_coils(*args, **kwargs)

__all__ = [
    "AxisGAConfig",
    "EvalConfig",
    "PsiFitConfig",
    "SurfaceScanConfig",
    "BoozerConfig",
    "DiagnosticsConfig",
    "evaluate_case_file",
    "evaluate_coils",
]
