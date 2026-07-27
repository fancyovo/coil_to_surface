import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from .config import AxisGAConfig, EvalConfig, PsiFitConfig, SurfaceScanConfig, BoozerConfig, DiagnosticsConfig, VolumeQSConfig
from .score import ScoreConfig, coil_geometry_metrics, evaluate_quality_score
from .volume_score import VolumeScoreConfig, evaluate_volume_quality_score


def evaluate_case_file(*args, **kwargs):
    from .pipeline import evaluate_case_file as _evaluate_case_file

    return _evaluate_case_file(*args, **kwargs)


def evaluate_coils(*args, **kwargs):
    from .pipeline import evaluate_coils as _evaluate_coils

    return _evaluate_coils(*args, **kwargs)


def evaluate_coil_quality(*args, **kwargs):
    from .pipeline import evaluate_coil_quality as _evaluate_coil_quality

    return _evaluate_coil_quality(*args, **kwargs)

__all__ = [
    "AxisGAConfig",
    "EvalConfig",
    "PsiFitConfig",
    "VolumeQSConfig",
    "SurfaceScanConfig",
    "BoozerConfig",
    "DiagnosticsConfig",
    "ScoreConfig",
    "VolumeScoreConfig",
    "coil_geometry_metrics",
    "evaluate_quality_score",
    "evaluate_volume_quality_score",
    "evaluate_case_file",
    "evaluate_coils",
    "evaluate_coil_quality",
]
