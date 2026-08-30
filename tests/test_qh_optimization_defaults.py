from __future__ import annotations

from pathlib import Path
import re

import pytest

from flow_matching.optimization import (
    CURRENT_NATIVE_SCORE_ABI,
    CURRENT_NATIVE_SCORE_LIBRARY_SHA256,
    CURRENT_QH_PROTOCOL_ID,
    DEPRECATED_NATIVE_SCORE_ABI,
    QH_OPTIMIZATION_DEFAULTS,
    describe_qh_optimization_protocol,
    describe_qh_screening_protocol,
    validate_qh_direction_count,
    validate_qh_native_score_abi,
    validate_qh_resume_protocol,
)
from scripts.optimize_flow_latent import (
    build_parser as optimizer_parser,
    parse_arguments as optimizer_arguments,
)
from scripts.optimize_flow_prior_standard_adam import parse_arguments as legacy_arguments
from scripts.screen_flow_starts import build_parser as screening_parser
from scripts.flow_runtime import repository_provenance


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_current_qh_protocol_is_the_public_default() -> None:
    defaults = QH_OPTIMIZATION_DEFAULTS
    assert defaults.candidate_count == 32
    assert defaults.iterations == 200
    assert defaults.directions == 64
    assert defaults.perturbation == 0.005
    assert defaults.learning_rate == 0.02
    assert defaults.beta1 == 0.7
    assert defaults.beta2 == 0.999
    assert defaults.flow_steps == 128
    assert defaults.gradient_mode == "random-orthogonal"
    assert CURRENT_QH_PROTOCOL_ID == "qh-flow-screen32-adam200-64d-abi11-v1"
    assert CURRENT_NATIVE_SCORE_ABI == 11
    assert CURRENT_NATIVE_SCORE_LIBRARY_SHA256 == (
        "1c6c78b0dee662233215a56dbdc1e50b8ed29f2d0eee9ae8c4ff7d0403b895ed"
    )


def test_command_line_defaults_use_the_validated_protocol() -> None:
    screening = screening_parser().parse_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--lib",
            "libstellarator_gpu.so",
            "--out-dir",
            "screen",
            "--nfp",
            "4",
            "--n-base-coils",
            "3",
            "--seed",
            "1",
        ]
    )
    optimization = optimizer_parser().parse_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--initial-case",
            "screen/selected_start.json",
            "--lib",
            "libstellarator_gpu.so",
            "--out-dir",
            "optimized",
        ]
    )

    assert screening.candidate_count == QH_OPTIMIZATION_DEFAULTS.candidate_count
    assert screening.flow_steps == QH_OPTIMIZATION_DEFAULTS.flow_steps
    assert optimization.iterations == QH_OPTIMIZATION_DEFAULTS.iterations
    assert optimization.random_directions == QH_OPTIMIZATION_DEFAULTS.directions
    assert optimization.gradient_mode == "random-orthogonal"
    assert optimization.optimizer == "adam"
    assert optimization.perturbation == QH_OPTIMIZATION_DEFAULTS.perturbation
    assert optimization.learning_rate == QH_OPTIMIZATION_DEFAULTS.learning_rate
    assert optimization.beta1 == QH_OPTIMIZATION_DEFAULTS.beta1
    assert optimization.beta2 == QH_OPTIMIZATION_DEFAULTS.beta2
    assert optimization.flow_steps == QH_OPTIMIZATION_DEFAULTS.flow_steps

    compatibility = legacy_arguments(
        ["--checkpoint", "checkpoint.pt", "--out-dir", "optimized"]
    )
    assert compatibility.iterations == QH_OPTIMIZATION_DEFAULTS.iterations
    assert compatibility.directions == QH_OPTIMIZATION_DEFAULTS.directions
    assert compatibility.learning_rate == QH_OPTIMIZATION_DEFAULTS.learning_rate


def test_two_direction_protocol_is_hard_blocked() -> None:
    with pytest.raises(ValueError, match="deprecated historical evidence"):
        validate_qh_direction_count(2)
    with pytest.raises(ValueError, match="deprecated historical evidence"):
        optimizer_arguments(
            [
                "--checkpoint",
                "checkpoint.pt",
                "--initial-case",
                "selected_start.json",
                "--lib",
                "libstellarator_gpu.so",
                "--out-dir",
                "optimized",
                "--random-directions",
                "2",
            ]
        )
    with pytest.raises(ValueError, match="deprecated historical evidence"):
        legacy_arguments(
            [
                "--checkpoint",
                "checkpoint.pt",
                "--out-dir",
                "optimized",
                "--directions",
                "2",
            ]
        )


def test_default_protocol_is_machine_identifiable() -> None:
    defaults = QH_OPTIMIZATION_DEFAULTS
    description = describe_qh_optimization_protocol(
        parameter_space="latent",
        optimizer="adam",
        iterations=defaults.iterations,
        directions=defaults.directions,
        perturbation=defaults.perturbation,
        learning_rate=defaults.learning_rate,
        beta1=defaults.beta1,
        beta2=defaults.beta2,
        flow_steps=defaults.flow_steps,
        gradient_mode=defaults.gradient_mode,
        difference="centered",
    )
    assert description["id"] == CURRENT_QH_PROTOCOL_ID
    assert description["status"] == "current-default"
    assert description["requirements"] == {
        "native_score_abi": CURRENT_NATIVE_SCORE_ABI,
        "native_score_library_sha256": CURRENT_NATIVE_SCORE_LIBRARY_SHA256,
    }
    assert description["differences_from_current"] == {}


def test_nondefault_protocol_is_explicitly_unregistered() -> None:
    defaults = QH_OPTIMIZATION_DEFAULTS
    description = describe_qh_optimization_protocol(
        parameter_space="latent",
        optimizer="adam",
        iterations=defaults.iterations,
        directions=32,
        perturbation=defaults.perturbation,
        learning_rate=defaults.learning_rate,
        beta1=defaults.beta1,
        beta2=defaults.beta2,
        flow_steps=defaults.flow_steps,
        gradient_mode=defaults.gradient_mode,
        difference="centered",
    )
    assert description["id"] == "unregistered-experimental"
    assert description["status"] == "unregistered-experimental"
    assert description["differences_from_current"] == {
        "directions": {"expected": 64, "actual": 32}
    }


def test_native_score_contract_controls_protocol_identity() -> None:
    with pytest.raises(ValueError, match="ABI-10 is deprecated historical"):
        validate_qh_native_score_abi(DEPRECATED_NATIVE_SCORE_ABI)
    with pytest.raises(ValueError, match="ABI-10 is deprecated historical"):
        describe_qh_screening_protocol(
            candidate_count=QH_OPTIMIZATION_DEFAULTS.candidate_count,
            flow_steps=QH_OPTIMIZATION_DEFAULTS.flow_steps,
            native_score_abi=DEPRECATED_NATIVE_SCORE_ABI,
            native_score_library_sha256=(
                "565c32073b145d97a1f2244705fb06e4b3458ce798cd74d0c97ee4e0129dc729"
            ),
        )

    rebuilt = describe_qh_screening_protocol(
        candidate_count=QH_OPTIMIZATION_DEFAULTS.candidate_count,
        flow_steps=QH_OPTIMIZATION_DEFAULTS.flow_steps,
        native_score_library_sha256="0" * 64,
    )
    assert rebuilt["id"] == "unregistered-experimental"
    assert rebuilt["differences_from_current"] == {
        "native_score_library_sha256": {
            "expected": CURRENT_NATIVE_SCORE_LIBRARY_SHA256,
            "actual": "0" * 64,
        }
    }


def test_resume_requires_an_exact_nonhistorical_protocol() -> None:
    defaults = QH_OPTIMIZATION_DEFAULTS

    def describe(directions: int) -> dict[str, object]:
        return describe_qh_optimization_protocol(
            parameter_space="latent",
            optimizer="adam",
            iterations=defaults.iterations,
            directions=directions,
            perturbation=defaults.perturbation,
            learning_rate=defaults.learning_rate,
            beta1=defaults.beta1,
            beta2=defaults.beta2,
            flow_steps=defaults.flow_steps,
            gradient_mode=defaults.gradient_mode,
            difference="centered",
        )

    current = describe(defaults.directions)
    validate_qh_resume_protocol(current, current)

    with pytest.raises(ValueError, match="legacy or unclassified"):
        validate_qh_resume_protocol(None, current)
    with pytest.raises(ValueError, match="does not exactly match"):
        validate_qh_resume_protocol(describe(32), current)

    historical = {**current, "actual": {**current["actual"], "directions": 2}}
    with pytest.raises(ValueError, match="deprecated historical evidence"):
        validate_qh_resume_protocol(historical, current)

    abi10 = {
        **current,
        "requirements": {
            "native_score_abi": 10,
            "native_score_library_sha256": (
                "565c32073b145d97a1f2244705fb06e4b3458ce798cd74d0c97ee4e0129dc729"
            ),
        },
    }
    with pytest.raises(ValueError, match="ABI-10 is deprecated historical"):
        validate_qh_resume_protocol(abi10, current)


def test_active_runtime_contract_contains_no_abi10_default() -> None:
    header = (REPO_ROOT / "gpu_backend" / "include" / "coil_field.h").read_text(
        encoding="utf-8"
    )
    wrapper = (
        REPO_ROOT / "gpu_backend" / "python" / "stellarator_gpu.py"
    ).read_text(encoding="utf-8")
    assert "#define SGPU_SCORE_ABI_VERSION 11u" in header
    assert "SGPU_SCORE_ABI_VERSION = 11" in wrapper

    historical_hash = (
        "565c32073b145d97a1f2244705fb06e4b3458ce798cd74d0c97ee4e0129dc729"
    )
    for relative in (
        "flow_matching/optimization.py",
        "scripts/slurm_flow_prior_standard_adam.sh",
        "scripts/slurm_flow_prior_local_full_gradient_adam.sh",
    ):
        assert historical_hash not in (REPO_ROOT / relative).read_text(encoding="utf-8")


def test_repository_provenance_records_commit_and_tracked_state() -> None:
    provenance = repository_provenance(REPO_ROOT)
    assert provenance["available"] is True
    assert len(provenance["commit"]) == 40
    assert isinstance(provenance["tracked_dirty"], bool)


def test_historical_two_direction_launchers_cannot_run() -> None:
    marker = "DEPRECATED HISTORICAL 2-DIRECTION PROTOCOL"
    historical_patterns = (
        re.compile(r"--random-directions[ =]+2(?=\D|$)"),
        re.compile(r"--directions[ =]+2(?=\D|$)"),
        re.compile(r"DIRECTIONS(?:=|:-)2(?=\D|$)"),
        re.compile(r"central2"),
    )

    matched = []
    for path in sorted((REPO_ROOT / "scripts").glob("*.sh")):
        text = path.read_text(encoding="utf-8")
        token_offsets = [
            match.start()
            for pattern in historical_patterns
            if (match := pattern.search(text)) is not None
        ]
        if not token_offsets:
            continue
        matched.append(path.name)
        assert marker in text, f"{path.name} contains a 2-direction route without a marker"
        assert text.index("exit 64") < min(token_offsets), (
            f"{path.name} can reach a historical 2-direction configuration"
        )

    assert matched == [
        "slurm_iota_cubic_adam200.sh",
        "slurm_iota_cubic_direction_compare1000.sh",
        "slurm_summary1_flow_pairs.sh",
        "submit_score_fast_beta1_long.sh",
        "submit_score_fast_optimizer_matrix.sh",
    ]
