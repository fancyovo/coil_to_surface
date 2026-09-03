from __future__ import annotations

import json
from pathlib import Path

from scripts.run_axisflip_rl_latent_adam200 import (
    BETA1,
    BETA2,
    DIRECTIONS,
    FLOW_STEPS,
    ITERATIONS,
    LEARNING_RATE,
    NFP,
    N_BASE_COILS,
    PERTURBATION,
    PROTOCOL_ID,
    SCREEN_COUNT,
    case_indices,
    optimization_command,
    screen_command,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_two_workers_cover_eight_unique_interleaved_cases() -> None:
    left = case_indices(0, 2, 4)
    right = case_indices(1, 2, 4)
    assert left == [0, 2, 4, 6]
    assert right == [1, 3, 5, 7]
    assert sorted(left + right) == list(range(8))


def test_commands_pin_screen32_and_current_latent_recipe(tmp_path: Path) -> None:
    screen = screen_command(
        checkpoint=tmp_path / "checkpoint.pt",
        score_lib=tmp_path / "lib.so",
        out_dir=tmp_path / "screen",
        seed=17,
    )
    optimize = optimization_command(
        checkpoint=tmp_path / "checkpoint.pt",
        score_lib=tmp_path / "lib.so",
        initial_case=tmp_path / "selected_start.json",
        out_dir=tmp_path / "optimize",
        seed=18,
        max_wall_s=7200.0,
    )

    assert option_value(screen, "--nfp") == str(NFP)
    assert option_value(screen, "--n-base-coils") == str(N_BASE_COILS)
    assert option_value(screen, "--candidate-count") == str(SCREEN_COUNT)
    assert option_value(screen, "--flow-steps") == str(FLOW_STEPS)
    assert option_value(optimize, "--iterations") == str(ITERATIONS)
    assert option_value(optimize, "--random-directions") == str(DIRECTIONS)
    assert option_value(optimize, "--perturbation") == str(PERTURBATION)
    assert option_value(optimize, "--learning-rate") == str(LEARNING_RATE)
    assert option_value(optimize, "--beta1") == str(BETA1)
    assert option_value(optimize, "--beta2") == str(BETA2)
    assert option_value(optimize, "--flow-steps") == str(FLOW_STEPS)
    assert option_value(optimize, "--recorded-initial-score-tolerance") == "0.1"
    assert "--flow-pipeline" in optimize
    assert option_value(optimize, "--gradient-mode") == "random-orthogonal"


def test_spec_and_students_launcher_match_registered_experiment() -> None:
    spec = json.loads(
        (REPO_ROOT / "evaluation" / "axisflip_rl_round12_latent_adam200_abi11_v1.json").read_text(
            encoding="utf-8"
        )
    )
    launcher = (
        REPO_ROOT / "scripts" / "slurm_axisflip_rl_latent_adam200_students.sh"
    ).read_text(encoding="utf-8")

    assert spec["protocol_id"] == PROTOCOL_ID
    assert spec["frozen_flow"]["outer_round"] == 12
    assert spec["sample_plan"]["total_trajectories"] == 8
    assert "#SBATCH --partition=Students" in launcher
    assert "#SBATCH --qos=qos_stu_medium_2gpu" in launcher
    assert "#SBATCH --array=0-1" in launcher
    assert "#SBATCH --gres=gpu:RTX5090:1" in launcher
    assert "--worker-count 2" in launcher
    assert 'gpu_selector="${CUDA_VISIBLE_DEVICES:-}"' in launcher
    assert launcher.count('nvidia-smi --id="$gpu_selector"') == 4
