from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from flow_matching.axis_prior_rl import (
    N_BASE_COILS,
    TOKEN_DIM,
    fit_prior_normalizer,
    inverse_tokens,
    model_config,
    online_objective,
    random_permute_coils,
    transform_tokens,
)
from scripts.axisflip_prior_online_rl import (
    FORMAT,
    PROTOCOL_ID,
    compact_native,
    load_adam20_centers,
    save_online_checkpoint,
    summarize_round,
    trajectory_reward_weights,
    wilson_interval,
)
from scripts.flow_runtime import repository_provenance
from scripts.train_axisflip_prior_flow import distribution_stable


def teacher_tokens(count: int = 16) -> np.ndarray:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(count, N_BASE_COILS, TOKEN_DIM)).astype(np.float32)
    values[..., -1] = 100000.0
    return values


def test_prior_normalizer_keeps_current_trainable_with_fixed_q0_current() -> None:
    values = teacher_tokens()
    normalizer = fit_prior_normalizer(values)
    assert normalizer.std[-1] == 100000.0
    normalized = transform_tokens(values, normalizer)
    np.testing.assert_allclose(normalized[..., -1], 0.0)
    changed = normalized[:1].copy()
    changed[0, :, -1] = [0.3, -0.1, -0.2]
    physical = inverse_tokens(changed, normalizer)
    assert not np.allclose(physical[0, :, -1], physical[0, 0, -1])
    assert np.isclose(np.sum(np.abs(physical[0, :, -1])), 300000.0, rtol=1.0e-6)
    roundtrip = transform_tokens(physical, normalizer)
    np.testing.assert_allclose(roundtrip, changed, rtol=1.0e-5, atol=1.0e-5)


def test_random_permutation_preserves_each_coil_token() -> None:
    values = torch.arange(4 * 3 * 100, dtype=torch.float32).reshape(4, 3, 100)
    permuted = random_permute_coils(values, generator=torch.Generator().manual_seed(9))
    for row in range(len(values)):
        assert {tuple(token.tolist()) for token in values[row]} == {
            tuple(token.tolist()) for token in permuted[row]
        }


class ZeroVelocity(nn.Module):
    def forward(self, tokens: torch.Tensor, time: torch.Tensor, nfp: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(tokens)


def test_online_objective_uses_declared_mixture() -> None:
    data = torch.randn(8, 3, 100)
    loss, terms = online_objective(
        ZeroVelocity(),
        current_data=data,
        improved_data=data,
        improved_weights=torch.ones(8),
        q0_data=data,
        feature_weights=torch.ones(100),
        improvement_fraction=0.10,
        q0_fraction=0.05,
    )
    expected = (
        0.85 * terms["current_loss"]
        + 0.10 * terms["improvement_loss"]
        + 0.05 * terms["q0_loss"]
    )
    assert torch.allclose(loss.detach(), expected)


def test_small_model_matches_previous_online_architecture() -> None:
    from flow_matching.model import CoilFlowTransformer

    model = CoilFlowTransformer(**model_config())
    assert model.parameter_count == 5_761_380


def test_trajectory_weights_are_softmax_like_with_rollout_normalization() -> None:
    weights, reference = trajectory_reward_weights(
        np.asarray([80.0, 70.0, 0.0]),
        np.asarray([2, 2, 1]),
        temperature=10.0,
        epsilon=0.01,
    )
    assert reference == 80.0
    np.testing.assert_allclose(
        weights,
        [(1.0 + 0.01) / 2.0, (np.exp(-1.0) + 0.01) / 2.0, np.exp(-8.0) + 0.01],
    )
    assert weights[0] > weights[1] > weights[2] > 0.0


def test_wilson_interval_and_distribution_stability_guards() -> None:
    lower, upper = wilson_interval(8, 10)
    assert 0.0 < lower < 0.8 < upper < 1.0
    assert not distribution_stable([0.2, 0.201])
    assert distribution_stable([0.2, 0.205, 0.198])
    assert not distribution_stable([0.2, 0.24, 0.21])


def test_native_compaction_removes_nonfinite_json_values() -> None:
    compact = compact_native(
        {
            "status": "no_surface",
            "score": float("nan"),
            "components": {"volume_qs": float("inf"), "coil": None},
            "diagnostics": {"iota_min": float("-inf"), "iota_max": 0.2},
        }
    )
    assert compact == {
        "status": "no_surface",
        "score": 0.0,
        "components": {"volume_qs": None, "coil": None},
        "iota_min": None,
        "iota_max": 0.2,
    }
    json.dumps(compact, allow_nan=False)


def test_adam20_replay_loads_all_21_contiguous_formal_centers(tmp_path: Path) -> None:
    trajectory = tmp_path / "trajectory"
    trajectory.mkdir()
    normalizer = fit_prior_normalizer(teacher_tokens())
    base = teacher_tokens(1)[0].astype(np.float64)
    for step in range(21):
        physical = base.copy()
        physical[0, 0] += 0.01 * step
        payload = {
            "raw": {
                "x": physical[:, :33].tolist(),
                "y": physical[:, 33:66].tolist(),
                "z": physical[:, 66:99].tolist(),
                "current": physical[:, 99].tolist(),
            },
            "original_space_local_gradient_adam": {
                "iteration": step,
                "native_score": {
                    "status": "ok",
                    "score": 40.0 + step,
                    "components": {"volume_qs": 20.0 + step, "coil": 60.0 - step},
                },
            },
        }
        (trajectory / f"step_{step:04d}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    targets, scores, volume_qs, coil, steps = load_adam20_centers(
        tmp_path, normalizer
    )
    assert targets.shape == (21, 3, 100)
    np.testing.assert_array_equal(steps, np.arange(21))
    np.testing.assert_allclose(scores, np.arange(40.0, 61.0))
    np.testing.assert_allclose(volume_qs, np.arange(20.0, 41.0))
    np.testing.assert_allclose(coil, np.arange(60.0, 39.0, -1.0))
    assert not np.array_equal(targets[0], targets[-1])


def test_registered_rl_and_long_horizon_protocols_pin_current_optimizer() -> None:
    root = Path(__file__).resolve().parents[1]
    online = json.loads(
        (root / "evaluation" / "axisflip_prior_distilled_online_adam20_rwcfm_abi11_v1.json").read_text(
            encoding="utf-8"
        )
    )
    long_run = json.loads(
        (root / "evaluation" / "axisflip_v4_representative_adam2000_abi11_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert online["adam20"]["score_abi"] == 11
    assert online["adam20"]["iterations"] == 20
    assert online["adam20"]["directions"] == 64
    assert long_run["optimizer"]["score_abi"] == 11
    assert long_run["optimizer"]["iterations"] == 2000
    assert long_run["optimizer"]["directions"] == 64
    trajectory = json.loads(
        (root / "evaluation" / "axisflip_r012_trajectory_online_rl_r04_abi11_v1.json").read_text(
            encoding="utf-8"
        )
    )
    continuation = json.loads(
        (root / "evaluation" / "axisflip_r012_top2_adam3000_r04_abi11_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert trajectory["online_round"]["replay_capacity_rollouts"] == 512
    assert trajectory["online_round"]["reward"] == {
        "temperature": 7.5,
        "epsilon": 0.01,
        "weight": "(epsilon + exp((clip(score,0,100)-pool_max_score)/temperature)) / rollout_length",
    }
    assert continuation["optimizer"]["new_iterations"] == 3000
    assert continuation["optimizer"]["score_library_sha256"] == trajectory["evaluator"]["library_sha256"]


def test_online_checkpoint_retains_continuous_optimizer_state(tmp_path: Path) -> None:
    from flow_matching.model import CoilFlowTransformer

    config = model_config(width=16, layers=1, heads=4, hidden=32)
    model = CoilFlowTransformer(**config)
    ema = CoilFlowTransformer(**config)
    ema.load_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-5)
    sum(parameter.square().sum() for parameter in model.parameters()).backward()
    optimizer.step()
    path = tmp_path / "round_001.pt"
    save_online_checkpoint(
        path,
        model=model,
        ema=ema,
        optimizer=optimizer,
        normalizer=fit_prior_normalizer(teacher_tokens()),
        checkpoint={"step": 20},
        outer_round=1,
        train_steps=250,
        code_commit="abc123",
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["stage"] == "online_policy"
    assert checkpoint["outer_round"] == 1
    assert checkpoint["step"] == 270
    assert checkpoint["distillation_step"] == 20
    assert checkpoint["online_optimizer"]["state"]


def test_round_summary_builds_weighted_batch_and_replay(tmp_path: Path) -> None:
    manifest = {
        "format": FORMAT,
        "protocol": {"id": PROTOCOL_ID},
        "condition": {"nfp": 8, "n_base_coils": 3},
        "repository": repository_provenance(Path(__file__).resolve().parents[1]),
        "update": {
            "reward_temperature": 7.5,
            "reward_epsilon": 0.01,
            "base_batch_per_gpu": 256,
            "trajectory_microbatch_per_gpu": 256,
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    np.savez(
        tmp_path / "replay_pool.npz",
        targets=np.empty((0, 3, 100), dtype=np.float32),
        scores=np.empty(0, dtype=np.float64),
        volume_qs=np.empty(0, dtype=np.float64),
        coil=np.empty(0, dtype=np.float64),
        rollout_ids=np.empty(0, dtype="U64"),
        steps=np.empty(0, dtype=np.int16),
        rollout_lengths=np.empty(0, dtype=np.int16),
        rollout_valid=np.empty(0, dtype=np.bool_),
    )
    for worker in range(4):
        worker_dir = tmp_path / "rounds" / "round_000" / f"worker_{worker:02d}"
        worker_dir.mkdir(parents=True)
        rng = np.random.default_rng(100 + worker)
        current = rng.normal(size=(16, 3, 100)).astype(np.float32)
        improved = current + 0.01
        valid = np.arange(16) % 2 == 0
        initial = np.arange(worker * 16, worker * 16 + 16, dtype=np.float64)
        best = initial + valid.astype(np.float64)
        statuses = np.where(valid, "ok", "no_axis")
        sample_ids = np.asarray(
            [f"r000_w{worker:02d}_i{index:02d}" for index in range(16)], dtype="U64"
        )
        trajectory_targets = []
        trajectory_scores = []
        trajectory_ids = []
        trajectory_steps = []
        trajectory_lengths = []
        trajectory_valid = []
        for index, sample_id in enumerate(sample_ids):
            length = 21 if valid[index] else 1
            trajectory_targets.append(np.repeat(improved[index : index + 1], length, axis=0))
            trajectory_scores.append(np.linspace(initial[index], best[index], length))
            trajectory_ids.append(np.full(length, sample_id, dtype="U64"))
            trajectory_steps.append(np.arange(length, dtype=np.int16))
            trajectory_lengths.append(np.full(length, length, dtype=np.int16))
            trajectory_valid.append(np.full(length, valid[index], dtype=np.bool_))
        trajectory_scores_array = np.concatenate(trajectory_scores)
        np.savez(
            worker_dir / "batch.npz",
            current=current,
            improved=improved,
            valid=valid,
            initial_scores=initial,
            best_scores=best,
            initial_volume_qs=initial + 10.0,
            initial_coil=initial + 20.0,
            best_volume_qs=best + 10.0,
            best_coil=best + 20.0,
            statuses=statuses,
            sample_ids=sample_ids,
            trajectory_targets=np.concatenate(trajectory_targets),
            trajectory_scores=trajectory_scores_array,
            trajectory_volume_qs=trajectory_scores_array + 10.0,
            trajectory_coil=trajectory_scores_array + 20.0,
            trajectory_ids=np.concatenate(trajectory_ids),
            trajectory_steps=np.concatenate(trajectory_steps),
            trajectory_lengths=np.concatenate(trajectory_lengths),
            trajectory_valid=np.concatenate(trajectory_valid),
        )
        summary = {
            "format": FORMAT,
            "stage": "worker_complete",
            "round": 0,
            "worker": worker,
            "sample_count": 16,
            "flow_wall_s": 1.0,
            "score_wall_s": 2.0,
            "adam20_wall_s": 3.0,
            "batch_file": str((worker_dir / "batch.npz").relative_to(tmp_path)),
        }
        (worker_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    summarize_round(SimpleNamespace(run_root=tmp_path, round_index=0))
    training = np.load(tmp_path / "rounds" / "round_000" / "training_data.npz")
    assert training["current"].shape == (64, 3, 100)
    assert training["targets"].shape == (704, 3, 100)
    assert np.all(training["target_weights"] > 0.0)
    replay = np.load(tmp_path / "replay_pool.npz")
    assert replay["targets"].shape == (704, 3, 100)
    summary = json.loads(
        (tmp_path / "rounds" / "round_000" / "round_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["valid_count"] == 32
    assert summary["valid_rate"] == 0.5
    assert summary["training"]["replay_rollout_count_after_round"] == 64
    assert summary["training"]["trajectory_batch_per_gpu"] == 2816
