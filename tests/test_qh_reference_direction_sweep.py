from scripts.run_qh_reference_direction_sweep_shard import assign_shards, build_configs


def test_sweep_grid_has_all_27_unique_combinations() -> None:
    configs = build_configs()
    combinations = {
        (config["learning_rate"], config["beta1"], config["random_directions"])
        for config in configs
    }

    assert len(configs) == 27
    assert len(combinations) == 27


def test_six_gpu_shards_are_complete_and_balanced() -> None:
    configs = build_configs()
    shards = assign_shards(configs, 6)
    flattened = [config["run_name"] for shard in shards for config in shard]
    loads = [sum(float(config["balance_weight"]) for config in shard) for shard in shards]

    assert sorted(flattened) == sorted(config["run_name"] for config in configs)
    assert [len(shard) for shard in shards] == [4, 4, 4, 5, 5, 5]
    assert max(loads) - min(loads) <= 2.05
