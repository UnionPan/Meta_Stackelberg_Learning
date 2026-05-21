from fl_sandbox.config.schema import RunConfig


def test_run_config_normalizes_rlfl_schedule_for_rl_attacker():
    cfg = RunConfig.from_flat_dict(
        {
            "protocol": "rlfl",
            "attack_type": "rl",
            "warmup_rounds": 5,
            "rl_distribution_steps": None,
            "rl_attack_start_round": None,
            "rl_policy_train_end_round": None,
        }
    )

    assert cfg.attacker.rl_distribution_steps == 5
    assert cfg.attacker.rl_policy_train_end_round == 5
    assert cfg.attacker.rl_attack_start_round == 6


def test_run_config_resolves_default_attacker_count():
    clean = RunConfig.from_flat_dict({"attack_type": "clean", "num_attackers": None})
    malicious = RunConfig.from_flat_dict({"attack_type": "ipm", "num_attackers": None})
    explicit = RunConfig.from_flat_dict({"attack_type": "ipm", "num_attackers": 7})

    assert clean.resolved_num_attackers() == 0
    assert malicious.resolved_num_attackers() == 2
    assert explicit.resolved_num_attackers() == 7


def test_run_config_has_runner_sections_instead_of_flat_sandbox_config():
    cfg = RunConfig.from_flat_dict(
        {
            "dataset": "fmnist",
            "attack_type": "rl",
            "defense_type": "clipped_median",
            "num_clients": 12,
            "num_attackers": 3,
            "rounds": 9,
            "device": "cpu",
            "base_class": 2,
            "target_class": 8,
            "clipped_median_norm": 1.5,
        }
    )

    assert cfg.data.dataset == "fmnist"
    assert cfg.attacker.type == "rl"
    assert cfg.defender.type == "clipped_median"
    assert cfg.fl.num_clients == 12
    assert cfg.resolved_num_attackers() == 3
    assert cfg.runtime.rounds == 9
    assert cfg.runtime.device == "cpu"
    assert cfg.attacker.base_class == 2
    assert cfg.attacker.target_class == 8
    assert cfg.defender.clipped_median_norm == 1.5


def test_sandbox_config_is_removed_from_public_runner_api():
    import fl_sandbox.core.fl_runner as core_runner
    import fl_sandbox.federation.runner as federation_runner

    assert not hasattr(core_runner, "SandboxConfig")
    assert not hasattr(federation_runner, "SandboxConfig")
