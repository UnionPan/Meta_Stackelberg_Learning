from pathlib import Path

from fl_sandbox.config import load_run_config
from fl_sandbox.run.run_benchmark import benchmark_run_configs


def test_rlfl_paper_preset_contains_paper_scale_schedule():
    cfg = load_run_config("fl_sandbox/config/presets/rlfl_paper.yaml")

    assert cfg.protocol.name == "rlfl"
    assert cfg.attacker.type == "rl"
    assert cfg.runtime.rounds == 1000
    assert cfg.fl.num_clients == 100
    assert cfg.fl.num_attackers == 20
    assert cfg.fl.subsample_rate == 0.1
    assert cfg.attacker.rl_distribution_steps == 100
    assert cfg.attacker.rl_attack_start_round == 101
    assert cfg.attacker.rl_policy_train_end_round == 400


def test_benchmark_runner_expands_attack_defense_matrix_from_preset():
    configs = benchmark_run_configs(Path("fl_sandbox/config/presets/defense_suite.yaml"))
    names = [(cfg.attacker.type, cfg.defender.type) for cfg in configs]

    assert ("clean", "fedavg") in names
    assert ("ipm", "krum") in names
    assert ("dba", "fltrust") in names
    assert ("rl", "clipped_median") in names
