"""Tests for paper-aligned Meta-SG model-poisoning pretraining wiring."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import torch

from meta_sg.learning.best_response import AttackerBestResponse
from meta_sg.learning.config import MetaSGConfig, TD3Config
from meta_sg.learning.replay_buffer import ReplayBuffer
from meta_sg.learning.task_runner import _action_diagnostics
from meta_sg.learning.task_runner import AttackTaskRunner
from meta_sg.learning.td3 import TD3Agent
from meta_sg.games.trajectory import Trajectory, Transition
from meta_sg.simulation.stub import StubCoordinator
from meta_sg.strategies.types import ATTACK_DOMAIN, DefenseDecision


class ConstantPostMetricCoordinator(StubCoordinator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluate_calls = 0

    def evaluate_weights(self, weights):
        self.evaluate_calls += 1
        return {
            "clean_acc": 0.87,
            "clean_loss": 0.13,
            "backdoor_acc": 0.66,
        }


def test_attack_task_runner_threads_model_poisoning_reward_config_and_evaluator():
    created = []

    def factory():
        coord = ConstantPostMetricCoordinator(num_clients=6, num_attackers=1, seed=0)
        created.append(coord)
        return coord

    obs_dim = 1290
    act_dim = 3
    td3_cfg = TD3Config(hidden_dim=8, batch_size=2, buffer_capacity=16, warmup_steps=0)
    meta_cfg = MetaSGConfig(
        T=1,
        K=1,
        H_mnist=1,
        l=0,
        N_A=0,
        post_br_defender_updates=0,
        lambda_bd=0.0,
        reward_mode="accuracy",
        eval_every=1,
        warmup_steps=0,
    )
    attacker_buffers = {
        "ipm": ReplayBuffer(td3_cfg.buffer_capacity, obs_dim, act_dim),
    }
    runner = AttackTaskRunner(
        coordinator_factory=factory,
        td3_config=td3_cfg,
        meta_config=meta_cfg,
        obs_dim=obs_dim,
        act_dim=act_dim,
        attacker_agents={},
        attacker_buffers=attacker_buffers,
        best_response=AttackerBestResponse({}, {}, n_a=0),
    )
    defender = TD3Agent(obs_dim, act_dim, td3_cfg, torch.device("cpu"))

    result = runner.run(ATTACK_DOMAIN["ipm"], defender, seed_base=123)

    assert len(created) == 1
    assert created[0].evaluate_calls == 1
    assert result.mean_defender_reward == pytest.approx(0.87)
    assert result.mean_attacker_reward == pytest.approx(-0.87)
    assert result.diagnostics["clean_acc"] == pytest.approx(0.87)
    assert result.diagnostics["backdoor_acc"] == pytest.approx(0.66)


def test_pretraining_script_domain_contains_only_model_poisoning_attacks():
    from meta_sg.scripts.run_meta_sg_pretraining import poisoning_attack_domain

    domain = poisoning_attack_domain()

    assert [attack.name for attack in domain] == ["ipm", "lmp", "rl"]
    assert all(attack.objective == "untargeted" for attack in domain)


def test_pretraining_script_mixed_domain_combines_global_and_backdoor_attacks():
    from meta_sg.scripts.run_meta_sg_pretraining import attack_domain_from_name

    domain = attack_domain_from_name("mixed")

    assert [attack.name for attack in domain] == [
        "ipm",
        "lmp",
        "rl",
        "bfl",
        "dba",
        "rl_backdoor",
    ]


def test_pretraining_script_clean_mixed_domain_includes_clean_global_and_backdoor_attacks():
    from meta_sg.scripts.run_meta_sg_pretraining import attack_domain_from_name

    domain = attack_domain_from_name("clean_mixed")

    assert [attack.name for attack in domain] == [
        "clean",
        "ipm",
        "lmp",
        "rl",
        "bfl",
        "dba",
        "rl_backdoor",
    ]


def test_pretraining_script_clean_global_backdoor_mixed_domain_has_each_task_once():
    from meta_sg.scripts.run_meta_sg_pretraining import attack_domain_from_name

    domain = attack_domain_from_name("clean_global_backdoor_mixed")

    assert [attack.name for attack in domain] == [
        "clean",
        "ipm",
        "lmp",
        "rl",
        "bfl",
        "dba",
        "rl_backdoor",
        "mixed_backdoor",
    ]
    assert [attack.objective for attack in domain] == [
        "clean",
        "untargeted",
        "untargeted",
        "untargeted",
        "targeted",
        "targeted",
        "targeted",
        "targeted",
    ]


def test_pretraining_script_accepts_clean_global_backdoor_mixed_domain():
    from meta_sg.scripts.run_meta_sg_pretraining import parse_args

    args = parse_args(["--attack-domain", "clean_global_backdoor_mixed", "--K", "8"])

    assert args.attack_domain == "clean_global_backdoor_mixed"
    assert args.K == 8


def test_pretraining_script_clean_backdoor_domain_includes_clean_and_backdoor_attacks():
    from meta_sg.scripts.run_meta_sg_pretraining import attack_domain_from_name

    domain = attack_domain_from_name("clean_backdoor")

    assert [attack.name for attack in domain] == ["clean", "bfl", "dba", "rl_backdoor"]
    assert domain[0].objective == "clean"


def test_attack_task_runner_uses_no_attack_strategy_for_clean_task():
    def factory():
        return StubCoordinator(num_clients=6, num_attackers=1, seed=0)

    td3_cfg = TD3Config(hidden_dim=8, batch_size=2, buffer_capacity=16, warmup_steps=0)
    meta_cfg = MetaSGConfig(T=1, K=1, H_mnist=1, l=0, N_A=0, post_br_defender_updates=0)
    runner = AttackTaskRunner(
        coordinator_factory=factory,
        td3_config=td3_cfg,
        meta_config=meta_cfg,
        obs_dim=1290,
        act_dim=3,
        attacker_agents={},
        attacker_buffers={},
        best_response=AttackerBestResponse({}, {}, n_a=0),
    )

    assert runner._build_attack_strategy(ATTACK_DOMAIN["clean"]) is None


def test_pretraining_script_accepts_model_aware_neuroclip_and_log_epsilon_args():
    from meta_sg.scripts.run_meta_sg_pretraining import parse_args

    args = parse_args(
        [
            "--attack-domain",
            "clean_mixed",
            "--defender-third-action",
            "both",
            "--post-defense-mode",
            "model_aware_neuroclip",
            "--neuroclip-eps-min",
            "0.1",
            "--neuroclip-eps-max",
            "10.0",
            "--neuroclip-log-scale",
        ]
    )

    assert args.attack_domain == "clean_mixed"
    assert args.defender_third_action == "both"
    assert args.post_defense_mode == "model_aware_neuroclip"
    assert args.neuroclip_eps_min == pytest.approx(0.1)
    assert args.neuroclip_eps_max == pytest.approx(10.0)
    assert args.neuroclip_log_scale is True


def test_attack_task_runner_uses_model_aware_neuroclip_post_training_evaluator(monkeypatch):
    import meta_sg.learning.task_runner as task_runner

    calls = []

    class ModelAwareCoordinator(StubCoordinator):
        def __init__(self):
            super().__init__(num_clients=6, num_attackers=1, seed=0)
            self.runner = SimpleNamespace(model="base-model")

        def evaluate_weights(self, weights):
            return {
                "clean_acc": 0.1,
                "clean_loss": 0.9,
                "backdoor_acc": 0.8,
            }

        def evaluate_model(self, model, weights):
            return {
                "clean_acc": 0.93 if model == "wrapped-base-model" else 0.0,
                "clean_loss": 0.07,
                "backdoor_acc": 0.04,
            }

    def fake_apply_post_defense(model, defense_type, param, **kwargs):
        calls.append((model, defense_type, param, kwargs))
        return f"wrapped-{model}"

    monkeypatch.setattr(task_runner, "apply_post_defense", fake_apply_post_defense)

    td3_cfg = TD3Config(hidden_dim=8, batch_size=2, buffer_capacity=16, warmup_steps=0)
    meta_cfg = MetaSGConfig(
        T=1,
        K=1,
        H_mnist=1,
        l=0,
        N_A=0,
        post_br_defender_updates=0,
        lambda_bd=1.0,
        reward_mode="accuracy",
        eval_every=1,
        warmup_steps=0,
        defender_third_action="both",
        post_defense_mode="model_aware_neuroclip",
        eps_min=0.1,
        eps_max=10.0,
        eps_log_scale=True,
    )
    runner = AttackTaskRunner(
        coordinator_factory=lambda **_: ModelAwareCoordinator(),
        td3_config=td3_cfg,
        meta_config=meta_cfg,
        obs_dim=1290,
        act_dim=4,
        attacker_agents={},
        attacker_buffers={},
        best_response=AttackerBestResponse({}, {}, n_a=0),
    )
    defender = TD3Agent(1290, 4, td3_cfg, torch.device("cpu"))

    result = runner.run(ATTACK_DOMAIN["clean"], defender, seed_base=321)

    assert calls
    assert calls[0][0] == "base-model"
    assert calls[0][1] == "neuroclip"
    assert 0.1 <= calls[0][2] <= 10.0
    assert result.mean_defender_reward == pytest.approx(0.89)
    assert result.diagnostics["clean_acc"] == pytest.approx(0.93)
    assert result.diagnostics["backdoor_acc"] == pytest.approx(0.04)


def test_attack_task_runner_can_collect_multiple_support_episodes(monkeypatch):
    update_calls = 0
    original_update = TD3Agent.update

    def counting_update(self, *args, **kwargs):
        nonlocal update_calls
        update_calls += 1
        return original_update(self, *args, **kwargs)

    monkeypatch.setattr(TD3Agent, "update", counting_update)

    def factory():
        return StubCoordinator(num_clients=6, num_attackers=1, seed=0)

    obs_dim = 1290
    act_dim = 3
    td3_cfg = TD3Config(hidden_dim=8, batch_size=1, buffer_capacity=16, warmup_steps=0)
    meta_cfg = MetaSGConfig(
        T=1,
        K=1,
        H_mnist=1,
        l=2,
        N_A=0,
        post_br_defender_updates=0,
        support_episodes=3,
        warmup_steps=0,
    )
    runner = AttackTaskRunner(
        coordinator_factory=factory,
        td3_config=td3_cfg,
        meta_config=meta_cfg,
        obs_dim=obs_dim,
        act_dim=act_dim,
        attacker_agents={},
        attacker_buffers={"clean": ReplayBuffer(td3_cfg.buffer_capacity, obs_dim, act_dim)},
        best_response=AttackerBestResponse({}, {}, n_a=0),
    )
    defender = TD3Agent(obs_dim, act_dim, td3_cfg, torch.device("cpu"))

    result = runner.run(ATTACK_DOMAIN["clean"], defender, seed_base=123)

    assert result.trajectories_collected == 3
    assert result.transitions_collected == 3
    assert result.diagnostics["support_episodes"] == 3
    assert result.diagnostics["support_update_calls"] == 6
    assert update_calls == 6


def test_attack_task_runner_can_log_query_diagnostics_without_gating_reptile():
    def factory():
        return StubCoordinator(num_clients=6, num_attackers=1, seed=0)

    obs_dim = 1290
    act_dim = 3
    td3_cfg = TD3Config(hidden_dim=8, batch_size=1, buffer_capacity=16, warmup_steps=0)
    meta_cfg = MetaSGConfig(
        T=1,
        K=1,
        H_mnist=1,
        l=0,
        N_A=0,
        post_br_defender_updates=0,
        meta_objective="reptile",
        query_diagnostics_horizon=1,
        query_accept_margin=999.0,
        warmup_steps=0,
    )
    runner = AttackTaskRunner(
        coordinator_factory=factory,
        td3_config=td3_cfg,
        meta_config=meta_cfg,
        obs_dim=obs_dim,
        act_dim=act_dim,
        attacker_agents={},
        attacker_buffers={"clean": ReplayBuffer(td3_cfg.buffer_capacity, obs_dim, act_dim)},
        best_response=AttackerBestResponse({}, {}, n_a=0),
    )
    defender = TD3Agent(obs_dim, act_dim, td3_cfg, torch.device("cpu"))

    result = runner.run(ATTACK_DOMAIN["clean"], defender, seed_base=123)

    assert not np.isnan(result.query_gain)
    assert result.query_gain_accepted is False
    assert result.diagnostics["query_diagnostics_only"] == 1.0

    from meta_sg.learning.meta_sg_trainer import _meta_update_adapted_params, _query_metric_values

    params = _meta_update_adapted_params(
        [result],
        meta_objective="reptile",
        query_accept_margin=999.0,
    )
    metrics = _query_metric_values([result])

    assert params[0] is result.adapted_params
    assert metrics["query_gain_accept_rate"] == pytest.approx(0.0)
    assert metrics["query_attack_clean_gain_accepted"] == pytest.approx(0.0)


def test_meta_sg_trainer_uses_iid_attack_sampling_by_default():
    from meta_sg.learning.meta_sg_trainer import MetaSGTrainer
    from meta_sg.scripts.run_meta_sg_pretraining import poisoning_attack_domain

    trainer = object.__new__(MetaSGTrainer)
    trainer.attack_domain = poisoning_attack_domain()
    trainer.meta_cfg = MetaSGConfig(task_sampler="iid")
    np.random.seed(0)

    batch = trainer._sample_attack_types(3)

    names = [attack.name for attack in batch]
    assert names == ["ipm", "lmp", "ipm"]
    assert "rl" not in names


def test_meta_sg_trainer_can_stratify_attack_sampling_when_batch_covers_domain():
    from meta_sg.learning.meta_sg_trainer import MetaSGTrainer
    from meta_sg.scripts.run_meta_sg_pretraining import poisoning_attack_domain

    trainer = object.__new__(MetaSGTrainer)
    trainer.attack_domain = poisoning_attack_domain()
    trainer.meta_cfg = MetaSGConfig(task_sampler="stratified")
    np.random.seed(0)

    batch = trainer._sample_attack_types(3)

    assert sorted(attack.name for attack in batch) == ["ipm", "lmp", "rl"]


def test_pretraining_script_meta_config_uses_requested_horizon_for_each_dataset():
    from meta_sg.scripts.run_meta_sg_pretraining import build_meta_config, defender_action_dim, parse_args

    mnist_args = parse_args(["--backend", "stub", "--dataset", "mnist", "--H", "7"])
    cifar_args = parse_args(["--backend", "stub", "--dataset", "cifar10", "--H", "9"])
    server_lr_args = parse_args(
        ["--backend", "stub", "--dataset", "mnist", "--H", "7", "--defender-third-action", "server_lr"]
    )
    both_args = parse_args(
        [
            "--backend",
            "stub",
            "--dataset",
            "mnist",
            "--H",
            "7",
            "--attack-domain",
            "mixed",
            "--defender-third-action",
            "both",
        ]
    )
    stratified_args = parse_args(
        ["--backend", "stub", "--dataset", "mnist", "--H", "7", "--task-sampler", "stratified"]
    )
    support_args = parse_args(
        ["--backend", "stub", "--dataset", "mnist", "--H", "7", "--support-episodes", "3"]
    )
    diagnostic_args = parse_args(
        [
            "--backend",
            "stub",
            "--dataset",
            "mnist",
            "--H",
            "7",
            "--task-warmup-steps",
            "5",
            "--query-diagnostics-horizon",
            "50",
        ]
    )
    global_backdoor_args = parse_args(
        [
            "--backend",
            "fl_sandbox",
            "--attack-domain",
            "clean_global_backdoor_mixed",
            "--K",
            "8",
        ]
    )

    assert build_meta_config(mnist_args).H == 7
    assert build_meta_config(cifar_args).H == 9
    assert build_meta_config(mnist_args).support_episodes == 1
    assert build_meta_config(support_args).support_episodes == 3
    assert build_meta_config(diagnostic_args).warmup_steps == 5
    assert build_meta_config(diagnostic_args).query_diagnostics_horizon == 50
    assert build_meta_config(mnist_args).lambda_bd == pytest.approx(0.0)
    assert build_meta_config(mnist_args).reward_mode == "accuracy"
    assert build_meta_config(mnist_args).task_sampler == "iid"
    assert build_meta_config(stratified_args).task_sampler == "stratified"
    assert build_meta_config(server_lr_args).defender_third_action == "server_lr"
    assert build_meta_config(both_args).lambda_bd == pytest.approx(1.0)
    assert build_meta_config(both_args).native_sandbox_attacks is False
    assert build_meta_config(global_backdoor_args).lambda_bd == pytest.approx(1.0)
    assert build_meta_config(global_backdoor_args).native_sandbox_attacks is True
    assert defender_action_dim(server_lr_args) == 3
    assert defender_action_dim(both_args) == 4


def test_pretraining_script_defaults_to_paper_reptile_sampling_values():
    from meta_sg.scripts.run_meta_sg_pretraining import build_meta_config, parse_args

    args = parse_args(["--backend", "stub"])
    cfg = build_meta_config(args)

    assert cfg.K == 10
    assert cfg.meta_update_step == pytest.approx(1.0)
    assert cfg.task_sampler == "iid"
    assert args.device == "auto"


def test_pretraining_script_threads_server_lr_bounds_to_meta_config():
    from meta_sg.scripts.run_meta_sg_pretraining import build_meta_config, parse_args

    args = parse_args(
        [
            "--backend",
            "stub",
            "--defender-third-action",
            "both",
            "--server-lr-min",
            "0.6",
            "--server-lr-max",
            "1.0",
        ]
    )

    cfg = build_meta_config(args)

    assert cfg.server_lr_min == pytest.approx(0.6)
    assert cfg.server_lr_max == pytest.approx(1.0)


def test_pretraining_script_threads_server_lr_penalty_to_meta_config():
    from meta_sg.scripts.run_meta_sg_pretraining import build_meta_config, parse_args

    args = parse_args(
        [
            "--backend",
            "stub",
            "--defender-third-action",
            "both",
            "--server-lr-penalty-weight",
            "0.5",
        ]
    )

    cfg = build_meta_config(args)

    assert cfg.server_lr_penalty_weight == pytest.approx(0.5)


def test_pretraining_script_threads_query_gated_meta_objective_to_meta_config():
    from meta_sg.scripts.run_meta_sg_pretraining import build_meta_config, parse_args

    args = parse_args(
        [
            "--backend",
            "stub",
            "--meta-objective",
            "query_gated_reptile",
            "--query-horizon",
            "7",
            "--query-seed-offset",
            "12345",
            "--query-accept-margin",
            "0.002",
        ]
    )

    cfg = build_meta_config(args)

    assert cfg.meta_objective == "query_gated_reptile"
    assert cfg.query_horizon == 7
    assert cfg.query_seed_offset == 12345
    assert cfg.query_accept_margin == pytest.approx(0.002)


def test_pretraining_script_threads_clean_aware_query_gate_to_meta_config():
    from meta_sg.scripts.run_meta_sg_pretraining import build_meta_config, parse_args

    args = parse_args(
        [
            "--backend",
            "stub",
            "--meta-objective",
            "query_gated_reptile",
            "--query-clean-floor",
            "0.89",
            "--query-clean-drop-tolerance",
            "0.01",
        ]
    )

    cfg = build_meta_config(args)

    assert cfg.query_clean_floor == pytest.approx(0.89)
    assert cfg.query_clean_drop_tolerance == pytest.approx(0.01)


def test_pretraining_script_threads_backdoor_query_gate_to_meta_config():
    from meta_sg.scripts.run_meta_sg_pretraining import build_meta_config, parse_args

    args = parse_args(
        [
            "--backend",
            "stub",
            "--meta-objective",
            "query_gated_reptile",
            "--query-backdoor-ceiling",
            "0.05",
            "--query-backdoor-increase-tolerance",
            "0.01",
            "--query-backdoor-improvement-margin",
            "0.03",
        ]
    )

    cfg = build_meta_config(args)

    assert cfg.query_backdoor_ceiling == pytest.approx(0.05)
    assert cfg.query_backdoor_increase_tolerance == pytest.approx(0.01)
    assert cfg.query_backdoor_improvement_margin == pytest.approx(0.03)


def test_pretraining_script_threads_targeted_asr_query_objective_to_meta_config():
    from meta_sg.scripts.run_meta_sg_pretraining import build_meta_config, parse_args

    args = parse_args(
        [
            "--backend",
            "stub",
            "--meta-objective",
            "query_targeted_reptile",
            "--query-targeted-asr-reduction-margin",
            "0.04",
            "--query-targeted-min-base-backdoor",
            "0.50",
        ]
    )

    cfg = build_meta_config(args)

    assert cfg.meta_objective == "query_targeted_reptile"
    assert cfg.query_targeted_asr_reduction_margin == pytest.approx(0.04)
    assert cfg.query_targeted_min_base_backdoor == pytest.approx(0.50)


def test_pretraining_script_can_append_attack_context_names_to_meta_config():
    from meta_sg.scripts.run_meta_sg_pretraining import build_meta_config, parse_args

    args = parse_args(["--attack-domain", "mixed", "--attack-context"])

    cfg = build_meta_config(args)

    assert cfg.attack_context_names == ("ipm", "lmp", "rl", "bfl", "dba", "rl_backdoor")


def test_pretraining_script_can_append_clean_backdoor_attack_context_names():
    from meta_sg.scripts.run_meta_sg_pretraining import build_meta_config, parse_args

    args = parse_args(["--attack-domain", "clean_backdoor", "--attack-context"])

    cfg = build_meta_config(args)

    assert cfg.lambda_bd == pytest.approx(1.0)
    assert cfg.attack_context_names == ("clean", "bfl", "dba", "rl_backdoor")


def test_pretraining_script_supports_clean_global_attack_domain():
    from meta_sg.scripts.run_meta_sg_pretraining import (
        attack_domain_from_name,
        build_meta_config,
        parse_args,
    )

    args = parse_args(["--attack-domain", "clean_global", "--attack-context"])

    cfg = build_meta_config(args)
    attack_domain = attack_domain_from_name(args.attack_domain)

    assert [attack.name for attack in attack_domain] == ["clean", "ipm", "lmp", "rl"]
    assert cfg.lambda_bd == pytest.approx(0.0)
    assert cfg.attack_context_names == ("clean", "ipm", "lmp", "rl")


def test_pretraining_sandbox_config_disables_attackers_for_clean_task():
    from meta_sg.scripts.run_meta_sg_pretraining import build_sandbox_config, parse_args

    args = parse_args(["--num-attackers", "4", "--H", "1"])

    cfg = build_sandbox_config(args, attack_type=ATTACK_DOMAIN["clean"], horizon=1, seed=0)

    assert cfg.attacker.type == "clean"
    assert cfg.resolved_num_attackers() == 0


def test_direct_eval_script_threads_server_lr_bounds_to_bsmg_config():
    from meta_sg.scripts.evaluate_meta_sg_direct import _bsmg_config, parse_args

    args = parse_args(
        [
            "--checkpoint",
            "dummy.pt",
            "--output-json",
            "out.json",
            "--defender-third-action",
            "both",
            "--server-lr-min",
            "0.6",
            "--server-lr-max",
            "1.0",
        ]
    )
    cfg = _bsmg_config(args)

    assert cfg.server_lr_min == pytest.approx(0.6)
    assert cfg.server_lr_max == pytest.approx(1.0)


def test_direct_eval_script_threads_server_lr_penalty_to_bsmg_config():
    from meta_sg.scripts.evaluate_meta_sg_direct import _bsmg_config, parse_args

    args = parse_args(
        [
            "--checkpoint",
            "dummy.pt",
            "--output-json",
            "out.json",
            "--defender-third-action",
            "both",
            "--server-lr-penalty-weight",
            "0.5",
        ]
    )
    cfg = _bsmg_config(args)

    assert cfg.server_lr_penalty_weight == pytest.approx(0.5)


def test_direct_eval_records_requested_horizon_round_metrics_and_summary(monkeypatch):
    import meta_sg.scripts.evaluate_meta_sg_direct as direct_eval

    args = direct_eval.parse_args(
        [
            "--checkpoint",
            "dummy.pt",
            "--output-json",
            "out.json",
            "--H",
            "7",
            "--lambda-bd",
            "2.0",
            "--defender-third-action",
            "both",
        ]
    )
    scenario = direct_eval._scenarios(args)[0]

    class FakeEnv:
        def __init__(self):
            self.round = 0

        def reset(self, seed):
            del seed
            self.round = 0
            return np.zeros(3, dtype=np.float32)

        def step(self, defender_action, attacker_action):
            del defender_action, attacker_action
            self.round += 1
            info = {
                "clean_acc": 0.80 + 0.01 * self.round,
                "backdoor_acc": 0.10 - 0.01 * self.round,
                "server_lr_penalty": 0.001 * self.round,
                "defense_decision": DefenseDecision(
                    norm_bound_alpha=2.0 + self.round,
                    trimmed_mean_beta=0.1,
                    neuroclip_epsilon=4.0,
                    server_lr=0.8,
                ),
                "non_scalar_payload": {"must": "not leak"},
            }
            return (
                np.full(3, self.round, dtype=np.float32),
                0.5 + 0.01 * self.round,
                -0.5 - 0.01 * self.round,
                False,
                info,
            )

    class FakeDefender:
        def reset(self):
            return None

        def get_action(self, obs, noise=0.0):
            del obs, noise
            return np.zeros(4, dtype=np.float32)

    monkeypatch.setattr(direct_eval, "_make_env", lambda *args, **kwargs: FakeEnv())

    record = direct_eval._evaluate_scenario_at(
        args,
        FakeDefender(),
        scenario,
        seed=123,
        horizon=3,
    )

    assert record["horizon"] == 3
    assert len(record["round_metrics"]) == 3
    assert record["round_metrics"][-1]["round"] == 3
    assert record["round_metrics"][-1]["clean_acc"] == pytest.approx(0.83)
    assert "non_scalar_payload" not in record["round_metrics"][-1]
    json.dumps(record["round_metrics"], allow_nan=False)

    summary = direct_eval.summarize_evaluation_records(
        [record],
        checkpoint="final",
        master_seed=42,
        evaluation_seed=10042,
    )
    scenario_summary = summary["scenarios"][record["scenario"]]
    assert summary["single_seed"] is True
    assert summary["confidence_interval"] is None
    assert scenario_summary["rounds"] == 3
    assert scenario_summary["metrics"]["clean_acc"]["final"] == pytest.approx(0.83)
    assert scenario_summary["metrics"]["clean_acc"]["worst_round"] == 1
    json.dumps(summary, allow_nan=False)


def test_direct_eval_script_accepts_model_aware_neuroclip_overlay_args():
    from meta_sg.scripts.evaluate_meta_sg_direct import parse_args

    args = parse_args(
        [
            "--checkpoint",
            "dummy.pt",
            "--output-json",
            "out.json",
            "--post-defense-mode",
            "model_aware_neuroclip",
            "--fixed-neuroclip-epsilon",
            "0.075",
        ]
    )

    assert args.post_defense_mode == "model_aware_neuroclip"
    assert args.fixed_neuroclip_epsilon == pytest.approx(0.075)


def test_direct_eval_model_aware_neuroclip_evaluator_wraps_model_without_pruning(monkeypatch):
    import meta_sg.scripts.evaluate_meta_sg_direct as direct_eval

    calls = []
    coordinator = SimpleNamespace(
        runner=SimpleNamespace(model="base-model"),
        evaluate_weights=lambda weights: {
            "clean_acc": 0.1,
            "clean_loss": 0.9,
            "backdoor_acc": 0.8,
        },
        evaluate_model=lambda model, weights: {
            "clean_acc": 0.91 if model == "neuroclip-model" and weights == ["w"] else 0.0,
            "clean_loss": 0.09,
            "backdoor_acc": 0.02,
        },
    )

    def fake_apply_post_defense(model, defense_type, param, **kwargs):
        calls.append((model, defense_type, param, kwargs))
        return "neuroclip-model"

    monkeypatch.setattr(direct_eval, "apply_post_defense", fake_apply_post_defense)
    args = direct_eval.parse_args(
        [
            "--checkpoint",
            "dummy.pt",
            "--output-json",
            "out.json",
            "--post-defense-mode",
            "model_aware_neuroclip",
            "--fixed-neuroclip-epsilon",
            "0.075",
        ]
    )
    evaluator = direct_eval._post_training_evaluator_for_args(args, coordinator)

    metrics = evaluator(
        ["w"],
        DefenseDecision(
            norm_bound_alpha=1.0,
            trimmed_mean_beta=0.2,
            neuroclip_epsilon=None,
            server_lr=0.8,
        ),
    )

    assert calls == [("base-model", "neuroclip", pytest.approx(0.075), {})]
    assert metrics == {
        "clean_acc": pytest.approx(0.91),
        "clean_loss": pytest.approx(0.09),
        "backdoor_acc": pytest.approx(0.02),
    }


def test_direct_eval_model_aware_neuroclip_evaluator_uses_latest_runner_model(monkeypatch):
    import meta_sg.scripts.evaluate_meta_sg_direct as direct_eval

    runner = SimpleNamespace(model="initial-model")
    calls = []
    coordinator = SimpleNamespace(
        runner=runner,
        evaluate_model=lambda model, weights: {
            "clean_acc": 0.9 if model == "wrapped-latest-model" else 0.0,
            "clean_loss": 0.1,
            "backdoor_acc": 0.0,
        },
    )

    def fake_apply_post_defense(model, defense_type, param, **kwargs):
        calls.append(model)
        return f"wrapped-{model}"

    monkeypatch.setattr(direct_eval, "apply_post_defense", fake_apply_post_defense)
    args = direct_eval.parse_args(
        [
            "--checkpoint",
            "dummy.pt",
            "--output-json",
            "out.json",
            "--post-defense-mode",
            "model_aware_neuroclip",
            "--fixed-neuroclip-epsilon",
            "1000000.0",
        ]
    )
    evaluator = direct_eval._post_training_evaluator_for_args(args, coordinator)
    runner.model = "latest-model"

    metrics = evaluator(
        ["w"],
        DefenseDecision(norm_bound_alpha=1.0, trimmed_mean_beta=0.2),
    )

    assert calls == ["latest-model"]
    assert metrics["clean_acc"] == pytest.approx(0.9)


def test_direct_eval_script_threads_td3_adaptation_batch_and_warmup_options():
    from meta_sg.scripts.evaluate_meta_sg_direct import parse_args

    args = parse_args(
        [
            "--checkpoint",
            "dummy.pt",
            "--output-json",
            "out.json",
            "--few-shot",
            "--few-shot-method",
            "td3",
            "--adaptation-batch-size",
            "32",
            "--adaptation-warmup-steps",
            "5",
        ]
    )

    assert args.adaptation_batch_size == 32
    assert args.adaptation_warmup_steps == 5


def test_direct_eval_script_can_append_attack_context_names_to_bsmg_config():
    from meta_sg.scripts.evaluate_meta_sg_direct import _bsmg_config, parse_args

    args = parse_args(
        [
            "--checkpoint",
            "dummy.pt",
            "--output-json",
            "out.json",
            "--scenario-set",
            "mixed",
            "--attack-context",
        ]
    )

    cfg = _bsmg_config(args)

    assert cfg.attack_context_names == ("ipm", "lmp", "rl", "bfl", "dba", "rl_backdoor")


def test_direct_eval_script_accepts_scenario_filter_args():
    from meta_sg.scripts.evaluate_meta_sg_direct import parse_args

    args = parse_args(
        [
            "--checkpoint",
            "dummy.pt",
            "--output-json",
            "out.json",
            "--scenario-set",
            "mixed",
            "--scenario-filter",
            "bfl,dba,rl_backdoor",
        ]
    )

    assert args.scenario_filter == "bfl,dba,rl_backdoor"


def test_direct_eval_scenario_filter_keeps_mixed_attack_context_names():
    from meta_sg.scripts.evaluate_meta_sg_direct import _attack_context_names, _scenarios, parse_args

    args = parse_args(
        [
            "--checkpoint",
            "dummy.pt",
            "--output-json",
            "out.json",
            "--scenario-set",
            "mixed",
            "--attack-context",
            "--scenario-filter",
            "bfl,dba,rl_backdoor",
        ]
    )

    assert [scenario.name for scenario in _scenarios(args)] == ["bfl", "dba", "rl_backdoor"]
    assert _attack_context_names(args) == ("ipm", "lmp", "rl", "bfl", "dba", "rl_backdoor")


def test_direct_eval_script_accepts_guarded_few_shot_selection_args():
    from meta_sg.scripts.evaluate_meta_sg_direct import parse_args

    args = parse_args(
        [
            "--checkpoint",
            "dummy.pt",
            "--output-json",
            "out.json",
            "--few-shot",
            "--few-shot-selection",
            "guarded",
            "--selection-margin",
            "0.002",
            "--selection-horizon",
            "7",
            "--selection-seed-offset",
            "30000",
        ]
    )

    assert args.few_shot_selection == "guarded"
    assert args.selection_margin == pytest.approx(0.002)
    assert args.selection_horizon == 7
    assert args.selection_seed_offset == 30000


def test_direct_eval_script_accepts_sac_few_shot_args():
    from meta_sg.scripts.evaluate_meta_sg_direct import parse_args

    args = parse_args(
        [
            "--checkpoint",
            "dummy.pt",
            "--output-json",
            "out.json",
            "--few-shot",
            "--few-shot-method",
            "sac",
            "--sac-alpha",
            "0.1",
            "--sac-conditioned-sigma",
        ]
    )

    assert args.few_shot_method == "sac"
    assert args.sac_alpha == pytest.approx(0.1)
    assert args.sac_conditioned_sigma is True


def test_direct_eval_script_accepts_cem_offset_few_shot_args():
    from meta_sg.scripts.evaluate_meta_sg_direct import parse_args

    args = parse_args(
        [
            "--checkpoint",
            "dummy.pt",
            "--output-json",
            "out.json",
            "--few-shot",
            "--few-shot-method",
            "cem_offset",
            "--cem-iterations",
            "3",
            "--cem-population",
            "12",
            "--cem-elites",
            "4",
            "--cem-init-sigma",
            "0.15",
            "--cem-min-sigma",
            "0.03",
            "--cem-offset-bound",
            "0.5",
        ]
    )

    assert args.few_shot_method == "cem_offset"
    assert args.cem_iterations == 3
    assert args.cem_population == 12
    assert args.cem_elites == 4
    assert args.cem_init_sigma == pytest.approx(0.15)
    assert args.cem_min_sigma == pytest.approx(0.03)
    assert args.cem_offset_bound == pytest.approx(0.5)


def test_direct_eval_script_accepts_beta_offset_few_shot_args():
    from meta_sg.scripts.evaluate_meta_sg_direct import parse_args

    args = parse_args(
        [
            "--checkpoint",
            "dummy.pt",
            "--output-json",
            "out.json",
            "--few-shot",
            "--few-shot-method",
            "beta_offset",
            "--beta-offset-step",
            "0.25",
            "--beta-offset-max-steps",
            "2",
            "--beta-offset-asr-reduction-margin",
            "0.01",
            "--beta-offset-clean-floor",
            "0.89",
            "--beta-offset-clean-drop-tolerance",
            "0.02",
            "--beta-offset-deployment-beta-ceiling",
            "0.35",
            "--beta-offset-start-round",
            "10",
            "--beta-offset-end-round",
            "30",
        ]
    )

    assert args.few_shot_method == "beta_offset"
    assert args.beta_offset_step == pytest.approx(0.25)
    assert args.beta_offset_max_steps == 2
    assert args.beta_offset_asr_reduction_margin == pytest.approx(0.01)
    assert args.beta_offset_clean_floor == pytest.approx(0.89)
    assert args.beta_offset_clean_drop_tolerance == pytest.approx(0.02)
    assert args.beta_offset_deployment_beta_ceiling == pytest.approx(0.35)
    assert args.beta_offset_start_round == 10
    assert args.beta_offset_end_round == 30


def test_direct_eval_script_accepts_axis_offset_few_shot_args():
    from meta_sg.scripts.evaluate_meta_sg_direct import parse_args

    args = parse_args(
        [
            "--checkpoint",
            "dummy.pt",
            "--output-json",
            "out.json",
            "--few-shot",
            "--few-shot-method",
            "axis_offset",
            "--axis-offset-step",
            "0.25",
            "--axis-offset-max-steps",
            "2",
            "--axis-offset-asr-reduction-margin",
            "0.01",
            "--axis-offset-pessimistic-asr-margin",
            "0.0",
            "--axis-offset-clean-floor",
            "0.89",
            "--axis-offset-clean-drop-tolerance",
            "0.02",
        ]
    )

    assert args.few_shot_method == "axis_offset"
    assert args.axis_offset_step == pytest.approx(0.25)
    assert args.axis_offset_max_steps == 2
    assert args.axis_offset_asr_reduction_margin == pytest.approx(0.01)
    assert args.axis_offset_pessimistic_asr_margin == pytest.approx(0.0)
    assert args.axis_offset_clean_floor == pytest.approx(0.89)
    assert args.axis_offset_clean_drop_tolerance == pytest.approx(0.02)


def test_direct_eval_script_accepts_axis_rule_offset_few_shot_args():
    from meta_sg.scripts.evaluate_meta_sg_direct import parse_args

    args = parse_args(
        [
            "--checkpoint",
            "dummy.pt",
            "--output-json",
            "out.json",
            "--few-shot",
            "--few-shot-method",
            "axis_rule_offset",
            "--axis-offset-step",
            "0.25",
            "--axis-rule-beta-threshold",
            "0.35",
        ]
    )

    assert args.few_shot_method == "axis_rule_offset"
    assert args.axis_offset_step == pytest.approx(0.25)
    assert args.axis_rule_beta_threshold == pytest.approx(0.35)


def test_direct_eval_script_accepts_axis_rule_v2_offset_few_shot_args():
    from meta_sg.scripts.evaluate_meta_sg_direct import parse_args

    args = parse_args(
        [
            "--checkpoint",
            "dummy.pt",
            "--output-json",
            "out.json",
            "--few-shot",
            "--few-shot-method",
            "axis_rule_v2_offset",
            "--axis-offset-step",
            "0.25",
            "--axis-rule-beta-threshold",
            "0.35",
            "--axis-rule-low-beta-threshold",
            "0.34",
        ]
    )

    assert args.few_shot_method == "axis_rule_v2_offset"
    assert args.axis_offset_step == pytest.approx(0.25)
    assert args.axis_rule_beta_threshold == pytest.approx(0.35)
    assert args.axis_rule_low_beta_threshold == pytest.approx(0.34)


def test_direct_eval_script_accepts_physical_rule_target_few_shot_args():
    from meta_sg.scripts.evaluate_meta_sg_direct import parse_args

    args = parse_args(
        [
            "--checkpoint",
            "dummy.pt",
            "--output-json",
            "out.json",
            "--few-shot",
            "--few-shot-method",
            "physical_rule_target",
            "--physical-rule-low-beta-threshold",
            "0.34",
            "--physical-rule-beta-threshold",
            "0.35",
            "--physical-rule-low-alpha-target",
            "0.10",
            "--physical-rule-near-alpha-target",
            "0.15",
            "--physical-rule-beta-target",
            "0.38",
        ]
    )

    assert args.few_shot_method == "physical_rule_target"
    assert args.physical_rule_low_beta_threshold == pytest.approx(0.34)
    assert args.physical_rule_beta_threshold == pytest.approx(0.35)
    assert args.physical_rule_low_alpha_target == pytest.approx(0.10)
    assert args.physical_rule_near_alpha_target == pytest.approx(0.15)
    assert args.physical_rule_beta_target == pytest.approx(0.38)


def test_direct_eval_script_defaults_physical_rule_near_alpha_to_fine_sweep_choice():
    from meta_sg.scripts.evaluate_meta_sg_direct import parse_args

    args = parse_args(
        [
            "--checkpoint",
            "dummy.pt",
            "--output-json",
            "out.json",
            "--few-shot",
            "--few-shot-method",
            "physical_rule_target",
        ]
    )

    assert args.physical_rule_low_alpha_target == pytest.approx(0.10)
    assert args.physical_rule_near_alpha_target == pytest.approx(0.14)
    assert args.physical_rule_beta_target == pytest.approx(0.38)


def test_direct_eval_script_defaults_physical_target_selector_to_score_strict():
    from meta_sg.scripts.evaluate_meta_sg_direct import parse_args

    args = parse_args(
        [
            "--checkpoint",
            "dummy.pt",
            "--output-json",
            "out.json",
            "--few-shot",
            "--few-shot-method",
            "physical_target_selector",
        ]
    )

    assert args.physical_target_alpha_candidates == "0.10,0.12,0.14"
    assert args.physical_target_beta_target == pytest.approx(0.38)
    assert args.physical_target_clean_floor == pytest.approx(0.92)
    assert args.physical_target_clean_drop_tolerance == pytest.approx(0.04)
    assert args.physical_target_score_slack == pytest.approx(0.0)
    assert args.physical_target_end_round_candidates is None


def test_direct_eval_script_accepts_physical_target_selector_few_shot_args():
    from meta_sg.scripts.evaluate_meta_sg_direct import parse_args

    args = parse_args(
        [
            "--checkpoint",
            "dummy.pt",
            "--output-json",
            "out.json",
            "--few-shot",
            "--few-shot-method",
            "physical_target_selector",
            "--physical-target-alpha-candidates",
            "0.10,0.12,0.14",
            "--physical-target-beta-target",
            "0.38",
            "--physical-target-asr-reduction-margin",
            "0.005",
            "--physical-target-clean-floor",
            "0.92",
            "--physical-target-clean-drop-tolerance",
            "0.04",
            "--physical-target-score-slack",
            "0.02",
            "--physical-target-selection-mode",
            "deployment_clean_recovery",
            "--physical-target-deployment-clean-floor",
            "0.95",
            "--physical-target-deployment-clean-drop-tolerance",
            "0.02",
            "--physical-target-deployment-asr-ceiling",
            "0.30",
            "--physical-target-start-round",
            "0",
            "--physical-target-end-round",
            "35",
            "--physical-target-end-round-candidates",
            "full,35,40",
        ]
    )

    assert args.few_shot_method == "physical_target_selector"
    assert args.physical_target_alpha_candidates == "0.10,0.12,0.14"
    assert args.physical_target_beta_target == pytest.approx(0.38)
    assert args.physical_target_asr_reduction_margin == pytest.approx(0.005)
    assert args.physical_target_clean_floor == pytest.approx(0.92)
    assert args.physical_target_clean_drop_tolerance == pytest.approx(0.04)
    assert args.physical_target_score_slack == pytest.approx(0.02)
    assert args.physical_target_selection_mode == "deployment_clean_recovery"
    assert args.physical_target_deployment_clean_floor == pytest.approx(0.95)
    assert args.physical_target_deployment_clean_drop_tolerance == pytest.approx(0.02)
    assert args.physical_target_deployment_asr_ceiling == pytest.approx(0.30)
    assert args.physical_target_start_round == 0
    assert args.physical_target_end_round == 35
    assert args.physical_target_end_round_candidates == "full,35,40"


def test_parse_optional_int_candidates_accepts_full_and_dedupes_values():
    from meta_sg.scripts.evaluate_meta_sg_direct import _parse_optional_int_candidates

    assert _parse_optional_int_candidates("full, 35,40,35") == [None, 35, 40]
    assert _parse_optional_int_candidates("none,45") == [None, 45]
    assert _parse_optional_int_candidates(None) == []


def test_beta_offset_deployment_beta_ceiling_blocks_high_beta():
    from meta_sg.scripts.evaluate_meta_sg_direct import _beta_offset_deployment_beta_blocked

    assert _beta_offset_deployment_beta_blocked({"beta": 0.36}, beta_ceiling=0.35) is True
    assert _beta_offset_deployment_beta_blocked({"beta": 0.35}, beta_ceiling=0.35) is True
    assert _beta_offset_deployment_beta_blocked({"beta": 0.34}, beta_ceiling=0.35) is False
    assert _beta_offset_deployment_beta_blocked({"beta": float("nan")}, beta_ceiling=0.35) is False
    assert _beta_offset_deployment_beta_blocked({"beta": 0.36}, beta_ceiling=None) is False


def test_scheduled_action_offset_policy_applies_offset_only_inside_round_window():
    from meta_sg.scripts.evaluate_meta_sg_direct import ScheduledActionOffsetPolicy

    class BasePolicy:
        obs_dim = 2
        act_dim = 3

        def get_action(self, obs, noise=0.0):
            return np.asarray([0.0, 0.2, -0.2], dtype=np.float32)

    policy = ScheduledActionOffsetPolicy(
        BasePolicy(),
        offset=np.asarray([0.0, 0.25, 0.0], dtype=np.float32),
        start_round=1,
        end_round=3,
    )

    actions = [policy.get_action(np.zeros(2, dtype=np.float32)).tolist() for _ in range(4)]

    assert actions[0] == pytest.approx([0.0, 0.2, -0.2])
    assert actions[1] == pytest.approx([0.0, 0.45, -0.2])
    assert actions[2] == pytest.approx([0.0, 0.45, -0.2])
    assert actions[3] == pytest.approx([0.0, 0.2, -0.2])

    policy.reset()
    assert policy.get_action(np.zeros(2, dtype=np.float32)).tolist() == pytest.approx([0.0, 0.2, -0.2])


def test_scheduled_physical_target_policy_applies_targets_only_inside_round_window():
    from meta_sg.games.bsmg_env import BSMGConfig
    from meta_sg.scripts.evaluate_meta_sg_direct import ScheduledPhysicalTargetPolicy
    from meta_sg.strategies.types import DefenseDecision

    class BasePolicy:
        obs_dim = 2
        act_dim = 3

        def get_action(self, obs, noise=0.0):
            return np.asarray([0.5, -0.5, 0.25], dtype=np.float32)

    cfg = BSMGConfig(alpha_min=0.0, alpha_max=5.0, beta_min=0.0, beta_max=0.45)
    policy = ScheduledPhysicalTargetPolicy(
        BasePolicy(),
        cfg,
        target_alpha=0.10,
        target_beta=0.38,
        start_round=1,
        end_round=3,
    )

    actions = [policy.get_action(np.zeros(2, dtype=np.float32)) for _ in range(4)]
    decisions = [
        DefenseDecision.from_raw(
            action,
            alpha_min=cfg.alpha_min,
            alpha_max=cfg.alpha_max,
            beta_min=cfg.beta_min,
            beta_max=cfg.beta_max,
            eps_min=cfg.eps_min,
            eps_max=cfg.eps_max,
        )
        for action in actions
    ]

    assert decisions[0].norm_bound_alpha == pytest.approx(3.75)
    assert decisions[0].trimmed_mean_beta == pytest.approx(0.1125)
    assert decisions[1].norm_bound_alpha == pytest.approx(0.10)
    assert decisions[1].trimmed_mean_beta == pytest.approx(0.38)
    assert decisions[2].norm_bound_alpha == pytest.approx(0.10)
    assert decisions[2].trimmed_mean_beta == pytest.approx(0.38)
    assert decisions[3].norm_bound_alpha == pytest.approx(3.75)
    assert decisions[3].trimmed_mean_beta == pytest.approx(0.1125)

    policy.reset()
    reset_decision = DefenseDecision.from_raw(
        policy.get_action(np.zeros(2, dtype=np.float32)),
        alpha_min=cfg.alpha_min,
        alpha_max=cfg.alpha_max,
        beta_min=cfg.beta_min,
        beta_max=cfg.beta_max,
        eps_min=cfg.eps_min,
        eps_max=cfg.eps_max,
    )
    assert reset_decision.norm_bound_alpha == pytest.approx(3.75)


def test_direct_eval_script_accepts_residual_adapter_args():
    from meta_sg.scripts.evaluate_meta_sg_direct import parse_args

    args = parse_args(
        [
            "--checkpoint",
            "dummy.pt",
            "--output-json",
            "out.json",
            "--few-shot",
            "--few-shot-method",
            "residual_adapter",
            "--residual-bound",
            "0.2",
            "--residual-candidates",
            "5",
            "--residual-query-episodes",
            "2",
            "--residual-degradation-margin",
            "0.001",
            "--residual-objective-gate",
            "targeted",
            "--residual-selection-metric",
            "worst",
            "--residual-lcb-std-weight",
            "0.5",
        ]
    )

    assert args.few_shot_method == "residual_adapter"
    assert args.residual_bound == pytest.approx(0.2)
    assert args.residual_candidates == 5
    assert args.residual_query_episodes == 2
    assert args.residual_degradation_margin == pytest.approx(0.001)
    assert args.residual_objective_gate == "targeted"
    assert args.residual_selection_metric == "worst"
    assert args.residual_lcb_std_weight == pytest.approx(0.5)


def test_direct_eval_script_accepts_trained_residual_adapter_args():
    from meta_sg.scripts.evaluate_meta_sg_direct import parse_args

    args = parse_args(
        [
            "--checkpoint",
            "dummy.pt",
            "--output-json",
            "out.json",
            "--few-shot",
            "--few-shot-method",
            "trained_residual_adapter",
            "--residual-adapter-checkpoint",
            "adapter.pt",
            "--residual-dump-supervision-jsonl",
            "samples.jsonl",
        ]
    )

    assert args.few_shot_method == "trained_residual_adapter"
    assert args.residual_adapter_checkpoint == "adapter.pt"
    assert args.residual_dump_supervision_jsonl == "samples.jsonl"


def test_residual_objective_gate_can_limit_adapter_to_targeted_attacks():
    from meta_sg.scripts.evaluate_meta_sg_direct import Scenario, _residual_objective_allowed

    targeted = Scenario("bfl", ATTACK_DOMAIN["bfl"], "bfl", {}, seed=0)
    untargeted = Scenario("ipm", ATTACK_DOMAIN["ipm"], "ipm", {}, seed=0)

    assert _residual_objective_allowed("all", targeted) is True
    assert _residual_objective_allowed("all", untargeted) is True
    assert _residual_objective_allowed("targeted", targeted) is True
    assert _residual_objective_allowed("targeted", untargeted) is False
    assert _residual_objective_allowed("untargeted", targeted) is False
    assert _residual_objective_allowed("untargeted", untargeted) is True


def test_cem_update_distribution_uses_top_scoring_elites_and_sigma_floor():
    from meta_sg.scripts.evaluate_meta_sg_direct import _cem_update_distribution

    candidates = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [-0.4, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    scores = [0.1, 0.9, 1.0, -1.0]

    mean, sigma, elite_indices = _cem_update_distribution(
        candidates,
        scores,
        elite_count=2,
        min_sigma=0.05,
    )

    assert elite_indices == [2, 1]
    assert mean.tolist() == pytest.approx([0.3, 0.0, 0.0])
    assert sigma.tolist() == pytest.approx([0.1, 0.05, 0.05])


def test_residual_support_summary_tracks_score_and_metric_slopes():
    from meta_sg.scripts.evaluate_meta_sg_direct import _residual_support_summary

    records = [
        {
            "final_defense_score": 0.80,
            "final_clean_acc": 0.82,
            "final_backdoor_acc": 0.02,
            "mean_defender_reward": 0.70,
        },
        {
            "final_defense_score": 0.84,
            "final_clean_acc": 0.85,
            "final_backdoor_acc": 0.01,
            "mean_defender_reward": 0.74,
        },
    ]

    summary = _residual_support_summary(records, act_dim=3)

    assert summary["support_score_mean"] == pytest.approx(0.82)
    assert summary["support_score_slope"] == pytest.approx(0.04)
    assert summary["support_clean_slope"] == pytest.approx(0.03)
    assert summary["support_backdoor_slope"] == pytest.approx(-0.01)
    assert summary["direction"] == pytest.approx([0.04, 0.01, 0.03])


def test_residual_selection_requires_query_improvement_over_margin():
    from meta_sg.scripts.evaluate_meta_sg_direct import _select_residual_from_query_scores

    candidates = [
        {"offset": [0.0, 0.0, 0.0], "query_score_mean": 1.0},
        {"offset": [0.1, 0.0, 0.0], "query_score_mean": 1.0005},
        {"offset": [0.0, 0.1, 0.0], "query_score_mean": 1.0030},
    ]

    selected, decision = _select_residual_from_query_scores(candidates, margin=0.001)

    assert selected["offset"] == pytest.approx([0.0, 0.1, 0.0])
    assert decision["accepted"] is True
    assert decision["score_gain"] == pytest.approx(0.0030)


def test_residual_selection_can_use_worst_case_query_score():
    from meta_sg.scripts.evaluate_meta_sg_direct import _select_residual_from_query_scores

    candidates = [
        {"offset": [0.0, 0.0, 0.0], "query_scores": [1.0, 1.0]},
        {"offset": [0.2, 0.0, 0.0], "query_scores": [1.2, 0.7]},
        {"offset": [0.0, 0.2, 0.0], "query_scores": [1.03, 1.02]},
    ]

    selected, decision = _select_residual_from_query_scores(
        candidates,
        margin=0.001,
        metric="worst",
        lcb_std_weight=1.0,
    )

    assert selected["offset"] == pytest.approx([0.0, 0.2, 0.0])
    assert decision["accepted"] is True
    assert decision["selection_metric"] == "worst"
    assert decision["base_score"] == pytest.approx(1.0)
    assert decision["adapted_score"] == pytest.approx(1.02)
    assert decision["score_gain"] == pytest.approx(0.02)


def test_residual_selection_can_use_lower_confidence_bound():
    from meta_sg.scripts.evaluate_meta_sg_direct import _select_residual_from_query_scores

    candidates = [
        {"offset": [0.0, 0.0, 0.0], "query_scores": [1.0, 1.0]},
        {"offset": [0.2, 0.0, 0.0], "query_scores": [1.2, 0.8]},
        {"offset": [0.0, 0.2, 0.0], "query_scores": [1.04, 1.02]},
    ]

    selected, decision = _select_residual_from_query_scores(
        candidates,
        margin=0.001,
        metric="lcb",
        lcb_std_weight=1.0,
    )

    assert selected["offset"] == pytest.approx([0.0, 0.2, 0.0])
    assert decision["accepted"] is True
    assert decision["selection_metric"] == "lcb"
    assert decision["adapted_score"] > decision["base_score"]


def test_beta_offset_candidates_only_move_beta_dimension():
    from meta_sg.scripts.evaluate_meta_sg_direct import _beta_offset_candidates

    candidates = _beta_offset_candidates(act_dim=3, step=0.25, max_steps=2)

    assert [label for label, _ in candidates] == ["zero", "beta_plus_1", "beta_plus_2", "beta_minus_1"]
    expected = [
        [0.0, 0.0, 0.0],
        [0.0, 0.25, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, -0.25, 0.0],
    ]
    for (_, offset), expected_offset in zip(candidates, expected, strict=True):
        assert offset.tolist() == pytest.approx(expected_offset)


def test_beta_offset_selection_prefers_asr_reduction_with_clean_guard():
    from meta_sg.scripts.evaluate_meta_sg_direct import _select_beta_offset_from_query_records

    candidates = [
        {
            "offset_label": "zero",
            "offset": [0.0, 0.0, 0.0],
            "query_records": [
                {"final_clean_acc": 0.95, "final_backdoor_acc": 0.90, "final_defense_score": -0.85}
            ],
        },
        {
            "offset_label": "beta_plus_1",
            "offset": [0.0, 0.25, 0.0],
            "query_records": [
                {"final_clean_acc": 0.94, "final_backdoor_acc": 0.72, "final_defense_score": -0.50}
            ],
        },
        {
            "offset_label": "beta_plus_2",
            "offset": [0.0, 0.50, 0.0],
            "query_records": [
                {"final_clean_acc": 0.84, "final_backdoor_acc": 0.20, "final_defense_score": 0.44}
            ],
        },
    ]

    selected, decision = _select_beta_offset_from_query_records(
        candidates,
        asr_reduction_margin=0.05,
        clean_floor=0.89,
        clean_drop_tolerance=0.02,
    )

    assert selected["offset_label"] == "beta_plus_1"
    assert decision["accepted"] is True
    assert decision["selected"] == "adapted"
    assert decision["base_backdoor_mean"] == pytest.approx(0.90)
    assert decision["adapted_backdoor_mean"] == pytest.approx(0.72)
    assert decision["backdoor_reduction"] == pytest.approx(0.18)


def test_beta_offset_selection_rejects_without_margin():
    from meta_sg.scripts.evaluate_meta_sg_direct import _select_beta_offset_from_query_records

    candidates = [
        {
            "offset_label": "zero",
            "offset": [0.0, 0.0, 0.0],
            "query_records": [
                {"final_clean_acc": 0.95, "final_backdoor_acc": 0.90, "final_defense_score": -0.85}
            ],
        },
        {
            "offset_label": "beta_plus_1",
            "offset": [0.0, 0.25, 0.0],
            "query_records": [
                {"final_clean_acc": 0.95, "final_backdoor_acc": 0.88, "final_defense_score": -0.81}
            ],
        },
    ]

    selected, decision = _select_beta_offset_from_query_records(
        candidates,
        asr_reduction_margin=0.05,
        clean_floor=0.89,
        clean_drop_tolerance=0.02,
    )

    assert selected["offset_label"] == "zero"
    assert decision["accepted"] is False
    assert decision["selected"] == "base"


def test_axis_offset_candidates_cover_beta_and_alpha_directions():
    from meta_sg.scripts.evaluate_meta_sg_direct import _axis_offset_candidates

    candidates = _axis_offset_candidates(act_dim=3, step=0.25, max_steps=1)

    assert [label for label, _ in candidates] == [
        "zero",
        "alpha_minus_1",
        "alpha_plus_1",
        "beta_plus_1",
        "beta_minus_1",
        "beta_plus_1_alpha_minus_1",
    ]
    expected = [
        [0.0, 0.0, 0.0],
        [-0.25, 0.0, 0.0],
        [0.25, 0.0, 0.0],
        [0.0, 0.25, 0.0],
        [0.0, -0.25, 0.0],
        [-0.25, 0.25, 0.0],
    ]
    for (_, offset), expected_offset in zip(candidates, expected, strict=True):
        assert offset.tolist() == pytest.approx(expected_offset)


def test_axis_offset_selection_uses_pessimistic_asr_guard():
    from meta_sg.scripts.evaluate_meta_sg_direct import _select_axis_offset_from_query_records

    candidates = [
        {
            "offset_label": "zero",
            "offset": [0.0, 0.0, 0.0],
            "query_records": [
                {"final_clean_acc": 0.95, "final_backdoor_acc": 0.90, "final_defense_score": -0.85},
                {"final_clean_acc": 0.95, "final_backdoor_acc": 0.70, "final_defense_score": -0.45},
            ],
        },
        {
            "offset_label": "beta_plus_1",
            "offset": [0.0, 0.25, 0.0],
            "query_records": [
                {"final_clean_acc": 0.95, "final_backdoor_acc": 0.50, "final_defense_score": -0.05},
                {"final_clean_acc": 0.95, "final_backdoor_acc": 0.73, "final_defense_score": -0.51},
            ],
        },
        {
            "offset_label": "alpha_minus_1",
            "offset": [-0.25, 0.0, 0.0],
            "query_records": [
                {"final_clean_acc": 0.94, "final_backdoor_acc": 0.72, "final_defense_score": -0.50},
                {"final_clean_acc": 0.94, "final_backdoor_acc": 0.62, "final_defense_score": -0.30},
            ],
        },
    ]

    selected, decision = _select_axis_offset_from_query_records(
        candidates,
        asr_reduction_margin=0.05,
        pessimistic_asr_margin=0.0,
        clean_floor=0.89,
        clean_drop_tolerance=0.02,
    )

    assert selected["offset_label"] == "alpha_minus_1"
    assert decision["accepted"] is True
    assert decision["selected"] == "adapted"
    assert decision["backdoor_reduction_mean"] == pytest.approx(0.13)
    assert decision["backdoor_reduction_min"] == pytest.approx(0.08)


def test_axis_rule_offset_chooses_beta_plus_for_low_beta_and_alpha_minus_for_high_beta():
    from meta_sg.scripts.evaluate_meta_sg_direct import _axis_rule_offset_for_action

    low_label, low_offset = _axis_rule_offset_for_action(
        {"beta": 0.32},
        act_dim=3,
        step=0.25,
        beta_threshold=0.35,
    )
    high_label, high_offset = _axis_rule_offset_for_action(
        {"beta": 0.36},
        act_dim=3,
        step=0.25,
        beta_threshold=0.35,
    )

    assert low_label == "beta_plus_1"
    assert low_offset.tolist() == pytest.approx([0.0, 0.25, 0.0])
    assert high_label == "alpha_minus_1"
    assert high_offset.tolist() == pytest.approx([-0.25, 0.0, 0.0])


def test_axis_rule_offset_keeps_high_beta_dba_on_base_for_clean_guard():
    from meta_sg.scripts.evaluate_meta_sg_direct import _axis_rule_offset_for_action

    label, offset = _axis_rule_offset_for_action(
        {"beta": 0.36},
        act_dim=3,
        step=0.25,
        beta_threshold=0.35,
        attack_name="dba",
    )

    assert label == "zero"
    assert offset.tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_axis_rule_v2_offset_uses_combo_for_low_beta_and_alpha_for_near_beta():
    from meta_sg.scripts.evaluate_meta_sg_direct import _axis_rule_v2_offset_for_action

    low_label, low_offset = _axis_rule_v2_offset_for_action(
        {"beta": 0.329},
        act_dim=3,
        step=0.25,
        beta_threshold=0.35,
        low_beta_threshold=0.34,
        attack_name="rl_backdoor",
    )
    near_label, near_offset = _axis_rule_v2_offset_for_action(
        {"beta": 0.345},
        act_dim=3,
        step=0.25,
        beta_threshold=0.35,
        low_beta_threshold=0.34,
        attack_name="rl_backdoor",
    )

    assert low_label == "alpha_minus_1_beta_plus_1"
    assert low_offset.tolist() == pytest.approx([-0.25, 0.25, 0.0])
    assert near_label == "alpha_minus_1"
    assert near_offset.tolist() == pytest.approx([-0.25, 0.0, 0.0])


def test_axis_rule_v2_offset_keeps_near_beta_dba_on_base_for_clean_guard():
    from meta_sg.scripts.evaluate_meta_sg_direct import _axis_rule_v2_offset_for_action

    label, offset = _axis_rule_v2_offset_for_action(
        {"beta": 0.345},
        act_dim=3,
        step=0.25,
        beta_threshold=0.35,
        low_beta_threshold=0.34,
        attack_name="dba",
    )

    assert label == "zero"
    assert offset.tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_physical_target_policy_overrides_alpha_beta_raw_coordinates():
    from meta_sg.games.bsmg_env import BSMGConfig
    from meta_sg.scripts.evaluate_meta_sg_direct import PhysicalTargetPolicy
    from meta_sg.strategies.types import DefenseDecision

    class DummyPolicy:
        obs_dim = 3
        act_dim = 3

        def get_action(self, obs, noise=0.0):
            return np.array([0.5, -0.5, 0.25], dtype=np.float32)

    cfg = BSMGConfig(alpha_min=0.0, alpha_max=5.0, beta_min=0.0, beta_max=0.45)
    policy = PhysicalTargetPolicy(DummyPolicy(), cfg, target_alpha=0.10, target_beta=0.38)

    raw = policy.get_action(np.zeros(3, dtype=np.float32))
    decision = DefenseDecision.from_raw(
        raw,
        alpha_min=cfg.alpha_min,
        alpha_max=cfg.alpha_max,
        beta_min=cfg.beta_min,
        beta_max=cfg.beta_max,
        eps_min=cfg.eps_min,
        eps_max=cfg.eps_max,
    )

    assert raw[2] == pytest.approx(0.25)
    assert decision.norm_bound_alpha == pytest.approx(0.10)
    assert decision.trimmed_mean_beta == pytest.approx(0.38)


def test_physical_rule_target_selects_low_near_and_high_regimes():
    from meta_sg.scripts.evaluate_meta_sg_direct import _physical_rule_target_for_action

    low = _physical_rule_target_for_action(
        {"beta": 0.329},
        act_dim=3,
        step=0.25,
        low_beta_threshold=0.34,
        beta_threshold=0.35,
        low_alpha_target=0.10,
        near_alpha_target=0.15,
        beta_target=0.38,
        attack_name="rl_backdoor",
    )
    near = _physical_rule_target_for_action(
        {"beta": 0.345},
        act_dim=3,
        step=0.25,
        low_beta_threshold=0.34,
        beta_threshold=0.35,
        low_alpha_target=0.10,
        near_alpha_target=0.15,
        beta_target=0.38,
        attack_name="rl_backdoor",
    )
    near_dba = _physical_rule_target_for_action(
        {"beta": 0.345},
        act_dim=3,
        step=0.25,
        low_beta_threshold=0.34,
        beta_threshold=0.35,
        low_alpha_target=0.10,
        near_alpha_target=0.15,
        beta_target=0.38,
        attack_name="dba",
    )
    high = _physical_rule_target_for_action(
        {"beta": 0.36},
        act_dim=3,
        step=0.25,
        low_beta_threshold=0.34,
        beta_threshold=0.35,
        low_alpha_target=0.10,
        near_alpha_target=0.15,
        beta_target=0.38,
        attack_name="rl_backdoor",
    )

    assert low["selected_offset_label"] == "physical_a0.10_b0.38"
    assert low["target_alpha"] == pytest.approx(0.10)
    assert low["target_beta"] == pytest.approx(0.38)
    assert near["selected_offset_label"] == "physical_a0.15_b0.38"
    assert near["target_alpha"] == pytest.approx(0.15)
    assert near["target_beta"] == pytest.approx(0.38)
    assert near_dba["selected_offset_label"] == "zero"
    assert near_dba["target_alpha"] is None
    assert high["selected_offset_label"] == "alpha_minus_1"
    assert high["offset"].tolist() == pytest.approx([-0.25, 0.0, 0.0])


def test_physical_target_candidates_include_base_and_alpha_frontier():
    from meta_sg.scripts.evaluate_meta_sg_direct import _physical_target_candidates

    candidates = _physical_target_candidates(
        act_dim=3,
        alpha_candidates="0.10, 0.12,0.14,0.14",
        beta_target=0.38,
    )

    assert [candidate["offset_label"] for candidate in candidates] == [
        "zero",
        "physical_a0.10_b0.38",
        "physical_a0.12_b0.38",
        "physical_a0.14_b0.38",
    ]
    assert candidates[0]["target_alpha"] is None
    assert candidates[1]["target_alpha"] == pytest.approx(0.10)
    assert candidates[1]["target_beta"] == pytest.approx(0.38)
    assert candidates[1]["offset"] == pytest.approx([0.0, 0.0, 0.0])


def test_physical_target_candidates_expand_alpha_over_round_windows_once_per_base():
    from meta_sg.scripts.evaluate_meta_sg_direct import _physical_target_candidates

    candidates = _physical_target_candidates(
        act_dim=3,
        alpha_candidates="0.10",
        beta_target=0.38,
        start_round=0,
        end_round_candidates=[None, 40],
    )

    assert [candidate["offset_label"] for candidate in candidates] == [
        "zero",
        "physical_a0.10_b0.38_full",
        "physical_a0.10_b0.38_w0_40",
    ]
    assert candidates[0]["target_end_round"] is None
    assert candidates[1]["target_start_round"] == 0
    assert candidates[1]["target_end_round"] is None
    assert candidates[2]["target_start_round"] == 0
    assert candidates[2]["target_end_round"] == 40


def test_physical_target_policy_for_candidate_uses_candidate_window_over_args():
    from meta_sg.games.bsmg_env import BSMGConfig
    from meta_sg.scripts.evaluate_meta_sg_direct import (
        ScheduledPhysicalTargetPolicy,
        _physical_target_policy_for_candidate,
    )

    class DummyPolicy:
        obs_dim = 3
        act_dim = 3

        def get_action(self, obs, noise=0.0):
            return np.zeros(3, dtype=np.float32)

    args = SimpleNamespace(physical_target_start_round=0, physical_target_end_round=25)
    candidate = {
        "target_alpha": 0.10,
        "target_beta": 0.38,
        "target_start_round": 2,
        "target_end_round": 40,
    }

    policy = _physical_target_policy_for_candidate(args, DummyPolicy(), BSMGConfig(), candidate)

    assert isinstance(policy, ScheduledPhysicalTargetPolicy)
    assert policy.start_round == 2
    assert policy.end_round == 40


def test_physical_target_selection_prefers_clean_safe_candidate():
    from meta_sg.scripts.evaluate_meta_sg_direct import _select_physical_target_from_query_records

    candidates = [
        {
            "offset_label": "zero",
            "target_alpha": None,
            "target_beta": None,
            "offset": [0.0, 0.0, 0.0],
            "query_records": [
                {"final_clean_acc": 0.95, "final_backdoor_acc": 0.90, "final_defense_score": -0.85}
            ],
        },
        {
            "offset_label": "physical_a0.10_b0.38",
            "target_alpha": 0.10,
            "target_beta": 0.38,
            "offset": [0.0, 0.0, 0.0],
            "query_records": [
                {"final_clean_acc": 0.90, "final_backdoor_acc": 0.02, "final_defense_score": 0.86}
            ],
        },
        {
            "offset_label": "physical_a0.14_b0.38",
            "target_alpha": 0.14,
            "target_beta": 0.38,
            "offset": [0.0, 0.0, 0.0],
            "query_records": [
                {"final_clean_acc": 0.93, "final_backdoor_acc": 0.06, "final_defense_score": 0.81}
            ],
        },
    ]

    selected, decision = _select_physical_target_from_query_records(
        candidates,
        asr_reduction_margin=0.005,
        clean_floor=0.92,
        clean_drop_tolerance=0.04,
        score_slack=0.02,
    )

    assert selected["offset_label"] == "physical_a0.14_b0.38"
    assert decision["accepted"] is True
    assert decision["selected"] == "adapted"
    assert decision["selected_offset_label"] == "physical_a0.14_b0.38"
    assert decision["base_clean_mean"] == pytest.approx(0.95)
    assert decision["adapted_clean_mean"] == pytest.approx(0.93)
    assert decision["backdoor_reduction"] == pytest.approx(0.84)


def test_physical_target_selection_prefers_cleaner_candidate_within_score_slack():
    from meta_sg.scripts.evaluate_meta_sg_direct import _select_physical_target_from_query_records

    candidates = [
        {
            "offset_label": "zero",
            "target_alpha": None,
            "target_beta": None,
            "offset": [0.0, 0.0, 0.0],
            "query_records": [
                {"final_clean_acc": 0.95, "final_backdoor_acc": 0.90, "final_defense_score": -0.85}
            ],
        },
        {
            "offset_label": "physical_a0.10_b0.38",
            "target_alpha": 0.10,
            "target_beta": 0.38,
            "offset": [0.0, 0.0, 0.0],
            "query_records": [
                {"final_clean_acc": 0.930, "final_backdoor_acc": 0.010, "final_defense_score": 0.910}
            ],
        },
        {
            "offset_label": "physical_a0.12_b0.38",
            "target_alpha": 0.12,
            "target_beta": 0.38,
            "offset": [0.0, 0.0, 0.0],
            "query_records": [
                {"final_clean_acc": 0.945, "final_backdoor_acc": 0.020, "final_defense_score": 0.905}
            ],
        },
    ]

    selected, decision = _select_physical_target_from_query_records(
        candidates,
        asr_reduction_margin=0.005,
        clean_floor=0.92,
        clean_drop_tolerance=0.04,
        score_slack=0.02,
    )

    assert selected["offset_label"] == "physical_a0.12_b0.38"
    assert decision["selected_offset_label"] == "physical_a0.12_b0.38"
    assert decision["score_slack"] == pytest.approx(0.02)
    assert decision["adapted_clean_mean"] == pytest.approx(0.945)


def test_physical_target_deployment_clean_recovery_prefers_window_over_query_best():
    from meta_sg.scripts.evaluate_meta_sg_direct import _select_physical_target_with_deployment_records

    candidates = [
        {
            "offset_label": "zero",
            "target_alpha": None,
            "target_beta": None,
            "target_start_round": None,
            "target_end_round": None,
            "offset": [0.0, 0.0, 0.0],
            "query_records": [
                {"final_clean_acc": 0.943, "final_backdoor_acc": 0.443, "final_defense_score": 0.057}
            ],
            "deployment_record": {
                "final_clean_acc": 0.960,
                "final_backdoor_acc": 0.860,
                "final_defense_score": -0.760,
            },
        },
        {
            "offset_label": "physical_a0.10_b0.38_full",
            "target_alpha": 0.10,
            "target_beta": 0.38,
            "target_start_round": 0,
            "target_end_round": None,
            "offset": [0.0, 0.0, 0.0],
            "query_records": [
                {"final_clean_acc": 0.930, "final_backdoor_acc": 0.007, "final_defense_score": 0.917}
            ],
            "deployment_record": {
                "final_clean_acc": 0.940,
                "final_backdoor_acc": 0.077,
                "final_defense_score": 0.787,
            },
        },
        {
            "offset_label": "physical_a0.10_b0.38_w0_40",
            "target_alpha": 0.10,
            "target_beta": 0.38,
            "target_start_round": 0,
            "target_end_round": 40,
            "offset": [0.0, 0.0, 0.0],
            "query_records": [
                {"final_clean_acc": 0.920, "final_backdoor_acc": 0.063, "final_defense_score": 0.793}
            ],
            "deployment_record": {
                "final_clean_acc": 0.950,
                "final_backdoor_acc": 0.250,
                "final_defense_score": 0.450,
            },
        },
    ]

    selected, decision = _select_physical_target_with_deployment_records(
        candidates,
        asr_reduction_margin=0.005,
        clean_floor=0.92,
        clean_drop_tolerance=0.04,
        score_slack=0.0,
        deployment_clean_floor=0.95,
        deployment_clean_drop_tolerance=None,
        deployment_asr_ceiling=0.30,
    )

    assert selected["offset_label"] == "physical_a0.10_b0.38_w0_40"
    assert decision["accepted"] is True
    assert decision["selected_offset_label"] == "physical_a0.10_b0.38_w0_40"
    assert decision["query_selected_offset_label"] == "physical_a0.10_b0.38_full"
    assert decision["deployment_clean_acc"] == pytest.approx(0.95)
    assert decision["deployment_backdoor_acc"] == pytest.approx(0.25)
    assert decision["deployment_clean_floor"] == pytest.approx(0.95)
    assert decision["deployment_asr_ceiling"] == pytest.approx(0.30)


def test_physical_target_deployment_clean_drop_guard_falls_back_to_base():
    from meta_sg.scripts.evaluate_meta_sg_direct import _select_physical_target_with_deployment_records

    candidates = [
        {
            "offset_label": "zero",
            "target_alpha": None,
            "target_beta": None,
            "target_start_round": None,
            "target_end_round": None,
            "offset": [0.0, 0.0, 0.0],
            "query_records": [
                {"final_clean_acc": 0.930, "final_backdoor_acc": 0.967, "final_defense_score": -1.004}
            ],
            "deployment_record": {
                "final_clean_acc": 0.944,
                "final_backdoor_acc": 0.767,
                "final_defense_score": -0.590,
            },
        },
        {
            "offset_label": "physical_a0.10_b0.38_w0_40",
            "target_alpha": 0.10,
            "target_beta": 0.38,
            "target_start_round": 0,
            "target_end_round": 40,
            "offset": [0.0, 0.0, 0.0],
            "query_records": [
                {"final_clean_acc": 0.907, "final_backdoor_acc": 0.590, "final_defense_score": -0.273}
            ],
            "deployment_record": {
                "final_clean_acc": 0.916,
                "final_backdoor_acc": 0.020,
                "final_defense_score": 0.876,
            },
        },
    ]

    selected, decision = _select_physical_target_with_deployment_records(
        candidates,
        asr_reduction_margin=0.005,
        clean_floor=None,
        clean_drop_tolerance=0.04,
        score_slack=0.0,
        deployment_clean_floor=None,
        deployment_clean_drop_tolerance=0.02,
        deployment_asr_ceiling=0.30,
    )

    assert selected["offset_label"] == "zero"
    assert decision["accepted"] is False
    assert decision["selected"] == "base"
    assert decision["query_selected_offset_label"] == "physical_a0.10_b0.38_w0_40"
    assert decision["selection_stage"] == "deployment_rejected"
    assert decision["base_deployment_clean_acc"] == pytest.approx(0.944)
    assert decision["deployment_clean_drop"] == pytest.approx(0.0)
    assert decision["deployment_clean_drop_tolerance"] == pytest.approx(0.02)
    assert decision["deployment_eligible_count"] == 0


def test_axis_rule_offset_adaptation_uses_initial_beta_without_support_queries(monkeypatch):
    import meta_sg.scripts.evaluate_meta_sg_direct as direct_eval

    class DummyPolicy:
        obs_dim = 3
        act_dim = 3

        def get_action(self, obs, noise=0.0):
            return np.zeros(3, dtype=np.float32)

    args = SimpleNamespace(
        H=50,
        selection_horizon=50,
        seed=1012,
        axis_offset_step=0.25,
        axis_rule_beta_threshold=0.35,
        lambda_bd=2.0,
        defender_third_action="neuroclip",
        server_lr_min=0.0,
        server_lr_max=1.0,
        server_lr_penalty_weight=0.0,
    )
    scenario = direct_eval.Scenario(
        "rl_backdoor",
        direct_eval._attack_type("rl_backdoor"),
        "rl_backdoor",
        {},
        seed=1012,
    )

    def fail_if_support_query_runs(*args, **kwargs):
        raise AssertionError("axis_rule_offset should not run support queries")

    def fake_eval(args, policy, scenario):
        offset = np.asarray(getattr(policy, "offset", np.zeros(3)), dtype=np.float32)
        backdoor = 0.003 if offset.tolist() == pytest.approx([-0.25, 0.0, 0.0]) else 0.768
        return {
            "final_clean_acc": 0.92,
            "final_backdoor_acc": backdoor,
            "final_defense_score": 0.92 - 2.0 * backdoor,
        }

    monkeypatch.setattr(direct_eval, "_evaluate_scenario_at", fail_if_support_query_runs)
    monkeypatch.setattr(direct_eval, "_evaluate_scenario", fake_eval)
    monkeypatch.setattr(
        direct_eval,
        "_scenario_initial_action_diagnostics",
        lambda args, defender, scenario, *, horizon: {"beta": 0.36},
    )
    monkeypatch.setattr(direct_eval, "_action_diagnostics", lambda policy, obs, cfg: {})

    result = direct_eval._axis_rule_offset_adapt_and_evaluate(
        args,
        DummyPolicy(),
        scenario,
        probe_obs=np.zeros(3, dtype=np.float32),
    )

    assert result["selection"]["accepted"] is True
    assert result["selection"]["selected_offset_label"] == "alpha_minus_1"
    assert result["selected_offset"] == pytest.approx([-0.25, 0.0, 0.0])
    assert result["num_transitions"] == 0
    assert result["evaluation"]["final_backdoor_acc"] == pytest.approx(0.003)


def test_axis_rule_offset_adaptation_uses_base_for_high_beta_dba(monkeypatch):
    import meta_sg.scripts.evaluate_meta_sg_direct as direct_eval

    class DummyPolicy:
        obs_dim = 3
        act_dim = 3

        def get_action(self, obs, noise=0.0):
            return np.zeros(3, dtype=np.float32)

    args = SimpleNamespace(
        H=50,
        selection_horizon=50,
        seed=1012,
        axis_offset_step=0.25,
        axis_rule_beta_threshold=0.35,
        lambda_bd=2.0,
        defender_third_action="neuroclip",
        server_lr_min=0.0,
        server_lr_max=1.0,
        server_lr_penalty_weight=0.0,
    )
    scenario = direct_eval.Scenario(
        "dba",
        direct_eval._attack_type("dba"),
        "dba",
        {},
        seed=1012,
    )

    def fake_eval(args, policy, scenario):
        offset = np.asarray(getattr(policy, "offset", np.zeros(3)), dtype=np.float32)
        assert offset.tolist() == pytest.approx([0.0, 0.0, 0.0])
        return {
            "final_clean_acc": 0.965,
            "final_backdoor_acc": 0.025,
            "final_defense_score": 0.915,
        }

    monkeypatch.setattr(direct_eval, "_evaluate_scenario", fake_eval)
    monkeypatch.setattr(
        direct_eval,
        "_scenario_initial_action_diagnostics",
        lambda args, defender, scenario, *, horizon: {"beta": 0.36},
    )
    monkeypatch.setattr(direct_eval, "_action_diagnostics", lambda policy, obs, cfg: {})

    result = direct_eval._axis_rule_offset_adapt_and_evaluate(
        args,
        DummyPolicy(),
        scenario,
        probe_obs=np.zeros(3, dtype=np.float32),
    )

    assert result["selection"]["accepted"] is False
    assert result["selection"]["selected"] == "base"
    assert result["selection"]["selected_offset_label"] == "zero"
    assert result["selected_offset"] == pytest.approx([0.0, 0.0, 0.0])
    assert result["evaluation"]["final_clean_acc"] == pytest.approx(0.965)


def test_axis_rule_v2_offset_adaptation_uses_combo_for_low_beta_without_support_queries(monkeypatch):
    import meta_sg.scripts.evaluate_meta_sg_direct as direct_eval

    class DummyPolicy:
        obs_dim = 3
        act_dim = 3

        def get_action(self, obs, noise=0.0):
            return np.zeros(3, dtype=np.float32)

    args = SimpleNamespace(
        H=50,
        selection_horizon=50,
        seed=1212,
        axis_offset_step=0.25,
        axis_rule_beta_threshold=0.35,
        axis_rule_low_beta_threshold=0.34,
        lambda_bd=2.0,
        defender_third_action="neuroclip",
        server_lr_min=0.0,
        server_lr_max=1.0,
        server_lr_penalty_weight=0.0,
    )
    scenario = direct_eval.Scenario(
        "bfl",
        direct_eval._attack_type("bfl"),
        "bfl",
        {},
        seed=1212,
    )

    def fail_if_support_query_runs(*args, **kwargs):
        raise AssertionError("axis_rule_v2_offset should not run support queries")

    def fake_eval(args, policy, scenario):
        offset = np.asarray(getattr(policy, "offset", np.zeros(3)), dtype=np.float32)
        backdoor = 0.533 if offset.tolist() == pytest.approx([-0.25, 0.25, 0.0]) else 0.853
        return {
            "final_clean_acc": 0.956,
            "final_backdoor_acc": backdoor,
            "final_defense_score": 0.956 - 2.0 * backdoor,
        }

    monkeypatch.setattr(direct_eval, "_evaluate_scenario_at", fail_if_support_query_runs)
    monkeypatch.setattr(direct_eval, "_evaluate_scenario", fake_eval)
    monkeypatch.setattr(
        direct_eval,
        "_scenario_initial_action_diagnostics",
        lambda args, defender, scenario, *, horizon: {"beta": 0.329},
    )
    monkeypatch.setattr(direct_eval, "_action_diagnostics", lambda policy, obs, cfg: {})

    result = direct_eval._axis_rule_v2_offset_adapt_and_evaluate(
        args,
        DummyPolicy(),
        scenario,
        probe_obs=np.zeros(3, dtype=np.float32),
    )

    assert result["selection"]["accepted"] is True
    assert result["selection"]["selected_offset_label"] == "alpha_minus_1_beta_plus_1"
    assert result["selected_offset"] == pytest.approx([-0.25, 0.25, 0.0])
    assert result["num_transitions"] == 0
    assert result["evaluation"]["final_backdoor_acc"] == pytest.approx(0.533)


def test_physical_rule_target_adaptation_uses_physical_target_without_support_queries(monkeypatch):
    import meta_sg.scripts.evaluate_meta_sg_direct as direct_eval

    class DummyPolicy:
        obs_dim = 3
        act_dim = 3

        def get_action(self, obs, noise=0.0):
            return np.zeros(3, dtype=np.float32)

    args = SimpleNamespace(
        H=50,
        selection_horizon=50,
        seed=1212,
        axis_offset_step=0.25,
        physical_rule_low_beta_threshold=0.34,
        physical_rule_beta_threshold=0.35,
        physical_rule_low_alpha_target=0.10,
        physical_rule_near_alpha_target=0.15,
        physical_rule_beta_target=0.38,
        lambda_bd=2.0,
        defender_third_action="neuroclip",
        server_lr_min=0.0,
        server_lr_max=1.0,
        server_lr_penalty_weight=0.0,
    )
    scenario = direct_eval.Scenario(
        "rl_backdoor",
        direct_eval._attack_type("rl_backdoor"),
        "rl_backdoor",
        {},
        seed=1212,
    )

    def fail_if_support_query_runs(*args, **kwargs):
        raise AssertionError("physical_rule_target should not run support queries")

    def fake_eval(args, policy, scenario):
        assert getattr(policy, "target_alpha", None) == pytest.approx(0.10)
        assert getattr(policy, "target_beta", None) == pytest.approx(0.38)
        return {
            "final_clean_acc": 0.94,
            "final_backdoor_acc": 0.077,
            "final_defense_score": 0.94 - 2.0 * 0.077,
        }

    monkeypatch.setattr(direct_eval, "_evaluate_scenario_at", fail_if_support_query_runs)
    monkeypatch.setattr(direct_eval, "_evaluate_scenario", fake_eval)
    monkeypatch.setattr(
        direct_eval,
        "_scenario_initial_action_diagnostics",
        lambda args, defender, scenario, *, horizon: {"beta": 0.329},
    )
    monkeypatch.setattr(direct_eval, "_action_diagnostics", lambda policy, obs, cfg: {})

    result = direct_eval._physical_rule_target_adapt_and_evaluate(
        args,
        DummyPolicy(),
        scenario,
        probe_obs=np.zeros(3, dtype=np.float32),
    )

    assert result["selection"]["accepted"] is True
    assert result["selection"]["selected_offset_label"] == "physical_a0.10_b0.38"
    assert result["selection"]["target_alpha"] == pytest.approx(0.10)
    assert result["selection"]["target_beta"] == pytest.approx(0.38)
    assert result["num_transitions"] == 0
    assert result["evaluation"]["final_backdoor_acc"] == pytest.approx(0.077)


def test_physical_target_selector_adaptation_uses_query_clean_guard(monkeypatch):
    import meta_sg.scripts.evaluate_meta_sg_direct as direct_eval

    class DummyPolicy:
        obs_dim = 3
        act_dim = 3

        def get_action(self, obs, noise=0.0):
            return np.zeros(3, dtype=np.float32)

    args = SimpleNamespace(
        H=50,
        selection_horizon=50,
        selection_seed_offset=20_000,
        adaptation_episodes=1,
        seed=1212,
        physical_target_alpha_candidates="0.10,0.14",
        physical_target_beta_target=0.38,
        physical_target_asr_reduction_margin=0.005,
        physical_target_clean_floor=0.92,
        physical_target_clean_drop_tolerance=0.04,
        physical_target_start_round=0,
        physical_target_end_round=25,
        lambda_bd=2.0,
        defender_third_action="neuroclip",
        server_lr_min=0.0,
        server_lr_max=1.0,
        server_lr_penalty_weight=0.0,
    )
    scenario = direct_eval.Scenario(
        "rl_backdoor",
        direct_eval._attack_type("rl_backdoor"),
        "rl_backdoor",
        {},
        seed=1212,
    )

    def target_alpha(policy):
        value = getattr(policy, "target_alpha", None)
        return None if value is None else round(float(value), 2)

    def fake_evaluate_scenario_at(args, policy, scenario, *, seed, horizon):
        alpha = target_alpha(policy)
        by_alpha = {
            None: (0.95, 0.90),
            0.10: (0.90, 0.02),
            0.14: (0.93, 0.06),
        }
        clean, backdoor = by_alpha[alpha]
        return {
            "final_clean_acc": clean,
            "final_backdoor_acc": backdoor,
            "final_defense_score": clean - 2.0 * backdoor,
        }

    def fake_evaluate_scenario(args, policy, scenario):
        assert target_alpha(policy) == pytest.approx(0.14)
        assert getattr(policy, "start_round", None) == 0
        assert getattr(policy, "end_round", None) == 25
        return {
            "final_clean_acc": 0.932,
            "final_backdoor_acc": 0.07,
            "final_defense_score": 0.932 - 2.0 * 0.07,
        }

    monkeypatch.setattr(direct_eval, "_evaluate_scenario_at", fake_evaluate_scenario_at)
    monkeypatch.setattr(direct_eval, "_evaluate_scenario", fake_evaluate_scenario)
    monkeypatch.setattr(direct_eval, "_action_diagnostics", lambda policy, obs, cfg: {})

    result = direct_eval._physical_target_selector_adapt_and_evaluate(
        args,
        DummyPolicy(),
        scenario,
        probe_obs=np.zeros(3, dtype=np.float32),
    )

    assert result["method"] == "physical_target_selector"
    assert result["selection"]["accepted"] is True
    assert result["selection"]["selected_offset_label"] == "physical_a0.14_b0.38"
    assert result["target_alpha"] == pytest.approx(0.14)
    assert result["target_beta"] == pytest.approx(0.38)
    assert result["num_transitions"] == 50 * 1 * 3
    assert result["evaluation"]["final_backdoor_acc"] == pytest.approx(0.07)


def test_physical_target_selector_adaptation_can_select_round_window_candidate(monkeypatch):
    import meta_sg.scripts.evaluate_meta_sg_direct as direct_eval

    class DummyPolicy:
        obs_dim = 3
        act_dim = 3

        def get_action(self, obs, noise=0.0):
            return np.zeros(3, dtype=np.float32)

    args = SimpleNamespace(
        H=50,
        selection_horizon=50,
        selection_seed_offset=20_000,
        adaptation_episodes=1,
        seed=1212,
        physical_target_alpha_candidates="0.10",
        physical_target_beta_target=0.38,
        physical_target_asr_reduction_margin=0.005,
        physical_target_clean_floor=0.92,
        physical_target_clean_drop_tolerance=0.04,
        physical_target_score_slack=0.0,
        physical_target_start_round=0,
        physical_target_end_round=None,
        physical_target_end_round_candidates="full,40",
        lambda_bd=2.0,
        defender_third_action="neuroclip",
        server_lr_min=0.0,
        server_lr_max=1.0,
        server_lr_penalty_weight=0.0,
    )
    scenario = direct_eval.Scenario(
        "rl_backdoor",
        direct_eval._attack_type("rl_backdoor"),
        "rl_backdoor",
        {},
        seed=1212,
    )

    def policy_key(policy):
        alpha = getattr(policy, "target_alpha", None)
        alpha = None if alpha is None else round(float(alpha), 2)
        return alpha, getattr(policy, "end_round", None)

    def fake_evaluate_scenario_at(args, policy, scenario, *, seed, horizon):
        clean, backdoor = {
            (None, None): (0.95, 0.90),
            (0.10, None): (0.93, 0.04),
            (0.10, 40): (0.94, 0.01),
        }[policy_key(policy)]
        return {
            "final_clean_acc": clean,
            "final_backdoor_acc": backdoor,
            "final_defense_score": clean - 2.0 * backdoor,
        }

    def fake_evaluate_scenario(args, policy, scenario):
        assert policy_key(policy) == (0.10, 40)
        return {
            "final_clean_acc": 0.95,
            "final_backdoor_acc": 0.25,
            "final_defense_score": 0.45,
        }

    monkeypatch.setattr(direct_eval, "_evaluate_scenario_at", fake_evaluate_scenario_at)
    monkeypatch.setattr(direct_eval, "_evaluate_scenario", fake_evaluate_scenario)
    monkeypatch.setattr(direct_eval, "_action_diagnostics", lambda policy, obs, cfg: {})

    result = direct_eval._physical_target_selector_adapt_and_evaluate(
        args,
        DummyPolicy(),
        scenario,
        probe_obs=np.zeros(3, dtype=np.float32),
    )

    labels = [record["offset_label"] for record in result["candidate_scores"]]
    assert labels == ["zero", "physical_a0.10_b0.38_full", "physical_a0.10_b0.38_w0_40"]
    assert result["selection"]["accepted"] is True
    assert result["selection"]["selected_offset_label"] == "physical_a0.10_b0.38_w0_40"
    assert result["selection"]["target_end_round"] == 40
    assert result["physical_target_end_round"] == 40
    assert result["num_transitions"] == 50 * 1 * 3
    assert result["evaluation"]["final_clean_acc"] == pytest.approx(0.95)


def test_physical_target_selector_deployment_clean_recovery_mode_selects_clean_window(monkeypatch):
    import meta_sg.scripts.evaluate_meta_sg_direct as direct_eval

    class DummyPolicy:
        obs_dim = 3
        act_dim = 3

        def get_action(self, obs, noise=0.0):
            return np.zeros(3, dtype=np.float32)

    args = SimpleNamespace(
        H=50,
        selection_horizon=50,
        selection_seed_offset=20_000,
        adaptation_episodes=1,
        seed=1212,
        physical_target_alpha_candidates="0.10",
        physical_target_beta_target=0.38,
        physical_target_asr_reduction_margin=0.005,
        physical_target_clean_floor=0.92,
        physical_target_clean_drop_tolerance=0.04,
        physical_target_score_slack=0.0,
        physical_target_selection_mode="deployment_clean_recovery",
        physical_target_deployment_clean_floor=0.95,
        physical_target_deployment_asr_ceiling=0.30,
        physical_target_start_round=0,
        physical_target_end_round=None,
        physical_target_end_round_candidates="full,40",
        lambda_bd=2.0,
        defender_third_action="neuroclip",
        server_lr_min=0.0,
        server_lr_max=1.0,
        server_lr_penalty_weight=0.0,
    )
    scenario = direct_eval.Scenario(
        "rl_backdoor",
        direct_eval._attack_type("rl_backdoor"),
        "rl_backdoor",
        {},
        seed=1212,
    )

    def policy_key(policy):
        alpha = getattr(policy, "target_alpha", None)
        alpha = None if alpha is None else round(float(alpha), 2)
        return alpha, getattr(policy, "end_round", None)

    def fake_evaluate_scenario_at(args, policy, scenario, *, seed, horizon):
        clean, backdoor = {
            (None, None): (0.943, 0.443),
            (0.10, None): (0.930, 0.007),
            (0.10, 40): (0.920, 0.063),
        }[policy_key(policy)]
        return {
            "final_clean_acc": clean,
            "final_backdoor_acc": backdoor,
            "final_defense_score": clean - 2.0 * backdoor,
        }

    deployment_calls = []

    def fake_evaluate_scenario(args, policy, scenario):
        deployment_calls.append(policy_key(policy))
        clean, backdoor = {
            (None, None): (0.960, 0.860),
            (0.10, None): (0.940, 0.077),
            (0.10, 40): (0.950, 0.250),
        }[policy_key(policy)]
        return {
            "final_clean_acc": clean,
            "final_backdoor_acc": backdoor,
            "final_defense_score": clean - 2.0 * backdoor,
        }

    monkeypatch.setattr(direct_eval, "_evaluate_scenario_at", fake_evaluate_scenario_at)
    monkeypatch.setattr(direct_eval, "_evaluate_scenario", fake_evaluate_scenario)
    monkeypatch.setattr(direct_eval, "_action_diagnostics", lambda policy, obs, cfg: {})

    result = direct_eval._physical_target_selector_adapt_and_evaluate(
        args,
        DummyPolicy(),
        scenario,
        probe_obs=np.zeros(3, dtype=np.float32),
    )

    assert result["selection"]["mode"] == "deployment_clean_recovery_physical_target_selector"
    assert result["selection"]["selected_offset_label"] == "physical_a0.10_b0.38_w0_40"
    assert result["selection"]["query_selected_offset_label"] == "physical_a0.10_b0.38_full"
    assert result["physical_target_end_round"] == 40
    assert result["evaluation"]["final_clean_acc"] == pytest.approx(0.95)
    assert result["evaluation"]["final_backdoor_acc"] == pytest.approx(0.25)
    assert result["num_deployment_validations"] == 3
    assert result["num_transitions"] == 50 * 1 * 3 + 50 * 3
    assert deployment_calls == [(None, None), (0.10, None), (0.10, 40)]


def test_physical_target_selector_deployment_clean_drop_guard_keeps_base(monkeypatch):
    import meta_sg.scripts.evaluate_meta_sg_direct as direct_eval

    class DummyPolicy:
        obs_dim = 3
        act_dim = 3

        def get_action(self, obs, noise=0.0):
            return np.zeros(3, dtype=np.float32)

    args = SimpleNamespace(
        H=50,
        selection_horizon=50,
        selection_seed_offset=20_000,
        adaptation_episodes=1,
        seed=1312,
        physical_target_alpha_candidates="0.10",
        physical_target_beta_target=0.38,
        physical_target_asr_reduction_margin=0.005,
        physical_target_clean_floor=None,
        physical_target_clean_drop_tolerance=0.04,
        physical_target_score_slack=0.0,
        physical_target_selection_mode="deployment_clean_recovery",
        physical_target_deployment_clean_floor=None,
        physical_target_deployment_clean_drop_tolerance=0.02,
        physical_target_deployment_asr_ceiling=0.30,
        physical_target_start_round=0,
        physical_target_end_round=40,
        physical_target_end_round_candidates=None,
        lambda_bd=2.0,
        defender_third_action="neuroclip",
        server_lr_min=0.0,
        server_lr_max=1.0,
        server_lr_penalty_weight=0.0,
    )
    scenario = direct_eval.Scenario(
        "rl_backdoor",
        direct_eval._attack_type("rl_backdoor"),
        "rl_backdoor",
        {},
        seed=1312,
    )

    def policy_key(policy):
        alpha = getattr(policy, "target_alpha", None)
        alpha = None if alpha is None else round(float(alpha), 2)
        return alpha, getattr(policy, "end_round", None)

    def fake_evaluate_scenario_at(args, policy, scenario, *, seed, horizon):
        clean, backdoor = {
            (None, None): (0.930, 0.967),
            (0.10, 40): (0.907, 0.590),
        }[policy_key(policy)]
        return {
            "final_clean_acc": clean,
            "final_backdoor_acc": backdoor,
            "final_defense_score": clean - 2.0 * backdoor,
        }

    def fake_evaluate_scenario(args, policy, scenario):
        clean, backdoor = {
            (None, None): (0.944, 0.767),
            (0.10, 40): (0.916, 0.020),
        }[policy_key(policy)]
        return {
            "final_clean_acc": clean,
            "final_backdoor_acc": backdoor,
            "final_defense_score": clean - 2.0 * backdoor,
        }

    monkeypatch.setattr(direct_eval, "_evaluate_scenario_at", fake_evaluate_scenario_at)
    monkeypatch.setattr(direct_eval, "_evaluate_scenario", fake_evaluate_scenario)
    monkeypatch.setattr(direct_eval, "_action_diagnostics", lambda policy, obs, cfg: {})

    result = direct_eval._physical_target_selector_adapt_and_evaluate(
        args,
        DummyPolicy(),
        scenario,
        probe_obs=np.zeros(3, dtype=np.float32),
    )

    assert result["selection"]["accepted"] is False
    assert result["selection"]["selected_offset_label"] == "zero"
    assert result["selection"]["query_selected_offset_label"] == "physical_a0.10_b0.38"
    assert result["selection"]["selection_stage"] == "deployment_rejected"
    assert result["evaluation"]["final_clean_acc"] == pytest.approx(0.944)
    assert result["evaluation"]["final_backdoor_acc"] == pytest.approx(0.767)
    assert result["deployed_offset"] == pytest.approx([0.0, 0.0, 0.0])


def test_axis_offset_adaptation_prunes_candidate_after_pessimistic_asr_violation(monkeypatch):
    import meta_sg.scripts.evaluate_meta_sg_direct as direct_eval

    class DummyPolicy:
        obs_dim = 3
        act_dim = 3

        def get_action(self, obs, noise=0.0):
            return np.zeros(3, dtype=np.float32)

    args = SimpleNamespace(
        H=50,
        selection_horizon=50,
        seed=912,
        selection_seed_offset=20_000,
        adaptation_episodes=3,
        axis_offset_step=0.25,
        axis_offset_max_steps=1,
        axis_offset_asr_reduction_margin=0.005,
        axis_offset_pessimistic_asr_margin=0.0,
        axis_offset_clean_floor=0.89,
        axis_offset_clean_drop_tolerance=0.08,
        lambda_bd=2.0,
        defender_third_action="neuroclip",
        server_lr_min=0.0,
        server_lr_max=1.0,
        server_lr_penalty_weight=0.0,
    )
    scenario = direct_eval.Scenario(
        "rl_backdoor",
        direct_eval._attack_type("rl_backdoor"),
        "rl_backdoor",
        {},
        seed=100,
    )
    call_counts: dict[tuple[float, ...], int] = {}

    def offset_key(policy):
        return tuple(np.round(np.asarray(getattr(policy, "offset", np.zeros(3)), dtype=np.float32), 4).tolist())

    def fake_evaluate_scenario_at(args, policy, scenario, *, seed, horizon):
        key = offset_key(policy)
        call_counts[key] = call_counts.get(key, 0) + 1
        episode = seed - int(scenario.seed) - int(args.selection_seed_offset)
        backdoor_by_key = {
            (0.0, 0.0, 0.0): [0.90, 0.88, 0.86],
            (-0.25, 0.0, 0.0): [0.94, 0.80, 0.70],
            (0.25, 0.0, 0.0): [0.95, 0.95, 0.95],
            (0.0, 0.25, 0.0): [0.80, 0.74, 0.70],
            (0.0, -0.25, 0.0): [0.96, 0.96, 0.96],
            (-0.25, 0.25, 0.0): [0.93, 0.70, 0.65],
        }
        backdoor = backdoor_by_key[key][episode]
        return {
            "final_clean_acc": 0.95,
            "final_backdoor_acc": backdoor,
            "final_defense_score": 0.95 - 2.0 * backdoor,
        }

    def fake_evaluate_scenario(args, policy, scenario):
        key = offset_key(policy)
        backdoor = 0.70 if key == (0.0, 0.25, 0.0) else 0.90
        return {
            "final_clean_acc": 0.95,
            "final_backdoor_acc": backdoor,
            "final_defense_score": 0.95 - 2.0 * backdoor,
        }

    monkeypatch.setattr(direct_eval, "_evaluate_scenario_at", fake_evaluate_scenario_at)
    monkeypatch.setattr(direct_eval, "_evaluate_scenario", fake_evaluate_scenario)
    monkeypatch.setattr(direct_eval, "_action_diagnostics", lambda policy, obs, cfg: {})

    result = direct_eval._axis_offset_adapt_and_evaluate(
        args,
        DummyPolicy(),
        scenario,
        probe_obs=np.zeros(3, dtype=np.float32),
    )

    records = {item["offset_label"]: item for item in result["candidate_scores"]}
    assert records["alpha_minus_1"]["pruned"] is True
    assert records["alpha_minus_1"]["query_episodes_completed"] == 1
    assert records["beta_plus_1"]["pruned"] is False
    assert records["beta_plus_1"]["query_episodes_completed"] == 3
    assert result["selection"]["accepted"] is True
    assert result["selection"]["selected_offset_label"] == "beta_plus_1"
    assert result["num_transitions"] < 50 * 3 * 6
    assert call_counts[(-0.25, 0.0, 0.0)] == 1
    assert call_counts[(0.0, 0.25, 0.0)] == 3


def test_residual_adapter_uses_support_summary_and_query_guard(monkeypatch):
    import meta_sg.scripts.evaluate_meta_sg_direct as direct_eval

    class DummyPolicy:
        obs_dim = 2
        act_dim = 3

        def get_action(self, obs, noise=0.0):
            return np.zeros(3, dtype=np.float32)

    def fake_evaluate_scenario_at(args, policy, scenario, *, seed, horizon):
        offset = np.asarray(getattr(policy, "offset", np.zeros(3, dtype=np.float32)), dtype=np.float32)
        if seed < int(args.selection_seed_offset):
            support_step = seed - 10_000
            return {
                "final_defense_score": 0.80 + 0.04 * support_step,
                "final_clean_acc": 0.82 + 0.03 * support_step,
                "final_backdoor_acc": 0.02 - 0.01 * support_step,
                "mean_defender_reward": 0.70 + 0.04 * support_step,
            }
        return {
            "final_defense_score": 1.0 + max(0.0, float(offset[0])),
            "final_clean_acc": 1.0 + max(0.0, float(offset[0])),
            "final_backdoor_acc": 0.0,
            "mean_defender_reward": 1.0 + max(0.0, float(offset[0])),
        }

    def fake_evaluate_scenario(args, policy, scenario):
        offset = np.asarray(getattr(policy, "offset", np.zeros(3, dtype=np.float32)), dtype=np.float32)
        return {
            "final_defense_score": 1.0 + max(0.0, float(offset[0])),
            "final_clean_acc": 1.0 + max(0.0, float(offset[0])),
            "final_backdoor_acc": 0.0,
            "mean_defender_reward": 1.0 + max(0.0, float(offset[0])),
        }

    monkeypatch.setattr(direct_eval, "_evaluate_scenario_at", fake_evaluate_scenario_at)
    monkeypatch.setattr(direct_eval, "_evaluate_scenario", fake_evaluate_scenario)

    args = direct_eval.parse_args(
        [
            "--checkpoint",
            "dummy.pt",
            "--output-json",
            "out.json",
            "--few-shot",
            "--few-shot-method",
            "residual_adapter",
            "--adaptation-horizon",
            "1",
            "--adaptation-episodes",
            "2",
            "--residual-bound",
            "0.2",
            "--residual-candidates",
            "3",
            "--residual-query-episodes",
            "2",
            "--residual-degradation-margin",
            "0.001",
        ]
    )
    scenario = direct_eval.Scenario(
        name="ipm",
        attack_type=ATTACK_DOMAIN["ipm"],
        attack_name="ipm",
        patch={},
        seed=0,
    )

    result = direct_eval._residual_adapter_adapt_and_evaluate(
        args,
        DummyPolicy(),
        scenario,
        probe_obs=np.zeros(2, dtype=np.float32),
    )

    assert result["method"] == "residual_adapter"
    assert result["selection"]["accepted"] is True
    assert result["selection"]["score_gain"] > 0.001
    assert result["selected_offset"][0] > 0.0
    assert result["evaluation"]["final_defense_score"] > 1.0


def test_residual_adapter_can_dump_supervision_sample(monkeypatch, tmp_path):
    import json

    import meta_sg.scripts.evaluate_meta_sg_direct as direct_eval

    class DummyPolicy:
        obs_dim = 2
        act_dim = 3

        def get_action(self, obs, noise=0.0):
            return np.zeros(3, dtype=np.float32)

    def fake_evaluate_scenario_at(args, policy, scenario, *, seed, horizon):
        offset = np.asarray(getattr(policy, "offset", np.zeros(3, dtype=np.float32)), dtype=np.float32)
        if seed < int(args.selection_seed_offset):
            return {
                "final_defense_score": 0.8,
                "final_clean_acc": 0.8,
                "final_backdoor_acc": 0.01,
                "mean_defender_reward": 0.7,
            }
        return {
            "final_defense_score": 1.0 + max(0.0, float(offset[0])),
            "final_clean_acc": 1.0 + max(0.0, float(offset[0])),
            "final_backdoor_acc": 0.0,
            "mean_defender_reward": 1.0 + max(0.0, float(offset[0])),
        }

    def fake_evaluate_scenario(args, policy, scenario):
        offset = np.asarray(getattr(policy, "offset", np.zeros(3, dtype=np.float32)), dtype=np.float32)
        return {
            "final_defense_score": 1.0 + max(0.0, float(offset[0])),
            "final_clean_acc": 1.0 + max(0.0, float(offset[0])),
            "final_backdoor_acc": 0.0,
            "mean_defender_reward": 1.0 + max(0.0, float(offset[0])),
        }

    monkeypatch.setattr(direct_eval, "_evaluate_scenario_at", fake_evaluate_scenario_at)
    monkeypatch.setattr(direct_eval, "_evaluate_scenario", fake_evaluate_scenario)
    samples_path = tmp_path / "samples.jsonl"
    args = direct_eval.parse_args(
        [
            "--checkpoint",
            "dummy.pt",
            "--output-json",
            "out.json",
            "--few-shot",
            "--few-shot-method",
            "residual_adapter",
            "--adaptation-horizon",
            "1",
            "--adaptation-episodes",
            "1",
            "--residual-bound",
            "0.2",
            "--residual-candidates",
            "3",
            "--residual-query-episodes",
            "1",
            "--residual-degradation-margin",
            "0.001",
            "--residual-dump-supervision-jsonl",
            str(samples_path),
        ]
    )
    scenario = direct_eval.Scenario("bfl", ATTACK_DOMAIN["bfl"], "bfl", {}, seed=0)

    direct_eval._residual_adapter_adapt_and_evaluate(
        args,
        DummyPolicy(),
        scenario,
        probe_obs=np.zeros(2, dtype=np.float32),
    )

    records = [json.loads(line) for line in samples_path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["attack_name"] == "bfl"
    assert records[0]["attack_objective"] == "targeted"
    assert records[0]["selected_offset"][0] >= 0.0
    assert "support_summary" in records[0]
    assert "candidate_scores" in records[0]


def test_cem_guarded_selection_uses_independent_validation_seed(monkeypatch):
    import meta_sg.scripts.evaluate_meta_sg_direct as direct_eval

    class DummyPolicy:
        obs_dim = 2
        act_dim = 3

        def get_action(self, obs, noise=0.0):
            return np.zeros(3, dtype=np.float32)

    def score_for_policy(policy, *, seed):
        has_offset = hasattr(policy, "offset") and bool(np.any(np.asarray(policy.offset) != 0.0))
        if seed >= 20_000:
            return 0.50 if has_offset else 1.00
        return 0.90 if has_offset else 0.10

    def fake_evaluate_scenario_at(args, policy, scenario, *, seed, horizon):
        return {"final_defense_score": score_for_policy(policy, seed=seed)}

    def fake_evaluate_scenario(args, policy, scenario):
        return {"final_defense_score": 0.50 if hasattr(policy, "offset") else 1.00}

    monkeypatch.setattr(direct_eval, "_evaluate_scenario_at", fake_evaluate_scenario_at)
    monkeypatch.setattr(direct_eval, "_evaluate_scenario", fake_evaluate_scenario)

    args = direct_eval.parse_args(
        [
            "--checkpoint",
            "dummy.pt",
            "--output-json",
            "out.json",
            "--few-shot",
            "--few-shot-method",
            "cem_offset",
            "--few-shot-selection",
            "guarded",
            "--adaptation-horizon",
            "1",
            "--adaptation-episodes",
            "1",
            "--cem-iterations",
            "1",
            "--cem-population",
            "3",
            "--cem-elites",
            "1",
        ]
    )
    scenario = direct_eval.Scenario(
        name="ipm",
        attack_type=ATTACK_DOMAIN["ipm"],
        attack_name="ipm",
        patch={},
        seed=0,
    )

    result = direct_eval._cem_offset_adapt_and_evaluate(
        args,
        DummyPolicy(),
        scenario,
        probe_obs=np.zeros(2, dtype=np.float32),
    )

    selection = result["selection"]
    assert selection["accepted"] is False
    assert selection["selected"] == "base"
    assert selection["validation_seed"] == 20_000
    assert selection["adapted_support"]["mean_defense_score"] > selection["base_support"]["mean_defense_score"]
    assert selection["adapted_validation"]["final_defense_score"] < selection["base_validation"]["final_defense_score"]
    assert result["evaluation"]["final_defense_score"] == pytest.approx(1.0)


def test_guarded_few_shot_selection_requires_margin_before_accepting_adapted_policy():
    from meta_sg.scripts.evaluate_meta_sg_direct import _guarded_selection_decision

    rejected = _guarded_selection_decision(base_score=0.9000, adapted_score=0.9010, margin=0.0020)
    accepted = _guarded_selection_decision(base_score=0.9000, adapted_score=0.9025, margin=0.0020)

    assert rejected["accepted"] is False
    assert rejected["selected"] == "base"
    assert rejected["score_gain"] == pytest.approx(0.0010)
    assert accepted["accepted"] is True
    assert accepted["selected"] == "adapted"
    assert accepted["score_gain"] == pytest.approx(0.0025)


def test_action_offset_policy_adds_raw_action_delta_and_clips_to_action_bounds():
    from meta_sg.scripts.evaluate_meta_sg_direct import ActionOffsetPolicy

    class BasePolicy:
        obs_dim = 2
        act_dim = 3

        def get_action(self, obs, noise=0.0):
            assert noise == 0.0
            return np.asarray([0.8, -0.9, 0.1], dtype=np.float32)

    policy = ActionOffsetPolicy(BasePolicy(), np.asarray([0.5, -0.5, 0.2], dtype=np.float32))

    action = policy.get_action(np.zeros(2, dtype=np.float32), noise=0.0)

    assert action.tolist() == pytest.approx([1.0, -1.0, 0.3])
    assert policy.obs_dim == 2
    assert policy.act_dim == 3


def test_action_offset_candidates_include_zero_and_coordinate_steps():
    from meta_sg.scripts.evaluate_meta_sg_direct import _offset_candidates

    candidates = _offset_candidates(act_dim=3, step=0.25)

    assert np.allclose(
        np.stack(candidates),
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.25, 0.0, 0.0],
                [-0.25, 0.0, 0.0],
                [0.0, 0.25, 0.0],
                [0.0, -0.25, 0.0],
                [0.0, 0.0, 0.25],
                [0.0, 0.0, -0.25],
            ],
            dtype=np.float32,
        ),
    )


def test_sac_agent_can_initialize_from_td3_and_update_on_replay_buffer():
    from meta_sg.learning.sac import SACAgent

    obs_dim = 4
    act_dim = 2
    cfg = TD3Config(hidden_dim=8, batch_size=2, buffer_capacity=16, warmup_steps=0)
    td3 = TD3Agent(obs_dim, act_dim, cfg, torch.device("cpu"))
    sac = SACAgent.from_td3(td3, alpha=0.2)
    obs = np.linspace(-1.0, 1.0, obs_dim, dtype=np.float32)

    td3_action = td3.get_action(obs, noise=0.0)
    sac_action = sac.get_action(obs, noise=0.0)

    assert sac_action.shape == (act_dim,)
    assert np.allclose(sac_action, td3_action, atol=1e-5)

    buffer = ReplayBuffer(cfg.buffer_capacity, obs_dim, act_dim)
    for idx in range(3):
        state = np.full(obs_dim, idx, dtype=np.float32)
        action = sac.get_action(state, noise=0.0)
        buffer.add(state, action, 1.0, state + 0.1, False)

    losses = sac.update(buffer)

    assert "actor_loss" in losses
    assert "critic_loss" in losses
    assert "alpha" in losses


def test_pretraining_script_resolves_auto_device_to_cuda_when_available(monkeypatch):
    from meta_sg.scripts.run_meta_sg_pretraining import resolve_torch_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert str(resolve_torch_device("auto")) == "cuda:0"
    assert str(resolve_torch_device("cpu")) == "cpu"


def test_pretraining_script_threads_resolved_device_to_fl_sandbox_config(monkeypatch):
    from meta_sg.scripts.run_meta_sg_pretraining import build_sandbox_config, parse_args

    args = parse_args(
        [
            "--backend",
            "fl_sandbox",
            "--device",
            "cuda:0",
            "--fl-parallel-clients",
            "4",
            "--fl-num-workers",
            "2",
        ]
    )

    config = build_sandbox_config(args)

    assert config.runtime.device == "cuda:0"
    assert config.runtime.parallel_clients == 4
    assert config.runtime.num_workers == 2


def test_pretraining_sandbox_config_accepts_attack_horizon_and_seed_overrides():
    from meta_sg.scripts.run_meta_sg_pretraining import build_sandbox_config, parse_args

    args = parse_args(["--H", "20", "--seed", "7", "--batch-size", "16"])

    config = build_sandbox_config(
        args,
        attack_type=ATTACK_DOMAIN["bfl"],
        horizon=50,
        seed=3070000,
    )

    assert config.attacker.type == "bfl"
    assert config.runtime.rounds == 50
    assert config.runtime.seed == 3070000
    assert config.runtime.batch_size == 16
    assert config.attacker.rl_attack_start_round == 6
    assert config.attacker.rl_policy_train_end_round == 50


def test_pretraining_sandbox_config_can_override_fl_batch_size():
    from meta_sg.scripts.run_meta_sg_pretraining import build_sandbox_config, parse_args

    args = parse_args(["--batch-size", "16", "--fl-batch-size", "64"])

    config = build_sandbox_config(args)

    assert config.runtime.batch_size == 64


def test_attack_task_runner_threads_attack_horizon_and_seed_to_coordinator_factory():
    calls = []

    def factory(*, attack_type=None, horizon=None, seed=None):
        calls.append((attack_type.name if attack_type else None, horizon, seed))
        return ConstantPostMetricCoordinator(num_clients=6, num_attackers=1, seed=0)

    obs_dim = 1290
    act_dim = 3
    td3_cfg = TD3Config(hidden_dim=8, batch_size=2, buffer_capacity=16, warmup_steps=0)
    meta_cfg = MetaSGConfig(
        T=1,
        K=1,
        H_mnist=1,
        l=0,
        N_A=0,
        post_br_defender_updates=0,
        meta_objective="query_gated_reptile",
        query_horizon=2,
        query_seed_offset=50_000,
        lambda_bd=0.0,
        reward_mode="accuracy",
        eval_every=1,
        warmup_steps=0,
    )
    attacker_buffers = {
        "ipm": ReplayBuffer(td3_cfg.buffer_capacity, obs_dim, act_dim),
    }
    runner = AttackTaskRunner(
        coordinator_factory=factory,
        td3_config=td3_cfg,
        meta_config=meta_cfg,
        obs_dim=obs_dim,
        act_dim=act_dim,
        attacker_agents={},
        attacker_buffers=attacker_buffers,
        best_response=AttackerBestResponse({}, {}, n_a=0),
    )
    defender = TD3Agent(obs_dim, act_dim, td3_cfg, torch.device("cpu"))

    runner.run(ATTACK_DOMAIN["ipm"], defender, seed_base=123)

    assert calls == [
        ("ipm", 1, 123),
        ("ipm", 2, 50123),
        ("ipm", 2, 50123),
    ]


def test_meta_sg_config_has_single_reptile_step_field():
    cfg = MetaSGConfig()

    assert cfg.meta_update_step == pytest.approx(1.0)
    assert cfg.task_sampler == "iid"
    assert cfg.meta_objective == "reptile"
    assert cfg.query_horizon is None
    assert cfg.defender_third_action == "neuroclip"
    assert not hasattr(cfg, "kappa_D")
    assert not hasattr(cfg, "kappa_A")


def test_query_gated_reptile_selects_only_query_improving_adaptations():
    from meta_sg.learning.meta_sg_trainer import _meta_update_adapted_params
    from meta_sg.learning.task_runner import TaskResult

    accepted_params = {"actor.weight": torch.tensor([1.0])}
    rejected_params = {"actor.weight": torch.tensor([2.0])}
    attack = ATTACK_DOMAIN["ipm"]
    results = [
        TaskResult(
            attack_type=attack,
            adapted_params=accepted_params,
            mean_defender_reward=0.0,
            mean_attacker_reward=0.0,
            defender_reward_sum=0.0,
            trajectories_collected=1,
            transitions_collected=1,
            query_base_reward=0.80,
            query_adapted_reward=0.81,
            query_gain=0.01,
        ),
        TaskResult(
            attack_type=attack,
            adapted_params=rejected_params,
            mean_defender_reward=0.0,
            mean_attacker_reward=0.0,
            defender_reward_sum=0.0,
            trajectories_collected=1,
            transitions_collected=1,
            query_base_reward=0.80,
            query_adapted_reward=0.799,
            query_gain=-0.001,
        ),
    ]

    selected = _meta_update_adapted_params(
        results,
        meta_objective="query_gated_reptile",
        query_accept_margin=0.0,
    )

    assert selected == [accepted_params]


def test_query_gated_reptile_rejects_adaptation_below_clean_floor():
    from meta_sg.learning.meta_sg_trainer import _meta_update_adapted_params
    from meta_sg.learning.task_runner import TaskResult

    safe_params = {"actor.weight": torch.tensor([1.0])}
    risky_params = {"actor.weight": torch.tensor([2.0])}
    attack = ATTACK_DOMAIN["rl"]
    results = [
        TaskResult(
            attack_type=attack,
            adapted_params=safe_params,
            mean_defender_reward=0.0,
            mean_attacker_reward=0.0,
            defender_reward_sum=0.0,
            trajectories_collected=1,
            transitions_collected=1,
            query_base_reward=0.80,
            query_adapted_reward=0.81,
            query_gain=0.01,
            query_base_clean_acc=0.90,
            query_adapted_clean_acc=0.895,
        ),
        TaskResult(
            attack_type=attack,
            adapted_params=risky_params,
            mean_defender_reward=0.0,
            mean_attacker_reward=0.0,
            defender_reward_sum=0.0,
            trajectories_collected=1,
            transitions_collected=1,
            query_base_reward=0.80,
            query_adapted_reward=0.83,
            query_gain=0.03,
            query_base_clean_acc=0.90,
            query_adapted_clean_acc=0.79,
        ),
    ]

    selected = _meta_update_adapted_params(
        results,
        meta_objective="query_gated_reptile",
        query_accept_margin=0.0,
        query_clean_floor=0.89,
        query_clean_drop_tolerance=0.02,
    )

    assert selected == [safe_params]


def test_query_gated_reptile_rejects_adaptation_above_backdoor_ceiling():
    from meta_sg.learning.meta_sg_trainer import _meta_update_adapted_params
    from meta_sg.learning.task_runner import TaskResult

    safe_params = {"actor.weight": torch.tensor([1.0])}
    risky_params = {"actor.weight": torch.tensor([2.0])}
    attack = ATTACK_DOMAIN["bfl"]
    results = [
        TaskResult(
            attack_type=attack,
            adapted_params=safe_params,
            mean_defender_reward=0.0,
            mean_attacker_reward=0.0,
            defender_reward_sum=0.0,
            trajectories_collected=1,
            transitions_collected=1,
            query_base_reward=0.80,
            query_adapted_reward=0.82,
            query_gain=0.02,
            query_base_clean_acc=0.91,
            query_adapted_clean_acc=0.90,
            query_base_backdoor_acc=0.02,
            query_adapted_backdoor_acc=0.03,
        ),
        TaskResult(
            attack_type=attack,
            adapted_params=risky_params,
            mean_defender_reward=0.0,
            mean_attacker_reward=0.0,
            defender_reward_sum=0.0,
            trajectories_collected=1,
            transitions_collected=1,
            query_base_reward=0.80,
            query_adapted_reward=0.84,
            query_gain=0.04,
            query_base_clean_acc=0.92,
            query_adapted_clean_acc=0.91,
            query_base_backdoor_acc=0.02,
            query_adapted_backdoor_acc=0.08,
        ),
    ]

    selected = _meta_update_adapted_params(
        results,
        meta_objective="query_gated_reptile",
        query_accept_margin=0.0,
        query_clean_floor=0.89,
        query_clean_drop_tolerance=0.02,
        query_backdoor_ceiling=0.05,
        query_backdoor_increase_tolerance=0.02,
    )

    assert selected == [safe_params]


def test_query_gated_reptile_accepts_high_backdoor_when_adaptation_reduces_asr():
    from meta_sg.learning.meta_sg_trainer import _meta_update_adapted_params
    from meta_sg.learning.task_runner import TaskResult

    improved_params = {"actor.weight": torch.tensor([1.0])}
    insufficient_params = {"actor.weight": torch.tensor([2.0])}
    attack = ATTACK_DOMAIN["bfl"]
    results = [
        TaskResult(
            attack_type=attack,
            adapted_params=improved_params,
            mean_defender_reward=0.0,
            mean_attacker_reward=0.0,
            defender_reward_sum=0.0,
            trajectories_collected=1,
            transitions_collected=1,
            query_base_reward=-0.90,
            query_adapted_reward=-0.80,
            query_gain=0.10,
            query_base_clean_acc=0.95,
            query_adapted_clean_acc=0.94,
            query_base_backdoor_acc=0.90,
            query_adapted_backdoor_acc=0.82,
        ),
        TaskResult(
            attack_type=attack,
            adapted_params=insufficient_params,
            mean_defender_reward=0.0,
            mean_attacker_reward=0.0,
            defender_reward_sum=0.0,
            trajectories_collected=1,
            transitions_collected=1,
            query_base_reward=-0.90,
            query_adapted_reward=-0.88,
            query_gain=0.02,
            query_base_clean_acc=0.95,
            query_adapted_clean_acc=0.94,
            query_base_backdoor_acc=0.90,
            query_adapted_backdoor_acc=0.88,
        ),
    ]

    selected = _meta_update_adapted_params(
        results,
        meta_objective="query_gated_reptile",
        query_accept_margin=0.0,
        query_clean_floor=0.89,
        query_clean_drop_tolerance=0.02,
        query_backdoor_ceiling=0.65,
        query_backdoor_increase_tolerance=0.0,
        query_backdoor_improvement_margin=0.05,
    )

    assert selected == [improved_params]


def test_query_targeted_reptile_prefers_asr_reduction_over_score_gain():
    from meta_sg.learning.meta_sg_trainer import _meta_update_adapted_params
    from meta_sg.learning.task_runner import TaskResult

    asr_reduction_params = {"actor.weight": torch.tensor([1.0])}
    score_gain_params = {"actor.weight": torch.tensor([2.0])}
    attack = ATTACK_DOMAIN["bfl"]
    results = [
        TaskResult(
            attack_type=attack,
            adapted_params=score_gain_params,
            mean_defender_reward=0.0,
            mean_attacker_reward=0.0,
            defender_reward_sum=0.0,
            trajectories_collected=1,
            transitions_collected=1,
            query_base_reward=-0.90,
            query_adapted_reward=-0.82,
            query_gain=0.08,
            query_base_clean_acc=0.95,
            query_adapted_clean_acc=0.95,
            query_base_backdoor_acc=0.90,
            query_adapted_backdoor_acc=0.88,
        ),
        TaskResult(
            attack_type=attack,
            adapted_params=asr_reduction_params,
            mean_defender_reward=0.0,
            mean_attacker_reward=0.0,
            defender_reward_sum=0.0,
            trajectories_collected=1,
            transitions_collected=1,
            query_base_reward=-0.90,
            query_adapted_reward=-0.91,
            query_gain=-0.01,
            query_base_clean_acc=0.95,
            query_adapted_clean_acc=0.94,
            query_base_backdoor_acc=0.90,
            query_adapted_backdoor_acc=0.80,
        ),
    ]

    selected = _meta_update_adapted_params(
        results,
        meta_objective="query_targeted_reptile",
        query_accept_margin=0.0,
        query_clean_floor=0.89,
        query_clean_drop_tolerance=0.02,
        query_targeted_asr_reduction_margin=0.05,
        query_targeted_min_base_backdoor=0.50,
    )

    assert selected == [asr_reduction_params]


def test_query_targeted_reptile_keeps_clean_floor_as_hard_rollback():
    from meta_sg.learning.meta_sg_trainer import _meta_update_adapted_params
    from meta_sg.learning.task_runner import TaskResult

    params = {"actor.weight": torch.tensor([1.0])}
    selected = _meta_update_adapted_params(
        [
            TaskResult(
                attack_type=ATTACK_DOMAIN["bfl"],
                adapted_params=params,
                mean_defender_reward=0.0,
                mean_attacker_reward=0.0,
                defender_reward_sum=0.0,
                trajectories_collected=1,
                transitions_collected=1,
                query_base_reward=-0.90,
                query_adapted_reward=-0.70,
                query_gain=0.20,
                query_base_clean_acc=0.95,
                query_adapted_clean_acc=0.84,
                query_clean_drop=0.11,
                query_base_backdoor_acc=0.95,
                query_adapted_backdoor_acc=0.70,
            )
        ],
        meta_objective="query_targeted_reptile",
        query_accept_margin=0.0,
        query_clean_floor=0.89,
        query_clean_drop_tolerance=0.02,
        query_targeted_asr_reduction_margin=0.05,
        query_targeted_min_base_backdoor=0.50,
    )

    assert selected == []


def test_query_gated_reptile_backdoor_constraints_accept_only_guarded_improvements():
    from meta_sg.learning.task_runner import _query_backdoor_constraints_accept

    assert _query_backdoor_constraints_accept(
        base_backdoor=float("nan"),
        adapted_backdoor=float("nan"),
        backdoor_ceiling=None,
        backdoor_increase_tolerance=None,
    ) is True

    assert _query_backdoor_constraints_accept(
        base_backdoor=0.02,
        adapted_backdoor=0.09,
        backdoor_ceiling=0.05,
        backdoor_increase_tolerance=None,
    ) is False

    assert _query_backdoor_constraints_accept(
        base_backdoor=0.02,
        adapted_backdoor=0.03,
        backdoor_ceiling=0.05,
        backdoor_increase_tolerance=0.02,
    ) is True

    assert _query_backdoor_constraints_accept(
        base_backdoor=0.001762114537444934,
        adapted_backdoor=0.001762114537444934 + 1e-16,
        backdoor_ceiling=0.65,
        backdoor_increase_tolerance=0.0,
        backdoor_improvement_margin=None,
    ) is True

    assert _query_backdoor_constraints_accept(
        base_backdoor=0.90,
        adapted_backdoor=0.82,
        backdoor_ceiling=0.65,
        backdoor_increase_tolerance=0.0,
        backdoor_improvement_margin=0.05,
    ) is True

    assert _query_backdoor_constraints_accept(
        base_backdoor=0.90,
        adapted_backdoor=0.88,
        backdoor_ceiling=0.65,
        backdoor_increase_tolerance=0.0,
        backdoor_improvement_margin=0.05,
    ) is False

    assert _query_backdoor_constraints_accept(
        base_backdoor=0.02,
        adapted_backdoor=0.06,
        backdoor_ceiling=0.10,
        backdoor_increase_tolerance=0.02,
        backdoor_improvement_margin=None,
    ) is False


def test_query_metrics_report_per_attack_backdoor_diagnostics():
    from meta_sg.learning.meta_sg_trainer import _query_metric_values
    from meta_sg.learning.task_runner import TaskResult
    from meta_sg.strategies.types import AttackType

    bfl = ATTACK_DOMAIN["bfl"]
    rl = ATTACK_DOMAIN["rl"]
    rl_backdoor = AttackType(name="rl_backdoor", objective="targeted", adaptive=False)
    metrics = _query_metric_values(
        [
            TaskResult(
                attack_type=bfl,
                adapted_params={},
                mean_defender_reward=0.0,
                mean_attacker_reward=0.0,
                defender_reward_sum=0.0,
                trajectories_collected=1,
                transitions_collected=1,
                query_base_reward=0.10,
                query_adapted_reward=0.20,
                query_gain=0.10,
                query_base_backdoor_acc=0.70,
                query_adapted_backdoor_acc=0.40,
                query_backdoor_accepted=True,
                query_accepted=True,
            ),
            TaskResult(
                attack_type=rl,
                adapted_params={},
                mean_defender_reward=0.0,
                mean_attacker_reward=0.0,
                defender_reward_sum=0.0,
                trajectories_collected=1,
                transitions_collected=1,
                query_base_reward=0.30,
                query_adapted_reward=0.20,
                query_gain=-0.10,
                query_base_backdoor_acc=0.05,
                query_adapted_backdoor_acc=0.06,
                query_backdoor_accepted=False,
                query_accepted=False,
            ),
            TaskResult(
                attack_type=rl_backdoor,
                adapted_params={},
                mean_defender_reward=0.0,
                mean_attacker_reward=0.0,
                defender_reward_sum=0.0,
                trajectories_collected=1,
                transitions_collected=1,
                query_base_reward=-0.90,
                query_adapted_reward=-0.80,
                query_gain=0.10,
                query_base_backdoor_acc=0.90,
                query_adapted_backdoor_acc=0.80,
                query_backdoor_accepted=True,
                query_accepted=True,
            ),
        ]
    )

    assert metrics["query_attack_bfl_gain"] == pytest.approx(0.10)
    assert metrics["query_attack_bfl_base_backdoor_acc"] == pytest.approx(0.70)
    assert metrics["query_attack_bfl_adapted_backdoor_acc"] == pytest.approx(0.40)
    assert metrics["query_attack_bfl_backdoor_increase"] == pytest.approx(-0.30)
    assert metrics["query_attack_bfl_backdoor_reduction"] == pytest.approx(0.30)
    assert metrics["query_attack_bfl_backdoor_accepted"] == pytest.approx(1.0)
    assert metrics["query_attack_name__rl__accepted"] == pytest.approx(0.0)
    assert metrics["query_attack_name__rl__backdoor_accepted"] == pytest.approx(0.0)
    assert metrics["query_attack_name__rl_backdoor__accepted"] == pytest.approx(1.0)
    assert metrics["query_attack_name__rl_backdoor__backdoor_accepted"] == pytest.approx(1.0)
    assert metrics["query_attack_name__rl_backdoor__backdoor_reduction"] == pytest.approx(0.10)
    assert metrics["query_targeted_asr_reduction_mean"] == pytest.approx(0.20)
    assert metrics["query_targeted_asr_reduction_min"] == pytest.approx(0.10)
    assert metrics["query_targeted_asr_reduction_max"] == pytest.approx(0.30)
    assert metrics["query_attack_rl_gain"] == pytest.approx(-0.10)
    assert metrics["query_attack_name__rl__gain"] == pytest.approx(-0.10)


def test_query_rollout_metrics_use_final_clean_and_mean_reward():
    from meta_sg.learning.task_runner import _query_rollout_metrics

    traj = Trajectory(
        attack_type=ATTACK_DOMAIN["rl"],
        transitions=[
            Transition(
                state=np.zeros(2, dtype=np.float32),
                defender_action=np.zeros(3, dtype=np.float32),
                attacker_action=np.zeros(3, dtype=np.float32),
                defender_reward=0.10,
                attacker_reward=-0.10,
                next_state=np.zeros(2, dtype=np.float32),
                done=False,
                info={"clean_acc": 0.50, "backdoor_acc": 0.20},
            ),
            Transition(
                state=np.zeros(2, dtype=np.float32),
                defender_action=np.zeros(3, dtype=np.float32),
                attacker_action=np.zeros(3, dtype=np.float32),
                defender_reward=0.90,
                attacker_reward=-0.90,
                next_state=np.zeros(2, dtype=np.float32),
                done=True,
                info={"clean_acc": 0.91, "backdoor_acc": 0.03},
            ),
        ],
    )

    metrics = _query_rollout_metrics(traj)

    assert metrics["reward"] == pytest.approx(0.50)
    assert metrics["clean_acc"] == pytest.approx(0.91)
    assert metrics["backdoor_acc"] == pytest.approx(0.03)


def test_defender_action_diagnostics_report_transition_server_lr():
    transition = Transition(
        state=np.zeros(2, dtype=np.float32),
        defender_action=np.array([0.2, -0.2, 0.6], dtype=np.float32),
        attacker_action=np.zeros(3, dtype=np.float32),
        defender_reward=0.0,
        attacker_reward=0.0,
        next_state=np.zeros(2, dtype=np.float32),
        done=False,
        info={
            "defense_decision": DefenseDecision(
                norm_bound_alpha=3.0,
                trimmed_mean_beta=0.15,
                server_lr=0.8,
            )
        },
    )
    traj = Trajectory(attack_type=ATTACK_DOMAIN["ipm"], transitions=[transition])

    diagnostics = _action_diagnostics(traj)

    assert diagnostics["defender_alpha"] == pytest.approx(3.0)
    assert diagnostics["defender_beta"] == pytest.approx(0.15)
    assert diagnostics["defender_server_lr"] == pytest.approx(0.8)
    assert "defender_post_param" not in diagnostics


def test_pretraining_script_stub_smoke_writes_final_checkpoint(tmp_path):
    from meta_sg.scripts.run_meta_sg_pretraining import main

    main(
        [
            "--backend",
            "stub",
            "--output-dir",
            str(tmp_path),
            "--T",
            "1",
            "--K",
            "1",
            "--H",
            "2",
            "--l",
            "1",
            "--N-A",
            "1",
            "--post-br-defender-updates",
            "0",
            "--hidden-dim",
            "8",
            "--batch-size",
            "2",
            "--buffer-capacity",
            "32",
            "--num-clients",
            "6",
            "--num-attackers",
            "1",
            "--subsample-rate",
            "1.0",
            "--seed",
            "5",
            "--device",
            "cpu",
        ]
    )

    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    final_dir = run_dirs[0] / "final"
    assert (final_dir / "defender_meta.pt").exists()
    assert (final_dir / "attacker_ipm.pt").exists()
    assert (final_dir / "attacker_lmp.pt").exists()
    assert (final_dir / "attacker_rl.pt").exists()


def test_pretraining_script_writes_traceable_metrics_and_latest_checkpoint(tmp_path):
    from meta_sg.scripts.run_meta_sg_pretraining import main

    main(
        [
            "--backend",
            "stub",
            "--output-dir",
            str(tmp_path),
            "--T",
            "2",
            "--K",
            "1",
            "--H",
            "2",
            "--l",
            "1",
            "--N-A",
            "1",
            "--post-br-defender-updates",
            "0",
            "--hidden-dim",
            "8",
            "--batch-size",
            "2",
            "--buffer-capacity",
            "32",
            "--num-clients",
            "6",
            "--num-attackers",
            "1",
            "--subsample-rate",
            "1.0",
            "--seed",
            "9",
            "--device",
            "cpu",
            "--checkpoint-interval",
            "1",
            "--log-interval",
            "1",
        ]
    )

    run_dir = next(tmp_path.iterdir())
    metrics_path = run_dir / "metrics.jsonl"
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"
    latest_dir = run_dir / "checkpoints" / "latest"

    records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert [record["iteration"] for record in records] == [1, 2]
    assert all("reward_mean" in record for record in records)
    assert all(record["batch_attack_types"] for record in records)
    assert all("defender_alpha" in record for record in records)
    assert all("defender_beta" in record for record in records)
    assert all("defender_post_param" in record for record in records)
    assert all("defender_action_std_2" in record for record in records)
    assert (latest_dir / "defender_meta.pt").exists()
    assert (latest_dir / "attacker_ipm.pt").exists()
    assert json.loads(summary_path.read_text())["meta_iterations"] == 2
    assert json.loads(config_path.read_text())["args"]["checkpoint_interval"] == 1
    assert json.loads(config_path.read_text())["resolved_device"] == "cpu"
    record = records[-1]
    assert {"reward_min", "reward_max", "iteration_elapsed_seconds"} <= record.keys()
    assert {
        "defender_losses",
        "attacker_losses",
        "buffer_sizes",
        "task_records",
    } <= record.keys()
    assert len(record["task_records"]) == 1
    task = record["task_records"][0]
    assert {
        "attack_type",
        "elapsed_seconds",
        "transitions_collected",
        "diagnostics",
    } <= task.keys()
    json.dumps(record, allow_nan=False)


def test_pretraining_latest_only_checkpoint_replaces_history_and_records_iteration(tmp_path):
    from meta_sg.scripts.run_meta_sg_pretraining import main

    main(
        [
            "--backend",
            "stub",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "run",
            "--T",
            "2",
            "--K",
            "1",
            "--H",
            "1",
            "--l",
            "1",
            "--N-A",
            "1",
            "--post-br-defender-updates",
            "0",
            "--hidden-dim",
            "8",
            "--batch-size",
            "2",
            "--buffer-capacity",
            "32",
            "--num-clients",
            "6",
            "--num-attackers",
            "1",
            "--subsample-rate",
            "1.0",
            "--seed",
            "42",
            "--device",
            "cpu",
            "--checkpoint-interval",
            "1",
            "--latest-checkpoint-only",
        ]
    )

    checkpoint_root = tmp_path / "run" / "checkpoints"
    assert sorted(path.name for path in checkpoint_root.iterdir()) == ["latest"]
    metadata = json.loads((checkpoint_root / "latest" / "checkpoint.json").read_text())
    assert metadata["completed_iteration"] == 2
    assert metadata["master_seed"] == 42
    assert (checkpoint_root / "latest" / "defender_meta.pt").exists()


def test_pretraining_script_can_train_transition_server_lr_defender(tmp_path):
    from meta_sg.scripts.run_meta_sg_pretraining import main

    main(
        [
            "--backend",
            "stub",
            "--output-dir",
            str(tmp_path),
            "--T",
            "1",
            "--K",
            "1",
            "--H",
            "2",
            "--l",
            "1",
            "--N-A",
            "1",
            "--post-br-defender-updates",
            "0",
            "--hidden-dim",
            "8",
            "--batch-size",
            "2",
            "--buffer-capacity",
            "32",
            "--num-clients",
            "6",
            "--num-attackers",
            "1",
            "--subsample-rate",
            "1.0",
            "--seed",
            "11",
            "--device",
            "cpu",
            "--defender-third-action",
            "server_lr",
        ]
    )

    run_dir = next(tmp_path.iterdir())
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    config = json.loads((run_dir / "config.json").read_text())

    assert config["meta_config"]["defender_third_action"] == "server_lr"
    assert records[0]["defender_server_lr"] >= 0.0
    assert records[0]["defender_server_lr"] <= 1.0
    assert "defender_post_param" not in records[0]


def test_pretraining_script_logs_server_lr_penalty_diagnostics(tmp_path):
    from meta_sg.scripts.run_meta_sg_pretraining import main

    main(
        [
            "--backend",
            "stub",
            "--output-dir",
            str(tmp_path),
            "--T",
            "1",
            "--K",
            "1",
            "--H",
            "2",
            "--l",
            "1",
            "--N-A",
            "1",
            "--post-br-defender-updates",
            "0",
            "--hidden-dim",
            "8",
            "--batch-size",
            "2",
            "--buffer-capacity",
            "32",
            "--num-clients",
            "6",
            "--num-attackers",
            "1",
            "--subsample-rate",
            "1.0",
            "--seed",
            "12",
            "--device",
            "cpu",
            "--defender-third-action",
            "server_lr",
            "--server-lr-penalty-weight",
            "0.5",
        ]
    )

    run_dir = next(tmp_path.iterdir())
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]

    assert "server_lr_penalty" in records[0]
    assert records[0]["server_lr_penalty"] >= 0.0


def test_pretraining_script_logs_query_gated_meta_metrics(tmp_path):
    from meta_sg.scripts.run_meta_sg_pretraining import main

    main(
        [
            "--backend",
            "stub",
            "--output-dir",
            str(tmp_path),
            "--T",
            "1",
            "--K",
            "1",
            "--H",
            "2",
            "--l",
            "1",
            "--N-A",
            "1",
            "--post-br-defender-updates",
            "0",
            "--hidden-dim",
            "8",
            "--batch-size",
            "2",
            "--buffer-capacity",
            "32",
            "--num-clients",
            "6",
            "--num-attackers",
            "1",
            "--subsample-rate",
            "1.0",
            "--seed",
            "13",
            "--device",
            "cpu",
            "--meta-objective",
            "query_gated_reptile",
            "--query-horizon",
            "1",
        ]
    )

    run_dir = next(tmp_path.iterdir())
    record = json.loads((run_dir / "metrics.jsonl").read_text().splitlines()[0])
    config = json.loads((run_dir / "config.json").read_text())

    assert config["meta_config"]["meta_objective"] == "query_gated_reptile"
    assert "query_gain_mean" in record
    assert "query_accept_rate" in record
    assert "query_adapted_clean_acc_mean" in record
    assert "query_clean_drop_mean" in record
    assert "query_clean_accept_rate" in record
    assert record["query_accept_rate"] >= 0.0
    assert record["query_accept_rate"] <= 1.0


def test_checkpoint_selector_ranks_mean_score_and_worst_clean(tmp_path):
    from meta_sg.scripts.select_meta_sg_checkpoint import select_checkpoint

    t30 = tmp_path / "direct_t30.json"
    iter35 = tmp_path / "direct_iter35.json"
    iter45 = tmp_path / "direct_iter45.json"
    t30.write_text(
        json.dumps(
            [
                {"scenario": "rl", "final_clean_acc": 0.8933, "final_backdoor_acc": 0.0026, "final_defense_score": 0.8880},
                {"scenario": "bfl", "final_clean_acc": 0.9158, "final_backdoor_acc": 0.0088, "final_defense_score": 0.8982},
            ]
        ),
        encoding="utf-8",
    )
    iter35.write_text(
        json.dumps(
            [
                {"scenario": "rl", "final_clean_acc": 0.8900, "final_backdoor_acc": 0.0026, "final_defense_score": 0.8847},
                {"scenario": "bfl", "final_clean_acc": 0.9133, "final_backdoor_acc": 0.0035, "final_defense_score": 0.9063},
            ]
        ),
        encoding="utf-8",
    )
    iter45.write_text(
        json.dumps(
            [
                {"scenario": "rl", "final_clean_acc": 0.7967, "final_backdoor_acc": 0.0009, "final_defense_score": 0.7949},
                {"scenario": "bfl", "final_clean_acc": 0.9133, "final_backdoor_acc": 0.0035, "final_defense_score": 0.9063},
            ]
        ),
        encoding="utf-8",
    )

    mean_score = select_checkpoint([t30, iter35, iter45], metric="mean_score")
    worst_clean = select_checkpoint([t30, iter35, iter45], metric="worst_clean")
    floor_then_score = select_checkpoint(
        [t30, iter35, iter45],
        metric="clean_floor_then_score",
        clean_floor=0.88,
    )

    assert mean_score["selected"]["path"].endswith("direct_iter35.json")
    assert worst_clean["selected"]["path"].endswith("direct_t30.json")
    assert floor_then_score["selected"]["path"].endswith("direct_iter35.json")
    assert all(row["worst_clean"] >= 0.88 for row in floor_then_score["eligible"])


def test_checkpoint_selector_reports_worst_attack_and_targeted_backdoor(tmp_path):
    from meta_sg.scripts.select_meta_sg_checkpoint import summarize_eval_file

    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            [
                {"scenario": "clean", "final_clean_acc": 0.93, "final_backdoor_acc": 0.0, "final_defense_score": 0.93},
                {"scenario": "bfl", "final_clean_acc": 0.94, "final_backdoor_acc": 0.65, "final_defense_score": -0.35},
                {"scenario": "rl_backdoor", "final_clean_acc": 0.91, "final_backdoor_acc": 0.25, "final_defense_score": 0.41},
            ]
        ),
        encoding="utf-8",
    )

    summary = summarize_eval_file(path)

    assert summary["worst_attack_score"] == pytest.approx(-0.35)
    assert summary["max_targeted_backdoor"] == pytest.approx(0.65)
    assert summary["per_attack"]["bfl"]["backdoor"] == pytest.approx(0.65)
    assert summary["per_attack"]["rl_backdoor"]["score"] == pytest.approx(0.41)


def test_checkpoint_selector_backdoor_ceiling_then_score(tmp_path):
    from meta_sg.scripts.select_meta_sg_checkpoint import select_checkpoint

    safe = tmp_path / "safe.json"
    unsafe = tmp_path / "unsafe.json"
    safe.write_text(
        json.dumps(
            [
                {"scenario": "clean", "final_clean_acc": 0.91, "final_backdoor_acc": 0.0, "final_defense_score": 0.91},
                {"scenario": "bfl", "final_clean_acc": 0.90, "final_backdoor_acc": 0.20, "final_defense_score": 0.50},
            ]
        ),
        encoding="utf-8",
    )
    unsafe.write_text(
        json.dumps(
            [
                {"scenario": "clean", "final_clean_acc": 0.94, "final_backdoor_acc": 0.0, "final_defense_score": 0.94},
                {"scenario": "bfl", "final_clean_acc": 0.95, "final_backdoor_acc": 0.80, "final_defense_score": 0.70},
            ]
        ),
        encoding="utf-8",
    )

    report = select_checkpoint(
        [safe, unsafe],
        metric="backdoor_ceiling_then_score",
        clean_floor=0.89,
        backdoor_ceiling=0.30,
    )

    assert report["selected"]["path"] == str(safe)
    assert len(report["eligible"]) == 1


def test_direct_eval_mixed_scenarios_and_combined_action_space():
    from meta_sg.scripts.evaluate_meta_sg_direct import _action_delta, _scenarios, parse_args

    args = parse_args(["--checkpoint", "dummy.pt", "--output-json", "out.json", "--scenario-set", "mixed", "--defender-third-action", "both"])
    scenarios = _scenarios(args)

    assert [scenario.name for scenario in scenarios] == [
        "clean",
        "ipm",
        "lmp",
        "rl",
        "bfl",
        "dba",
        "rl_backdoor",
    ]
    assert _action_delta(
        {"alpha": 1.0, "beta": 0.1, "neuroclip": 2.0, "server_lr": 0.8},
        {"alpha": 2.0, "beta": 0.2, "neuroclip": 3.0, "server_lr": 0.5},
    ) == {
        "alpha": pytest.approx(1.0),
        "beta": pytest.approx(0.1),
        "neuroclip": pytest.approx(1.0),
        "server_lr": pytest.approx(-0.3),
    }


def test_pretraining_script_can_resume_from_checkpoint_with_global_iteration(tmp_path):
    from meta_sg.scripts.run_meta_sg_pretraining import main

    common_args = [
        "--backend",
        "stub",
        "--T",
        "1",
        "--K",
        "1",
        "--H",
        "1",
        "--l",
        "1",
        "--N-A",
        "1",
        "--post-br-defender-updates",
        "0",
        "--hidden-dim",
        "8",
        "--batch-size",
        "2",
        "--buffer-capacity",
        "32",
        "--num-clients",
        "6",
        "--num-attackers",
        "1",
        "--subsample-rate",
        "1.0",
        "--checkpoint-interval",
        "1",
        "--log-interval",
        "1",
        "--device",
        "cpu",
    ]
    first_root = tmp_path / "first"
    main([*common_args, "--output-dir", str(first_root), "--seed", "31"])
    first_run = next(first_root.iterdir())

    resumed_root = tmp_path / "resumed"
    main(
        [
            *common_args,
            "--output-dir",
            str(resumed_root),
            "--seed",
            "32",
            "--resume-from",
            str(first_run / "checkpoints" / "latest"),
            "--start-iteration",
            "1",
            "--total-iterations",
            "2",
        ]
    )
    resumed_run = next(resumed_root.iterdir())
    records = [
        json.loads(line)
        for line in (resumed_run / "metrics.jsonl").read_text().splitlines()
    ]
    config = json.loads((resumed_run / "config.json").read_text())

    assert records[0]["iteration"] == 2
    assert records[0]["local_iteration"] == 1
    assert (resumed_run / "checkpoints" / "iter_0002" / "defender_meta.pt").exists()
    assert config["resume_from"] == str(first_run / "checkpoints" / "latest")
    assert config["start_iteration"] == 1
    assert config["total_iterations"] == 2


def test_4d_global_backdoor_job_defaults_to_checkpoint_every_five_iterations():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_4d_both_global_backdoor_mixed_h200_job.sh"
    )

    assert 'CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5}"' in script.read_text()


def test_experiment_artifacts_status_preserves_stage_history(tmp_path):
    from meta_sg.scripts.experiment_artifacts import main

    status_path = tmp_path / "status.json"
    main(
        [
            "status",
            "--output",
            str(status_path),
            "--stage",
            "initializing",
            "--message",
            "run created",
        ]
    )
    main(
        [
            "status",
            "--output",
            str(status_path),
            "--stage",
            "training",
            "--message",
            "started",
            "--last-completed-iteration",
            "0",
        ]
    )

    status = json.loads(status_path.read_text())
    assert status["stage"] == "training"
    assert status["message"] == "started"
    assert status["last_completed_iteration"] == 0
    assert [entry["stage"] for entry in status["history"]] == [
        "initializing",
        "training",
    ]
    json.dumps(status, allow_nan=False)


def test_global_model_poisoning_h200_launcher_has_observable_job_contract():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_global_model_poisoning_h200_30c6a.sh"
    )
    text = script.read_text()

    assert 'CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10}"' in text
    assert "--latest-checkpoint-only" in text
    assert "--summary-json" in text
    assert "experiment_artifacts.py provenance" in text
    assert "experiment_artifacts.py status" in text
    assert "resource_metrics.csv" in text
    assert "--scenario-set model_poisoning" in text


def test_global_model_poisoning_h200_launcher_uses_weight_copy_for_stub_smoke():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_global_model_poisoning_h200_30c6a.sh"
    )
    text = script.read_text()

    assert 'if [[ "${BACKEND}" == "stub" ]]' in text
    assert 'POST_DEFENSE_MODE="weight_copy"' in text
    assert text.count('--post-defense-mode "${POST_DEFENSE_MODE}"') == 2


def test_curriculum_job_runs_three_domains_with_short_then_full_schedule():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_4d_curriculum_global_backdoor_mixed_h200_job.sh"
    )

    text = script.read_text()

    assert 'STAGE1_DOMAIN="${STAGE1_DOMAIN:-clean_global}"' in text
    assert 'STAGE2_DOMAIN="${STAGE2_DOMAIN:-clean_backdoor_mixed}"' in text
    assert 'STAGE3_DOMAIN="${STAGE3_DOMAIN:-clean_global_backdoor_mixed}"' in text
    assert 'STAGE1_T="${STAGE1_T:-10}"' in text
    assert 'STAGE2_T="${STAGE2_T:-10}"' in text
    assert 'STAGE3_T="${STAGE3_T:-30}"' in text
    assert 'STAGE1_K="${STAGE1_K:-4}"' in text
    assert 'STAGE2_K="${STAGE2_K:-5}"' in text
    assert 'STAGE3_K="${STAGE3_K:-8}"' in text
    assert 'STAGE3_META_STEP="${STAGE3_META_STEP:-0.1}"' in text
    assert '--resume-from "$resume_from"' in text
