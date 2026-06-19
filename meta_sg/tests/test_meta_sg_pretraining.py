"""Tests for paper-aligned Meta-SG model-poisoning pretraining wiring."""

import json

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

    assert build_meta_config(mnist_args).H == 7
    assert build_meta_config(cifar_args).H == 9
    assert build_meta_config(mnist_args).lambda_bd == pytest.approx(0.0)
    assert build_meta_config(mnist_args).reward_mode == "accuracy"
    assert build_meta_config(mnist_args).task_sampler == "iid"
    assert build_meta_config(stratified_args).task_sampler == "stratified"
    assert build_meta_config(server_lr_args).defender_third_action == "server_lr"
    assert build_meta_config(both_args).lambda_bd == pytest.approx(1.0)
    assert build_meta_config(both_args).native_sandbox_attacks is False
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


def test_meta_sg_config_has_single_reptile_step_field():
    cfg = MetaSGConfig()

    assert cfg.meta_update_step == pytest.approx(1.0)
    assert cfg.task_sampler == "iid"
    assert cfg.defender_third_action == "neuroclip"
    assert not hasattr(cfg, "kappa_D")
    assert not hasattr(cfg, "kappa_A")


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
