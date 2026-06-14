import csv
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from fl_sandbox.attacks.registry import create_attack
from fl_sandbox.attacks.rl_attacker import RLAttack
from fl_sandbox.attacks.rl_attacker import paper_attack as paper_attack_module
from fl_sandbox.attacks.rl_attacker.config import RLAttackerConfig
from fl_sandbox.attacks.rl_attacker.paper_attack import PaperRLAttack
from fl_sandbox.attacks.rl_attacker.simulator.paper_env import (
    PaperFLSimulator,
    decode_paper_action,
    transform_paper_reward,
)
from fl_sandbox.config import RunConfig
from fl_sandbox.run.run_experiment import parse_args
from fl_sandbox.runtime import RoundContext


def _write_distribution(root, *, count=8):
    train_dir = root / "train"
    train_dir.mkdir(parents=True)
    with (root / "data.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        for idx in range(count):
            writer.writerow([idx, idx % 2])
            image = np.full((28, 28), idx * 10, dtype=np.uint8)
            Image.fromarray(image, mode="L").save(train_dir / f"{idx}.png")
    (root / "metadata.json").write_text(
        json.dumps({"mean": [0.0], "std": [1.0]}),
        encoding="utf-8",
    )


def _ctx(round_idx, model, weights, config):
    return RoundContext(
        round_idx=round_idx,
        old_weights=[layer.copy() for layer in weights],
        benign_weights=[[layer.copy() for layer in weights]],
        selected_attacker_ids=[1],
        model=model,
        device=torch.device("cpu"),
        fl_config=config,
        defense_type="clipped_median",
        lr=0.01,
        server_lr=0.01,
        local_epochs=1,
        eval_loader=None,
    )


def test_cli_maps_distribution_dir_to_config(tmp_path):
    args = parse_args(
        [
            "--attack_type",
            "rl",
            "--distribution_dir",
            str(tmp_path),
            "--distribution_split",
            "train",
            "--rl_policy_warmup_steps",
            "64",
            "--rl_policy_warmup_random_steps",
            "16",
            "--start_round_idx",
            "101",
            "--rl_distribution_growth_mode",
            "full",
        ]
    )
    config = RunConfig.from_flat_dict(vars(args))

    assert config.attacker.rl_distribution_dir == str(tmp_path)
    assert config.attacker.rl_distribution_split == "train"
    assert config.attacker.rl_policy_warmup_steps == 64
    assert config.attacker.rl_policy_warmup_random_steps == 16
    assert config.attacker.rl_distribution_growth_mode == "full"
    assert config.runtime.start_round_idx == 101


def test_cli_maps_paper_reward_transform_to_config(tmp_path):
    args = parse_args(
        [
            "--attack_type",
            "rl",
            "--distribution_dir",
            str(tmp_path),
            "--rl_reward_transform",
            "tanh_delta",
            "--rl_reward_scale",
            "10.0",
        ]
    )
    config = RunConfig.from_flat_dict(vars(args))

    assert config.attacker.rl_reward_transform == "tanh_delta"
    assert config.attacker.rl_reward_scale == pytest.approx(10.0)


def test_rl_attack_requires_phase1_distribution_dir():
    config = RunConfig.from_flat_dict({"attack_type": "rl"})

    with pytest.raises(ValueError, match="rl_distribution_dir"):
        create_attack(config.attacker)


def test_registry_builds_paper_rl_attack_from_phase1_distribution(tmp_path):
    _write_distribution(tmp_path)
    config = RunConfig.from_flat_dict(
        {
            "attack_type": "rl",
            "rl_distribution_dir": str(tmp_path),
            "rl_policy_warmup_steps": 8,
            "rl_policy_warmup_random_steps": 2,
        }
    )

    attack = create_attack(config.attacker)

    assert isinstance(attack, PaperRLAttack)
    assert isinstance(attack, RLAttack)
    assert len(attack.distribution) == 8


def test_paper_rl_defaults_match_author_td3_training_setup(tmp_path):
    _write_distribution(tmp_path)
    config = RunConfig.from_flat_dict(
        {
            "attack_type": "rl",
            "rl_distribution_dir": str(tmp_path),
        }
    )

    attack = create_attack(config.attacker)

    assert attack.config.policy_lr == pytest.approx(1e-7)
    assert attack.config.critic_lr == pytest.approx(1e-7)
    assert attack.config.gamma == pytest.approx(1.0)
    assert attack.config.hidden_sizes == (256, 128)
    assert attack.config.train_freq_steps == 5
    assert attack.config.replay_capacity == 100_000
    assert attack.config.simulator_horizon == 1000
    assert attack.config.local_search_batch_size == 128


def test_paper_action_decoding_matches_clipped_median_formula():
    gamma, local_steps = decode_paper_action(np.asarray([0.0, 0.0], dtype=np.float32))

    assert gamma == pytest.approx(15.0)
    assert local_steps == 25


def test_paper_reward_transform_defaults_to_raw_loss_delta():
    config = RLAttackerConfig()

    assert transform_paper_reward(12.0, 2.0, config) == pytest.approx(10.0)


def test_paper_reward_transform_tanh_delta_scales_and_bounds_reward():
    config = RLAttackerConfig(reward_transform="tanh_delta", reward_scale=10.0)

    assert transform_paper_reward(12.0, 2.0, config) == pytest.approx(np.tanh(1.0))
    assert transform_paper_reward(1002.0, 2.0, config) == pytest.approx(1.0)
    assert transform_paper_reward(-998.0, 2.0, config) == pytest.approx(-1.0)


def test_paper_rl_attacker_accepts_known_defense_eval_set():
    config = RLAttackerConfig()

    for defense in [
        "fedavg",
        "median",
        "trimmed_mean",
        "krum",
        "clipped_median",
        "paper_norm_trimmed_mean",
    ]:
        config.validate_defense(defense)


def test_policy_simulator_resets_to_first_initial_weights_not_latest_round(tmp_path):
    _write_distribution(tmp_path)
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 2))
    initial_weights = [value.detach().numpy().copy() for value in model.state_dict().values()]
    later_weights = [value + 3.0 for value in initial_weights]
    run_config = RunConfig.from_flat_dict(
        {
            "attack_type": "rl",
            "defense_type": "clipped_median",
            "num_clients": 2,
            "num_attackers": 1,
            "subsample_rate": 1.0,
            "rl_distribution_dir": str(tmp_path),
            "rl_hidden_sizes": [8],
        }
    )
    attack = create_attack(run_config.attacker)

    env1 = attack._build_policy_env(_ctx(101, model, initial_weights, run_config))
    env2 = attack._build_policy_env(_ctx(102, model, later_weights, run_config))

    assert all(np.array_equal(a, b) for a, b in zip(env1.initial_weights, initial_weights))
    assert all(np.array_equal(a, b) for a, b in zip(env2.initial_weights, initial_weights))
    assert not all(np.array_equal(a, b) for a, b in zip(env2.initial_weights, later_weights))


def test_paper_rl_trains_offline_once_and_deploys_deterministically(tmp_path, monkeypatch):
    _write_distribution(tmp_path)
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 2))
    weights = [value.detach().numpy().copy() for value in model.state_dict().values()]
    run_config = RunConfig.from_flat_dict(
        {
            "attack_type": "rl",
            "defense_type": "clipped_median",
            "num_clients": 2,
            "num_attackers": 1,
            "subsample_rate": 1.0,
            "batch_size": 2,
            "lr": 0.01,
            "rl_distribution_dir": str(tmp_path),
            "rl_attack_start_round": 0,
            "rl_policy_warmup_steps": 4,
            "rl_policy_warmup_random_steps": 2,
            "rl_policy_train_steps_per_round": 1,
            "rl_simulator_horizon": 2,
            "rl_batch_size": 2,
            "rl_hidden_sizes": [8],
            "rl_replay_capacity": 32,
        }
    )
    attack = create_attack(run_config.attacker)
    calls = {"fallback": 0}

    def fake_fallback(ctx):
        calls["fallback"] += 1
        return []

    monkeypatch.setattr(attack, "fallback_old_weights", fake_fallback)

    ctx1 = _ctx(1, model, weights, run_config)
    attack.observe_round(ctx1)
    first_trainer = attack.trainer
    attack.observe_round(_ctx(2, model, weights, run_config))
    malicious = attack.execute(_ctx(2, model, weights, run_config))
    metrics = attack.after_round(ctx=ctx1, clean_loss_before=1.0, clean_loss=1.2)

    assert attack._policy_warmup_done is True
    assert attack.trainer is first_trainer
    assert len(malicious) == 1
    assert calls == {"fallback": 0}
    assert "fl_sandbox.attacks.rl_attacker.proxy.inversion" not in sys.modules
    assert metrics["rl_proxy_source"] == 1.0
    assert metrics["rl_proxy_buffer_size"] == 8.0
    assert metrics["rl_policy_warmup_done"] == 1.0
    assert metrics["rl_policy_frozen"] == 1.0


def test_paper_rl_runs_author_style_offline_train_once(tmp_path, monkeypatch):
    _write_distribution(tmp_path)
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 2))
    weights = [value.detach().numpy().copy() for value in model.state_dict().values()]
    run_config = RunConfig.from_flat_dict(
        {
            "attack_type": "rl",
            "defense_type": "clipped_median",
            "num_clients": 2,
            "num_attackers": 1,
            "subsample_rate": 1.0,
            "rl_distribution_dir": str(tmp_path),
            "rl_attack_start_round": 0,
            "rl_policy_warmup_steps": 20,
            "rl_policy_warmup_random_steps": 2,
            "rl_policy_train_steps_per_round": 0,
            "rl_policy_train_end_round": 400,
            "rl_simulator_horizon": 2,
            "rl_batch_size": 2,
            "rl_hidden_sizes": [8],
            "rl_replay_capacity": 32,
        }
    )
    attack = create_attack(run_config.attacker)
    calls = {"warmup": [], "collect": [], "update": []}

    class FakeStats:
        steps = 0
        reward_mean = 0.25

    class FakeTrainer:
        def ensure_initialized(self, obs_space, action_space):
            self.obs_shape = obs_space.shape
            self.action_shape = action_space.shape

        def warmup_collect(self, env, random_steps):
            calls["warmup"].append(random_steps)
            return SimpleNamespace(steps=random_steps, reward_mean=0.25)

        def collect(self, env, steps):
            calls["collect"].append(steps)
            return SimpleNamespace(steps=steps, reward_mean=0.25)

        def update(self, gradient_steps):
            calls["update"].append(gradient_steps)

        def act(self, obs, *, deterministic=False):
            return np.asarray([0.0, 0.0], dtype=np.float32)

        def diagnostics(self):
            return {"trainer_collect_steps": float(sum(calls["warmup"]) + sum(calls["collect"]))}

    monkeypatch.setattr(paper_attack_module, "build_trainer", lambda config: FakeTrainer())

    attack.observe_round(_ctx(1, model, weights, run_config))
    attack.observe_round(_ctx(2, model, weights, run_config))
    attack.observe_round(_ctx(3, model, weights, run_config))

    assert calls == {
        "warmup": [2],
        "collect": [5, 5, 5, 3],
        "update": [5, 5, 5, 3],
    }
    assert attack._policy_warmup_done is True
    assert attack._policy_training_frozen is True


def test_paper_policy_training_saves_step_checkpoints(tmp_path, monkeypatch):
    _write_distribution(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 2))
    weights = [value.detach().numpy().copy() for value in model.state_dict().values()]
    run_config = RunConfig.from_flat_dict(
        {
            "attack_type": "rl",
            "defense_type": "clipped_median",
            "num_clients": 2,
            "num_attackers": 1,
            "subsample_rate": 1.0,
            "rl_distribution_dir": str(tmp_path),
            "rl_policy_warmup_steps": 10,
            "rl_policy_warmup_random_steps": 0,
            "rl_policy_warmup_checkpoint_interval": 5,
            "rl_policy_warmup_checkpoint_dir": str(checkpoint_dir),
            "rl_train_freq_steps": 5,
            "rl_batch_size": 2,
            "rl_hidden_sizes": [8],
            "rl_replay_capacity": 32,
        }
    )
    attack = create_attack(run_config.attacker)
    saved = []

    class FakeTrainer:
        def ensure_initialized(self, obs_space, action_space):
            pass

        def warmup_collect(self, env, random_steps):
            return SimpleNamespace(steps=random_steps, reward_mean=0.0)

        def collect(self, env, steps):
            return SimpleNamespace(steps=steps, reward_mean=0.0)

        def update(self, gradient_steps):
            pass

        def save(self, path):
            saved.append(path)

        def diagnostics(self):
            return {}

    monkeypatch.setattr(paper_attack_module, "build_trainer", lambda config: FakeTrainer())

    attack.observe_round(_ctx(1, model, weights, run_config))

    assert saved == [
        str(checkpoint_dir / "rl_policy_step_000005.pt"),
        str(checkpoint_dir / "rl_policy_step_000010.pt"),
    ]


def test_paper_policy_training_uses_proxy_eval_not_real_eval_loader(tmp_path):
    _write_distribution(tmp_path)
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 2))
    weights = [value.detach().numpy().copy() for value in model.state_dict().values()]
    run_config = RunConfig.from_flat_dict(
        {
            "attack_type": "rl",
            "defense_type": "clipped_median",
            "num_clients": 2,
            "num_attackers": 1,
            "subsample_rate": 1.0,
            "rl_distribution_dir": str(tmp_path),
        }
    )
    attack = create_attack(run_config.attacker)
    ctx = _ctx(1, model, weights, run_config)
    ctx.eval_loader = object()

    env = attack._build_policy_env(ctx)

    assert env.simulator.eval_loader is None


def test_paper_policy_budget_includes_random_warmup(tmp_path, monkeypatch):
    _write_distribution(tmp_path)
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 2))
    weights = [value.detach().numpy().copy() for value in model.state_dict().values()]
    run_config = RunConfig.from_flat_dict(
        {
            "attack_type": "rl",
            "defense_type": "clipped_median",
            "num_clients": 2,
            "num_attackers": 1,
            "subsample_rate": 1.0,
            "rl_distribution_dir": str(tmp_path),
            "rl_policy_warmup_steps": 5,
            "rl_policy_warmup_random_steps": 2,
            "rl_policy_train_steps_per_round": 10,
            "rl_policy_train_end_round": 4,
        }
    )
    attack = create_attack(run_config.attacker)
    calls = {"warmup": [], "collect": [], "update": []}

    class FakeTrainer:
        def ensure_initialized(self, obs_space, action_space):
            pass

        def warmup_collect(self, env, random_steps):
            calls["warmup"].append(random_steps)
            return SimpleNamespace(steps=random_steps, reward_mean=0.0)

        def collect(self, env, steps):
            calls["collect"].append(steps)
            return SimpleNamespace(steps=steps, reward_mean=0.0)

        def update(self, gradient_steps):
            calls["update"].append(gradient_steps)

        def diagnostics(self):
            return {}

    monkeypatch.setattr(paper_attack_module, "build_trainer", lambda config: FakeTrainer())

    attack.observe_round(_ctx(1, model, weights, run_config))
    attack.observe_round(_ctx(2, model, weights, run_config))

    assert calls == {"warmup": [2], "collect": [3], "update": [3]}
    assert attack._policy_train_steps_completed == 5
    assert attack._policy_training_frozen is True


def test_paper_rl_loads_checkpoint_without_offline_simulator(tmp_path, monkeypatch):
    _write_distribution(tmp_path)
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"placeholder")
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 2))
    weights = [value.detach().numpy().copy() for value in model.state_dict().values()]
    run_config = RunConfig.from_flat_dict(
        {
            "attack_type": "rl",
            "defense_type": "clipped_median",
            "num_clients": 2,
            "num_attackers": 1,
            "subsample_rate": 1.0,
            "rl_distribution_dir": str(tmp_path),
            "rl_policy_checkpoint_path": str(checkpoint),
            "rl_policy_warmup_steps": 4,
            "rl_policy_warmup_random_steps": 2,
            "rl_hidden_sizes": [8],
        }
    )
    attack = create_attack(run_config.attacker)
    calls = {"load": [], "warmup": 0}

    class FakeTrainer:
        def ensure_initialized(self, obs_space, action_space):
            self.obs_shape = obs_space.shape
            self.action_shape = action_space.shape

        def load(self, path):
            calls["load"].append(path)

        def warmup_collect(self, *args, **kwargs):
            calls["warmup"] += 1
            raise AssertionError("checkpoint deployment must not run simulator warmup")

        def collect(self, *args, **kwargs):
            raise AssertionError("checkpoint deployment must not collect simulator transitions")

        def update(self, *args, **kwargs):
            raise AssertionError("checkpoint deployment must not update policy")

        def act(self, obs, *, deterministic=False):
            assert deterministic is True
            return np.asarray([0.0, 0.0], dtype=np.float32)

        def diagnostics(self):
            return {"trainer_collect_steps": 0.0, "trainer_update_steps": 0.0}

    def fail_simulator(*args, **kwargs):
        raise AssertionError("checkpoint deployment must not construct PaperFLSimulator")

    monkeypatch.setattr(paper_attack_module, "build_trainer", lambda config: FakeTrainer())
    monkeypatch.setattr(paper_attack_module, "PaperFLSimulator", fail_simulator)

    attack.observe_round(_ctx(1, model, weights, run_config))

    assert attack._policy_warmup_done is True
    assert calls == {"load": [str(checkpoint)], "warmup": 0}


def test_paper_simulator_parallelizes_benign_updates_when_configured():
    import threading
    import time

    simulator = object.__new__(PaperFLSimulator)
    simulator.fl_config = SimpleNamespace(runtime=SimpleNamespace(parallel_clients=4))
    simulator.device = torch.device("cpu")
    thread_ids = set()

    def fake_benign_update(old_weights):
        time.sleep(0.02)
        thread_ids.add(threading.get_ident())
        return [old_weights[0].copy()]

    simulator._simulate_benign_update = fake_benign_update
    weights = [np.asarray([1.0], dtype=np.float32)]

    updates = simulator._simulate_benign_updates(weights, benign_count=4)

    assert len(updates) == 4
    assert len(thread_ids) > 1
