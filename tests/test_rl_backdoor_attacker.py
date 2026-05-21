from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from fl_sandbox.attacks.registry import ATTACK_CHOICES, create_attack
from fl_sandbox.config import RunConfig


def test_rl_backdoor_is_public_attack_type():
    assert "rl_backdoor" in ATTACK_CHOICES
    cfg = RunConfig.from_flat_dict({"attack_type": "rl_backdoor"}).attacker
    attack = create_attack(cfg)
    assert attack.attack_type == "rl_backdoor"
    assert attack.name == "RLBackdoor"


def test_rl_backdoor_action_maps_to_paper_ranges():
    from fl_sandbox.attacks.rl_backdoor.action import decode_backdoor_action

    low = decode_backdoor_action(np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32))
    high = decode_backdoor_action(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))

    assert low.poison_frac == 0.0
    assert high.poison_frac == 1.0
    assert low.local_lr == 0.0
    assert high.local_lr == 0.1
    assert low.local_epochs == 1
    assert high.local_epochs == 11
    assert low.boost == 0.0
    assert high.boost == 10.0


def test_rl_backdoor_reward_prefers_asr_gain_and_penalizes_clean_drop():
    from fl_sandbox.attacks.rl_backdoor.reward import BackdoorRewardFn, BackdoorRewardInputs

    reward = BackdoorRewardFn(clean_weight=1.0, norm_weight=0.2)
    good = reward(BackdoorRewardInputs(asr_before=0.2, asr_after=0.8, clean_before=0.8, clean_after=0.78, norm_ratio=1.0))
    bad = reward(BackdoorRewardInputs(asr_before=0.2, asr_after=0.3, clean_before=0.8, clean_after=0.5, norm_ratio=2.0))

    assert good > bad
    assert good > 0.0
    assert bad < 0.0


def test_rl_backdoor_uses_one_shared_action_for_all_selected_attackers(monkeypatch):
    from fl_sandbox.attacks.rl_backdoor.attack import RLBackdoorAttack

    old = [np.array([1.0, 2.0], dtype=np.float32)]
    trained = [np.array([2.0, 4.0], dtype=np.float32)]
    calls = []

    def fake_train_on_loader(ctx, loader):
        calls.append((ctx.lr, ctx.local_epochs, loader))
        return [layer.copy() for layer in trained]

    monkeypatch.setattr("fl_sandbox.attacks.rl_backdoor.attack.train_on_loader", fake_train_on_loader)
    loaders = {1: object(), 3: object()}
    ctx = SimpleNamespace(
        selected_attacker_ids=[1, 3],
        old_weights=old,
        lr=0.05,
        local_epochs=1,
        attacker_action=None,
        poisoned_train_iters={"global_by_attacker": loaders},
        benign_weights=[],
    )

    attack = RLBackdoorAttack(default_action=(-1.0, 0.0, -1.0, 0.0))
    outputs = attack.execute(ctx)

    assert len(outputs) == 2
    assert len(calls) == 2
    assert calls[0][1:] == (1, loaders[1])
    assert calls[1][1:] == (1, loaders[3])
    np.testing.assert_allclose([calls[0][0], calls[1][0]], [0.05, 0.05])
    np.testing.assert_allclose(outputs[0][0], outputs[1][0])
    np.testing.assert_allclose(outputs[0][0], np.array([6.0, 12.0], dtype=np.float32))


def test_rl_backdoor_does_not_norm_clip_fedavg_updates(monkeypatch):
    from fl_sandbox.attacks.rl_backdoor.attack import RLBackdoorAttack

    old = [np.array([0.0], dtype=np.float32)]
    trained = [np.array([1.0], dtype=np.float32)]
    benign = [[np.array([0.1], dtype=np.float32)]]

    monkeypatch.setattr(
        "fl_sandbox.attacks.rl_backdoor.attack.train_on_loader",
        lambda ctx, loader: [layer.copy() for layer in trained],
    )
    ctx = SimpleNamespace(
        selected_attacker_ids=[1],
        old_weights=old,
        lr=0.05,
        local_epochs=1,
        attacker_action=None,
        defense_type="fedavg",
        poisoned_train_iters={"global_by_attacker": {1: object()}},
        benign_weights=benign,
    )

    outputs = RLBackdoorAttack(default_action=(1.0, 0.0, -1.0, 0.0)).execute(ctx)

    np.testing.assert_allclose(outputs[0][0], np.array([5.0], dtype=np.float32))


def test_rl_backdoor_elite_policy_saves_and_loads_best_action(tmp_path):
    from fl_sandbox.attacks.rl_backdoor.policy import EliteBackdoorPolicy

    path = tmp_path / "policy.json"
    policy = EliteBackdoorPolicy(default_action=(0.0, 0.0, 0.0, 0.0))
    policy.observe(action=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32), reward=0.2)
    policy.observe(action=np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32), reward=0.8)
    policy.save(path)

    loaded = EliteBackdoorPolicy.load(path)

    np.testing.assert_allclose(loaded.act(), np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32))
    assert loaded.best_reward == 0.8


def test_rl_backdoor_exports_td3_policy_components():
    import fl_sandbox.attacks.rl_backdoor as package
    import fl_sandbox.attacks as attacks

    assert hasattr(package, "BackdoorRLConfig")
    assert hasattr(package, "BackdoorObservationBuilder")
    assert hasattr(package, "BackdoorPolicyGymEnv")
    assert hasattr(package, "TD3BackdoorPolicy")
    assert hasattr(attacks, "BackdoorRLConfig")
    assert hasattr(attacks, "TD3BackdoorPolicy")


def test_rl_backdoor_config_builds_paper_style_td3_spaces():
    from fl_sandbox.attacks.rl_backdoor.config import BackdoorRLConfig

    config = BackdoorRLConfig(projection_dim=8)

    assert config.algorithm == "td3"
    assert config.action_dim == 4
    assert config.history_window == 1
    assert config.observation_dim == 2 * 8 + 4 + 5
    np.testing.assert_allclose(config.action_low, -np.ones(4, dtype=np.float32))
    np.testing.assert_allclose(config.action_high, np.ones(4, dtype=np.float32))


def test_rl_backdoor_observation_uses_tail_layers_attacker_info_and_feedback():
    from fl_sandbox.attacks.rl_backdoor.config import BackdoorRLConfig
    from fl_sandbox.attacks.rl_backdoor.observation import BackdoorObservationBuilder

    config = BackdoorRLConfig(projection_dim=3, history_window=1, seed=7)
    builder = BackdoorObservationBuilder(config)
    old = [
        np.array([100.0, 200.0], dtype=np.float32),
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([3.0], dtype=np.float32),
    ]
    new = [
        np.array([-100.0, -200.0], dtype=np.float32),
        np.array([2.0, 4.0], dtype=np.float32),
        np.array([6.0], dtype=np.float32),
    ]
    same_tail_new = [
        np.array([999.0, 888.0], dtype=np.float32),
        new[1].copy(),
        new[2].copy(),
    ]

    obs = builder.build(
        weights=old,
        previous_weights=old,
        last_action=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        round_idx=0,
        total_rounds=10,
        sampled_attacker_count=1,
        num_attackers=2,
        sampled_client_count=5,
        clean_acc=0.7,
        asr=0.2,
    )
    changed = builder.build(
        weights=new,
        previous_weights=old,
        last_action=np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32),
        round_idx=1,
        total_rounds=10,
        sampled_attacker_count=2,
        num_attackers=2,
        sampled_client_count=4,
        clean_acc=0.6,
        asr=0.8,
    )
    builder.reset()
    same_tail = builder.build(
        weights=same_tail_new,
        previous_weights=old,
        last_action=np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32),
        round_idx=1,
        total_rounds=10,
        sampled_attacker_count=2,
        num_attackers=2,
        sampled_client_count=4,
        clean_acc=0.6,
        asr=0.8,
    )

    assert obs.shape == (config.observation_dim,)
    assert changed.shape == (config.observation_dim,)
    np.testing.assert_allclose(changed[-5:], np.array([1.0, 0.5, 0.1, 0.6, 0.8], dtype=np.float32))
    np.testing.assert_allclose(changed, same_tail)
    assert not np.allclose(obs, changed)


def test_rl_backdoor_policy_env_steps_real_fl_runner_with_td3_action():
    from fl_sandbox.attacks.rl_backdoor.config import BackdoorRLConfig
    from fl_sandbox.attacks.rl_backdoor.env import BackdoorPolicyGymEnv

    class FakeSummary:
        clean_acc = 0.72
        backdoor_acc = 0.55
        benign_update_norms = [1.0]
        malicious_update_norms = [1.2]

    class FakeRunner:
        attacker_ids = [0, 1]

        def __init__(self):
            self.current_weights = [np.array([1.0, 2.0], dtype=np.float32)]
            self.actions = []

        def run_round(self, round_idx, *, attack, evaluate, attacker_action):
            self.actions.append(np.asarray(attacker_action, dtype=np.float32))
            self.current_weights = [self.current_weights[0] + 1.0]
            return FakeSummary()

        def _sample_clients(self, round_idx):
            return [0, 2]

    runner = FakeRunner()
    env = BackdoorPolicyGymEnv(
        runner_factory=lambda seed_offset=0: runner,
        config=BackdoorRLConfig(projection_dim=4, history_window=1, train_horizon=2),
    )

    obs, info = env.reset()
    next_obs, reward, terminated, truncated, step_info = env.step(np.array([1.0, 0.0, -1.0, 0.5], dtype=np.float32))

    assert obs.shape == env.observation_space.shape
    assert next_obs.shape == env.observation_space.shape
    np.testing.assert_allclose(runner.actions[0], np.array([1.0, 0.0, -1.0, 0.5], dtype=np.float32))
    assert reward > 0.0
    assert not terminated
    assert not truncated
    assert step_info["asr"] == 0.55
    assert info["round_idx"] == 0


def test_td3_backdoor_policy_uses_trainer_for_train_act_and_checkpoint(tmp_path):
    from fl_sandbox.attacks.rl_backdoor.config import BackdoorRLConfig
    from fl_sandbox.attacks.rl_backdoor.policy import TD3BackdoorPolicy

    class FakeTrainer:
        def __init__(self):
            self.initialized = False
            self.saved_path = None
            self.loaded_path = None
            self.collected = 0
            self.updated = 0

        def ensure_initialized(self, obs_space, action_space):
            self.initialized = True
            self.obs_shape = obs_space.shape
            self.action_shape = action_space.shape

        def collect(self, env, steps):
            self.collected += steps
            return SimpleNamespace(steps=steps, reward_mean=0.25)

        def update(self, gradient_steps):
            self.updated += gradient_steps
            return SimpleNamespace(gradient_steps=gradient_steps, loss=0.1)

        def act(self, obs, *, deterministic=False):
            return np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

        def save(self, path):
            self.saved_path = path

        def load(self, path):
            self.loaded_path = path

        def diagnostics(self):
            return {"trainer_algorithm_id": 1.0}

    class FakeEnv:
        observation_space = SimpleNamespace(shape=(12,), low=np.full(12, -np.inf), high=np.full(12, np.inf))
        action_space = SimpleNamespace(shape=(4,), low=-np.ones(4), high=np.ones(4))

    trainer = FakeTrainer()
    policy = TD3BackdoorPolicy(config=BackdoorRLConfig(train_steps=5, train_freq_steps=2), trainer_factory=lambda config: trainer)
    stats = policy.train(FakeEnv())
    action = policy.act(np.zeros(12, dtype=np.float32))
    path = tmp_path / "policy.pt"
    policy.save(path)
    policy.load(path, FakeEnv().observation_space, FakeEnv().action_space)

    assert trainer.initialized
    assert trainer.collected == 5
    assert trainer.updated == 2
    assert stats.collect.steps == 5
    np.testing.assert_allclose(action, np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
    assert trainer.saved_path == str(path)
    assert trainer.loaded_path == str(path)


def test_td3_backdoor_rollout_records_every_round_and_rl_telemetry():
    from fl_sandbox.scripts.run_rl_backdoor_td3 import _rollout

    class FakeSummary:
        def __init__(self, round_idx):
            self.round_idx = round_idx
            self.clean_loss = 1.0 / round_idx
            self.clean_acc = 0.7 + round_idx / 100.0
            self.backdoor_acc = 0.2 + round_idx / 100.0
            self.round_seconds = float(round_idx)
            self.sampled_clients = [0, 1]
            self.selected_attackers = [0]
            self.benign_update_norms = [1.0]
            self.malicious_update_norms = [1.5]
            self.malicious_cosines_to_benign = [0.25]
            self.attack_metrics = {"attack_metric": float(round_idx)}

    class FakeRunner:
        def __init__(self):
            self.evaluated = []

        def run_round(self, round_idx, *, attack, evaluate, attacker_action):
            self.evaluated.append(evaluate)
            return FakeSummary(round_idx)

    runner = FakeRunner()

    def action_fn(previous):
        del previous
        return np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32), {
            "rl_action_raw_0": 0.1,
            "rl_real_reward": 0.5,
            "rl_observation_norm": 2.0,
        }

    result = _rollout(runner, object(), 3, action_fn=action_fn, eval_every=100)

    assert runner.evaluated == [True, True, True]
    assert [point["round"] for point in result["series"]] == [1.0, 2.0, 3.0]
    assert result["series"][0]["clean_loss"] == 1.0
    assert np.isclose(result["series"][0]["backdoor_acc"], 0.21)
    assert result["series"][0]["rl_action_raw_0"] == 0.1
    assert result["series"][0]["rl_real_reward"] == 0.5
    assert result["series"][0]["rl_observation_norm"] == 2.0
    assert result["series"][0]["mean_benign_norm"] == 1.0
    assert result["series"][0]["mean_malicious_norm"] == 1.5
