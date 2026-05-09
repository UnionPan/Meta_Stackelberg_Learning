import tempfile

import gymnasium as gym
import numpy as np

from fl_sandbox.attacks.rl_attacker.config import RLAttackerConfig
from fl_sandbox.attacks.rl_attacker.trainer import build_trainer


class ToyContinuousEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        action = np.asarray(action, dtype=np.float32)
        reward = float(1.0 - np.mean(np.square(action)))
        obs = np.full(4, self.steps / 10.0, dtype=np.float32)
        return obs, reward, self.steps >= 2, False, {}


def _small_config(algorithm: str) -> RLAttackerConfig:
    return RLAttackerConfig(
        algorithm=algorithm,
        replay_capacity=64,
        batch_size=4,
        hidden_sizes=(16, 16),
        policy_lr=1e-3,
        critic_lr=1e-3,
    )


def test_sac_trainer_collects_updates_and_round_trips_policy():
    env = ToyContinuousEnv()
    trainer = build_trainer(_small_config("sac"))
    trainer.ensure_initialized(env.observation_space, env.action_space)

    before = trainer.act(np.zeros(4, dtype=np.float32), deterministic=True)
    collect_stats = trainer.collect(env, steps=4)
    update_stats = trainer.update(gradient_steps=1)

    assert collect_stats.steps == 4
    assert update_stats.gradient_steps == 1
    assert np.isfinite(update_stats.loss)
    assert trainer.diagnostics()["trainer_replay_size"] >= 4

    with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
        trainer.save(handle.name)
        restored = build_trainer(_small_config("sac"))
        restored.ensure_initialized(env.observation_space, env.action_space)
        restored.load(handle.name)
        after = restored.act(np.zeros(4, dtype=np.float32), deterministic=True)

    assert np.allclose(after, trainer.act(np.zeros(4, dtype=np.float32), deterministic=True))
    assert before.shape == after.shape


def test_td3_trainer_collects_and_updates_with_bounded_actions():
    env = ToyContinuousEnv()
    trainer = build_trainer(_small_config("td3"))
    trainer.ensure_initialized(env.observation_space, env.action_space)

    action = trainer.act(np.zeros(4, dtype=np.float32), deterministic=False)
    collect_stats = trainer.collect(env, steps=4)
    update_stats = trainer.update(gradient_steps=1)

    assert np.all(action >= env.action_space.low)
    assert np.all(action <= env.action_space.high)
    assert collect_stats.steps == 4
    assert update_stats.gradient_steps == 1
    assert np.isfinite(update_stats.loss)
    assert trainer.diagnostics()["trainer_algorithm_id"] == 1.0
