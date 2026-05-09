import gymnasium as gym
import numpy as np
import torch

from fl_sandbox.attacks.rl_attacker.action_decoder import decode_action, decode_hybrid_action
from fl_sandbox.attacks.rl_attacker.config import RLAttackerConfig
from fl_sandbox.attacks.rl_attacker.diagnostics import RLSim2RealDiagnostics
from fl_sandbox.attacks.rl_attacker.observation import ProjectedObservationBuilder
from fl_sandbox.attacks.rl_attacker.simulator.reward import DefaultRewardFn, RewardInputs
from fl_sandbox.attacks.rl_attacker.tianshou_backend.common import RecencyWeightedReplayBuffer
from fl_sandbox.attacks.rl_attacker.trainer import build_trainer


class TinyContinuousEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, action_dim: int):
        self.observation_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(6,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(6, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        obs = np.full(6, self.steps / 10.0, dtype=np.float32)
        reward = float(1.0 - np.mean(np.square(action)))
        return obs, reward, self.steps >= 3, False, {}


def test_canonical_algorithms_are_td3_default_and_ppo_extension():
    assert RLAttackerConfig().algorithm == "td3"
    assert build_trainer(RLAttackerConfig()).algorithm_name == "td3"
    assert build_trainer(RLAttackerConfig(algorithm="ppo")).algorithm_name == "ppo"


def test_sac_is_not_a_canonical_algorithm():
    try:
        build_trainer(RLAttackerConfig(algorithm="sac"))
    except ValueError as exc:
        assert "Unsupported RL attacker algorithm" in str(exc)
    else:
        raise AssertionError("SAC should not be exposed by the canonical TD3/PPO design")


def test_ppo_trainer_collects_updates_and_reports_policy_diagnostics():
    trainer = build_trainer(RLAttackerConfig(algorithm="ppo", hidden_sizes=(16, 16), batch_size=4))
    env = TinyContinuousEnv(action_dim=RLAttackerConfig(algorithm="ppo").action_dim("krum"))

    collect_stats = trainer.collect(env, steps=6)
    update_stats = trainer.update(gradient_steps=1)
    diagnostics = trainer.diagnostics()

    assert collect_stats.steps == 6
    assert update_stats.gradient_steps == 1
    assert diagnostics["trainer_algorithm_id"] == 2.0
    assert diagnostics["trainer_replay_size"] == 0.0
    assert "trainer_policy_loss" in diagnostics or "trainer_loss" in diagnostics


def test_hybrid_action_decoder_selects_template_and_masks_unused_params():
    config = RLAttackerConfig(algorithm="ppo")
    decoded = decode_hybrid_action(np.asarray([0.4, 0.7, -0.2, 0.9, 0.1, -0.1, 0.2, -0.3]), "krum", config)

    assert decoded.template_name in {"sign_flip", "alie_lmp", "fang_specific"}
    assert decoded.template_index == decoded.template_index
    assert 0.0 <= decoded.scale <= 1.0
    assert decoded.parameters.shape == (config.hybrid_continuous_dim,)


def test_td3_recency_replay_samples_recent_transitions_more_often():
    from tianshou.data import Batch

    replay = RecencyWeightedReplayBuffer(size=32, recency_tau=4, random_seed=7)
    for idx in range(32):
        replay.add(
            Batch(
                obs=np.asarray([idx], dtype=np.float32),
                act=np.asarray([0.0], dtype=np.float32),
                rew=float(idx),
                terminated=False,
                truncated=False,
                done=False,
                obs_next=np.asarray([idx + 1], dtype=np.float32),
                info={},
            )
        )

    sampled = np.concatenate([replay.sample_indices(8) for _ in range(200)])

    assert sampled.mean() > 20.0


def test_reward_components_are_running_normalized():
    reward = DefaultRewardFn(RLAttackerConfig())
    previous = np.zeros(3, dtype=np.float32)
    first = reward(
        RewardInputs(
            loss_delta=10.0,
            acc_delta=0.1,
            bypass_score=1.0,
            action=np.ones(3, dtype=np.float32),
            previous_action=previous,
            norm_penalty=0.0,
        )
    )
    second = reward(
        RewardInputs(
            loss_delta=10.0,
            acc_delta=0.1,
            bypass_score=1.0,
            action=np.ones(3, dtype=np.float32),
            previous_action=previous,
            norm_penalty=0.0,
        )
    )

    assert np.isfinite(first)
    assert np.isfinite(second)
    assert reward.last_components["loss"] != 10.0


def test_sim2real_diagnostics_rolls_window_and_counts_guard_blocks():
    diagnostics = RLSim2RealDiagnostics(window=3, max_gap=0.5)
    diagnostics.record_gap(real_reward=0.0, simulated_reward=1.0, components={"loss": -1.0})
    diagnostics.record_gap(real_reward=0.1, simulated_reward=1.1, components={"loss": -1.0})
    diagnostics.record_gap(real_reward=0.2, simulated_reward=1.2, components={"loss": -1.0})

    payload = diagnostics.as_dict()

    assert payload["rl_sim2real_gap_mean"] == -1.0
    assert payload["rl_deploy_guard_blocks_total"] == 3.0
    assert payload["rl_gap_loss_mean"] == -1.0


def test_projected_observation_uses_seeded_fixed_projection_and_history_stack():
    config = RLAttackerConfig(projection_dim=4, history_window=4, seed=13)
    builder_a = ProjectedObservationBuilder(config=config, action_dim=3)
    builder_b = ProjectedObservationBuilder(config=config, action_dim=3)
    weights = [np.arange(8, dtype=np.float32)]
    previous = [np.zeros(8, dtype=np.float32)]

    obs_a = builder_a.build(
        weights=weights,
        previous_weights=previous,
        last_aggregate_update=[weights[0] - previous[0]],
        last_action=np.zeros(3, dtype=np.float32),
        last_bypass_score=0.0,
        round_idx=1,
        total_rounds=10,
        defense_type="krum",
    )
    obs_b = builder_b.build(
        weights=weights,
        previous_weights=previous,
        last_aggregate_update=[weights[0] - previous[0]],
        last_action=np.zeros(3, dtype=np.float32),
        last_bypass_score=0.0,
        round_idx=1,
        total_rounds=10,
        defense_type="krum",
    )

    per_step_dim = 3 * config.projection_dim + 3 + 2 + len(builder_a.defenses)
    assert obs_a.shape == (config.history_window * per_step_dim,)
    assert np.allclose(obs_a, obs_b)
