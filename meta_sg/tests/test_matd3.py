import numpy as np
import pytest
import torch

from meta_sg.games.bsmg_env import BSMGConfig, BSMGEnv
from meta_sg.learning.config import TD3Config
from meta_sg.learning.matd3 import JointReplayBuffer, MATD3AgentPair
from meta_sg.scripts.run_matd3_colearning import (
    _build_env,
    _br_stage_steps,
    _clip_learned_defender_action,
    _phase_step_budget,
    compare_best_response_defenders,
    compare_fixed_defender_baseline,
    compare_fixed_attacker_defenders,
    compare_with_baselines,
    parse_args,
    prefer_eval_summary,
)
from meta_sg.simulation.stub import StubCoordinator
from meta_sg.strategies.attacks.adaptive import AdaptiveAttackStrategy
from meta_sg.strategies.defenses.paper import PaperDefenseStrategy
from meta_sg.strategies.types import ATTACK_DOMAIN


def _make_env(seed: int = 0):
    attacker_agent = _ZeroAgent()
    attack_type = ATTACK_DOMAIN["rl"]
    env = BSMGEnv(
        coordinator=StubCoordinator(num_clients=6, num_attackers=2, subsample_rate=1.0, seed=seed),
        attack_type=attack_type,
        attack_strategy=AdaptiveAttackStrategy(attack_type, attacker_agent),
        defense_strategy=PaperDefenseStrategy(),
        config=BSMGConfig(horizon=4, reward_mode="loss", lambda_bd=0.0, history_len=0),
    )
    return env


def test_joint_replay_buffer_samples_both_agents():
    buffer = JointReplayBuffer(capacity=8, obs_dim=5, defender_act_dim=3, attacker_act_dim=3)
    obs = np.zeros(5, dtype=np.float32)
    next_obs = np.ones(5, dtype=np.float32)

    buffer.add(
        obs=obs,
        defender_action=np.zeros(3, dtype=np.float32),
        attacker_action=np.ones(3, dtype=np.float32),
        defender_reward=0.25,
        attacker_reward=-0.25,
        next_obs=next_obs,
        done=False,
    )

    batch = buffer.sample(1, device=torch.device("cpu"))

    assert batch.obs.shape == (1, 5)
    assert batch.defender_action.shape == (1, 3)
    assert batch.attacker_action.shape == (1, 3)
    assert batch.defender_reward.item() == pytest.approx(0.25)
    assert batch.attacker_reward.item() == pytest.approx(-0.25)


def test_matd3_pair_collects_and_updates_on_bsmg_env():
    env = _make_env(seed=3)
    obs = env.reset(seed=3)
    cfg = TD3Config(
        hidden_dim=16,
        batch_size=2,
        buffer_capacity=32,
        warmup_steps=0,
        exploration_noise=0.05,
    )
    pair = MATD3AgentPair(obs_dim=obs.shape[0], defender_act_dim=3, attacker_act_dim=3, config=cfg)
    buffer = JointReplayBuffer(32, obs.shape[0], 3, 3)

    for _ in range(4):
        defender_action, attacker_action = pair.get_actions(obs, noise=0.1)
        next_obs, defender_reward, attacker_reward, done, _ = env.step(defender_action, attacker_action)
        buffer.add(obs, defender_action, attacker_action, defender_reward, attacker_reward, next_obs, done)
        obs = env.reset(seed=4) if done else next_obs

    stats = pair.update(buffer)

    assert "defender_critic_loss" in stats
    assert "attacker_critic_loss" in stats


def test_matd3_pair_can_update_defender_only_for_fixed_attacker():
    env = _make_env(seed=5)
    obs = env.reset(seed=5)
    cfg = TD3Config(hidden_dim=16, batch_size=2, buffer_capacity=32, warmup_steps=0)
    pair = MATD3AgentPair(obs_dim=obs.shape[0], defender_act_dim=3, attacker_act_dim=3, config=cfg)
    buffer = JointReplayBuffer(32, obs.shape[0], 3, 3)

    for _ in range(4):
        defender_action, _ = pair.get_actions(obs, noise=0.1)
        attacker_action = np.zeros(3, dtype=np.float32)
        next_obs, defender_reward, attacker_reward, done, _ = env.step(defender_action, attacker_action)
        buffer.add(obs, defender_action, attacker_action, defender_reward, attacker_reward, next_obs, done)
        obs = env.reset(seed=6) if done else next_obs

    stats = pair.update(buffer, update_attacker=False)

    assert "defender_critic_loss" in stats
    assert "attacker_critic_loss" not in stats


def test_matd3_pair_can_update_attacker_only_against_fixed_defender():
    env = _make_env(seed=8)
    obs = env.reset(seed=8)
    cfg = TD3Config(hidden_dim=16, batch_size=2, buffer_capacity=32, warmup_steps=0)
    pair = MATD3AgentPair(obs_dim=obs.shape[0], defender_act_dim=3, attacker_act_dim=3, config=cfg)
    buffer = JointReplayBuffer(32, obs.shape[0], 3, 3)
    fixed_defender = np.zeros(3, dtype=np.float32)

    for _ in range(4):
        _, attacker_action = pair.get_actions(obs, noise=0.1)
        next_obs, defender_reward, attacker_reward, done, _ = env.step(fixed_defender, attacker_action)
        buffer.add(obs, fixed_defender, attacker_action, defender_reward, attacker_reward, next_obs, done)
        obs = env.reset(seed=9) if done else next_obs

    stats = pair.update(
        buffer,
        update_defender=False,
        update_attacker=True,
        fixed_defender_action=fixed_defender,
    )

    assert "attacker_critic_loss" in stats
    assert "defender_critic_loss" not in stats


def test_compare_with_baselines_reports_reward_improvement():
    matd3 = {
        "mean_defender_reward": -1.0,
        "final_clean_acc": 0.6,
    }
    baselines = {
        "fixed_mid": {
            "mean_defender_reward": -1.5,
            "final_clean_acc": 0.4,
        }
    }

    comparison = compare_with_baselines(matd3, baselines)

    assert comparison["best_baseline"] == "fixed_mid"
    assert comparison["defender_reward_improvement_vs_best_baseline"] == pytest.approx(0.5)
    assert comparison["final_clean_acc_improvement_vs_best_baseline"] == pytest.approx(0.2)


def test_prefer_eval_summary_uses_deterministic_eval_when_present():
    summary = {
        "mean_defender_reward": -2.0,
        "eval": {"mean_defender_reward": -1.0, "final_clean_acc": 0.7},
    }

    preferred = prefer_eval_summary(summary)

    assert preferred["mean_defender_reward"] == pytest.approx(-1.0)
    assert preferred["final_clean_acc"] == pytest.approx(0.7)


def test_compare_fixed_attacker_defenders_reports_co_learning_delta():
    comparison = compare_fixed_attacker_defenders(
        co_learning_eval={"mean_defender_reward": -1.0, "final_clean_acc": 0.6},
        fixed_attacker_eval={"mean_defender_reward": -1.2, "final_clean_acc": 0.55},
    )

    assert comparison["metric_source"] == "fixed_attacker_eval"
    assert comparison["co_learning_reward_improvement"] == pytest.approx(0.2)
    assert comparison["co_learning_final_clean_acc_improvement"] == pytest.approx(0.05)


def test_compare_fixed_defender_baseline_reports_co_learning_delta():
    comparison = compare_fixed_defender_baseline(
        co_learning_eval={"mean_defender_reward": -1.0, "final_clean_acc": 0.6},
        fixed_defender_eval={"mean_defender_reward": -1.3, "final_clean_acc": 0.45},
    )

    assert comparison["metric_source"] == "learned_attacker_eval"
    assert comparison["co_learning_reward_improvement"] == pytest.approx(0.3)
    assert comparison["co_learning_final_clean_acc_improvement"] == pytest.approx(0.15)


def test_compare_best_response_defenders_reports_co_learning_delta():
    comparison = compare_best_response_defenders(
        co_learning_br_eval={"mean_defender_reward": -2.0, "final_clean_acc": 0.52},
        fixed_defender_br_eval={"mean_defender_reward": -2.4, "final_clean_acc": 0.48},
    )

    assert comparison["metric_source"] == "attacker_best_response_eval"
    assert comparison["co_learning_br_reward_improvement"] == pytest.approx(0.4)
    assert comparison["co_learning_br_final_clean_acc_improvement"] == pytest.approx(0.04)


def test_runner_can_enable_history_features_for_adaptive_policies():
    no_history_args = parse_args(["--backend", "stub", "--history-len", "0"])
    history_args = parse_args(["--backend", "stub", "--history-len", "2"])

    no_history_obs = _build_env(no_history_args, seed=1).reset(seed=1)
    history_obs = _build_env(history_args, seed=1).reset(seed=1)

    assert history_obs.shape[0] > no_history_obs.shape[0]


def test_runner_can_select_targeted_attack_and_backdoor_penalty():
    args = parse_args(["--backend", "stub", "--attack-name", "brl", "--lambda-bd", "1.5"])

    env = _build_env(args, seed=2)

    assert env.attack_type.name == "brl"
    assert env.attack_type.objective == "targeted"
    assert env.config.lambda_bd == pytest.approx(1.5)


def test_runner_can_select_accuracy_reward_mode_for_global_attack():
    args = parse_args(["--backend", "stub", "--reward-mode", "accuracy"])

    env = _build_env(args, seed=3)

    assert env.config.reward_mode == "accuracy"


def test_runner_clips_learned_defender_actions_to_configured_bounds():
    args = parse_args(
        [
            "--defender-action-low",
            "0.0",
            "-1.0",
            "-0.5",
            "--defender-action-high",
            "1.0",
            "0.0",
            "0.5",
        ]
    )

    clipped = _clip_learned_defender_action(args, np.asarray([-0.8, 0.8, 0.9], dtype=np.float32))

    assert clipped.tolist() == pytest.approx([0.0, 0.0, 0.5])


def test_br_train_steps_overrides_episode_horizon_budget():
    args = parse_args(["--br-train-episodes", "3", "--br-train-horizon", "7", "--br-train-steps", "80"])

    assert _phase_step_budget(args, episodes_attr="br_train_episodes", horizon=7, steps_attr="br_train_steps") == 80


def test_br_train_step_budget_defaults_to_episodes_times_horizon():
    args = parse_args(["--br-train-episodes", "3", "--br-train-horizon", "7"])

    assert _phase_step_budget(args, episodes_attr="br_train_episodes", horizon=7, steps_attr="br_train_steps") == 21


def test_br_stage_steps_are_sorted_unique_and_within_budget():
    args = parse_args(["--br-train-steps", "20", "--br-stage-steps", "10", "5", "10", "25"])

    assert _br_stage_steps(args, total_steps=20) == [5, 10, 20]


class _ZeroAgent:
    def get_action(self, obs, noise: float = 0.0):
        del obs, noise
        return np.zeros(3, dtype=np.float32)
