import os
import subprocess
import sys

import numpy as np
import pytest

from meta_sg.learning.config import TD3Config
from meta_sg.learning.td3 import TD3Agent
from meta_sg.scripts.run_stackelberg_defender_td3 import (
    DEFAULT_PRETRAINED_ATTACKER_CHECKPOINT,
    choose_training_action,
    initialize_defender_actor_constant3,
    parse_args,
    resolve_attacker_checkpoint,
    resolve_distribution_dir,
)


def test_resolve_attacker_checkpoint_uses_repository_80k_policy_by_default():
    path = resolve_attacker_checkpoint("")

    assert path == DEFAULT_PRETRAINED_ATTACKER_CHECKPOINT
    assert path.name == "rl_policy_latest.pt"
    assert path.exists()


def test_resolve_attacker_checkpoint_rejects_missing_explicit_path(tmp_path):
    missing = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError):
        resolve_attacker_checkpoint(str(missing))


def test_resolve_distribution_dir_reads_repository_checkpoint_config():
    path = resolve_distribution_dir("", DEFAULT_PRETRAINED_ATTACKER_CHECKPOINT)

    assert path.name == "mnist_clipping_median_q_0.1_init_pre_label"
    assert path.exists()


def test_initialize_defender_actor_constant3_sets_all_raw_dimensions():
    agent = TD3Agent(obs_dim=4, act_dim=3, config=TD3Config(hidden_dim=8, batch_size=2))
    obs_a = np.zeros(4, dtype=np.float32)
    obs_b = np.ones(4, dtype=np.float32)

    initialize_defender_actor_constant3(agent, raw_action=np.asarray([-0.8, -1.0, 0.0], dtype=np.float32))

    assert agent.get_action(obs_a, noise=0.0).tolist() == pytest.approx([-0.8, -0.999, 0.0], abs=1e-5)
    assert agent.get_action(obs_b, noise=0.0).tolist() == pytest.approx([-0.8, -0.999, 0.0], abs=1e-5)


def test_choose_training_action_uses_uniform_random_during_warmup(monkeypatch):
    agent = TD3Agent(obs_dim=4, act_dim=3, config=TD3Config(hidden_dim=8, batch_size=2))
    obs = np.zeros(4, dtype=np.float32)
    args = parse_args(["--warmup-steps", "2", "--exploration-noise", "0.0"])
    monkeypatch.setattr(np.random, "uniform", lambda low, high, size: np.asarray([-0.5, 0.25, 0.75]))

    warmup = choose_training_action(args, agent, obs, global_step=1)
    policy = choose_training_action(args, agent, obs, global_step=2)

    assert warmup.tolist() == pytest.approx([-0.5, 0.25, 0.75])
    assert policy.tolist() != pytest.approx([-0.5, 0.25, 0.75])


def test_default_initial_action_keeps_beta_exploration_near_paper_trim_prior():
    args = parse_args([])

    assert args.initial_raw_action == pytest.approx([-0.95, -0.111111, 1.0], abs=1e-5)


def test_default_neuroclip_epsilon_avoids_self_damaging_lower_bound():
    args = parse_args([])

    assert args.neuroclip_eps_min == pytest.approx(2.0)


def test_runner_help_suppresses_tensorflow_startup_noise():
    env = dict(os.environ)
    env.pop("TF_CPP_MIN_LOG_LEVEL", None)
    result = subprocess.run(
        [sys.executable, "-m", "meta_sg.scripts.run_stackelberg_defender_td3", "--help"],
        cwd=str(DEFAULT_PRETRAINED_ATTACKER_CHECKPOINT.parents[5]),
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0
    assert "usage: run_stackelberg_defender_td3.py" in result.stdout
    assert "oneDNN" not in result.stderr
    assert "absl::InitializeLog" not in result.stderr
