import os
import subprocess
import sys

import numpy as np
import pytest
import torch

import meta_sg.scripts.run_stackelberg_defender_td3 as defender_runner
from meta_sg.learning.config import TD3Config
from meta_sg.learning.td3 import TD3Agent
from meta_sg.scripts.run_stackelberg_defender_td3 import (
    DEFAULT_PRETRAINED_ATTACKER_CHECKPOINT,
    acquire_run_lock,
    choose_continuous_training_action,
    choose_training_action,
    compute_defender_reward,
    evaluate_defender_online_adapt,
    initialize_defender_actor_constant3,
    parse_args,
    release_run_lock,
    resolve_attacker_checkpoint,
    resolve_distribution_dir,
    restore_env_training_snapshot,
    run_clean_warmup,
    save_latest_training_checkpoint,
    select_train_defender,
    train_defender,
)


def test_resolve_attacker_checkpoint_uses_configured_default(tmp_path, monkeypatch):
    checkpoint = tmp_path / "rl_policy_latest.pt"
    torch.save({}, checkpoint)
    monkeypatch.setattr(defender_runner, "DEFAULT_PRETRAINED_ATTACKER_CHECKPOINT", checkpoint)

    path = resolve_attacker_checkpoint("")

    assert path == checkpoint
    assert path.name == "rl_policy_latest.pt"
    assert path.exists()


def test_resolve_attacker_checkpoint_rejects_missing_explicit_path(tmp_path):
    missing = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError):
        resolve_attacker_checkpoint(str(missing))


def test_resolve_distribution_dir_reads_checkpoint_config(tmp_path):
    distribution_dir = tmp_path / "mnist_clipping_median_q_0.1_init_pre_label"
    distribution_dir.mkdir()
    checkpoint = tmp_path / "rl_policy_latest.pt"
    torch.save({"config": {"rl_distribution_dir": str(distribution_dir)}}, checkpoint)

    path = resolve_distribution_dir("", checkpoint)

    assert path == distribution_dir


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


def test_skip_fixed_baselines_flag_is_explicitly_available():
    args = parse_args(["--skip-fixed-baselines"])

    assert args.skip_fixed_baselines is True


def test_nonfinite_loss_gets_large_negative_reward_instead_of_zero():
    args = parse_args(["--nonfinite-loss-penalty", "123.0"])

    reward = compute_defender_reward(args, {"post_clean_loss": float("inf"), "post_clean_acc": 0.0})

    assert reward == pytest.approx(-123.0)


def test_continuous_safe_fallback_uses_fixed_action_after_collapse():
    class DummyDefender:
        def get_action(self, _obs, noise=0.0):
            return np.asarray([0.75, -0.25, 0.5], dtype=np.float32)

    args = parse_args(
        [
            "--safe-fallback-acc-threshold",
            "0.2",
            "--continuous-safe-action",
            "0.1",
            "0.45",
            "2.0",
            "--warmup-steps",
            "0",
        ]
    )

    raw_action, used_safe = choose_continuous_training_action(
        args,
        DummyDefender(),
        np.zeros(4, dtype=np.float32),
        global_step=50,
        previous_info={"post_clean_acc": 0.1, "post_clean_loss": 1.0},
    )

    assert used_safe is True
    assert raw_action.tolist() == pytest.approx([-1.0, 1.0, -1.0])


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


def test_save_latest_training_checkpoint_overwrites_single_file(tmp_path):
    agent = TD3Agent(obs_dim=4, act_dim=3, config=TD3Config(hidden_dim=8, batch_size=2))
    rows = [{"step": 1, "alpha": 0.1}, {"step": 2, "alpha": 0.2}]
    args = parse_args(["--checkpoint-every", "1000"])

    save_latest_training_checkpoint(
        tmp_path,
        args=args,
        defender=agent,
        buffer=None,
        rows=rows[:1],
        global_step=1,
        episode=0,
    )
    save_latest_training_checkpoint(
        tmp_path,
        args=args,
        defender=agent,
        buffer=None,
        rows=rows,
        global_step=2,
        episode=1,
    )

    ckpt_path = tmp_path / "latest_checkpoint.pt"
    partial_path = tmp_path / "rounds_partial.csv"
    assert ckpt_path.exists()
    assert partial_path.exists()
    assert not (tmp_path / "checkpoint_000001.pt").exists()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["global_step"] == 2
    assert ckpt["episode"] == 1
    assert ckpt["rows"] == rows
    assert "defender" in ckpt
    assert ckpt["replay_buffer"] is None


def test_save_latest_training_checkpoint_writes_checkpoint_atomically(tmp_path, monkeypatch):
    import meta_sg.scripts.run_stackelberg_defender_td3 as runner

    agent = TD3Agent(obs_dim=4, act_dim=3, config=TD3Config(hidden_dim=8, batch_size=2))
    args = parse_args(["--checkpoint-every", "1000"])
    original_save = runner.torch.save
    saved_paths = []

    def recording_save(obj, path):
        saved_paths.append(path.name)
        original_save(obj, path)

    monkeypatch.setattr(runner.torch, "save", recording_save)

    save_latest_training_checkpoint(
        tmp_path,
        args=args,
        defender=agent,
        buffer=None,
        rows=[{"step": 1, "alpha": 0.1}],
        global_step=1,
        episode=0,
    )

    assert saved_paths == ["latest_checkpoint.pt.tmp"]
    assert (tmp_path / "latest_checkpoint.pt").exists()
    assert not (tmp_path / "latest_checkpoint.pt.tmp").exists()


def test_save_latest_training_checkpoint_includes_continuous_env_snapshot(tmp_path):
    from meta_sg.simulation.types import SimulationSnapshot

    class DummyCoordinator:
        def snapshot(self):
            return SimulationSnapshot(
                round_idx=17,
                weights=[np.asarray([1.0, 2.0], dtype=np.float32)],
                rng_state=None,
            )

    class DummyEnv:
        def __init__(self):
            self.coordinator = DummyCoordinator()
            self._round = 3

    agent = TD3Agent(obs_dim=4, act_dim=3, config=TD3Config(hidden_dim=8, batch_size=2))
    args = parse_args(["--checkpoint-every", "1000", "--train-mode", "continuous_online"])
    obs = np.asarray([9.0, 8.0, 7.0, 6.0], dtype=np.float32)

    save_latest_training_checkpoint(
        tmp_path,
        args=args,
        defender=agent,
        buffer=None,
        rows=[{"step": 1, "alpha": 0.1}],
        global_step=1,
        episode=0,
        env=DummyEnv(),
        obs=obs,
    )

    ckpt = torch.load(tmp_path / "latest_checkpoint.pt", map_location="cpu", weights_only=False)
    snapshot = ckpt["env_snapshot"]
    assert snapshot["coordinator_round_idx"] == 17
    assert snapshot["env_round"] == 3
    assert snapshot["weights"][0].tolist() == pytest.approx([1.0, 2.0])
    assert snapshot["obs"].tolist() == pytest.approx([9.0, 8.0, 7.0, 6.0])


def test_run_lock_rejects_active_pid_and_cleans_stale_pid(tmp_path):
    lock_path = tmp_path / "training.lock"
    lock_path.write_text(str(os.getpid()), encoding="utf-8")

    with pytest.raises(RuntimeError, match="already running"):
        acquire_run_lock(tmp_path)

    lock_path.write_text("999999999", encoding="utf-8")
    acquired = acquire_run_lock(tmp_path)

    assert acquired == lock_path
    assert lock_path.read_text(encoding="utf-8").strip() == str(os.getpid())

    release_run_lock(acquired)
    assert not lock_path.exists()


def test_run_clean_warmup_reuses_checkpoint_on_resume(tmp_path, monkeypatch):
    args = parse_args(["--resume-checkpoint", str(tmp_path / "latest_checkpoint.pt")])
    weights = [np.asarray([1.0, 2.0], dtype=np.float32)]
    metrics = {"clean_loss": 1.2, "clean_acc": 0.8, "backdoor_acc": 0.0}
    torch.save({"weights": weights, "metrics": metrics}, tmp_path / "warmup_checkpoint.pt")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("clean warmup should be loaded from checkpoint on resume")

    monkeypatch.setattr("meta_sg.scripts.run_stackelberg_defender_td3.FLSandboxCoordinatorAdapter", fail_if_called)

    loaded_weights, loaded_metrics = run_clean_warmup(args, output_dir=tmp_path)

    assert loaded_metrics == metrics
    assert loaded_weights[0].tolist() == pytest.approx([1.0, 2.0])


def test_train_defender_stops_at_requested_global_step():
    class DummyDefender:
        def get_action(self, _obs, noise=0.0):
            return np.zeros(3, dtype=np.float32)

        def update(self, _buffer):
            return {}

    class DummyBuffer:
        def add(self, *_args):
            pass

    class DummyEnv:
        def reset(self, seed=None):
            return np.zeros(4, dtype=np.float32)

        def step(self, raw_action, _attacker_action):
            info = {
                "round": 1,
                "clean_loss": 1.0,
                "clean_acc": 0.5,
                "post_clean_loss": 1.0,
                "post_clean_acc": 0.5,
                "backdoor_acc": 0.0,
                "attack_success_rate": 0.0,
                "malicious_update_norm": 0.0,
            }
            return np.zeros(4, dtype=np.float32), 0.0, -1.0, False, info

    args = parse_args(
        [
            "--episodes",
            "10",
            "--horizon",
            "5",
            "--stop-at-step",
            "3",
            "--warmup-steps",
            "0",
            "--print-every",
            "0",
        ]
    )

    rows = train_defender(args, DummyEnv(), DummyDefender(), DummyBuffer())

    assert [row["step"] for row in rows] == [1, 2, 3]


def test_continuous_online_training_resets_only_once_and_runs_total_steps():
    class DummyDefender:
        def __init__(self):
            self.update_calls = 0

        def get_action(self, _obs, noise=0.0):
            return np.zeros(3, dtype=np.float32)

        def update(self, _buffer):
            self.update_calls += 1
            return {"critic_loss": float(self.update_calls)}

    class DummyBuffer:
        def __init__(self):
            self.items = []

        def add(self, *args):
            self.items.append(args)

    class DummyEnv:
        def __init__(self):
            self.reset_calls = []
            self.round = 0

        def reset(self, seed=None):
            self.reset_calls.append(seed)
            self.round = 0
            return np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        def step(self, raw_action, _attacker_action):
            self.round += 1
            local_round = ((self.round - 1) % 2) + 1
            info = {
                "round": local_round,
                "clean_loss": 1.0,
                "clean_acc": 0.5,
                "post_clean_loss": 1.0,
                "post_clean_acc": 0.5,
                "backdoor_acc": 0.0,
                "attack_success_rate": 0.0,
                "malicious_update_norm": 0.0,
            }
            obs = np.asarray([float(self.round), 0.0, 0.0, 0.0], dtype=np.float32)
            return obs, 0.0, -1.0, local_round == 2, info

    args = parse_args(
        [
            "--train-mode",
            "continuous_online",
            "--episodes",
            "3",
            "--horizon",
            "2",
            "--online-update-interval",
            "2",
            "--updates-per-interval",
            "3",
            "--warmup-steps",
            "0",
            "--print-every",
            "0",
        ]
    )
    env = DummyEnv()
    defender = DummyDefender()
    buffer = DummyBuffer()

    rows = select_train_defender(args, env, defender, buffer)

    assert env.reset_calls == [7]
    assert [row["step"] for row in rows] == [1, 2, 3, 4, 5, 6]
    assert [row["round"] for row in rows] == [1, 2, 3, 4, 5, 6]
    assert defender.update_calls == 9


def test_continuous_checkpoint_saved_after_horizon_boundary_is_resume_ready(tmp_path):
    from meta_sg.simulation.types import SimulationSnapshot
    from meta_sg.learning.replay_buffer import ReplayBuffer

    class DummyCoordinator:
        def __init__(self):
            self.round_idx = 100

        def snapshot(self):
            return SimulationSnapshot(
                round_idx=self.round_idx,
                weights=[np.asarray([float(self.round_idx)], dtype=np.float32)],
                rng_state=None,
            )

    class DummyEnv:
        def __init__(self):
            self.coordinator = DummyCoordinator()
            self._round = 0

        def reset(self, seed=None):
            del seed
            self._round = 0
            return np.zeros(4, dtype=np.float32)

        def step(self, raw_action, _attacker_action):
            del raw_action
            self._round += 1
            self.coordinator.round_idx += 1
            info = {
                "round": self._round,
                "clean_loss": 1.0,
                "clean_acc": 0.5,
                "post_clean_loss": 1.0,
                "post_clean_acc": 0.5,
                "backdoor_acc": 0.0,
                "attack_success_rate": 0.0,
                "malicious_update_norm": 0.0,
            }
            return np.zeros(4, dtype=np.float32), 0.0, -1.0, self._round >= 2, info

    args = parse_args(
        [
            "--train-mode",
            "continuous_online",
            "--episodes",
            "1",
            "--horizon",
            "2",
            "--checkpoint-every",
            "2",
            "--warmup-steps",
            "0",
            "--print-every",
            "0",
        ]
    )

    defender = TD3Agent(obs_dim=4, act_dim=3, config=TD3Config(hidden_dim=8, batch_size=2))
    buffer = ReplayBuffer(capacity=8, obs_dim=4, act_dim=3)

    rows = select_train_defender(args, DummyEnv(), defender, buffer, output_dir=tmp_path)

    ckpt = torch.load(tmp_path / "latest_checkpoint.pt", map_location="cpu", weights_only=False)
    assert [row["round"] for row in rows] == [1, 2]
    assert ckpt["env_snapshot"]["coordinator_round_idx"] == 102
    assert ckpt["env_snapshot"]["env_round"] == 0


def test_restore_continuous_snapshot_normalizes_legacy_boundary_round():
    from types import SimpleNamespace
    from meta_sg.simulation.types import SimulationSnapshot

    class DummyCoordinator:
        def __init__(self):
            self.restored = None

        def restore(self, snapshot):
            self.restored = snapshot

    class DummyEnv:
        def __init__(self):
            self.config = SimpleNamespace(horizon=100)
            self.coordinator = DummyCoordinator()
            self._round = -1
            self._obs = None
            self._obs_dim = None

    env = DummyEnv()
    obs = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    restored_obs = restore_env_training_snapshot(
        env,
        {
            "coordinator_round_idx": 1100,
            "weights": [np.asarray([5.0], dtype=np.float32)],
            "env_round": 100,
            "obs": obs,
        },
    )

    assert isinstance(env.coordinator.restored, SimulationSnapshot)
    assert env.coordinator.restored.round_idx == 1100
    assert env._round == 0
    assert restored_obs.tolist() == pytest.approx(obs.tolist())


def test_online_adapt_eval_updates_policy_without_episode_reset():
    class DummyDefender:
        def __init__(self):
            self.update_calls = 0

        def get_action(self, _obs, noise=0.0):
            return np.zeros(3, dtype=np.float32)

        def update(self, _buffer):
            self.update_calls += 1
            return {"critic_loss": float(self.update_calls)}

    class DummyBuffer:
        def __init__(self):
            self.items = []

        def add(self, *args):
            self.items.append(args)

    class DummyEnv:
        def __init__(self):
            self.reset_calls = []
            self.round = 0

        def reset(self, seed=None):
            self.reset_calls.append(seed)
            self.round = 0
            return np.zeros(4, dtype=np.float32)

        def step(self, raw_action, _attacker_action):
            del raw_action
            self.round += 1
            info = {
                "round": self.round,
                "clean_loss": 1.0,
                "clean_acc": 0.5,
                "post_clean_loss": 1.0,
                "post_clean_acc": 0.5,
                "backdoor_acc": 0.0,
                "attack_success_rate": 0.0,
                "malicious_update_norm": 0.0,
            }
            return np.zeros(4, dtype=np.float32), 0.0, -1.0, False, info

    args = parse_args(
        [
            "--eval-online-update-interval",
            "2",
            "--eval-online-updates-per-interval",
            "3",
            "--eval-online-exploration-noise",
            "0.0",
            "--eval-horizon",
            "5",
            "--print-every",
            "0",
        ]
    )
    env = DummyEnv()
    defender = DummyDefender()
    buffer = DummyBuffer()

    rows = evaluate_defender_online_adapt(args, env, defender, buffer)

    assert env.reset_calls == [10007]
    assert [row["step"] for row in rows] == [1, 2, 3, 4, 5]
    assert defender.update_calls == 6
    assert [row["online_adapt"] for row in rows] == [1, 1, 1, 1, 1]
