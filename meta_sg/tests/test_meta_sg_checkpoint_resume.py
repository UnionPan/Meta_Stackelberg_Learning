import json
import random

import numpy as np
import pytest
import torch

from meta_sg.learning.config import MetaSGConfig, TD3Config
from meta_sg.learning.meta_sg_trainer import MetaSGTrainer
from meta_sg.learning.replay_buffer import ReplayBuffer
from meta_sg.simulation.stub import StubCoordinator
from meta_sg.strategies.types import ATTACK_DOMAIN


def _filled_replay_buffer() -> ReplayBuffer:
    buffer = ReplayBuffer(capacity=7, obs_dim=2, act_dim=1)
    for index in range(10):
        buffer.add(
            np.array([index, index + 0.5], dtype=np.float32),
            np.array([index / 10], dtype=np.float32),
            float(index),
            np.array([index + 1, index + 1.5], dtype=np.float32),
            index % 3 == 0,
        )
    return buffer


def test_replay_buffer_hdf5_round_trip_preserves_state_and_sampling(tmp_path):
    source = _filled_replay_buffer()
    path = tmp_path / "buffer.hdf5"

    source.save(path)
    restored = ReplayBuffer.load(path, capacity=7, obs_dim=2, act_dim=1)

    assert len(restored) == len(source) == 7
    assert restored.capacity == source.capacity
    assert restored.tianshou_buffer.last_index == source.tianshou_buffer.last_index
    np.random.seed(123)
    expected = source.sample(5)
    np.random.seed(123)
    actual = restored.sample(5)
    for left, right in zip(expected, actual):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize(
    ("capacity", "obs_dim", "act_dim"),
    [(8, 2, 1), (7, 3, 1), (7, 2, 2)],
)
def test_replay_buffer_load_rejects_invariant_mismatch(
    tmp_path,
    capacity,
    obs_dim,
    act_dim,
):
    source = _filled_replay_buffer()
    path = tmp_path / "buffer.hdf5"
    source.save(path)

    with pytest.raises(ValueError, match="replay buffer invariant"):
        ReplayBuffer.load(
            path,
            capacity=capacity,
            obs_dim=obs_dim,
            act_dim=act_dim,
        )


def _make_trainer(tmp_path) -> MetaSGTrainer:
    def coordinator_factory(*, attack_type, horizon, seed):
        del attack_type, horizon
        return StubCoordinator(
            num_clients=3,
            num_attackers=1,
            layer_shapes=[(2,), (1,)],
            seed=seed,
        )

    return MetaSGTrainer(
        coordinator_factory=coordinator_factory,
        attack_domain=[ATTACK_DOMAIN["rl"]],
        meta_config=MetaSGConfig(
            T=1,
            K=1,
            H_mnist=1,
            l=1,
            N_A=1,
            post_br_defender_updates=0,
        ),
        td3_config=TD3Config(
            hidden_dim=8,
            batch_size=2,
            buffer_capacity=7,
            warmup_steps=0,
        ),
        obs_dim=2,
        act_dim=3,
        device=torch.device("cpu"),
        checkpoint_master_seed=42,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        checkpoint_interval=1,
    )


def _fill_attacker_buffers(trainer: MetaSGTrainer) -> None:
    for buffer in trainer.attacker_buffers.values():
        for index in range(5):
            buffer.add(
                np.array([index, index + 1], dtype=np.float32),
                np.array([0.1, 0.2, 0.3], dtype=np.float32),
                float(index),
                np.array([index + 1, index + 2], dtype=np.float32),
                False,
            )


def test_checkpoint_manifest_contains_complete_training_state(tmp_path):
    trainer = _make_trainer(tmp_path)
    _fill_attacker_buffers(trainer)
    checkpoint = tmp_path / "checkpoint"

    trainer.save(checkpoint, completed_iteration=3)

    metadata = json.loads((checkpoint / "checkpoint.json").read_text())
    assert metadata["schema_version"] == 2
    assert metadata["completed_iteration"] == 3
    assert metadata["invariants"] == {
        "obs_dim": 2,
        "defender_act_dim": 3,
        "attacker_act_dim": 3,
        "buffer_capacity": 7,
        "attackers": ["rl"],
    }
    assert (checkpoint / "rng_state.pt").exists()
    assert (checkpoint / "attacker_buffer_rl.hdf5").exists()


def test_checkpoint_load_restores_rng_and_replay_buffers(tmp_path):
    trainer = _make_trainer(tmp_path)
    _fill_attacker_buffers(trainer)
    checkpoint = tmp_path / "checkpoint"
    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)
    trainer.save(checkpoint, completed_iteration=3)

    expected_python = random.random()
    expected_numpy = np.random.random(5)
    expected_torch = torch.rand(5)
    expected_cuda = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None
    )
    saved_lengths = {
        name: len(buffer) for name, buffer in trainer.attacker_buffers.items()
    }

    random.seed(201)
    np.random.seed(202)
    torch.manual_seed(203)
    for buffer in trainer.attacker_buffers.values():
        buffer.add(np.zeros(2), np.zeros(3), -1.0, np.zeros(2), True)

    trainer.load(checkpoint)

    assert random.random() == expected_python
    np.testing.assert_array_equal(np.random.random(5), expected_numpy)
    torch.testing.assert_close(torch.rand(5), expected_torch, rtol=0, atol=0)
    if expected_cuda is not None:
        actual_cuda = torch.cuda.get_rng_state_all()
        assert len(actual_cuda) == len(expected_cuda)
        for actual, expected in zip(actual_cuda, expected_cuda):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert {
        name: len(buffer) for name, buffer in trainer.attacker_buffers.items()
    } == saved_lengths
    assert trainer.resume_fidelity == "continuous"


def test_checkpoint_validation_rejects_before_mutating_trainer(tmp_path):
    trainer = _make_trainer(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    trainer.save(checkpoint, completed_iteration=3)
    before = {
        key: value.clone() for key, value in trainer.defender.get_params().items()
    }
    metadata_path = checkpoint / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["invariants"]["obs_dim"] += 1
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="obs_dim"):
        trainer.load(checkpoint)

    after = trainer.defender.get_params()
    for key, value in before.items():
        torch.testing.assert_close(after[key], value)


def test_latest_checkpoint_rollback_preserves_previous_complete_state(
    tmp_path,
    monkeypatch,
):
    trainer = _make_trainer(tmp_path)
    _fill_attacker_buffers(trainer)
    latest = tmp_path / "checkpoints" / "latest"
    trainer.save(latest, completed_iteration=1, replace=True)

    def fail_save(_path):
        raise OSError("injected replay save failure")

    monkeypatch.setattr(trainer.attacker_buffers["rl"], "save", fail_save)
    with pytest.raises(OSError, match="injected"):
        trainer.save(latest, completed_iteration=2, replace=True)

    metadata = json.loads((latest / "checkpoint.json").read_text())
    assert metadata["completed_iteration"] == 1
    leftovers = [
        path.name
        for path in latest.parent.iterdir()
        if path.name.startswith(".latest.")
    ]
    assert leftovers == []


def test_model_only_checkpoint_requires_explicit_discontinuous_resume(tmp_path):
    trainer = _make_trainer(tmp_path)
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    trainer.defender.save(str(legacy / "defender_meta.pt"))
    for name, agent in trainer.attacker_agents.items():
        agent.save(str(legacy / f"attacker_{name}.pt"))
    (legacy / "checkpoint.json").write_text(
        json.dumps({"completed_iteration": 1, "master_seed": 42})
    )

    with pytest.raises(ValueError, match="model-only"):
        trainer.load(legacy)

    trainer.load(legacy, allow_model_only=True)
    assert trainer.resume_fidelity == "model_only_discontinuity"


def test_model_only_resume_cli_requires_checkpoint():
    from meta_sg.scripts.run_meta_sg_pretraining import parse_args

    assert parse_args([]).allow_model_only_resume is False
    with pytest.raises(SystemExit):
        parse_args(["--allow-model-only-resume"])


def _stub_cli_args(output_dir, run_name, *, iterations):
    return [
        "--backend",
        "stub",
        "--output-dir",
        str(output_dir),
        "--run-name",
        run_name,
        "--T",
        str(iterations),
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
        "2",
        "--latest-checkpoint-only",
        "--log-interval",
        "1",
        "--seed",
        "77",
        "--device",
        "cpu",
    ]


def test_deterministic_stub_resume_matches_uninterrupted_training(tmp_path):
    from meta_sg.scripts.run_meta_sg_pretraining import main

    uninterrupted_root = tmp_path / "uninterrupted"
    main(_stub_cli_args(uninterrupted_root, "run", iterations=4))
    uninterrupted = [
        json.loads(line)
        for line in (uninterrupted_root / "run" / "metrics.jsonl")
        .read_text()
        .splitlines()
    ]

    split_root = tmp_path / "split"
    main(_stub_cli_args(split_root, "first", iterations=2))
    checkpoint = split_root / "first" / "checkpoints" / "latest"
    main(
        [
            *_stub_cli_args(split_root, "second", iterations=2),
            "--resume-from",
            str(checkpoint),
            "--start-iteration",
            "2",
            "--total-iterations",
            "4",
        ]
    )
    resumed = [
        json.loads(line)
        for line in (split_root / "second" / "metrics.jsonl").read_text().splitlines()
    ]

    assert [record["iteration"] for record in resumed] == [3, 4]
    for expected, actual in zip(uninterrupted[2:], resumed):
        assert actual["batch_attack_types"] == expected["batch_attack_types"]
        assert actual["reward_mean"] == pytest.approx(
            expected["reward_mean"], abs=1e-12
        )
        assert actual["reptile_delta_norm"] == pytest.approx(
            expected["reptile_delta_norm"], abs=1e-12
        )
        assert actual["reptile_actor_delta_norm"] == pytest.approx(
            expected["reptile_actor_delta_norm"], abs=1e-12
        )
        assert actual["reptile_critic_delta_norm"] == pytest.approx(
            expected["reptile_critic_delta_norm"], abs=1e-12
        )
        assert actual["buffer_sizes"] == expected["buffer_sizes"]
