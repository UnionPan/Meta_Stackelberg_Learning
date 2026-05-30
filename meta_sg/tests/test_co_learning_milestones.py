import numpy as np
import pytest

from meta_sg.co_learning.milestones import (
    FixedClippedMedianPolicy,
    decode_henger_attacker_action,
    decode_m1_defender_action,
    poison_survival_cosine,
    run_m1_episode,
    sweep_fixed_defenders,
)
from meta_sg.learning.config import TD3Config
from meta_sg.learning.td3 import TD3Agent
from meta_sg.simulation.stub import StubCoordinator


def test_henger_attacker_action_decodes_to_epsilon_and_local_steps():
    low = decode_henger_attacker_action(np.asarray([-1.0, -1.0], dtype=np.float32))
    mid = decode_henger_attacker_action(np.asarray([0.0, 0.0], dtype=np.float32))
    high = decode_henger_attacker_action(np.asarray([1.0, 1.0], dtype=np.float32))

    assert low.epsilon == pytest.approx(0.1)
    assert low.local_steps == 1
    assert mid.epsilon == pytest.approx(15.0)
    assert mid.local_steps == 25
    assert high.epsilon == pytest.approx(29.9)
    assert high.local_steps == 49


def test_m1_defender_action_decodes_to_clipped_median_parameters():
    low = decode_m1_defender_action(np.asarray([-1.0, -1.0], dtype=np.float32))
    mid = decode_m1_defender_action(np.asarray([0.0, 0.0], dtype=np.float32))
    high = decode_m1_defender_action(np.asarray([1.0, 1.0], dtype=np.float32))

    assert low.clip_radius == pytest.approx(0.5)
    assert low.trim_ratio == pytest.approx(0.0)
    assert mid.clip_radius == pytest.approx(2.5)
    assert mid.trim_ratio == pytest.approx(0.2)
    assert high.clip_radius == pytest.approx(4.5)
    assert high.trim_ratio == pytest.approx(0.4)


def test_fixed_clipped_median_policy_returns_raw_action_for_requested_parameters():
    policy = FixedClippedMedianPolicy(clip_radius=3.5, trim_ratio=0.1)

    raw = policy.get_action(np.zeros(4, dtype=np.float32))
    decoded = decode_m1_defender_action(raw)

    assert raw.shape == (2,)
    assert decoded.clip_radius == pytest.approx(3.5)
    assert decoded.trim_ratio == pytest.approx(0.1)


def test_poison_survival_cosine_reports_alignment_between_malicious_and_aggregate():
    value = poison_survival_cosine(
        malicious_update=np.asarray([1.0, 0.0], dtype=np.float32),
        aggregate_update=np.asarray([1.0, 1.0], dtype=np.float32),
    )

    assert value == pytest.approx(2 ** -0.5)


def test_run_m1_episode_populates_attacker_buffer_and_diagnostics():
    coord = StubCoordinator(num_clients=6, num_attackers=2, subsample_rate=1.0, seed=3)
    obs_dim = coord.spec.empty_weights()[-1].size + coord.spec.empty_weights()[-2].size
    attacker = TD3Agent(
        obs_dim=obs_dim,
        act_dim=2,
        config=TD3Config(hidden_dim=16, batch_size=4, warmup_steps=0),
    )

    result = run_m1_episode(
        coordinator=coord,
        attacker=attacker,
        defender=FixedClippedMedianPolicy(clip_radius=3.5, trim_ratio=0.1),
        horizon=3,
        seed=11,
    )

    assert result.transitions_collected == 3
    assert result.attacker_buffer_size == 3
    assert result.clip_radius == pytest.approx(3.5)
    assert result.trim_ratio == pytest.approx(0.1)
    assert 0.1 <= result.mean_epsilon <= 29.9
    assert 1.0 <= result.mean_local_steps <= 49.0
    assert -1.0 <= result.mean_survival <= 1.0
    assert np.isfinite(result.mean_attacker_reward)


def test_run_m1_episode_can_update_attacker_policy_from_collected_buffer():
    coord = StubCoordinator(num_clients=6, num_attackers=2, subsample_rate=1.0, seed=4)
    obs_dim = coord.spec.empty_weights()[-1].size + coord.spec.empty_weights()[-2].size
    attacker = TD3Agent(
        obs_dim=obs_dim,
        act_dim=2,
        config=TD3Config(hidden_dim=16, batch_size=2, warmup_steps=0),
    )
    before = {
        key: value.clone()
        for key, value in attacker.get_params().items()
        if key.startswith("actor.")
    }

    result = run_m1_episode(
        coordinator=coord,
        attacker=attacker,
        defender=FixedClippedMedianPolicy(clip_radius=2.5, trim_ratio=0.2),
        horizon=4,
        seed=12,
        br_updates=2,
    )

    after = {
        key: value
        for key, value in attacker.get_params().items()
        if key.startswith("actor.")
    }
    changed = any(not np.allclose(before[key].numpy(), after[key].numpy()) for key in before)
    assert changed
    assert result.attacker_updates == 2
    assert result.attacker_update_losses


def test_sweep_fixed_defenders_returns_one_record_per_radius():
    td3_config = TD3Config(hidden_dim=16, batch_size=4, warmup_steps=0)

    def make_coordinator():
        return StubCoordinator(num_clients=6, num_attackers=2, subsample_rate=1.0, seed=5)

    records = sweep_fixed_defenders(
        coordinator_factory=make_coordinator,
        radii=[0.5, 4.5],
        trim_ratio=0.2,
        horizon=2,
        td3_config=td3_config,
        seed=17,
    )

    assert len(records) == 2
    assert [record.clip_radius for record in records] == pytest.approx([0.5, 4.5])
    assert all(record.trim_ratio == pytest.approx(0.2) for record in records)
    assert all(record.transitions_collected == 2 for record in records)
    assert all(record.attacker_buffer_size == 2 for record in records)
