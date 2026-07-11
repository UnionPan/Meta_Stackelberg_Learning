import numpy as np

from meta_stackelberg.agents.td3.replay import TD3ReplayBuffer, flatten_observation


def test_flatten_observation_uses_explicit_key_order() -> None:
    observation = {
        'b': np.array([3.0], dtype=np.float32),
        'a': np.array([1.0, 2.0], dtype=np.float32),
    }
    np.testing.assert_array_equal(flatten_observation(observation, ('a', 'b')), [1.0, 2.0, 3.0])


def test_replay_copies_transitions_enforces_role_and_samples_replayably() -> None:
    first = TD3ReplayBuffer(8, obs_dim=2, action_dim=1, role='attacker', seed=7)
    second = TD3ReplayBuffer(8, obs_dim=2, action_dim=1, role='attacker', seed=7)
    for index in range(5):
        obs = np.array([index, index + 1], dtype=np.float32)
        action = np.array([index / 10], dtype=np.float32)
        for buffer in (first, second):
            buffer.add(obs, action, float(index), obs + 1, index == 4, generation=3, role='attacker')
        obs[:] = 99
        action[:] = 99
    a = first.sample(3)
    b = second.sample(3)
    np.testing.assert_array_equal(a.observations, b.observations)
    np.testing.assert_array_equal(a.actions, b.actions)
    assert np.all(a.generations == 3)
    try:
        first.add(np.zeros(2), np.zeros(1), 0.0, np.zeros(2), False, generation=0, role='defender')
    except ValueError as error:
        assert 'role' in str(error)
    else:
        raise AssertionError('accepted cross-role transition')


def test_replay_capacity_and_small_batch_fail_fast() -> None:
    buffer = TD3ReplayBuffer(2, obs_dim=1, action_dim=1, role='defender', seed=1)
    for value in range(3):
        buffer.add([value], [0.0], 0.0, [value + 1], False, generation=0, role='defender')
    assert len(buffer) == 2
    try:
        buffer.sample(3)
    except ValueError as error:
        assert 'batch' in str(error)
    else:
        raise AssertionError('sampled beyond buffer size')


def test_replay_snapshot_restores_contents_cursor_generation_and_rng_exactly() -> None:
    buffer = TD3ReplayBuffer(4, obs_dim=2, action_dim=1, role='attacker', seed=11)
    for index in range(5):
        buffer.add(
            [index, index + 1], [index / 10], float(index),
            [index + 1, index + 2], index == 4,
            generation=index // 2, role='attacker',
        )
    snapshot = buffer.snapshot()
    fingerprint = buffer.fingerprint()
    expected = buffer.sample(3)
    buffer.add([9, 9], [0.9], 9.0, [10, 10], False,
               generation=9, role='attacker')
    assert buffer.fingerprint() != fingerprint

    buffer.restore(snapshot)
    assert buffer.fingerprint() == fingerprint
    actual = buffer.sample(3)

    assert buffer.fingerprint() != fingerprint  # sampling advances only RNG state
    np.testing.assert_array_equal(actual.observations, expected.observations)
    np.testing.assert_array_equal(actual.actions, expected.actions)
    np.testing.assert_array_equal(actual.generations, expected.generations)
