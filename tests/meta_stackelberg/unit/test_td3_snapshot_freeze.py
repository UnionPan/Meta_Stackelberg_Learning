import numpy as np

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.agents.td3.replay import TD3Batch


def _agent(role='attacker') -> TD3Agent:
    return TD3Agent(
        obs_dim=2, action_dim=3, role=role, seed=17,
        hidden_sizes=(8,), learning_rate=1e-3, gamma=0.99,
        tau=0.005, policy_delay=2, target_policy_noise=0.2, noise_clip=0.5,
    )


def _batch() -> TD3Batch:
    return TD3Batch(
        np.zeros((4, 2), np.float32), np.zeros((4, 3), np.float32),
        np.ones((4, 1), np.float32), np.ones((4, 2), np.float32),
        np.zeros((4, 1), np.float32), np.zeros((4, 1), np.int64),
    )


def test_snapshot_restore_recovers_full_fingerprint_and_action() -> None:
    agent = _agent()
    snapshot = agent.snapshot()
    fingerprint = agent.fingerprint()
    action = agent.act(np.array([0.2, -0.1], dtype=np.float32), deterministic=True)
    agent.update(_batch())
    assert agent.fingerprint() != fingerprint
    agent.restore(snapshot)
    assert agent.fingerprint() == fingerprint
    np.testing.assert_array_equal(
        agent.act(np.array([0.2, -0.1], dtype=np.float32), deterministic=True), action,
    )


def test_freeze_guard_detects_online_target_optimizer_and_counter_mutation() -> None:
    agent = _agent()
    guard = agent.freeze_guard()
    guard.verify()
    agent.update(_batch())
    try:
        guard.verify()
    except RuntimeError as error:
        assert 'attacker' in str(error)
    else:
        raise AssertionError('freeze guard missed mutation')
