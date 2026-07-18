import numpy as np
import torch

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.agents.td3.replay import TD3Batch


def _batch(done: bool = False) -> TD3Batch:
    rng = np.random.default_rng(3)
    return TD3Batch(
        observations=rng.normal(size=(8, 4)).astype(np.float32),
        actions=np.clip(rng.normal(size=(8, 3)), -1, 1).astype(np.float32),
        rewards=rng.normal(size=(8, 1)).astype(np.float32),
        next_observations=rng.normal(size=(8, 4)).astype(np.float32),
        dones=np.full((8, 1), done, dtype=np.float32),
        generations=np.zeros((8, 1), dtype=np.int64),
    )


def _agent(seed: int = 5) -> TD3Agent:
    return TD3Agent(
        obs_dim=4, action_dim=3, role='defender', seed=seed,
        hidden_sizes=(16, 16), learning_rate=1e-3, gamma=0.99,
        tau=0.005, policy_delay=2, target_policy_noise=0.2,
        noise_clip=0.5,
    )


def test_actor_is_3d_bounded_and_update_uses_delayed_policy() -> None:
    agent = _agent()
    action = agent.act(np.ones(4, dtype=np.float32), deterministic=True)
    assert action.shape == (3,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
    first = agent.update(_batch())
    second = agent.update(_batch())
    assert first.actor_updated is False
    assert second.actor_updated is True
    assert np.isfinite(first.critic_loss)
    assert np.isfinite(second.actor_loss)


def test_terminal_target_does_not_bootstrap() -> None:
    agent = _agent()
    rewards = torch.tensor([[1.0], [2.0]])
    target_q = torch.tensor([[100.0], [100.0]])
    dones = torch.ones((2, 1))
    torch.testing.assert_close(agent.compute_td_target(rewards, dones, target_q), rewards)


def test_seeded_agents_update_exactly() -> None:
    first = _agent(11)
    second = _agent(11)
    stats_a = first.update(_batch())
    stats_b = second.update(_batch())
    assert stats_a == stats_b
    assert first.fingerprint() == second.fingerprint()


def test_online_logit_penalty_is_optional_and_rejects_invalid_values() -> None:
    baseline = _agent(23)
    regularized = _agent(23)
    baseline.update(_batch())
    regularized.update(_batch(), actor_logit_l2=0.1)
    baseline.update(_batch())
    regularized.update(_batch(), actor_logit_l2=0.1)
    assert baseline.fingerprint() != regularized.fingerprint()

    with np.testing.assert_raises(ValueError):
        regularized.update(_batch(), actor_logit_l2=-1.0)
    with np.testing.assert_raises(ValueError):
        regularized.update(
            _batch(), actor_logit_l2=0.1, actor_logit_l2_mask=(1.0,),
        )


def test_uniform_learning_start_actions_cover_box_and_resume_exactly() -> None:
    first = _agent(17)
    second = _agent(17)
    actions = np.stack([first.sample_uniform_action() for _ in range(64)])
    assert actions.shape == (64, 3)
    assert np.all(actions >= -1.0) and np.all(actions <= 1.0)
    assert np.ptp(actions, axis=0).min() > 1.5
    for _ in range(64):
        second.sample_uniform_action()
    assert first.fingerprint() == second.fingerprint()

    snapshot = first.snapshot()
    expected = first.sample_uniform_action()
    first.restore(snapshot)
    np.testing.assert_array_equal(first.sample_uniform_action(), expected)
