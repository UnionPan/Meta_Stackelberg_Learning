import numpy as np

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.agents.td3.replay import TD3ReplayBuffer
from meta_stackelberg.stackelberg.policy_online import PolicyOnlineAdaptationRunner


def _agent(role, seed):
    return TD3Agent(
        obs_dim=3, action_dim=3, role=role, seed=seed,
        hidden_sizes=(8,), learning_rate=0.001, gamma=0.99, tau=0.005,
        policy_delay=2, target_policy_noise=0.2, noise_clip=0.5,
    )


def test_online_T_times_l_is_exact_sequential_adaptation_budget() -> None:
    meta = _agent('defender', 1)
    attacker = _agent('attacker', 2)
    replay = TD3ReplayBuffer(
        64, obs_dim=3, action_dim=3, role='defender', seed=3,
    )
    meta_before = meta.fingerprint()
    attacker_before = attacker.fingerprint()
    calls = []

    def collect(adapted, frozen, target, iteration, local_step, global_step):
        del frozen
        calls.append((iteration, local_step, global_step))
        for index in range(4):
            obs = np.full(3, index / 10, dtype=np.float32)
            target.add(
                obs, adapted.act(obs, deterministic=True), 1.0,
                obs + 0.01, False, generation=iteration, role='defender',
            )

    result = PolicyOnlineAdaptationRunner(
        online_T=2, online_l=3, online_steps=6,
        batch_size=4, adaptation_step=0.01,
    ).run(
        meta_defender=meta, attacker=attacker, replay=replay,
        collect_fresh=collect,
    )

    assert calls == [
        (0, 0, 0), (0, 1, 1), (0, 2, 2),
        (1, 0, 3), (1, 1, 4), (1, 2, 5),
    ]
    assert len(result.iterations) == 2
    assert all(len(item.updates) == 3 for item in result.iterations)
    assert result.adapted_defender.update_count == 6
    assert result.adapted_defender.fingerprint() != meta_before
    assert meta.fingerprint() == meta_before
    assert attacker.fingerprint() == attacker_before


def test_online_runner_rejects_inconsistent_total_steps() -> None:
    try:
        PolicyOnlineAdaptationRunner(
            online_T=10, online_l=10, online_steps=10,
            batch_size=4, adaptation_step=0.01,
        )
    except ValueError as error:
        assert 'online_steps' in str(error)
    else:
        raise AssertionError('accepted online_steps != online_T * online_l')
