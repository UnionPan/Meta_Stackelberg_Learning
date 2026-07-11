from collections import Counter

import numpy as np

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.agents.td3.replay import TD3ReplayBuffer
from meta_stackelberg.stackelberg.policy_algorithm2 import PolicyMetaSGAlgorithm2


def _agent(role, seed):
    return TD3Agent(
        obs_dim=3, action_dim=3, role=role, seed=seed,
        hidden_sizes=(8,), learning_rate=0.001, gamma=0.99, tau=0.005,
        policy_delay=2, target_policy_noise=0.2, noise_clip=0.5,
    )


def test_concrete_algorithm2_runs_exact_T_K_l_and_freezes_responses() -> None:
    defender = _agent('defender', 1)
    attackers = {'a': _agent('attacker', 2), 'b': _agent('attacker', 3)}
    defender_before = defender.fingerprint()
    attacker_before = {key: value.fingerprint() for key, value in attackers.items()}
    calls = []

    def replay_factory(task, iteration):
        return TD3ReplayBuffer(
            32, obs_dim=3, action_dim=3, role='defender',
            seed=10 + iteration + (task == 'b'),
        )

    def collect(task, step, adapted, frozen_response, replay, iteration):
        del frozen_response
        calls.append((task, step, iteration))
        for index in range(4):
            obs = np.full(3, index / 10, dtype=np.float32)
            replay.add(
                obs, adapted.act(obs, deterministic=True), 1.0,
                obs + 0.01, False, generation=iteration, role='defender',
            )

    result = PolicyMetaSGAlgorithm2(
        T=2, K=2, l=2, batch_size=4,
        kappa=0.001, meta_update_step=1.0,
    ).run(
        defender=defender,
        response_policies=attackers,
        sample_tasks=lambda iteration, count: ('a', 'b'),
        replay_factory=replay_factory,
        collect_task=collect,
    )

    assert len(calls) == 2 * 2 * 2
    assert Counter(step for _, step, _ in calls) == {0: 4, 1: 4}
    assert defender.fingerprint() != defender_before
    assert {key: value.fingerprint() for key, value in attackers.items()} == attacker_before
    assert len(result.iterations) == 2
    assert all(len(item.tasks) == 2 for item in result.iterations)
