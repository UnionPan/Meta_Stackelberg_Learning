from collections import Counter

import numpy as np

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.agents.td3.replay import TD3ReplayBuffer
from meta_stackelberg.stackelberg.policy_algorithm1 import PolicyMetaSGAlgorithm1


def _agent(role, seed):
    return TD3Agent(
        obs_dim=3, action_dim=3, role=role, seed=seed,
        hidden_sizes=(8,), learning_rate=0.001, gamma=0.99, tau=0.005,
        policy_delay=2, target_policy_noise=0.2, noise_clip=0.5,
    )


def test_concrete_algorithm1_runs_adapt_br_leader_in_paper_order() -> None:
    defender = _agent('defender', 1)
    attacker = _agent('attacker', 2)
    defender_before = defender.fingerprint()
    attacker_before = attacker.fingerprint()
    calls = []
    response_defenders = []

    def replay_factory(task, role, phase, iteration):
        del task, phase
        return TD3ReplayBuffer(
            32, obs_dim=3, action_dim=3, role=role,
            seed=10 + iteration + len(calls),
        )

    def add(replay, role, policy, generation):
        for index in range(4):
            obs = np.full(3, index / 10, dtype=np.float32)
            replay.add(
                obs, policy.act(obs, deterministic=True), 1.0,
                obs + 0.01, False, generation=generation, role=role,
            )

    def collect_adaptation(task, adapted, frozen_attacker, replay, iteration):
        del frozen_attacker
        calls.append(('adapt', task, iteration))
        add(replay, 'defender', adapted, iteration)

    def collect_response(task, step, frozen_defender, current_attacker, replay, iteration):
        response_defenders.append(frozen_defender.fingerprint())
        calls.append(('response', task, step, iteration))
        add(replay, 'attacker', current_attacker, iteration)

    def collect_leader(task, adapted, frozen_br, replay, iteration):
        del frozen_br
        calls.append(('leader', task, iteration))
        add(replay, 'defender', adapted, iteration)

    result = PolicyMetaSGAlgorithm1(
        N_D=1, K=1, N_A=2,
        batch_size=4, eta=0.01, kappa_A=0.001, kappa_D=0.001,
    ).run(
        defender=defender,
        attackers={'rl': attacker},
        sample_tasks=lambda iteration, count: ('rl',),
        replay_factory=replay_factory,
        collect_adaptation=collect_adaptation,
        collect_response=collect_response,
        collect_leader=collect_leader,
        independent_attacker_objective=lambda task, policy: float(
            policy.act(np.zeros(3, dtype=np.float32), deterministic=True)[0]
        ),
    )

    assert [call[0] for call in calls] == ['adapt', 'response', 'response', 'leader']
    assert Counter(call[0] for call in calls) == {'adapt': 1, 'response': 2, 'leader': 1}
    assert defender.fingerprint() != defender_before
    assert attacker.fingerprint() != attacker_before
    task_trace = result.iterations[0].tasks[0]
    assert task_trace.response.update_stats.__len__() == 2
    assert response_defenders == [
        task_trace.adaptation.adapted_defender_fingerprint,
        task_trace.adaptation.adapted_defender_fingerprint,
    ]
    assert response_defenders[0] != defender_before
    assert task_trace.adaptation.attacker_fingerprint == task_trace.response.initial_attacker_fingerprint
    assert result.iterations[0].leader.task_updates[0].best_response_fingerprint == attacker.fingerprint()


def test_algorithm1_rejects_algorithm2_l_alias() -> None:
    try:
        PolicyMetaSGAlgorithm1(
            N_D=1, K=1, N_A=1, adaptation_l=10,
            batch_size=4, eta=0.01, kappa_A=0.001, kappa_D=0.001,
        )
    except TypeError:
        pass
    else:
        raise AssertionError('Algorithm 1 accepted Algorithm 2 l alias')


def test_algorithm1_resume_uses_global_nd_indices_and_boundary_callback() -> None:
    defender = _agent('defender', 41)
    attacker = _agent('attacker', 42)
    callbacks = []

    def replay_factory(task, role, phase, iteration):
        del task, phase
        return TD3ReplayBuffer(
            16, obs_dim=3, action_dim=3, role=role, seed=100 + iteration,
        )

    def add(replay, role, policy, generation):
        for index in range(4):
            obs = np.full(3, index / 10, dtype=np.float32)
            replay.add(obs, policy.act(obs, deterministic=True), 1.0,
                       obs, False, generation=generation, role=role)

    result = PolicyMetaSGAlgorithm1(
        N_D=3, K=1, N_A=1, batch_size=4,
        eta=0.01, kappa_A=0.001, kappa_D=0.001,
    ).run(
        defender=defender, attackers={'rl': attacker},
        sample_tasks=lambda iteration, count: ('rl',),
        replay_factory=replay_factory,
        collect_adaptation=lambda task, policy, frozen, replay, iteration: add(
            replay, 'defender', policy, iteration,
        ),
        collect_response=lambda task, step, frozen, policy, replay, iteration: add(
            replay, 'attacker', policy, iteration,
        ),
        collect_leader=lambda task, policy, frozen, replay, iteration: add(
            replay, 'defender', policy, iteration,
        ),
        independent_attacker_objective=lambda task, policy: 0.0,
        start_iteration=2,
        iteration_callback=lambda trace, policy, responses: callbacks.append(
            (trace.leader_iteration, policy.fingerprint()),
        ),
    )
    assert [item.leader_iteration for item in result.iterations] == [2]
    assert callbacks[0][0] == 2
