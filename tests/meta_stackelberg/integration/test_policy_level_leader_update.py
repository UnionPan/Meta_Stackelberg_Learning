import numpy as np

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.agents.td3.replay import TD3ReplayBuffer
from meta_stackelberg.stackelberg.policy_leader import (
    PolicyLeaderTask,
    PolicyLeaderTrainer,
)


def _agent(role, seed):
    return TD3Agent(
        obs_dim=3, action_dim=3, role=role, seed=seed,
        hidden_sizes=(8,), learning_rate=0.001, gamma=0.99, tau=0.005,
        policy_delay=2, target_policy_noise=0.2, noise_clip=0.5,
    )


def _replay(seed):
    return TD3ReplayBuffer(
        16, obs_dim=3, action_dim=3, role='defender', seed=seed,
    )


def test_leader_update_freezes_each_br_and_updates_meta_defender_once() -> None:
    defender = _agent('defender', 1)
    attackers = (_agent('attacker', 2), _agent('attacker', 3))
    attacker_before = tuple(agent.fingerprint() for agent in attackers)
    defender_before = defender.fingerprint()
    tasks = tuple(PolicyLeaderTask(
        f'task-{index}', defender.clone(), attacker, _replay(index + 4),
    )
                  for index, attacker in enumerate(attackers))

    def collect(task_id, task_defender, frozen_attacker, replay):
        del frozen_attacker
        offset = 0.1 if task_id == 'task-0' else -0.1
        for index in range(4):
            obs = np.full(3, index / 10 + offset, dtype=np.float32)
            action = task_defender.act(obs, deterministic=True)
            replay.add(
                obs, action, 1.0 + offset, obs + 0.01, False,
                generation=0, role='defender',
            )

    result = PolicyLeaderTrainer(
        defender_updates=1, batch_size=4, kappa_D=0.5,
    ).train(defender=defender, tasks=tasks, collect_fresh=collect)

    assert defender.fingerprint() != defender_before
    assert tuple(agent.fingerprint() for agent in attackers) == attacker_before
    assert len(result.task_updates) == 2
    assert all(update.defender_update_count == 1 for update in result.task_updates)
    assert result.meta_defender_fingerprint == defender.fingerprint()


def test_leader_update_detects_mutated_best_response() -> None:
    defender = _agent('defender', 1)
    attacker = _agent('attacker', 2)
    task = PolicyLeaderTask('task', defender.clone(), attacker, _replay(3))

    def mutate(task_id, task_defender, frozen_attacker, replay):
        del task_id, task_defender, replay
        next(frozen_attacker.actor.parameters()).data.add_(1.0)

    try:
        PolicyLeaderTrainer(
            defender_updates=1, batch_size=1, kappa_D=0.5,
        ).train(defender=defender, tasks=(task,), collect_fresh=mutate)
    except RuntimeError as error:
        assert 'attacker freeze fingerprint changed' in str(error)
    else:
        raise AssertionError('mutated best response was not detected')


def test_kappa_D_is_applied_once_as_task_optimizer_step_size() -> None:
    defender = _agent('defender', 31)
    attacker = _agent('attacker', 32)
    replay = _replay(33)
    task = PolicyLeaderTask('task', defender.clone(), attacker, replay)
    before = next(defender.critic1.parameters()).detach().clone()

    def collect(task_id, task_defender, frozen_attacker, target_replay):
        del task_id, task_defender, frozen_attacker
        for index in range(4):
            obs = np.full(3, 0.1 * index, dtype=np.float32)
            target_replay.add(
                obs, np.full(3, 0.2, dtype=np.float32), 1.0,
                obs + 0.01, False, generation=0, role='defender',
            )

    PolicyLeaderTrainer(
        defender_updates=1, batch_size=4, kappa_D=0.001,
    ).train(defender=defender, tasks=(task,), collect_fresh=collect)

    delta = (next(defender.critic1.parameters()).detach() - before).abs().max().item()
    assert 0.0005 < delta < 0.0011
