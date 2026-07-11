import torch

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.stackelberg.algorithm2 import reptile_update_td3


def _agent(seed: int) -> TD3Agent:
    return TD3Agent(
        obs_dim=3,
        action_dim=3,
        role='defender',
        seed=seed,
        hidden_sizes=(8,),
        learning_rate=0.001,
        gamma=0.99,
        tau=0.005,
        policy_delay=2,
        target_policy_noise=0.2,
        noise_clip=0.5,
    )


def _fill(agent: TD3Agent, value: float) -> None:
    for module in (
        agent.actor, agent.actor_target, agent.critic1, agent.critic2,
        agent.critic1_target, agent.critic2_target,
    ):
        for parameter in module.parameters():
            parameter.data.fill_(value)


def test_reptile_update_averages_isolated_task_parameters() -> None:
    meta = _agent(1)
    _fill(meta, 0.0)
    task_one = _agent(2)
    task_two = _agent(3)
    _fill(task_one, 1.0)
    _fill(task_two, 3.0)
    meta_before = meta.snapshot()
    one_before = task_one.fingerprint()
    two_before = task_two.fingerprint()

    reptile_update_td3(
        meta,
        (task_one.snapshot(), task_two.snapshot()),
        meta_step=0.5,
    )

    for parameter in meta.actor.parameters():
        assert torch.allclose(parameter, torch.ones_like(parameter))
    assert task_one.fingerprint() == one_before
    assert task_two.fingerprint() == two_before
    assert meta.actor_optimizer.state_dict() == meta_before.actor_optimizer
    assert meta.update_count == meta_before.update_count


def test_reptile_update_rejects_role_mismatch_and_empty_tasks() -> None:
    meta = _agent(1)
    attacker = TD3Agent(
        obs_dim=3, action_dim=3, role='attacker', seed=2,
        hidden_sizes=(8,), learning_rate=0.001, gamma=0.99, tau=0.005,
        policy_delay=2, target_policy_noise=0.2, noise_clip=0.5,
    )
    try:
        reptile_update_td3(meta, (), meta_step=1.0)
        raise AssertionError('empty task snapshots must fail')
    except ValueError:
        pass
    try:
        reptile_update_td3(meta, (attacker.snapshot(),), meta_step=1.0)
        raise AssertionError('role mismatch must fail')
    except ValueError:
        pass
