import numpy as np

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.agents.td3.replay import TD3ReplayBuffer, flatten_observation
from meta_stackelberg.experiments.paper_meta_sg import (
    ATTACKER_OBSERVATION_KEYS,
    DEFENDER_OBSERVATION_KEYS,
    PaperTD3TrajectoryCollector,
)
from tests.meta_stackelberg.integration.test_paper_bsmg_environment import _make_env


def _agent(obs_dim, role, seed):
    return TD3Agent(
        obs_dim=obs_dim, action_dim=3, role=role, seed=seed,
        hidden_sizes=(8,), learning_rate=0.001, gamma=0.99, tau=0.005,
        policy_delay=2, target_policy_noise=0.2, noise_clip=0.5,
    )


def test_scaled_collector_executes_both_3d_policies_each_fl_round() -> None:
    env = _make_env(seed=19)
    defender_dim = len(flatten_observation(
        env.defender_observation(), DEFENDER_OBSERVATION_KEYS,
    ))
    pending_probe = _make_env(seed=19).begin_round(np.zeros(3, dtype=np.float32))
    attacker_dim = len(flatten_observation(
        pending_probe.attacker_observation, ATTACKER_OBSERVATION_KEYS,
    ))
    defender = _agent(defender_dim, 'defender', 1)
    attacker = _agent(attacker_dim, 'attacker', 2)
    defender_replay = TD3ReplayBuffer(
        8, obs_dim=defender_dim, action_dim=3, role='defender', seed=3,
    )
    attacker_replay = TD3ReplayBuffer(
        8, obs_dim=attacker_dim, action_dim=3, role='attacker', seed=4,
    )

    trajectory = PaperTD3TrajectoryCollector().collect(
        env=env, defender=defender, attacker=attacker,
        defender_replay=defender_replay, attacker_replay=attacker_replay,
        generation=7, deterministic=True,
    )

    assert len(trajectory.steps) == env.horizon == 2
    assert len(defender_replay) == len(attacker_replay) == 2
    assert all(step.defender_raw_action.shape == (3,) for step in trajectory.steps)
    assert all(step.attacker_raw_action.shape == (3,) for step in trajectory.steps)
    assert defender_replay.sample(2).generations.tolist() == [[7], [7]]
    assert attacker_replay.sample(2).generations.tolist() == [[7], [7]]
