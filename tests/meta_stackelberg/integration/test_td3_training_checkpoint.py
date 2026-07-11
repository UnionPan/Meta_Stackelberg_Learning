import numpy as np

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.agents.td3.checkpoint import (
    load_td3_training_checkpoint,
    save_td3_training_checkpoint,
)
from meta_stackelberg.agents.td3.replay import TD3ReplayBuffer


def _agent():
    return TD3Agent(
        obs_dim=2, action_dim=3, role='attacker', seed=1,
        hidden_sizes=(8,), learning_rate=0.001, gamma=0.99, tau=0.005,
        policy_delay=2, target_policy_noise=0.2, noise_clip=0.5,
    )


def test_disk_checkpoint_restores_agent_replay_rng_and_metadata_exactly(tmp_path) -> None:
    agent = _agent()
    replay = TD3ReplayBuffer(
        8, obs_dim=2, action_dim=3, role='attacker', seed=2,
    )
    for index in range(5):
        replay.add(
            np.full(2, index, np.float32), np.full(3, index / 10, np.float32),
            float(index), np.full(2, index + 1, np.float32), False,
            generation=index, role='attacker',
        )
    agent_before = agent.fingerprint()
    replay_before = replay.fingerprint()
    path = tmp_path / 'checkpoint.pt'

    saved = save_td3_training_checkpoint(
        path,
        agent=agent,
        replay=replay,
        metadata={'phase': 'best_response', 'iteration': 3},
    )
    agent.set_learning_rate(0.2)
    replay.sample(2)
    loaded = load_td3_training_checkpoint(
        path, agent=agent, replay=replay,
    )

    assert path.is_file()
    assert saved.agent_fingerprint == loaded.agent_fingerprint == agent_before
    assert saved.replay_fingerprint == loaded.replay_fingerprint == replay_before
    assert agent.fingerprint() == agent_before
    assert replay.fingerprint() == replay_before
    assert dict(loaded.metadata) == {'phase': 'best_response', 'iteration': 3}


def test_checkpoint_rejects_role_mismatch(tmp_path) -> None:
    attacker = _agent()
    replay = TD3ReplayBuffer(8, obs_dim=2, action_dim=3, role='attacker', seed=2)
    path = tmp_path / 'checkpoint.pt'
    save_td3_training_checkpoint(path, agent=attacker, replay=replay, metadata={})
    defender = TD3Agent(
        obs_dim=2, action_dim=3, role='defender', seed=3,
        hidden_sizes=(8,), learning_rate=0.001, gamma=0.99, tau=0.005,
        policy_delay=2, target_policy_noise=0.2, noise_clip=0.5,
    )
    defender_replay = TD3ReplayBuffer(
        8, obs_dim=2, action_dim=3, role='defender', seed=4,
    )
    try:
        load_td3_training_checkpoint(path, agent=defender, replay=defender_replay)
    except ValueError as error:
        assert 'role' in str(error)
    else:
        raise AssertionError('loaded attacker checkpoint into Defender state')
