import numpy as np
import pytest

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.agents.td3.config import PaperMetaSGConfig
from meta_stackelberg.agents.td3.replay import flatten_observation
from meta_stackelberg.experiments.paper_meta_sg import (
    ATTACKER_OBSERVATION_KEYS,
    DEFENDER_OBSERVATION_KEYS,
    PaperOnlineAdaptationTrainingRunner,
)
from meta_stackelberg.experiments.deterministic_paper_env import make_deterministic_paper_env


def _make_env(seed):
    return make_deterministic_paper_env(seed=seed)


def _agent(obs_dim, role, seed):
    return TD3Agent(
        obs_dim=obs_dim, action_dim=3, role=role, seed=seed,
        hidden_sizes=(8,), learning_rate=0.001, gamma=0.99, tau=0.005,
        policy_delay=2, target_policy_noise=0.2, noise_clip=0.5,
    )


def test_real_online_runner_executes_T_l_H_and_trajectory_batch_budget() -> None:
    config = PaperMetaSGConfig().scaled_online(
        online_T=2, online_H=2, online_l=2, online_steps=4,
        td3_batch_size=4, learning_starts=4, replay_capacity=128,
    )
    defender_dim = len(flatten_observation(
        _make_env(seed=1).defender_observation(), DEFENDER_OBSERVATION_KEYS,
    ))
    attacker_dim = len(flatten_observation(
        _make_env(seed=1).begin_round(np.zeros(3, np.float32)).attacker_observation,
        ATTACKER_OBSERVATION_KEYS,
    ))
    defender = _agent(defender_dim, 'defender', 1)
    attacker = _agent(attacker_dim, 'attacker', 2)
    defender_before = defender.fingerprint()
    attacker_before = attacker.fingerprint()

    def env_factory(task, seed, horizon):
        del task
        env = _make_env(seed=seed)
        env.horizon = horizon
        return env

    result = PaperOnlineAdaptationTrainingRunner(
        config=config,
        env_factory=env_factory,
        defender_obs_dim=defender_dim,
        attacker_obs_dim=attacker_dim,
        support_seeds=tuple(range(4000, 4003)),
    ).run(task='rl', meta_defender=defender, attacker=attacker)

    assert result.trajectory_count == 3
    assert result.fl_round_count == 6
    assert result.trajectories_per_update == 2
    assert result.adaptation.total_updates == 4
    assert result.adaptation.adapted_defender.fingerprint() != defender_before
    assert defender.fingerprint() == defender_before
    assert attacker.fingerprint() == attacker_before


def test_online_runner_uses_sb3_replacement_when_batch_exceeds_replay() -> None:
    config = PaperMetaSGConfig().scaled_online(
        online_T=2, online_H=2, online_l=2, online_steps=4,
        td3_batch_size=8, learning_starts=2, replay_capacity=128,
    )
    defender_dim = len(flatten_observation(
        _make_env(seed=1).defender_observation(), DEFENDER_OBSERVATION_KEYS,
    ))
    attacker_dim = len(flatten_observation(
        _make_env(seed=1).begin_round(
            np.zeros(3, np.float32),
        ).attacker_observation,
        ATTACKER_OBSERVATION_KEYS,
    ))

    def env_factory(task, seed, horizon):
        del task
        env = _make_env(seed=seed)
        env.horizon = horizon
        return env

    result = PaperOnlineAdaptationTrainingRunner(
        config=config,
        env_factory=env_factory,
        defender_obs_dim=defender_dim,
        attacker_obs_dim=attacker_dim,
        support_seeds=tuple(range(6000, 6002)),
    ).run(
        task='rl',
        meta_defender=_agent(defender_dim, 'defender', 31),
        attacker=_agent(attacker_dim, 'attacker', 32),
    )

    assert result.trajectories_per_update == 1
    assert result.trajectory_count == 2
    assert result.fl_round_count == 4


def test_online_runner_resumes_exactly_from_outer_iteration_checkpoint(
    tmp_path,
) -> None:
    config = PaperMetaSGConfig().scaled_online(
        online_T=2, online_H=2, online_l=2, online_steps=4,
        td3_batch_size=4, learning_starts=4, replay_capacity=128,
    )
    defender_dim = len(flatten_observation(
        _make_env(seed=1).defender_observation(), DEFENDER_OBSERVATION_KEYS,
    ))
    attacker_dim = len(flatten_observation(
        _make_env(seed=1).begin_round(np.zeros(3, np.float32)).attacker_observation,
        ATTACKER_OBSERVATION_KEYS,
    ))
    defender = _agent(defender_dim, 'defender', 11)
    attacker = _agent(attacker_dim, 'attacker', 12)
    seeds = tuple(range(5000, 5003))

    def env_factory(task, seed, horizon):
        del task
        env = _make_env(seed=seed)
        env.horizon = horizon
        return env

    uninterrupted = PaperOnlineAdaptationTrainingRunner(
        config=config,
        env_factory=env_factory,
        defender_obs_dim=defender_dim,
        attacker_obs_dim=attacker_dim,
        support_seeds=seeds,
    ).run(task='rl', meta_defender=defender, attacker=attacker)

    def interrupted_factory(task, seed, horizon):
        if seed >= 5002:
            raise RuntimeError('simulated interruption')
        return env_factory(task, seed, horizon)

    checkpoint = tmp_path / 'online.pt'
    with pytest.raises(RuntimeError, match='simulated interruption'):
        PaperOnlineAdaptationTrainingRunner(
            config=config,
            env_factory=interrupted_factory,
            defender_obs_dim=defender_dim,
            attacker_obs_dim=attacker_dim,
            support_seeds=seeds,
        ).run(
            task='rl', meta_defender=defender, attacker=attacker,
            checkpoint_path=checkpoint,
        )

    resumed = PaperOnlineAdaptationTrainingRunner(
        config=config,
        env_factory=env_factory,
        defender_obs_dim=defender_dim,
        attacker_obs_dim=attacker_dim,
        support_seeds=seeds,
    ).run(
        task='rl', meta_defender=defender, attacker=attacker,
        checkpoint_path=checkpoint, resume=True,
    )

    assert resumed.support_seeds == uninterrupted.support_seeds
    assert resumed.adaptation.total_updates == 4
    assert resumed.adaptation.adapted_defender_fingerprint == (
        uninterrupted.adaptation.adapted_defender_fingerprint
    )
