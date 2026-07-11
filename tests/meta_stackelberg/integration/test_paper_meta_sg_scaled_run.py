import numpy as np

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.agents.td3.replay import TD3ReplayBuffer, flatten_observation
from meta_stackelberg.agents.td3.config import PaperMetaSGConfig
from meta_stackelberg.experiments.paper_meta_sg import (
    ATTACKER_OBSERVATION_KEYS,
    DEFENDER_OBSERVATION_KEYS,
    PaperTD3TrajectoryCollector,
    evaluate_scaled_conformance,
)
from meta_stackelberg.experiments.scientific_gate import (
    QueryEvidencePlan,
    evaluate_frozen_policy,
)
from meta_stackelberg.stackelberg.algorithm1 import MetaSGAlgorithm1
from meta_stackelberg.stackelberg.algorithm2 import MetaSGAlgorithm2
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


def test_immutable_paper_scaled_configuration_matches_execution_traces() -> None:
    config = PaperMetaSGConfig().scaled(
        T=2, K=2, H=8, l=2, N_A=2, N_D=2,
        workers=6, untargeted_attackers=3, sample_size=4,
        td3_batch_size=16, learning_starts=16, hidden_sizes=(64, 64),
        replay_capacity=4096,
    )
    algorithm1 = MetaSGAlgorithm1(N_D=2, K=2, N_A=2).run(
        sample_tasks=lambda iteration, count: tuple(range(count)),
        adapt_defender=lambda task, iteration: (task, iteration),
        update_attacker=lambda task, defender, step: step,
        estimate_defender_gradient=lambda task, defender, response: response,
        apply_leader_update=lambda iteration, gradients: None,
    )
    algorithm2 = MetaSGAlgorithm2(T=2, K=2, l=2, meta_step=1.0).run(
        sample_tasks=lambda iteration, count: tuple(range(count)),
        clone_for_task=lambda task: [task],
        adapt_task=lambda task, clone, step: clone.append(step),
        apply_meta_update=lambda iteration, policies, meta_step: None,
    )
    env = _make_env(seed=29)
    env.horizon = config.H
    defender_dim = len(flatten_observation(
        env.defender_observation(), DEFENDER_OBSERVATION_KEYS,
    ))
    probe = _make_env(seed=29)
    probe.horizon = config.H
    attacker_dim = len(flatten_observation(
        probe.begin_round(np.zeros(3, dtype=np.float32)).attacker_observation,
        ATTACKER_OBSERVATION_KEYS,
    ))
    defender = _agent(defender_dim, 'defender', 11)
    attacker = _agent(attacker_dim, 'attacker', 12)
    defender_replay = TD3ReplayBuffer(
        32, obs_dim=defender_dim, action_dim=3, role='defender', seed=13,
    )
    attacker_replay = TD3ReplayBuffer(
        32, obs_dim=attacker_dim, action_dim=3, role='attacker', seed=14,
    )
    trajectory = PaperTD3TrajectoryCollector().collect(
        env=env, defender=defender, attacker=attacker,
        defender_replay=defender_replay, attacker_replay=attacker_replay,
        generation=0, deterministic=True,
    )

    result = evaluate_scaled_conformance(
        config=config, algorithm1=algorithm1, algorithm2=algorithm2,
        trajectory=trajectory,
    )

    assert result.passed
    assert result.scale_provenance == 'scaled-conformance-only-v1'
    assert all(observed == expected for _, passed, observed, expected in result.checks if passed)


def test_real_environment_query_trajectories_keep_policy_frozen() -> None:
    probe = _make_env(seed=41)
    defender_dim = len(flatten_observation(
        probe.defender_observation(), DEFENDER_OBSERVATION_KEYS,
    ))
    attacker_probe = _make_env(seed=41).begin_round(np.zeros(3, dtype=np.float32))
    attacker_dim = len(flatten_observation(
        attacker_probe.attacker_observation, ATTACKER_OBSERVATION_KEYS,
    ))
    defender = _agent(defender_dim, 'defender', 21)
    attacker = _agent(attacker_dim, 'attacker', 22)
    before = attacker.fingerprint()

    def query(policy, seed):
        env = _make_env(seed=seed)
        defender_replay = TD3ReplayBuffer(
            8, obs_dim=defender_dim, action_dim=3, role='defender', seed=seed,
        )
        attacker_replay = TD3ReplayBuffer(
            8, obs_dim=attacker_dim, action_dim=3, role='attacker', seed=seed + 100,
        )
        trajectory = PaperTD3TrajectoryCollector().collect(
            env=env, defender=defender, attacker=policy,
            defender_replay=defender_replay, attacker_replay=attacker_replay,
            generation=0, deterministic=True,
        )
        actions = np.concatenate([
            step.attacker_raw_action for step in trajectory.steps
        ])
        return trajectory.attacker_return, actions

    evidence = evaluate_frozen_policy(
        label='attacker_query', policy=attacker,
        plan=QueryEvidencePlan((1, 2), (101, 102)), query=query,
    )

    assert evidence.query_seeds == (101, 102)
    assert all(len(actions) == 6 for actions in evidence.action_trajectory)
    assert attacker.fingerprint() == before == evidence.policy_fingerprint
