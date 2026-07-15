import pytest

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.agents.td3.config import PaperMetaSGConfig
from meta_stackelberg.agents.td3.replay import TD3ReplayBuffer, flatten_observation
from meta_stackelberg.experiments.paper_meta_sg import (
    ATTACKER_OBSERVATION_KEYS,
    DEFENDER_OBSERVATION_KEYS,
    ScaledPaperMetaSGTrainingRunner,
    PaperTD3TrajectoryCollector,
)
from meta_stackelberg.experiments.scaled_training_checkpoint import (
    load_scaled_training_checkpoint,
)
from meta_stackelberg.experiments.scientific_gate import (
    QueryEvidencePlan,
    evaluate_frozen_pair,
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


def test_scaled_runner_drives_real_algorithm1_and_algorithm2_trajectories() -> None:
    config = PaperMetaSGConfig().scaled(
        T=1, K=1, H=2, l=1, N_A=1, N_D=1,
        workers=4, untargeted_attackers=2, sample_size=4,
        td3_batch_size=4, learning_starts=4, hidden_sizes=(8,),
        replay_capacity=128,
    )
    probe = _make_env(seed=1)
    defender_dim = len(flatten_observation(
        probe.defender_observation(), DEFENDER_OBSERVATION_KEYS,
    ))
    attacker_dim = len(flatten_observation(
        _make_env(seed=1).begin_round(
            __import__('numpy').zeros(3, dtype='float32'),
        ).attacker_observation,
        ATTACKER_OBSERVATION_KEYS,
    ))
    defender = _agent(defender_dim, 'defender', 2)
    attacker = _agent(attacker_dim, 'attacker', 3)
    initial_defender = defender.fingerprint()
    initial_attacker = attacker.fingerprint()

    def env_factory(task, seed, horizon):
        del task
        env = _make_env(seed=seed)
        env.horizon = horizon
        return env

    result = ScaledPaperMetaSGTrainingRunner(
        config=config,
        env_factory=env_factory,
        defender_obs_dim=defender_dim,
        attacker_obs_dim=attacker_dim,
        support_seed=1000,
        query_seeds=(101, 102),
    ).run(
        initial_defender=defender,
        initial_attackers={'rl': attacker},
        sample_tasks=lambda iteration, count: ('rl',),
    )

    assert result.algorithm1_defender.fingerprint() != initial_defender
    assert result.algorithm2_defender.fingerprint() != initial_defender
    assert result.algorithm1_attackers['rl'].fingerprint() != initial_attacker
    assert len(result.algorithm1.iterations) == config.N_D
    assert len(result.algorithm2.iterations) == config.T
    assert result.trajectory_count == 8
    assert set(result.support_seeds).isdisjoint({101, 102})


def test_declared_2_2_8_2_2_2_training_scale_executes_real_rollouts() -> None:
    config = PaperMetaSGConfig().scaled(
        T=2, K=2, H=8, l=2, N_A=2, N_D=2,
        workers=6, untargeted_attackers=3, sample_size=4,
        td3_batch_size=16, learning_starts=16, hidden_sizes=(8,),
        replay_capacity=4096,
    )
    probe = _make_env(seed=1)
    defender_dim = len(flatten_observation(
        probe.defender_observation(), DEFENDER_OBSERVATION_KEYS,
    ))
    attacker_dim = len(flatten_observation(
        _make_env(seed=1).begin_round(
            __import__('numpy').zeros(3, dtype='float32'),
        ).attacker_observation,
        ATTACKER_OBSERVATION_KEYS,
    ))
    defender = _agent(defender_dim, 'defender', 20)
    attackers = {
        'rl-a': _agent(attacker_dim, 'attacker', 21),
        'rl-b': _agent(attacker_dim, 'attacker', 22),
    }

    def env_factory(task, seed, horizon):
        del task
        env = _make_env(seed=seed)
        env.horizon = horizon
        return env

    result = ScaledPaperMetaSGTrainingRunner(
        config=config, env_factory=env_factory,
        defender_obs_dim=defender_dim, attacker_obs_dim=attacker_dim,
        support_seed=2000, query_seeds=(101, 102),
    ).run(
        initial_defender=defender,
        initial_attackers=attackers,
        sample_tasks=lambda iteration, count: ('rl-a', 'rl-b'),
    )

    assert result.trajectories_per_update == 2
    assert result.trajectory_count == 48
    assert len(result.support_seeds) == len(set(result.support_seeds))
    assert all(len(item.tasks) == 2 for item in result.algorithm1.iterations)
    assert all(len(item.tasks) == 2 for item in result.algorithm2.iterations)

    def pair_query(current_defender, current_attacker, seed):
        env = env_factory('rl-a', seed, config.H)
        defender_replay = TD3ReplayBuffer(
            32, obs_dim=defender_dim, action_dim=3, role='defender', seed=seed,
        )
        attacker_replay = TD3ReplayBuffer(
            32, obs_dim=attacker_dim, action_dim=3, role='attacker', seed=seed + 100,
        )
        trajectory = PaperTD3TrajectoryCollector().collect(
            env=env, defender=current_defender, attacker=current_attacker,
            defender_replay=defender_replay, attacker_replay=attacker_replay,
            generation=0, deterministic=True,
        )
        return (
            trajectory.defender_return,
            trajectory.attacker_return,
            __import__('numpy').concatenate([
                step.defender_raw_action for step in trajectory.steps
            ]),
            __import__('numpy').concatenate([
                step.attacker_raw_action for step in trajectory.steps
            ]),
        )

    plan = QueryEvidencePlan(result.support_seeds, (101, 102))
    initial_evidence = evaluate_frozen_pair(
        label='initial', defender=defender, attacker=attackers['rl-a'],
        plan=plan, query=pair_query,
    )
    learned_evidence = evaluate_frozen_pair(
        label='learned', defender=result.algorithm1_defender,
        attacker=result.algorithm1_attackers['rl-a'],
        plan=plan, query=pair_query,
    )
    assert initial_evidence.query_seeds == learned_evidence.query_seeds == (101, 102)
    assert initial_evidence.attacker_action_trajectories != learned_evidence.attacker_action_trajectories


def test_scaled_runner_checkpoint_is_complete_resumable_and_lightweight(
    tmp_path,
) -> None:
    config = PaperMetaSGConfig().scaled(
        T=1, K=1, H=2, l=1, N_A=1, N_D=1,
        workers=4, untargeted_attackers=2, sample_size=4,
        td3_batch_size=2, learning_starts=2, hidden_sizes=(8,),
        replay_capacity=128,
    )
    probe = _make_env(seed=1)
    defender_dim = len(flatten_observation(
        probe.defender_observation(), DEFENDER_OBSERVATION_KEYS,
    ))
    attacker_dim = len(flatten_observation(
        _make_env(seed=1).begin_round(
            __import__('numpy').zeros(3, dtype='float32'),
        ).attacker_observation,
        ATTACKER_OBSERVATION_KEYS,
    ))
    defender = _agent(defender_dim, 'defender', 50)
    attacker = _agent(attacker_dim, 'attacker', 51)

    def env_factory(task, seed, horizon):
        del task
        env = _make_env(seed=seed)
        env.horizon = horizon
        return env

    checkpoint_path = tmp_path / 'training.pt'
    first = ScaledPaperMetaSGTrainingRunner(
        config=config,
        env_factory=env_factory,
        defender_obs_dim=defender_dim,
        attacker_obs_dim=attacker_dim,
        support_seed=3000,
    ).run(
        initial_defender=defender,
        initial_attackers={'rl': attacker},
        sample_tasks=lambda iteration, count: ('rl',),
        checkpoint_path=checkpoint_path,
    )
    checkpoint = load_scaled_training_checkpoint(checkpoint_path)
    assert checkpoint.phase == 'complete'
    assert checkpoint.algorithm1_completed == checkpoint.algorithm2_completed == 1
    assert checkpoint.algorithm1_iterations[0].tasks[0].adaptation.adapted_defender is None
    assert checkpoint.algorithm1_iterations[0].tasks[0].response.approximate_best_response is None
    assert checkpoint.algorithm2_iterations[0].tasks[0].adapted_snapshot is None
    assert checkpoint.config_signature['workers'] == 4
    assert checkpoint.config_signature['protocol_signature'] == {}

    resumed = ScaledPaperMetaSGTrainingRunner(
        config=config,
        env_factory=env_factory,
        defender_obs_dim=defender_dim,
        attacker_obs_dim=attacker_dim,
        support_seed=3000,
    ).run(
        initial_defender=defender,
        initial_attackers={'rl': attacker},
        sample_tasks=lambda iteration, count: ('rl',),
        checkpoint_path=checkpoint_path,
        resume=True,
    )
    assert resumed.support_seeds == first.support_seeds
    assert resumed.algorithm1_defender.fingerprint() == first.algorithm1_defender.fingerprint()
    assert resumed.algorithm2_defender.fingerprint() == first.algorithm2_defender.fingerprint()
    assert resumed.algorithm1_attackers['rl'].fingerprint() == first.algorithm1_attackers['rl'].fingerprint()

    incompatible = ScaledPaperMetaSGTrainingRunner(
        config=config,
        env_factory=env_factory,
        defender_obs_dim=defender_dim,
        attacker_obs_dim=attacker_dim,
        support_seed=3000,
        protocol_signature={'partition_seed': 999},
    )
    with pytest.raises(ValueError, match='configuration mismatch'):
        incompatible.run(
            initial_defender=defender,
            initial_attackers={'rl': attacker},
            sample_tasks=lambda iteration, count: ('rl',),
            checkpoint_path=checkpoint_path,
            resume=True,
        )


def test_scaled_runner_trains_meta_sg_without_running_algorithm2(tmp_path) -> None:
    config = PaperMetaSGConfig().scaled(
        T=2, K=1, H=2, l=1, N_A=1, N_D=1,
        workers=4, untargeted_attackers=2, sample_size=4,
        td3_batch_size=2, learning_starts=2, hidden_sizes=(8,),
        replay_capacity=128,
    )
    probe = _make_env(seed=1)
    defender_dim = len(flatten_observation(
        probe.defender_observation(), DEFENDER_OBSERVATION_KEYS,
    ))
    attacker_dim = len(flatten_observation(
        _make_env(seed=1).begin_round(
            __import__('numpy').zeros(3, dtype='float32'),
        ).attacker_observation,
        ATTACKER_OBSERVATION_KEYS,
    ))
    defender = _agent(defender_dim, 'defender', 70)
    attacker = _agent(attacker_dim, 'attacker', 71)

    def env_factory(task, seed, horizon):
        del task
        env = _make_env(seed=seed)
        env.horizon = horizon
        return env

    checkpoint_path = tmp_path / 'meta-sg.pt'
    result = ScaledPaperMetaSGTrainingRunner(
        config=config,
        env_factory=env_factory,
        defender_obs_dim=defender_dim,
        attacker_obs_dim=attacker_dim,
        support_seed=4000,
    ).run(
        initial_defender=defender,
        initial_attackers={'rl': attacker},
        sample_tasks=lambda iteration, count: ('rl',),
        training_method='meta-sg',
        checkpoint_path=checkpoint_path,
    )

    checkpoint = load_scaled_training_checkpoint(checkpoint_path)
    assert checkpoint.phase == 'complete'
    assert checkpoint.config_signature['training_method'] == 'meta-sg'
    assert checkpoint.algorithm1_completed == 1
    assert checkpoint.algorithm2_completed == 0
    assert len(result.algorithm1.iterations) == 1
    assert result.algorithm2.iterations == ()
    assert result.algorithm1_defender.fingerprint() != defender.fingerprint()
    assert result.algorithm2_defender.fingerprint() == defender.fingerprint()
    assert result.trajectory_count == 3


def test_scaled_runner_trains_meta_rl_without_running_algorithm1(tmp_path) -> None:
    config = PaperMetaSGConfig().scaled(
        T=2, K=1, H=2, l=1, N_A=1, N_D=1,
        workers=4, untargeted_attackers=2, sample_size=4,
        td3_batch_size=2, learning_starts=2, hidden_sizes=(8,),
        replay_capacity=128,
    )
    defender_dim = len(flatten_observation(
        _make_env(seed=1).defender_observation(), DEFENDER_OBSERVATION_KEYS,
    ))
    attacker_dim = len(flatten_observation(
        _make_env(seed=1).begin_round(
            __import__('numpy').zeros(3, dtype='float32'),
        ).attacker_observation,
        ATTACKER_OBSERVATION_KEYS,
    ))
    defender = _agent(defender_dim, 'defender', 80)
    attacker = _agent(attacker_dim, 'attacker', 81)

    def env_factory(task, seed, horizon):
        del task
        env = _make_env(seed=seed)
        env.horizon = horizon
        return env

    checkpoint_path = tmp_path / 'meta-rl.pt'
    result = ScaledPaperMetaSGTrainingRunner(
        config=config,
        env_factory=env_factory,
        defender_obs_dim=defender_dim,
        attacker_obs_dim=attacker_dim,
        support_seed=5000,
    ).run(
        initial_defender=defender,
        initial_attackers={'fixed': attacker},
        sample_tasks=lambda iteration, count: ('fixed',),
        training_method='meta-rl',
        checkpoint_path=checkpoint_path,
    )

    checkpoint = load_scaled_training_checkpoint(checkpoint_path)
    assert checkpoint.phase == 'complete'
    assert checkpoint.config_signature['training_method'] == 'meta-rl'
    assert checkpoint.algorithm1_completed == 0
    assert checkpoint.algorithm2_completed == 2
    assert result.algorithm1.iterations == ()
    assert len(result.algorithm2.iterations) == 2
    assert result.algorithm1_defender.fingerprint() == defender.fingerprint()
    assert result.algorithm2_defender.fingerprint() != defender.fingerprint()
    assert result.trajectory_count == 2
