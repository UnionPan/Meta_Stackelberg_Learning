import numpy as np

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.agents.td3.config import PaperMetaSGConfig
from meta_stackelberg.agents.td3.replay import flatten_observation
from meta_stackelberg.experiments.paper_meta_sg import (
    ATTACKER_OBSERVATION_KEYS,
    DEFENDER_OBSERVATION_KEYS,
    ScaledPaperMetaSGTrainingRunner,
)
from meta_stackelberg.experiments.scientific_gate import (
    QueryEvidencePlan,
    ScientificGateThresholds,
)
from meta_stackelberg.experiments.scientific_run import PaperScientificGateRunner
from meta_stackelberg.experiments.deterministic_paper_env import make_deterministic_paper_env


def _make_env(seed):
    return make_deterministic_paper_env(seed=seed)


def _agent(obs_dim, role, seed):
    return TD3Agent(
        obs_dim=obs_dim, action_dim=3, role=role, seed=seed,
        hidden_sizes=(8,), learning_rate=0.001, gamma=0.99, tau=0.005,
        policy_delay=2, target_policy_noise=0.2, noise_clip=0.5,
    )


def _constant_defender(source, raw_action):
    result = source.clone()
    for parameter in result.actor.parameters():
        parameter.data.zero_()
    final = tuple(result.actor.modules())[-1]
    final.bias.data.copy_(
        __import__('torch').atanh(__import__('torch').tensor(raw_action) * 0.999)
    )
    return result


def _constant_attacker(source, raw_action):
    result = source.clone()
    for parameter in result.actor.parameters():
        parameter.data.zero_()
    final = tuple(result.actor.modules())[-1]
    final.bias.data.copy_(
        __import__('torch').atanh(__import__('torch').tensor(raw_action) * 0.999)
    )
    return result


def test_scientific_runner_executes_fresh_responses_equal_budgets_and_six_gates() -> None:
    config = PaperMetaSGConfig().scaled(
        T=1, K=1, H=2, l=2, N_A=2, N_D=1,
        workers=4, untargeted_attackers=2, sample_size=4,
        td3_batch_size=4, learning_starts=4, hidden_sizes=(8,),
        replay_capacity=512,
    )
    probe = _make_env(seed=1)
    defender_dim = len(flatten_observation(
        probe.defender_observation(), DEFENDER_OBSERVATION_KEYS,
    ))
    attacker_dim = len(flatten_observation(
        _make_env(seed=1).begin_round(np.zeros(3, np.float32)).attacker_observation,
        ATTACKER_OBSERVATION_KEYS,
    ))
    learned = _agent(defender_dim, 'defender', 1)
    random = _agent(defender_dim, 'defender', 3)
    attacker = _agent(attacker_dim, 'attacker', 4)

    def env_factory(task, seed, horizon):
        del task
        env = _make_env(seed=seed)
        env.horizon = horizon
        return env

    runner = PaperScientificGateRunner(
        config=config,
        env_factory=env_factory,
        defender_obs_dim=defender_dim,
        attacker_obs_dim=attacker_dim,
        evidence_plan=QueryEvidencePlan(tuple(range(1000, 1200)), (101, 102)),
        thresholds=ScientificGateThresholds(
            attacker_improvement=0.001,
            response_difference=0.001,
            defender_adaptation_improvement=0.001,
            meta_advantage=0.001,
            oracle_regret=0.1,
            action_difference=0.001,
            attacker_plateau_gap=0.001,
        ),
    )
    result = runner.run(
        task='rl',
        learned_defender=learned,
        random_defender=random,
        initial_attacker=attacker,
        specialized_defenders={
            'low': _constant_defender(learned, (-0.5, 0.0, 0.0)),
            'high': _constant_defender(learned, (0.5, 0.0, 0.0)),
        },
        attacker_oracle_policies={
            'low-gamma': _constant_attacker(attacker, (-0.8, 0.0, 0.0)),
            'high-gamma': _constant_attacker(attacker, (0.8, 0.0, 0.0)),
        },
    )

    assert len(result.gate.checks) == 6
    assert not result.gate.passed
    assert 'attacker_oracle' in result.evidence
    assert set(result.evidence) >= {
        'attacker_initial', 'attacker_br', 'defender_a_response',
        'defender_b_response', 'defender_initial', 'defender_adapted',
        'meta_adapted', 'random_adapted', 'no_adaptation',
        'learned_defender', 'specialized_oracle',
    }
    assert result.budgets['meta_adapted'] == result.budgets['random_adapted']
    assert result.budgets['meta_adapted'].fl_rounds == result.budgets['no_adaptation'].fl_rounds
    assert len(set(result.adaptation_seed_blocks.values())) == 1
    assert result.fresh_response_count >= 6
    assert set(result.used_support_seeds).isdisjoint(result.query_seeds)


def test_declared_scaled_training_flows_into_independent_scientific_gate() -> None:
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
        _make_env(seed=1).begin_round(np.zeros(3, np.float32)).attacker_observation,
        ATTACKER_OBSERVATION_KEYS,
    ))
    initial_defender = _agent(defender_dim, 'defender', 31)
    random_defender = _agent(defender_dim, 'defender', 32)
    initial_attackers = {
        'rl-a': _agent(attacker_dim, 'attacker', 33),
        'rl-b': _agent(attacker_dim, 'attacker', 34),
    }

    def env_factory(task, seed, horizon):
        del task
        env = _make_env(seed=seed)
        env.horizon = horizon
        return env

    training = ScaledPaperMetaSGTrainingRunner(
        config=config, env_factory=env_factory,
        defender_obs_dim=defender_dim, attacker_obs_dim=attacker_dim,
        support_seed=2000, query_seeds=(101, 102),
    ).run(
        initial_defender=initial_defender,
        initial_attackers=initial_attackers,
        sample_tasks=lambda iteration, count: ('rl-a', 'rl-b'),
    )
    result = PaperScientificGateRunner(
        config=config, env_factory=env_factory,
        defender_obs_dim=defender_dim, attacker_obs_dim=attacker_dim,
        evidence_plan=QueryEvidencePlan(tuple(range(5000, 5500)), (101, 102)),
        thresholds=ScientificGateThresholds(
            attacker_improvement=0.001,
            response_difference=0.001,
            defender_adaptation_improvement=0.001,
            meta_advantage=0.001,
            oracle_regret=0.1,
            action_difference=0.001,
            attacker_plateau_gap=0.001,
        ),
    ).run(
        task='rl-a',
        learned_defender=training.algorithm1_defender,
        random_defender=random_defender,
        initial_attacker=initial_attackers['rl-a'],
        specialized_defenders={
            'low': _constant_defender(initial_defender, (-0.5, 0.0, 0.0)),
            'high': _constant_defender(initial_defender, (0.5, 0.0, 0.0)),
        },
        attacker_oracle_policies={
            'low-gamma': _constant_attacker(
                initial_attackers['rl-a'], (-0.8, 0.0, 0.0),
            ),
            'high-gamma': _constant_attacker(
                initial_attackers['rl-a'], (0.8, 0.0, 0.0),
            ),
        },
    )

    assert training.trajectory_count == 48
    assert len(result.gate.checks) == 6
    assert not result.gate.passed
    checks = {check.name: check for check in result.gate.checks}
    assert checks['attacker_best_response'].passed  # finite-oracle plateau branch
    assert not checks['behavior_and_objective_signal'].passed
    assert result.attacker_oracle_label in {'low-gamma', 'high-gamma'}
    assert result.budgets['meta_adapted'] == result.budgets['random_adapted']
    assert result.budgets['meta_adapted'].fl_rounds == 32
    assert set(training.support_seeds).isdisjoint(result.query_seeds)
    assert set(result.used_support_seeds).isdisjoint(result.query_seeds)
