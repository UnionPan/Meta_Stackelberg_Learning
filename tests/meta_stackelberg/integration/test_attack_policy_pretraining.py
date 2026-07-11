import numpy as np

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.agents.td3.replay import flatten_observation
from meta_stackelberg.experiments.attack_pretraining import (
    AttackPolicyPretrainingConfig,
    AttackPolicyPretrainer,
    build_attack_type_domain,
    fixed_pretraining_aggregator,
)
from meta_stackelberg.experiments.deterministic_paper_env import (
    make_deterministic_paper_env,
)
from meta_stackelberg.experiments.paper_meta_sg import (
    ATTACKER_OBSERVATION_KEYS,
    DEFENDER_OBSERVATION_KEYS,
)


def _agent(obs_dim: int, role: str, seed: int) -> TD3Agent:
    return TD3Agent(
        obs_dim=obs_dim, action_dim=3, role=role, seed=seed,
        hidden_sizes=(8,), learning_rate=0.001, gamma=0.99, tau=0.005,
        policy_delay=2, target_policy_noise=0.2, noise_clip=0.5,
    )


def test_pretraining_uses_fl_round_budget_not_best_response_count() -> None:
    probe = make_deterministic_paper_env(seed=1, horizon=4)
    defender_dim = len(flatten_observation(
        probe.defender_observation(), DEFENDER_OBSERVATION_KEYS,
    ))
    attacker_dim = len(flatten_observation(
        probe.begin_round(np.zeros(3, dtype=np.float32)).attacker_observation,
        ATTACKER_OBSERVATION_KEYS,
    ))
    defender = _agent(defender_dim, 'defender', 2)
    attacker = _agent(attacker_dim, 'attacker', 3)
    defender_before = defender.fingerprint()
    attacker_before = attacker.fingerprint()

    result = AttackPolicyPretrainer(AttackPolicyPretrainingConfig(
        fl_rounds=4,
        batch_size=2,
        learning_starts=2,
        train_freq=1,
        gradient_steps=1,
        replay_capacity=32,
    )).train(
        label='krum',
        origin='pretrained-against-krum',
        env=make_deterministic_paper_env(seed=5, horizon=4),
        defender=defender,
        attacker=attacker,
        aggregator=fixed_pretraining_aggregator('krum', byzantine_count=0),
        replay_seed=9,
    )

    assert result.fl_round_count == 4
    assert result.replay_transition_count == 4
    assert result.td3_update_count == 3
    assert result.label == 'krum'
    assert result.origin == 'pretrained-against-krum'
    assert defender.fingerprint() == defender_before
    assert attacker.fingerprint() != attacker_before
    assert result.protocol == 'sequential-fl-round-td3-pretraining-v1'
    domain = build_attack_type_domain((result,))
    assert set(domain.snapshots) == {'krum'}
    assert domain.origins == {'krum': 'pretrained-against-krum'}
    assert domain.protocol == 'sequential-fl-round-td3-pretraining-v1'
