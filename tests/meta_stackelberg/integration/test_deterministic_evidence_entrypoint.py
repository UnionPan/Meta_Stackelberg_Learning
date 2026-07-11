from meta_stackelberg.agents.td3.config import PaperMetaSGConfig
from meta_stackelberg.experiments.scaled_evidence import (
    run_deterministic_scaled_evidence,
)
from meta_stackelberg.experiments.scientific_gate import ScientificGateThresholds
from meta_stackelberg.experiments.attack_domain import AttackTypeDomainSource
from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.agents.td3.replay import flatten_observation
from meta_stackelberg.experiments.deterministic_paper_env import make_deterministic_paper_env
from meta_stackelberg.experiments.paper_meta_sg import ATTACKER_OBSERVATION_KEYS
import numpy as np


def test_single_entrypoint_runs_training_and_scientific_evidence() -> None:
    config = PaperMetaSGConfig().scaled(
        T=1, K=1, H=2, l=2, N_A=2, N_D=1,
        workers=4, untargeted_attackers=2, sample_size=4,
        td3_batch_size=4, learning_starts=4, hidden_sizes=(8,),
        replay_capacity=512,
    )
    result = run_deterministic_scaled_evidence(
        config=config,
        thresholds=ScientificGateThresholds(
            0.001, 0.001, 0.001, 0.001, 0.1, 0.001,
            attacker_plateau_gap=0.001,
        ),
        query_seeds=(101, 102),
        training_support_seed=2000,
        scientific_support_seeds=tuple(range(5000, 5200)),
        seed=7,
    )

    assert len(result.training.algorithm1.iterations) == 1
    assert len(result.training.algorithm2.iterations) == 1
    assert len(result.scientific.gate.checks) == 6
    assert result.parameter_snapshot['N_A'] == 2
    assert result.parameter_snapshot['H'] == 2
    assert result.query_seeds == (101, 102)


def test_entrypoint_restores_pretrained_attack_type_domain_and_provenance() -> None:
    config = PaperMetaSGConfig().scaled(
        T=1, K=1, H=1, l=1, N_A=1, N_D=1,
        workers=4, untargeted_attackers=2, sample_size=4,
        td3_batch_size=1, learning_starts=1, hidden_sizes=(8,),
        replay_capacity=64,
    )
    env = make_deterministic_paper_env(seed=1, horizon=1)
    attacker_observation = env.observation_encoder.attacker_observation(
        env.defender_observation(), malicious_count=0,
        defender_raw_action=np.zeros(3, dtype=np.float32),
    )
    obs_dim = len(flatten_observation(
        attacker_observation, ATTACKER_OBSERVATION_KEYS,
    ))
    attacker = TD3Agent(
        obs_dim=obs_dim, action_dim=3, role='attacker', seed=7,
        hidden_sizes=(8,), learning_rate=0.001, gamma=0.99, tau=0.005,
        policy_delay=2, target_policy_noise=0.2, noise_clip=0.5,
    )
    domain = AttackTypeDomainSource.from_policies(
        {'krum-pretrained': attacker},
        origins={'krum-pretrained': 'pretrained-against-krum'},
    )
    result = run_deterministic_scaled_evidence(
        config=config,
        thresholds=ScientificGateThresholds(0, 0, 0, 0, 1, 0),
        query_seeds=(101,), training_support_seed=1000,
        scientific_support_seeds=tuple(range(2000, 2100)),
        seed=9, attack_domain=domain,
    )
    assert result.parameter_snapshot['attack_type_origins'] == {
        'krum-pretrained': 'pretrained-against-krum',
    }
    assert set(result.training.algorithm1_attackers) == {'krum-pretrained'}
