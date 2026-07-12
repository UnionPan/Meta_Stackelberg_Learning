from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.agents.td3.config import PaperMetaSGConfig
from meta_stackelberg.experiments.paper_mnist_backdoor_env import (
    PaperMNISTBackdoorEnvironmentFactory,
    make_whitebox_mnist_datasets,
)
from meta_stackelberg.experiments.paper_mnist_backdoor_scientific import (
    run_mnist_whitebox_scientific_evidence,
)
from meta_stackelberg.experiments.scientific_gate import (
    QueryEvidencePlan,
    ScientificGateThresholds,
)
from meta_stackelberg.experiments.whitebox_backdoor_evidence import (
    WhiteBoxSafetyThresholds,
)


def _mnist_like(samples: int, seed: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(samples, 1, 28, 28, generator=generator)
    labels = torch.arange(samples, dtype=torch.long) % 10
    return TensorDataset(inputs, labels)


def _agent(obs_dim: int, role: str, seed: int) -> TD3Agent:
    return TD3Agent(
        obs_dim=obs_dim,
        action_dim=3,
        role=role,
        seed=seed,
        hidden_sizes=(8,),
        learning_rate=0.001,
        gamma=0.99,
        tau=0.005,
        policy_delay=2,
        target_policy_noise=0.2,
        noise_clip=0.5,
    )


def _constant(source: TD3Agent, raw_action: tuple[float, float, float]) -> TD3Agent:
    result = source.clone()
    for parameter in result.actor.parameters():
        parameter.data.zero_()
    final = tuple(result.actor.modules())[-1]
    final.bias.data.copy_(torch.atanh(torch.tensor(raw_action) * 0.999))
    return result


def test_whitebox_scientific_runner_attaches_query_metrics_to_all_comparisons() -> None:
    datasets = make_whitebox_mnist_datasets(
        train_dataset=_mnist_like(40, 7),
        held_out_test=_mnist_like(20, 8),
        reward_samples=20,
        seed=17,
    )
    factory = PaperMNISTBackdoorEnvironmentFactory(
        datasets=datasets,
        partition_seed=18,
        model_seed=99,
        workers=4,
        backdoor_attackers=1,
        sample_size=4,
        fl_batch_size=10,
        malicious_batch_size=10,
    )
    config = PaperMetaSGConfig().scaled(
        T=1,
        K=1,
        H=1,
        l=1,
        N_A=1,
        N_D=1,
        workers=4,
        untargeted_attackers=1,
        sample_size=4,
        td3_batch_size=1,
        learning_starts=1,
        hidden_sizes=(8,),
        replay_capacity=256,
    )
    learned = _agent(factory.defender_observation_dim, 'defender', 1)
    random_defender = _agent(factory.defender_observation_dim, 'defender', 2)
    attacker = _agent(factory.attacker_observation_dim, 'attacker', 3)

    result = run_mnist_whitebox_scientific_evidence(
        config=config,
        environment_factory=factory,
        evidence_plan=QueryEvidencePlan(tuple(range(1_000, 1_200)), (101, 102)),
        meta_thresholds=ScientificGateThresholds(
            attacker_improvement=0.001,
            response_difference=0.001,
            defender_adaptation_improvement=0.001,
            meta_advantage=0.001,
            oracle_regret=0.1,
            action_difference=0.001,
            attacker_plateau_gap=0.001,
        ),
        safety_thresholds=WhiteBoxSafetyThresholds(0.0, 1.0, 0.0),
        learned_defender=learned,
        random_defender=random_defender,
        initial_attacker=attacker,
        specialized_defenders={
            'tight': _constant(learned, (-0.8, 0.0, -0.8)),
            'loose': _constant(learned, (0.8, 0.0, 0.8)),
        },
        attacker_oracle_policies={
            'low-poison': _constant(attacker, (-0.8, 0.0, -0.9)),
            'high-poison': _constant(attacker, (0.8, 0.0, -0.9)),
        },
        task='brl-norm',
        query_batch_size=10,
    )

    required = {
        'attacker_initial', 'attacker_br', 'attacker_oracle',
        'defender_a_response', 'defender_b_response',
        'defender_initial', 'defender_adapted',
        'meta_adapted', 'random_adapted', 'no_adaptation',
        'learned_defender', 'specialized_oracle',
    }
    assert set(result.metrics) == required
    assert all(item.query_seeds == (101, 102) for item in result.metrics.values())
    assert all(len(item.per_seed) == 2 for item in result.metrics.values())
    assert len(result.meta_sg.gate.checks) == 6
    assert len(result.safety_gate.checks) == 3
    assert result.passed == (result.meta_sg.gate.passed and result.safety_gate.passed)
    assert set(result.meta_sg.used_support_seeds).isdisjoint({101, 102})
