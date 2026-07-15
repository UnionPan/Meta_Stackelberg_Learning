import numpy as np
import torch
from torch.utils.data import TensorDataset

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.agents.td3.replay import flatten_observation
from meta_stackelberg.experiments.model_poisoning_evaluation import (
    ModelPoisoningScenario,
    evaluate_model_poisoning_scenario,
    summarize_model_poisoning_evaluation,
)
from meta_stackelberg.experiments.paper_meta_sg import (
    ATTACKER_OBSERVATION_KEYS,
    DEFENDER_OBSERVATION_KEYS,
)
from meta_stackelberg.experiments.paper_mnist_env import (
    PaperMNISTEnvironmentFactory,
)


def _datasets(samples=400):
    generator = torch.Generator().manual_seed(7)
    inputs = torch.randn(samples, 1, 28, 28, generator=generator)
    labels = torch.arange(samples) % 10
    full = TensorDataset(inputs, labels)
    return full, TensorDataset(inputs[:40], labels[:40])


def _agent(obs_dim, role, seed):
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


def test_fixed_ipm_final_evaluation_is_scalar_only_and_freezes_policies() -> None:
    train, test = _datasets()
    factory = PaperMNISTEnvironmentFactory(
        train_dataset=train,
        root_dataset=test,
        partition_seed=8,
        model_seed=9,
        workers=20,
        untargeted_attackers=10,
        sample_size=10,
        fl_batch_size=16,
        local_search_batch_size=8,
    )
    probe = factory.make(seed=1, horizon=2)
    defender_dim = len(flatten_observation(
        probe.defender_observation(), DEFENDER_OBSERVATION_KEYS,
    ))
    attacker_probe = factory.make(seed=1, horizon=2)
    attacker_dim = len(flatten_observation(
        attacker_probe.begin_round(
            np.zeros(3, dtype=np.float32),
        ).attacker_observation,
        ATTACKER_OBSERVATION_KEYS,
    ))
    defender = _agent(defender_dim, 'defender', 1)
    attacker = _agent(attacker_dim, 'attacker', 2)
    defender_before = defender.fingerprint()
    attacker_before = attacker.fingerprint()

    record = evaluate_model_poisoning_scenario(
        scenario=ModelPoisoningScenario('ipm', 'ipm', 'rl-a', 2.0),
        factory=factory,
        test_dataset=test,
        defender=defender,
        attacker=attacker,
        seed=101,
        horizon=2,
    )

    assert record['scenario'] == 'ipm'
    assert len(record['round_metrics']) == 2
    assert record['test_examples'] == 40
    assert all(row['sampled_malicious_clients'] >= 0 for row in record['round_metrics'])
    assert all(isinstance(row['clean_accuracy'], float) for row in record['round_metrics'])
    assert defender.fingerprint() == defender_before
    assert attacker.fingerprint() == attacker_before


def test_model_poisoning_summary_requires_and_aggregates_baseline_families() -> None:
    def record(name, family, accuracy):
        return {
            'scenario': name,
            'attack_family': family,
            'round_metrics': [
                {'clean_accuracy': accuracy, 'clean_loss': 1.0},
                {'clean_accuracy': accuracy + 0.1, 'clean_loss': 0.8},
            ],
            'final_raw_clean_accuracy': accuracy + 0.1,
            'final_delivered_clean_accuracy': accuracy,
            'final_raw_clean_loss': 0.8,
            'final_delivered_clean_loss': 0.9,
            'mean_defender_reward': -0.9,
            'mean_attacker_reward': 0.1,
        }

    summary = summarize_model_poisoning_evaluation({
        'clean': record('clean', 'clean', 0.9),
        'ipm': record('ipm', 'ipm', 0.7),
        'lmp': record('lmp', 'lmp', 0.6),
        'rl-a': record('rl-a', 'rl', 0.5),
        'rl-b': record('rl-b', 'rl', 0.4),
    }, checkpoint='final.pt', seed=101, horizon=2)

    assert summary['single_seed'] is True
    assert summary['confidence_interval'] is None
    assert summary['worst_rl_scenario'] == 'rl-b'
    assert summary['scenarios']['ipm']['rounds'] == 2
