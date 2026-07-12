from __future__ import annotations

import json

import torch
from torch.utils.data import TensorDataset

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.experiments.attack_domain import AttackTypeDomainSource
from meta_stackelberg.experiments.paper_mnist_backdoor_env import (
    PaperMNISTBackdoorEnvironmentFactory,
    make_whitebox_mnist_datasets,
)
from meta_stackelberg.experiments.paper_mnist_backdoor_meta_sg import (
    MNISTWhiteBoxMetaSGConfig,
    run_mnist_whitebox_backdoor_meta_sg,
)


def _mnist_like(samples: int, seed: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(samples, 1, 28, 28, generator=generator)
    labels = torch.arange(samples, dtype=torch.long) % 10
    return TensorDataset(inputs, labels)


def _attacker(obs_dim: int, seed: int) -> TD3Agent:
    return TD3Agent(
        obs_dim=obs_dim,
        action_dim=3,
        role='attacker',
        seed=seed,
        hidden_sizes=(8,),
        learning_rate=0.001,
        gamma=0.99,
        tau=0.005,
        policy_delay=2,
        target_policy_noise=0.2,
        noise_clip=0.5,
    )


def test_mnist_whitebox_runner_executes_algorithm1_against_adapted_defender(
    tmp_path,
) -> None:
    datasets = make_whitebox_mnist_datasets(
        train_dataset=_mnist_like(80, 7),
        held_out_test=_mnist_like(20, 8),
        reward_samples=20,
        seed=17,
    )
    factory = PaperMNISTBackdoorEnvironmentFactory(
        datasets=datasets,
        partition_seed=18,
        model_seed=99,
        workers=4,
        backdoor_attackers=2,
        sample_size=4,
        fl_batch_size=8,
        malicious_batch_size=8,
    )
    config = MNISTWhiteBoxMetaSGConfig(
        N_D=1,
        K=1,
        N_A=1,
        H=1,
        td3_batch_size=1,
        learning_starts=1,
        replay_capacity=64,
        hidden_sizes=(8,),
    )
    attack_domain = AttackTypeDomainSource.from_policies(
        {
            'brl-norm': _attacker(factory.attacker_observation_dim, 31),
            'brl-neuroclip': _attacker(factory.attacker_observation_dim, 32),
        },
        origins={
            'brl-norm': 'pretrained-against-norm-bounding',
            'brl-neuroclip': 'pretrained-against-neuroclip',
        },
    )

    result = run_mnist_whitebox_backdoor_meta_sg(
        environment_factory=factory,
        attack_domain=attack_domain,
        output_dir=tmp_path,
        seed=9,
        support_seed=1_000,
        config=config,
    )

    assert result.defender_action_dim == 3
    assert result.attacker_action_dim == 3
    assert len(result.algorithm1.iterations) == 1
    task_trace = result.algorithm1.iterations[0].tasks[0]
    assert task_trace.adaptation.adapted_defender_fingerprint != (
        task_trace.adaptation.initial_defender_fingerprint
    )
    assert len(task_trace.response.update_stats) == 1
    assert task_trace.response.initial_attacker_fingerprint != (
        task_trace.response.adapted_attacker_fingerprint
    )
    assert result.trajectory_count == 3
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest['protocol'] == 'mnist-whitebox-real-data-v1'
    assert manifest['algorithm'] == 'meta-sg-algorithm1-reptile'
    assert manifest['attack_origins'] == {
        'brl-neuroclip': 'pretrained-against-neuroclip',
        'brl-norm': 'pretrained-against-norm-bounding',
    }
    assert manifest['query_data_used_for_training'] is False
