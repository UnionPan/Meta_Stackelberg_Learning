from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.agents.td3.config import PaperMetaSGConfig
from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.experiments.attack_pretraining import AttackPolicyPretrainingConfig
from meta_stackelberg.experiments.backdoor_attack_pretraining import (
    BackdoorAttackPretrainingTask,
    fixed_backdoor_pretraining_defense,
    pretrain_backdoor_attack_type_domain,
)
from meta_stackelberg.experiments.paper_mnist_backdoor_env import (
    PaperMNISTBackdoorEnvironmentFactory,
    make_whitebox_mnist_datasets,
)
from meta_stackelberg.federated.types import ClientUpdate


def _mnist_like(samples: int, seed: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(samples, 1, 28, 28, generator=generator)
    labels = torch.arange(samples, dtype=torch.long) % 10
    return TensorDataset(inputs, labels)


def _updates(values: tuple[float, ...]) -> tuple[ClientUpdate, ...]:
    return tuple(
        ClientUpdate(index, ModelState.from_tensors([np.array([value])]), 1)
        for index, value in enumerate(values)
    )


def test_fixed_backdoor_pretraining_defenses_have_distinct_execution() -> None:
    norm = fixed_backdoor_pretraining_defense(
        'norm-bounding', clip_radius=1.0,
    )
    neuroclip = fixed_backdoor_pretraining_defense(
        'neuroclip', epsilon=2.0,
    )

    np.testing.assert_allclose(
        norm.aggregate(_updates((-10.0, 0.0, 2.0))).vector(),
        np.array([0.0]),
    )
    np.testing.assert_allclose(
        neuroclip.aggregate(_updates((-10.0, 0.0, 2.0))).vector(),
        np.array([-8.0 / 3.0]),
    )
    assert norm.spec == {
        'defense': 'norm-bounding', 'clip_radius': 1.0,
    }
    assert neuroclip.spec == {
        'defense': 'neuroclip', 'epsilon': 2.0,
    }


def test_backdoor_pretraining_builds_two_paper_attack_origins(tmp_path) -> None:
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
    config = AttackPolicyPretrainingConfig(
        fl_rounds=2,
        batch_size=1,
        learning_starts=1,
        train_freq=1,
        gradient_steps=1,
        replay_capacity=32,
    )

    result = pretrain_backdoor_attack_type_domain(
        config=config,
        paper=PaperMetaSGConfig(),
        environment_factory=factory,
        tasks=(
            BackdoorAttackPretrainingTask(
                'brl-norm', 'norm-bounding', clip_radius=1.0,
            ),
            BackdoorAttackPretrainingTask(
                'brl-neuroclip', 'neuroclip', epsilon=2.0,
            ),
        ),
        hidden_sizes=(8,),
        seed=20,
        checkpoint_directory=tmp_path,
        checkpoint_interval=1,
    )

    assert tuple(result.domain.snapshots) == ('brl-norm', 'brl-neuroclip')
    assert result.domain.origins == {
        'brl-norm': 'pretrained-against-norm-bounding',
        'brl-neuroclip': 'pretrained-against-neuroclip',
    }
    assert [task.fl_round_count for task in result.tasks] == [2, 2]
    assert [task.td3_update_count for task in result.tasks] == [2, 2]
    assert result.total_fl_round_count == 4
    assert (tmp_path / 'brl-norm.pt').is_file()
    assert (tmp_path / 'brl-neuroclip.pt').is_file()

    resumed = pretrain_backdoor_attack_type_domain(
        config=config,
        paper=PaperMetaSGConfig(),
        environment_factory=factory,
        tasks=(
            BackdoorAttackPretrainingTask(
                'brl-norm', 'norm-bounding', clip_radius=1.0,
            ),
            BackdoorAttackPretrainingTask(
                'brl-neuroclip', 'neuroclip', epsilon=2.0,
            ),
        ),
        hidden_sizes=(8,),
        seed=20,
        checkpoint_directory=tmp_path,
        checkpoint_interval=1,
        resume_checkpoints=True,
    )
    for label, snapshot in result.domain.snapshots.items():
        first = _snapshot_agent(factory.attacker_observation_dim, snapshot, 1)
        second = _snapshot_agent(
            factory.attacker_observation_dim,
            resumed.domain.snapshots[label],
            2,
        )
        assert first.fingerprint() == second.fingerprint()


def _snapshot_agent(obs_dim: int, snapshot, seed: int) -> TD3Agent:
    agent = TD3Agent(
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
    agent.restore(snapshot)
    return agent
