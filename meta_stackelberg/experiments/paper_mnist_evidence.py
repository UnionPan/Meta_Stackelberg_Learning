"""One-call Meta-SG evidence protocol on the canonical paper MNIST factory."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np

from meta_stackelberg.agents.td3.config import ScaledMetaSGConfig
from meta_stackelberg.agents.td3.replay import flatten_observation
from meta_stackelberg.experiments.paper_meta_sg import (
    ATTACKER_OBSERVATION_KEYS,
    DEFENDER_OBSERVATION_KEYS,
)
from meta_stackelberg.experiments.attack_domain import AttackTypeDomainSource
from meta_stackelberg.experiments.paper_mnist_env import (
    PaperMNISTDatasets,
    PaperMNISTEnvironmentFactory,
)
from meta_stackelberg.experiments.scaled_evidence import (
    ScaledEvidenceResult,
    run_scaled_evidence,
)
from meta_stackelberg.experiments.scientific_gate import ScientificGateThresholds


def run_paper_mnist_scaled_evidence(
    *,
    config: ScaledMetaSGConfig,
    datasets: PaperMNISTDatasets,
    thresholds: ScientificGateThresholds,
    query_seeds: tuple[int, ...],
    training_support_seed: int,
    scientific_support_seeds: tuple[int, ...],
    seed: int,
    partition_seed: int,
    model_seed: int,
    local_search_learning_rate: float = 0.01,
    local_search_batch_size: int = 128,
    local_search_trajectories: int = 1,
    local_search_gradient_norm_cap: float = 1.0,
    attack_domain: AttackTypeDomainSource | None = None,
    device: str = 'cpu',
    training_checkpoint_path: str | None = None,
    resume_training: bool = False,
    training_checkpoint_interval: int = 1,
    training_method: str = 'both',
) -> ScaledEvidenceResult:
    paper = config.paper_reference
    factory = PaperMNISTEnvironmentFactory(
        train_dataset=datasets.client_train,
        root_dataset=datasets.root,
        partition_seed=partition_seed,
        model_seed=model_seed,
        workers=config.workers,
        untargeted_attackers=config.untargeted_attackers,
        sample_size=config.sample_size,
        non_iid_q=paper.non_iid_q,
        fl_batch_size=paper.fl_batch_size,
        local_iterations=paper.local_iterations,
        client_learning_rate=paper.client_learning_rate,
        local_search_learning_rate=local_search_learning_rate,
        local_search_batch_size=local_search_batch_size,
        local_search_trajectories=local_search_trajectories,
        local_search_gradient_norm_cap=local_search_gradient_norm_cap,
        device=device,
    )
    probe = factory.make(seed=seed, horizon=config.H, task_id='mnist-probe')
    defender_observation = probe.defender_observation()
    defender_obs_dim = len(flatten_observation(
        defender_observation, DEFENDER_OBSERVATION_KEYS,
    ))
    attacker_observation = probe.observation_encoder.attacker_observation(
        defender_observation,
        malicious_count=0,
        defender_raw_action=np.zeros(3, dtype=np.float32),
    )
    attacker_obs_dim = len(flatten_observation(
        attacker_observation, ATTACKER_OBSERVATION_KEYS,
    ))

    def env_factory(task, rollout_seed, horizon):
        return factory.make(
            seed=rollout_seed, horizon=horizon, task_id=str(task),
        )

    result = run_scaled_evidence(
        config=config,
        thresholds=thresholds,
        query_seeds=query_seeds,
        training_support_seed=training_support_seed,
        scientific_support_seeds=scientific_support_seeds,
        seed=seed,
        env_factory=env_factory,
        defender_obs_dim=defender_obs_dim,
        attacker_obs_dim=attacker_obs_dim,
        attack_domain=attack_domain,
        device=device,
        training_checkpoint_path=training_checkpoint_path,
        resume_training=resume_training,
        training_checkpoint_interval=training_checkpoint_interval,
        training_method=training_method,
        training_protocol_signature={
            'dataset': 'MNIST',
            'partition_seed': partition_seed,
            'model_seed': model_seed,
            'device': str(device),
            'local_search_learning_rate': local_search_learning_rate,
            'local_search_batch_size': local_search_batch_size,
            'local_search_trajectories': local_search_trajectories,
            'local_search_gradient_norm_cap': local_search_gradient_norm_cap,
            'client_train_samples': len(datasets.client_train),
            'root_samples': len(datasets.root),
        },
    )
    parameters = dict(result.parameter_snapshot)
    parameters.update({
        'dataset': 'MNIST',
        'client_train_samples': len(datasets.client_train),
        'root_samples': len(datasets.root),
        'test_samples': len(datasets.test),
        'partition_seed': partition_seed,
        'model_seed': model_seed,
        'defender_obs_dim': defender_obs_dim,
        'attacker_obs_dim': attacker_obs_dim,
        'device': str(device),
        'local_search_gradient_norm_cap': local_search_gradient_norm_cap,
        'data_provenance': {
            key: getattr(datasets.provenance, key)
            for key in datasets.provenance.__dataclass_fields__
        },
    })
    return ScaledEvidenceResult(
        result.training,
        result.scientific,
        MappingProxyType(parameters),
        result.query_seeds,
        protocol='paper-mnist-scaled-meta-sg-evidence-v1',
    )
