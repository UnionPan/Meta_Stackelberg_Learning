"""Real support and frozen-query execution for an IPM follower response."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Subset, TensorDataset

from meta_stackelberg.agents import IPMScalePolicySnapshot
from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.feedback import IPMAttackerProxy, SupportEpisodeFeedback
from meta_stackelberg.federated.aggregation.fedavg import FedAvg
from meta_stackelberg.federated.clients.sampling import UniformClientSampler
from meta_stackelberg.federated.clients.trainer import TorchLocalTrainer
from meta_stackelberg.federated.data.partitioning import iid_partition
from meta_stackelberg.federated.engine.round_engine import RoundEngine
from meta_stackelberg.federated.engine.server_optimizer import ServerSGD
from meta_stackelberg.federated.episode import EpisodeRunner, EpisodeSpec, FederatedTrajectory
from meta_stackelberg.federated.evaluation.classification import ClassificationEvaluator
from meta_stackelberg.federated.models.parameters import TorchParameterCodec
from meta_stackelberg.federated.types import RoundState
from meta_stackelberg.security.attacks.ipm import IPMAttack
from meta_stackelberg.security.defenses.clipped_trimmed_mean import ClippedTrimmedMean
from meta_stackelberg.security.engine.attack_round_engine import AttackRoundEngine
from meta_stackelberg.security.population import FixedMaliciousPopulation
from meta_stackelberg.security.types import AttackKnowledge
from meta_stackelberg.stackelberg import DefenderCommitment


CANDIDATE_SCALES = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0)
SUPPORT_SEEDS = (701, 702, 703)
QUERY_SEEDS = (711, 712, 713)
_MALICIOUS = frozenset({0, 1, 2})


@dataclass(frozen=True)
class FrozenQueryRecord:
    seed: int
    scale: float
    clean_loss: float
    clean_accuracy: float
    attack_loss: float
    attack_accuracy: float
    sampled_clients: tuple[tuple[int, ...], ...]

    @property
    def harm(self) -> float:
        return self.attack_loss - self.clean_loss


@dataclass(frozen=True)
class FrozenResponseEvaluation:
    commitment_fingerprint: str
    follower_snapshot: IPMScalePolicySnapshot
    records: tuple[FrozenQueryRecord, ...]
    protocol: str = 'frozen-ipm-query-v1'

    @property
    def mean_harm(self) -> float:
        return sum(record.harm for record in self.records) / len(self.records)


def make_ipm_support_feedback(
    commitment: DefenderCommitment,
    scale: float,
    seed: int,
) -> SupportEpisodeFeedback:
    commitment.verify()
    trajectory, _ = _run_episode(commitment, scale, seed, attack=True, evaluate=False)
    result = IPMAttackerProxy().evaluate_trajectory(trajectory, scale)
    commitment.verify()
    return result


def evaluate_frozen_ipm_response(
    commitment: DefenderCommitment,
    follower_snapshot: IPMScalePolicySnapshot,
    query_seeds: tuple[int, ...],
) -> FrozenResponseEvaluation:
    if not query_seeds or len(query_seeds) != len(set(query_seeds)):
        raise ValueError('query seeds must be non-empty and unique')
    commitment.verify()
    records = []
    for seed in query_seeds:
        clean_trajectory, clean_metrics = _run_episode(
            commitment, follower_snapshot.scale, seed, attack=False, evaluate=True,
        )
        attack_trajectory, attack_metrics = _run_episode(
            commitment, follower_snapshot.scale, seed, attack=True, evaluate=True,
        )
        clean_samples = tuple(step.sampled_clients for step in clean_trajectory.transitions)
        attack_samples = tuple(step.sampled_clients for step in attack_trajectory.transitions)
        if clean_samples != attack_samples:
            raise ValueError('clean and attack query sampling do not match')
        records.append(FrozenQueryRecord(
            seed, follower_snapshot.scale,
            clean_metrics.loss, clean_metrics.accuracy,
            attack_metrics.loss, attack_metrics.accuracy,
            attack_samples,
        ))
    commitment.verify()
    return FrozenResponseEvaluation(
        commitment.policy_fingerprint,
        follower_snapshot,
        tuple(records),
    )


def _run_episode(commitment, scale, seed, *, attack, evaluate):
    train = _dataset(0.0)
    held_out = _dataset(0.2) if evaluate else None
    partitions = iid_partition(len(train), 6, RandomSource(77))
    datasets = {index: Subset(train, values) for index, values in enumerate(partitions)}
    codec = TorchParameterCodec()
    source = RandomSource(seed)
    state = RoundState(0, codec.capture(_model_factory()), source.capture())

    def trainer(client_ids):
        selected = {client_id: datasets[client_id] for client_id in client_ids}
        return TorchLocalTrainer(
            model_factory=_model_factory,
            client_datasets=selected,
            codec=codec,
            learning_rate=0.2,
            local_epochs=1,
            batch_size=8,
        )

    aggregator = ClippedTrimmedMean(
        commitment.action.clip_radius,
        commitment.action.trim_ratio,
    )
    sampler = UniformClientSampler(num_clients=6)
    if attack:
        engine = AttackRoundEngine(
            sampler=sampler,
            benign_trainer=trainer(frozenset(datasets) - _MALICIOUS),
            malicious_generator=IPMAttack(
                scale=scale,
                num_examples_by_client={client_id: len(datasets[client_id]) for client_id in _MALICIOUS},
            ),
            population=FixedMaliciousPopulation(_MALICIOUS),
            knowledge=AttackKnowledge(allows_benign_updates=True),
            aggregator=aggregator,
            server_optimizer=ServerSGD(),
        )
    else:
        engine = RoundEngine(
            sampler=sampler,
            trainer=trainer(frozenset(datasets)),
            aggregator=aggregator,
            server_optimizer=ServerSGD(),
        )
    trajectory = EpisodeRunner(engine).run(
        EpisodeSpec('ipm-best-response', 8, 4, 1.0, state),
        source,
    )
    metrics = None
    if held_out is not None:
        metrics = ClassificationEvaluator(
            model_factory=_model_factory,
            dataset=held_out,
            codec=codec,
            batch_size=24,
        ).evaluate(trajectory.final_state.global_model)
    return trajectory, metrics


def _dataset(offset_start: float) -> TensorDataset:
    offsets = torch.linspace(offset_start, offset_start + 1.0, steps=60)
    negative = torch.stack((-1.0 - offsets, -0.5 - 0.25 * offsets), dim=1)
    positive = -negative
    return TensorDataset(
        torch.cat((negative, positive), dim=0),
        torch.cat((torch.zeros(60), torch.ones(60))).long(),
    )


def _model_factory() -> torch.nn.Module:
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    return model
