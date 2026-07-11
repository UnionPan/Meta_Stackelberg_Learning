from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Subset, TensorDataset

from meta_stackelberg.core.random_state import RandomSnapshot, RandomSource
from meta_stackelberg.federated.aggregation.coordinate_median import CoordinateMedian
from meta_stackelberg.federated.aggregation.fedavg import FedAvg
from meta_stackelberg.federated.clients.sampling import UniformClientSampler
from meta_stackelberg.federated.clients.trainer import TorchLocalTrainer
from meta_stackelberg.federated.data.partitioning import iid_partition
from meta_stackelberg.federated.engine.round_engine import RoundEngine
from meta_stackelberg.federated.engine.server_optimizer import ServerSGD
from meta_stackelberg.federated.episode import EpisodeRunner, EpisodeSpec
from meta_stackelberg.federated.evaluation.classification import ClassificationEvaluator
from meta_stackelberg.federated.models.parameters import TorchParameterCodec
from meta_stackelberg.federated.types import RoundState
from meta_stackelberg.experiments.defense_response_matrix import (
    DefenseGridPoint,
    MatrixGateThresholds,
    RawDefenseObservation,
    evaluate_defense_response_matrix,
    evaluate_task_matrix_gate,
)
from meta_stackelberg.security.attacks.delta_reversal import DeltaReversalAttack
from meta_stackelberg.security.attacks.ipm import IPMAttack
from meta_stackelberg.security.attacks.lmp import LMPAttack
from meta_stackelberg.security.defenses.clipped_trimmed_mean import ClippedTrimmedMean
from meta_stackelberg.security.engine.attack_round_engine import AttackRoundEngine
from meta_stackelberg.security.population import FixedMaliciousPopulation
from meta_stackelberg.security.training import ScopedLocalTrainer
from meta_stackelberg.security.types import AttackKnowledge


SEEDS_BY_TASK = {
    'delta-reversal': (301, 302, 303),
    'ipm': (501, 502, 503),
    'lmp': (501, 502, 503),
}
CLIP_RADII = (0.01, 0.1, 10.0)
TRIM_RATIOS = (0.0, 0.2, 0.4)
MALICIOUS_CLIENTS = frozenset({0, 1, 2})
THRESHOLDS = MatrixGateThresholds(
    aggregate_span=1e-6,
    metric_span=1e-4,
    active_cell_delta=1e-5,
    min_active_cell_ratio=0.25,
)


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


@dataclass(frozen=True)
class _Run:
    observation: RawDefenseObservation
    trajectory: object


def _run(
    task: str,
    seed: int,
    branch: str,
    point: DefenseGridPoint,
    *,
    reference: bool,
) -> _Run:
    train_dataset = _dataset(0.0)
    held_out = _dataset(0.2)
    partitions = iid_partition(len(train_dataset), 6, RandomSource(77))
    datasets = {
        client_id: Subset(train_dataset, indices)
        for client_id, indices in enumerate(partitions)
    }
    codec = TorchParameterCodec()
    source = RandomSource(seed)
    state = RoundState(
        round_index=0,
        global_model=codec.capture(_model_factory()),
        random_snapshot=source.capture(),
    )

    def trainer(client_ids=None) -> TorchLocalTrainer:
        selected = datasets if client_ids is None else {
            client_id: datasets[client_id] for client_id in client_ids
        }
        return TorchLocalTrainer(
            model_factory=_model_factory,
            client_datasets=selected,
            codec=codec,
            learning_rate=0.2,
            local_epochs=1,
            batch_size=8,
        )

    if reference:
        aggregator = CoordinateMedian() if task == 'lmp' else FedAvg()
    else:
        aggregator = ClippedTrimmedMean(point.clip_radius, point.trim_ratio)
    sampler = UniformClientSampler(num_clients=6)
    if branch == 'clean':
        engine = RoundEngine(
            sampler=sampler,
            trainer=trainer(),
            aggregator=aggregator,
            server_optimizer=ServerSGD(),
        )
    else:
        benign_ids = frozenset(set(datasets) - set(MALICIOUS_CLIENTS))
        if task == 'delta-reversal':
            generator = DeltaReversalAttack(
                trainer=ScopedLocalTrainer(trainer(MALICIOUS_CLIENTS), MALICIOUS_CLIENTS),
                budget=2.0,
            )
            knowledge = AttackKnowledge(allows_global_model=True, allows_local_data=True)
        elif task == 'ipm':
            generator = IPMAttack(
                scale=3.0,
                num_examples_by_client={client_id: len(datasets[client_id]) for client_id in MALICIOUS_CLIENTS},
            )
            knowledge = AttackKnowledge(allows_benign_updates=True)
        elif task == 'lmp':
            generator = LMPAttack(
                scale=5.0,
                num_examples_by_client={client_id: len(datasets[client_id]) for client_id in MALICIOUS_CLIENTS},
            )
            knowledge = AttackKnowledge(allows_global_model=True, allows_benign_updates=True)
        else:
            raise ValueError(task)
        engine = AttackRoundEngine(
            sampler=sampler,
            benign_trainer=trainer(benign_ids),
            malicious_generator=generator,
            population=FixedMaliciousPopulation(MALICIOUS_CLIENTS),
            knowledge=knowledge,
            aggregator=aggregator,
            server_optimizer=ServerSGD(),
        )
    trajectory = EpisodeRunner(engine).run(
        EpisodeSpec(
            task_id=task,
            horizon=8,
            sample_size=4,
            server_lr=1.0,
            initial_state=state,
        ),
        source,
    )
    metrics = ClassificationEvaluator(
        model_factory=_model_factory,
        dataset=held_out,
        codec=codec,
        batch_size=24,
    ).evaluate(trajectory.final_state.global_model)
    if reference:
        clipped = (0.0,) * 8
        trim_counts = (0,) * 8
        retained = (4,) * 8
    else:
        summaries = tuple(
            ClippedTrimmedMean(point.clip_radius, point.trim_ratio).summarize(
                transition.benign_updates + transition.malicious_updates
            )
            for transition in trajectory.transitions
        )
        clipped = tuple(summary.clipping.clipped_client_fraction for summary in summaries)
        trim_counts = tuple(summary.trimming.per_tail_trim_count for summary in summaries)
        retained = tuple(summary.trimming.retained_count for summary in summaries)
    return _Run(RawDefenseObservation(
        task_id=task,
        seed=seed,
        grid_point=point,
        branch=branch,
        final_clean_loss=metrics.loss,
        final_clean_accuracy=metrics.accuracy,
        attack_metric=metrics.loss,
        aggregate_norms=tuple(
            float(np.linalg.norm(step.aggregate_delta.vector().astype(np.float64)))
            for step in trajectory.transitions
        ),
        clipped_fractions=clipped,
        per_tail_trim_counts=trim_counts,
        retained_counts=retained,
        sampled_clients=tuple(step.sampled_clients for step in trajectory.transitions),
        final_random_snapshot=trajectory.final_state.random_snapshot,
        final_model_vector=trajectory.final_state.global_model.vector(),
    ), trajectory)


def build_untargeted_matrix(task: str):
    reference_point = DefenseGridPoint(CLIP_RADII[-1], 0.0)
    return evaluate_defense_response_matrix(
        task_id=task,
        seeds=SEEDS_BY_TASK[task],
        clip_radii=CLIP_RADII,
        trim_ratios=TRIM_RATIOS,
        observation_factory=lambda seed, point, branch: _run(
            task, seed, branch, point, reference=False
        ).observation,
        reference_factory=lambda seed, branch: _run(
            task, seed, branch, reference_point, reference=True
        ).observation,
        evaluation_protocol='held-out-clean-loss-v1',
        attack_metric_direction='higher_is_worse',
    )


def _assert_snapshots_equal(left: RandomSnapshot, right: RandomSnapshot) -> None:
    assert left.python_state == right.python_state
    assert left.numpy_state == right.numpy_state
    assert torch.equal(left.torch_cpu_state, right.torch_cpu_state)


def _assert_matrices_exact(left, right) -> None:
    assert left.task_id == right.task_id
    assert len(left.points) == len(right.points)
    for first, second in zip(left.points, right.points):
        assert first.attack_harm == second.attack_harm
        assert first.clip_cost == second.clip_cost
        assert first.trim_cost == second.trim_cost
        a, b = first.observation, second.observation
        assert a.final_clean_loss == b.final_clean_loss
        assert a.final_clean_accuracy == b.final_clean_accuracy
        assert a.attack_metric == b.attack_metric
        assert a.aggregate_norms == b.aggregate_norms
        assert a.clipped_fractions == b.clipped_fractions
        assert a.per_tail_trim_counts == b.per_tail_trim_counts
        assert a.retained_counts == b.retained_counts
        assert a.sampled_clients == b.sampled_clients
        np.testing.assert_array_equal(a.final_model_vector, b.final_model_vector)
        _assert_snapshots_equal(a.final_random_snapshot, b.final_random_snapshot)


def test_untargeted_attack_matrices_are_complete_replayable_and_controllable() -> None:
    for task in ('delta-reversal', 'ipm', 'lmp'):
        matrix = build_untargeted_matrix(task)
        replay = build_untargeted_matrix(task)
        gate = evaluate_task_matrix_gate(matrix, THRESHOLDS)

        assert len(matrix.points) == len(SEEDS_BY_TASK[task]) * 3 * 3 * 2
        assert len(matrix.references) == len(SEEDS_BY_TASK[task]) * 2
        _assert_matrices_exact(matrix, replay)
        assert gate.passed
        assert gate.clip_dimension_active
        assert gate.trim_dimension_active
        for seed in SEEDS_BY_TASK[task]:
            clean, attack = (
                observation
                for observation in matrix.references
                if observation.seed == seed
            )
            assert clean.final_clean_accuracy == 1.0
            assert attack.final_clean_loss > clean.final_clean_loss
            assert clean.sampled_clients == attack.sampled_clients
            _assert_snapshots_equal(
                clean.final_random_snapshot,
                attack.final_random_snapshot,
            )
