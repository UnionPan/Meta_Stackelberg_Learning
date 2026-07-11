from itertools import product

import numpy as np
import torch
from torch.utils.data import Subset, TensorDataset

from meta_stackelberg.core.random_state import RandomSnapshot, RandomSource
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
from meta_stackelberg.experiments.defense_response_surface import (
    ClipControllabilityGate,
    ClipResponsePoint,
    RawClipObservation,
    evaluate_clip_response_surface,
)
from meta_stackelberg.security.attacks.delta_reversal import DeltaReversalAttack
from meta_stackelberg.security.defenses.clipping import ClippedAggregator
from meta_stackelberg.security.engine.attack_round_engine import AttackRoundEngine
from meta_stackelberg.security.population import FixedMaliciousPopulation
from meta_stackelberg.security.training import ScopedLocalTrainer
from meta_stackelberg.security.types import AttackKnowledge


SEEDS = (301, 302, 303)
RADII = (0.001, 0.003, 0.01, 0.03, 10.0)
MALICIOUS_CLIENTS = frozenset({0, 1, 2})


def _dataset(offset_start: float) -> TensorDataset:
    offsets = torch.linspace(offset_start, offset_start + 1.0, steps=60)
    negative = torch.stack((-1.0 - offsets, -0.5 - 0.25 * offsets), dim=1)
    positive = -negative
    return TensorDataset(
        torch.cat((negative, positive), dim=0),
        torch.cat((torch.zeros(60, dtype=torch.long), torch.ones(60, dtype=torch.long))),
    )


def _model_factory() -> torch.nn.Module:
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    return model


def _run_observation(
    seed: int,
    radius: float,
    branch: str,
    *,
    use_fedavg: bool = False,
) -> RawClipObservation:
    train_dataset = _dataset(0.0)
    held_out_dataset = _dataset(0.2)
    partitions = iid_partition(len(train_dataset), 6, RandomSource(77))
    client_datasets = {
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

    def trainer(client_ids: frozenset[int] | None = None) -> TorchLocalTrainer:
        datasets = client_datasets
        if client_ids is not None:
            datasets = {client_id: client_datasets[client_id] for client_id in client_ids}
        return TorchLocalTrainer(
            model_factory=_model_factory,
            client_datasets=datasets,
            codec=codec,
            learning_rate=0.2,
            local_epochs=1,
            batch_size=8,
        )

    aggregator = FedAvg() if use_fedavg else ClippedAggregator(FedAvg(), radius)
    sampler = UniformClientSampler(num_clients=6)
    if branch == 'clean':
        engine = RoundEngine(
            sampler=sampler,
            trainer=trainer(),
            aggregator=aggregator,
            server_optimizer=ServerSGD(),
        )
    else:
        benign_ids = frozenset(set(client_datasets) - set(MALICIOUS_CLIENTS))
        malicious_trainer = ScopedLocalTrainer(trainer(MALICIOUS_CLIENTS), MALICIOUS_CLIENTS)
        engine = AttackRoundEngine(
            sampler=sampler,
            benign_trainer=trainer(benign_ids),
            malicious_generator=DeltaReversalAttack(trainer=malicious_trainer, budget=2.0),
            population=FixedMaliciousPopulation(MALICIOUS_CLIENTS),
            knowledge=AttackKnowledge(allows_global_model=True, allows_local_data=True),
            aggregator=aggregator,
            server_optimizer=ServerSGD(),
        )
    trajectory = EpisodeRunner(engine).run(
        EpisodeSpec(
            task_id='e2.1-clip-response-surface',
            horizon=8,
            sample_size=4,
            server_lr=1.0,
            initial_state=state,
        ),
        source,
    )
    metrics = ClassificationEvaluator(
        model_factory=_model_factory,
        dataset=held_out_dataset,
        codec=codec,
        batch_size=24,
    ).evaluate(trajectory.final_state.global_model)
    aggregate_norms = tuple(
        float(np.linalg.norm(transition.aggregate_delta.vector().astype(np.float64)))
        for transition in trajectory.transitions
    )
    if use_fedavg:
        clipped_fractions = (0.0,) * len(trajectory.transitions)
    else:
        clipped_fractions = tuple(
            ClippedAggregator(FedAvg(), radius).summarize(
                transition.benign_updates + transition.malicious_updates
            ).clipped_client_fraction
            for transition in trajectory.transitions
        )
    return RawClipObservation(
        seed=seed,
        radius=radius,
        branch=branch,
        final_clean_loss=metrics.loss,
        final_clean_accuracy=metrics.accuracy,
        aggregate_norms=aggregate_norms,
        clipped_client_fractions=clipped_fractions,
        sampled_clients=tuple(step.sampled_clients for step in trajectory.transitions),
        final_random_snapshot=trajectory.final_state.random_snapshot,
    )


def _point(surface, seed: int, radius: float, branch: str) -> ClipResponsePoint:
    return next(
        point
        for point in surface.points
        if (
            point.observation.seed,
            point.observation.radius,
            point.observation.branch,
        ) == (seed, radius, branch)
    )


def _assert_snapshots_equal(left: RandomSnapshot, right: RandomSnapshot) -> None:
    assert left.python_state == right.python_state
    _assert_nested_equal(left.numpy_state, right.numpy_state)
    assert torch.equal(left.torch_cpu_state, right.torch_cpu_state)


def _assert_nested_equal(left, right) -> None:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        np.testing.assert_array_equal(left, right)
        return
    if isinstance(left, dict) or isinstance(right, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
        return
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right):
            _assert_nested_equal(left_value, right_value)
        return
    assert left == right


def _assert_observations_exact(left: RawClipObservation, right: RawClipObservation) -> None:
    assert left.seed == right.seed
    assert left.radius == right.radius
    assert left.branch == right.branch
    assert left.final_clean_loss == right.final_clean_loss
    assert left.final_clean_accuracy == right.final_clean_accuracy
    assert left.aggregate_norms == right.aggregate_norms
    assert left.clipped_client_fractions == right.clipped_client_fractions
    assert left.sampled_clients == right.sampled_clients
    _assert_snapshots_equal(left.final_random_snapshot, right.final_random_snapshot)


def test_clip_radius_has_a_matched_deterministic_non_flat_response_surface() -> None:
    arguments = {
        'radii': RADII,
        'seeds': SEEDS,
        'observation_factory': _run_observation,
        'task_fingerprint': 'delta-reversal-fedavg-fixture-v1',
        'evaluation_protocol': 'held-out-clean-loss-v1',
    }
    surface = evaluate_clip_response_surface(**arguments)
    replay = evaluate_clip_response_surface(**arguments)

    assert surface.reference_radius == RADII[-1]
    assert ClipControllabilityGate().evaluate(surface)
    assert len(surface.points) == len(SEEDS) * len(RADII) * 2

    for left, right in zip(surface.points, replay.points):
        assert left.attack_harm == right.attack_harm
        assert left.defense_cost == right.defense_cost
        _assert_observations_exact(left.observation, right.observation)

    for seed, radius in product(SEEDS, RADII):
        clean = _point(surface, seed, radius, 'clean').observation
        attack = _point(surface, seed, radius, 'attack').observation
        assert clean.sampled_clients == attack.sampled_clients
        _assert_snapshots_equal(clean.final_random_snapshot, attack.final_random_snapshot)

    for seed in SEEDS:
        for branch in ('clean', 'attack'):
            reference = _point(surface, seed, RADII[-1], branch).observation
            fedavg = _run_observation(seed, RADII[-1], branch, use_fedavg=True)
            assert reference.final_clean_loss == fedavg.final_clean_loss
            assert reference.final_clean_accuracy == fedavg.final_clean_accuracy
            assert reference.aggregate_norms == fedavg.aggregate_norms
            assert reference.sampled_clients == fedavg.sampled_clients
            _assert_snapshots_equal(reference.final_random_snapshot, fedavg.final_random_snapshot)
