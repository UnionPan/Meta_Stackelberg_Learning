import numpy as np
import torch
from torch.utils.data import Subset, TensorDataset

from meta_stackelberg.core.random_state import RandomSnapshot, RandomSource
from meta_stackelberg.evaluation.targeted import TargetedAttackEvaluator
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
from meta_stackelberg.security.attacks.backdoor import BackdoorLocalUpdateGenerator
from meta_stackelberg.security.attacks.dba import (
    DistributedBackdoorUpdateGenerator,
    DistributedTriggerPlan,
)
from meta_stackelberg.security.data.poisoning import SourceTargetPoisonedDataset
from meta_stackelberg.security.data.trigger import PatchTrigger
from meta_stackelberg.security.defenses.clipped_trimmed_mean import ClippedTrimmedMean
from meta_stackelberg.security.engine.attack_round_engine import AttackRoundEngine
from meta_stackelberg.security.population import FixedMaliciousPopulation
from meta_stackelberg.security.training import ScopedLocalTrainer
from meta_stackelberg.security.types import AttackKnowledge


SEEDS_BY_TASK = {'bfl': (401, 402, 403), 'dba': (601, 602, 603)}
CLIP_RADII = (0.01, 0.1, 10.0)
TRIM_RATIOS = (0.0, 0.2, 0.4)
MALICIOUS_CLIENTS = frozenset({0, 1, 2})
BFL_TRIGGER = PatchTrigger(row=6, column=6, height=2, width=2, value=5.0)
DBA_SUB_TRIGGERS = (
    PatchTrigger(row=6, column=0, height=2, width=2, value=1.5),
    PatchTrigger(row=6, column=3, height=2, width=2, value=1.5),
    PatchTrigger(row=6, column=6, height=2, width=2, value=1.5),
)
DBA_PLAN = DistributedTriggerPlan(
    DBA_SUB_TRIGGERS,
    {client_id: client_id for client_id in MALICIOUS_CLIENTS},
    MALICIOUS_CLIENTS,
)
THRESHOLDS = MatrixGateThresholds(1e-6, 1e-4, 1e-5, 0.25)


def _dataset(offset: float) -> TensorDataset:
    levels = torch.linspace(0.5 + offset, 1.0 + offset, 60)
    source = -levels[:, None, None, None].expand(-1, 1, 8, 8).clone()
    target = levels[:, None, None, None].expand(-1, 1, 8, 8).clone()
    return TensorDataset(
        torch.cat((source, target)),
        torch.cat((torch.zeros(60), torch.ones(60))).long(),
    )


def _model_factory() -> torch.nn.Module:
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(64, 2))
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].bias.zero_()
    return model


def _run(
    task: str,
    seed: int,
    branch: str,
    point: DefenseGridPoint,
    *,
    reference: bool,
) -> RawDefenseObservation:
    train = _dataset(0.0)
    held_out = _dataset(0.1)
    partition_seed = 91 if task == 'bfl' else 92
    partitions = iid_partition(len(train), 6, RandomSource(partition_seed))
    clean_datasets = {
        client_id: Subset(train, indices)
        for client_id, indices in enumerate(partitions)
    }
    trigger_for_client = (
        (lambda client_id: BFL_TRIGGER)
        if task == 'bfl'
        else DBA_PLAN.trigger_for
    )
    poison_seed_base = 10_000 if task == 'bfl' else 20_000
    poisoned_datasets = {
        client_id: SourceTargetPoisonedDataset(
            dataset=clean_datasets[client_id],
            trigger=trigger_for_client(client_id),
            source_class=0,
            target_class=1,
            poison_fraction=1.0,
            rng=RandomSource(poison_seed_base + client_id),
        )
        for client_id in MALICIOUS_CLIENTS
    }
    codec = TorchParameterCodec()
    source = RandomSource(seed)
    state = RoundState(
        round_index=0,
        global_model=codec.capture(_model_factory()),
        random_snapshot=source.capture(),
    )

    def trainer(datasets) -> TorchLocalTrainer:
        return TorchLocalTrainer(
            model_factory=_model_factory,
            client_datasets=datasets,
            codec=codec,
            learning_rate=0.08,
            local_epochs=2,
            batch_size=8,
        )

    aggregator = FedAvg() if reference else ClippedTrimmedMean(
        point.clip_radius,
        point.trim_ratio,
    )
    sampler = UniformClientSampler(num_clients=6)
    if branch == 'clean':
        engine = RoundEngine(
            sampler=sampler,
            trainer=trainer(clean_datasets),
            aggregator=aggregator,
            server_optimizer=ServerSGD(),
        )
    else:
        benign_ids = frozenset(set(clean_datasets) - set(MALICIOUS_CLIENTS))
        malicious_trainer = ScopedLocalTrainer(
            trainer(poisoned_datasets),
            MALICIOUS_CLIENTS,
        )
        if task == 'bfl':
            generator = BackdoorLocalUpdateGenerator(
                trainer=malicious_trainer,
                source_class=0,
                target_class=1,
                poison_fraction=1.0,
            )
        else:
            generator = DistributedBackdoorUpdateGenerator(
                trainer=malicious_trainer,
                plan=DBA_PLAN,
                source_class=0,
                target_class=1,
                poison_fraction=1.0,
            )
        engine = AttackRoundEngine(
            sampler=sampler,
            benign_trainer=trainer({
                client_id: clean_datasets[client_id] for client_id in benign_ids
            }),
            malicious_generator=generator,
            population=FixedMaliciousPopulation(MALICIOUS_CLIENTS),
            knowledge=AttackKnowledge(allows_global_model=True, allows_local_data=True),
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
    final_model = trajectory.final_state.global_model
    clean_metrics = ClassificationEvaluator(
        model_factory=_model_factory,
        dataset=held_out,
        codec=codec,
        batch_size=24,
    ).evaluate(final_model)

    def asr(trigger) -> float:
        return TargetedAttackEvaluator(
            model_factory=_model_factory,
            dataset=held_out,
            codec=codec,
            trigger=trigger,
            source_class=0,
            target_class=1,
            batch_size=24,
        ).evaluate(final_model).attack_success_rate

    evaluation_trigger = BFL_TRIGGER if task == 'bfl' else DBA_PLAN.full_trigger
    attack_metric = asr(evaluation_trigger)
    components = {}
    if task == 'dba':
        components = {
            f'sub_trigger_{index}_asr': asr(trigger)
            for index, trigger in enumerate(DBA_SUB_TRIGGERS)
        }
        components['full_trigger_asr'] = attack_metric
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
    return RawDefenseObservation(
        task_id=task,
        seed=seed,
        grid_point=point,
        branch=branch,
        final_clean_loss=clean_metrics.loss,
        final_clean_accuracy=clean_metrics.accuracy,
        attack_metric=attack_metric,
        aggregate_norms=tuple(
            float(np.linalg.norm(step.aggregate_delta.vector().astype(np.float64)))
            for step in trajectory.transitions
        ),
        clipped_fractions=clipped,
        per_tail_trim_counts=trim_counts,
        retained_counts=retained,
        sampled_clients=tuple(step.sampled_clients for step in trajectory.transitions),
        final_random_snapshot=trajectory.final_state.random_snapshot,
        final_model_vector=final_model.vector(),
        metric_components=components,
    )


def build_backdoor_matrix(task: str):
    reference_point = DefenseGridPoint(CLIP_RADII[-1], 0.0)
    return evaluate_defense_response_matrix(
        task_id=task,
        seeds=SEEDS_BY_TASK[task],
        clip_radii=CLIP_RADII,
        trim_ratios=TRIM_RATIOS,
        observation_factory=lambda seed, point, branch: _run(
            task, seed, branch, point, reference=False
        ),
        reference_factory=lambda seed, branch: _run(
            task, seed, branch, reference_point, reference=True
        ),
        evaluation_protocol='source-only-asr-v1',
        attack_metric_direction='higher_is_worse',
    )


def _assert_snapshots_equal(left: RandomSnapshot, right: RandomSnapshot) -> None:
    assert left.python_state == right.python_state
    assert left.numpy_state == right.numpy_state
    assert torch.equal(left.torch_cpu_state, right.torch_cpu_state)


def _assert_matrix_replay(left, right) -> None:
    assert len(left.points) == len(right.points)
    for first, second in zip(left.points, right.points):
        assert first.attack_harm == second.attack_harm
        a, b = first.observation, second.observation
        assert a.final_clean_loss == b.final_clean_loss
        assert a.final_clean_accuracy == b.final_clean_accuracy
        assert a.attack_metric == b.attack_metric
        assert a.metric_components == b.metric_components
        assert a.aggregate_norms == b.aggregate_norms
        assert a.clipped_fractions == b.clipped_fractions
        assert a.per_tail_trim_counts == b.per_tail_trim_counts
        assert a.retained_counts == b.retained_counts
        assert a.sampled_clients == b.sampled_clients
        np.testing.assert_array_equal(a.final_model_vector, b.final_model_vector)
        _assert_snapshots_equal(a.final_random_snapshot, b.final_random_snapshot)


def test_backdoor_matrices_are_complete_replayable_and_controllable() -> None:
    for task in ('bfl', 'dba'):
        matrix = build_backdoor_matrix(task)
        replay = build_backdoor_matrix(task)
        gate = evaluate_task_matrix_gate(matrix, THRESHOLDS)

        assert len(matrix.points) == len(SEEDS_BY_TASK[task]) * 3 * 3 * 2
        assert len(matrix.references) == len(SEEDS_BY_TASK[task]) * 2
        _assert_matrix_replay(matrix, replay)
        assert gate.passed
        assert gate.clip_dimension_active
        assert gate.trim_dimension_active
        for seed in SEEDS_BY_TASK[task]:
            clean, attack = (
                observation for observation in matrix.references if observation.seed == seed
            )
            assert clean.final_clean_accuracy == 1.0
            assert attack.attack_metric > clean.attack_metric
            assert attack.final_clean_accuracy >= 0.95
            assert clean.sampled_clients == attack.sampled_clients
            _assert_snapshots_equal(clean.final_random_snapshot, attack.final_random_snapshot)
            if task == 'dba':
                assert set(attack.metric_components) == {
                    'sub_trigger_0_asr',
                    'sub_trigger_1_asr',
                    'sub_trigger_2_asr',
                    'full_trigger_asr',
                }
