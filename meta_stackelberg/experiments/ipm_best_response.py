"""Real support and frozen-query execution for an IPM follower response."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.utils.data import Subset, TensorDataset

from meta_stackelberg.agents import IPMScalePolicy, IPMScalePolicySnapshot
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
from meta_stackelberg.security.defenses.actions import DefenseAction
from meta_stackelberg.security.defenses.clipped_trimmed_mean import ClippedTrimmedMean
from meta_stackelberg.security.engine.attack_round_engine import AttackRoundEngine
from meta_stackelberg.security.population import FixedMaliciousPopulation
from meta_stackelberg.security.types import AttackKnowledge
from meta_stackelberg.stackelberg import (
    BestResponseResult,
    CandidateIPMBestResponseSolver,
    DefenderCommitment,
    FixedIPMResponseOracle,
)


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


@dataclass(frozen=True)
class CommitmentResponseCurve:
    commitment: DefenderCommitment
    response: BestResponseResult
    initial_query: FrozenResponseEvaluation
    adapted_query: FrozenResponseEvaluation
    fixed_queries: tuple[tuple[float, FrozenResponseEvaluation], ...]


@dataclass(frozen=True)
class E3IPMResponseCurveResult:
    passed: bool
    commitments: tuple[CommitmentResponseCurve, ...]
    failed_requirements: tuple[str, ...]
    minimum_query_improvement: float = 1e-4


@dataclass(frozen=True)
class CommitmentOracleRegret:
    commitment_id: str
    adapted_scale: float
    initial_harm: float
    adapted_harm: float
    oracle_harm: float
    oracle_scales: tuple[float, ...]
    oracle_regret: float
    query_harm_span: float
    query_plateau: bool
    strictly_improved: bool


@dataclass(frozen=True)
class E3OracleRegretGateResult:
    passed: bool
    commitments: tuple[CommitmentOracleRegret, ...]
    independently_improved_commitments: tuple[str, ...]
    leader_dependent_response: bool
    failed_requirements: tuple[str, ...]


def evaluate_e3_oracle_regret_gate(
    curve_result: E3IPMResponseCurveResult,
    *,
    required_commitment_ids: tuple[str, ...],
    required_candidate_scales: tuple[float, ...],
    oracle_regret_tolerance: float = 1e-6,
    plateau_tolerance: float = 1e-6,
    strict_improvement_margin: float = 1e-4,
) -> E3OracleRegretGateResult:
    required_ids = _required_ids(required_commitment_ids)
    candidates = _candidate_scales(required_candidate_scales)
    regret_tolerance = _nonnegative_finite(oracle_regret_tolerance, 'oracle regret tolerance')
    plateau_limit = _nonnegative_finite(plateau_tolerance, 'plateau tolerance')
    improvement_margin = _nonnegative_finite(strict_improvement_margin, 'strict improvement margin')
    curve_by_id = {curve.commitment.commitment_id: curve for curve in curve_result.commitments}
    if len(curve_by_id) != len(curve_result.commitments) or set(curve_by_id) != set(required_ids):
        raise ValueError('commitment coverage does not match required ids')
    diagnostics = []
    for commitment_id in required_ids:
        curve = curve_by_id[commitment_id]
        fingerprint = curve.commitment.policy_fingerprint
        _validate_query_evaluation(curve.initial_query, fingerprint, 'initial')
        _validate_query_evaluation(curve.adapted_query, fingerprint, 'adapted')
        query_seeds = tuple(record.seed for record in curve.initial_query.records)
        if tuple(record.seed for record in curve.adapted_query.records) != query_seeds:
            raise ValueError('initial and adapted query seed sets do not match')
        adapted_scale = curve.response.adapted_follower_snapshot.scale
        if adapted_scale not in candidates:
            raise ValueError('adapted scale is absent from required candidate set')
        if curve.adapted_query.follower_snapshot.scale != adapted_scale:
            raise ValueError('adapted query scale does not match follower snapshot')
        fixed = dict(curve.fixed_queries)
        if len(fixed) != len(curve.fixed_queries) or set(fixed) != set(candidates):
            raise ValueError(f'candidate coverage does not match for commitment {commitment_id}')
        for scale, evaluation in curve.fixed_queries:
            _validate_query_evaluation(evaluation, fingerprint, 'candidate')
            if evaluation.follower_snapshot.scale != scale or any(
                record.scale != scale for record in evaluation.records
            ):
                raise ValueError('candidate scale does not match frozen query evaluation')
            if tuple(record.seed for record in evaluation.records) != query_seeds:
                raise ValueError('candidate query seed set does not match initial query')
        initial_harm = _finite_harm(curve.initial_query, 'initial harm')
        adapted_harm = _finite_harm(curve.adapted_query, 'adapted harm')
        harms = tuple((scale, _finite_harm(fixed[scale], 'candidate harm')) for scale in candidates)
        oracle_harm = max(harm for _, harm in harms)
        oracle_scales = tuple(scale for scale, harm in harms if harm == oracle_harm)
        span = oracle_harm - min(harm for _, harm in harms)
        regret = oracle_harm - adapted_harm
        diagnostics.append(CommitmentOracleRegret(
            commitment_id=commitment_id,
            adapted_scale=curve.response.adapted_follower_snapshot.scale,
            initial_harm=initial_harm,
            adapted_harm=adapted_harm,
            oracle_harm=oracle_harm,
            oracle_scales=oracle_scales,
            oracle_regret=regret,
            query_harm_span=span,
            query_plateau=span <= plateau_limit,
            strictly_improved=adapted_harm - initial_harm > improvement_margin,
        ))
    proxy_signatures = {
        (
            curve.response.adapted_follower_snapshot.scale,
            tuple(record.mean_scalar for record in curve.response.candidate_records),
        )
        for curve in curve_result.commitments
    }
    leader_dependent = len(required_ids) < 2 or len(proxy_signatures) > 1
    improved = tuple(item.commitment_id for item in diagnostics if item.strictly_improved)
    failed = []
    if any(item.oracle_regret > regret_tolerance for item in diagnostics):
        failed.append('one or more follower responses exceed oracle regret tolerance')
    if not improved:
        failed.append('no commitment strictly improves over initial follower')
    if not leader_dependent:
        failed.append('follower response does not depend on leader commitment')
    return E3OracleRegretGateResult(
        passed=not failed,
        commitments=tuple(diagnostics),
        independently_improved_commitments=improved,
        leader_dependent_response=leader_dependent,
        failed_requirements=tuple(failed),
    )


def _required_ids(values: tuple[str, ...]) -> tuple[str, ...]:
    result = tuple(values)
    if not result or any(not isinstance(value, str) or not value for value in result):
        raise ValueError('required commitment ids must be non-empty strings')
    if len(result) != len(set(result)):
        raise ValueError('required commitment ids must be unique')
    return result


def _candidate_scales(values: tuple[float, ...]) -> tuple[float, ...]:
    result = tuple(sorted(_positive_scale(value) for value in values))
    if not result or len(result) != len(set(result)):
        raise ValueError('required candidate scales must be non-empty and unique')
    return result


def _positive_scale(value: float) -> float:
    if isinstance(value, bool):
        raise TypeError('candidate scale must be a real number')
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError('candidate scale must be finite and positive')
    return result


def _nonnegative_finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f'{name} must be a real number')
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f'{name} must be finite and non-negative')
    return result


def _finite_harm(evaluation: FrozenResponseEvaluation, name: str) -> float:
    result = evaluation.mean_harm
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite')
    return result


def _validate_query_evaluation(
    evaluation: FrozenResponseEvaluation,
    expected_fingerprint: str,
    name: str,
) -> None:
    if evaluation.commitment_fingerprint != expected_fingerprint:
        raise ValueError(f'{name} query commitment fingerprint does not match')
    if not evaluation.records:
        raise ValueError(f'{name} query records must not be empty')
    seeds = tuple(record.seed for record in evaluation.records)
    if len(seeds) != len(set(seeds)):
        raise ValueError(f'{name} query seeds must be unique')
    scale = evaluation.follower_snapshot.scale
    if any(record.scale != scale for record in evaluation.records):
        raise ValueError(f'{name} query record scale does not match follower snapshot')


def run_e3_ipm_response_curve() -> E3IPMResponseCurveResult:
    declarations = (
        ('weak', DefenseAction(10.0, 0.0)),
        ('strong-clip', DefenseAction(0.1, 0.0)),
        ('trim', DefenseAction(10.0, 0.4)),
    )
    curves = []
    failed = []
    for name, action in declarations:
        commitment = DefenderCommitment.create(name, action)
        response = CandidateIPMBestResponseSolver(CANDIDATE_SCALES).solve(
            commitment,
            IPMScalePolicy(1.0),
            SUPPORT_SEEDS,
            make_ipm_support_feedback,
        )
        initial_query = evaluate_frozen_ipm_response(
            commitment, IPMScalePolicy(1.0).snapshot(), QUERY_SEEDS,
        )
        adapted_query = evaluate_frozen_ipm_response(
            commitment, response.adapted_follower_snapshot, QUERY_SEEDS,
        )
        fixed_queries = tuple(
            (
                scale,
                evaluate_frozen_ipm_response(
                    commitment,
                    FixedIPMResponseOracle(scale).solve(
                        commitment, IPMScalePolicy(1.0), (), make_ipm_support_feedback,
                    ).adapted_follower_snapshot,
                    QUERY_SEEDS,
                ),
            )
            for scale in CANDIDATE_SCALES
        )
        if adapted_query.mean_harm - initial_query.mean_harm <= 1e-4:
            failed.append(f'adapted query harm did not improve for {name}')
        curves.append(CommitmentResponseCurve(
            commitment, response, initial_query, adapted_query, fixed_queries,
        ))
    proxy_curves = {
        tuple(record.mean_scalar for record in curve.response.candidate_records)
        for curve in curves
    }
    if len(proxy_curves) < 2:
        failed.append('follower response does not depend on leader commitment')
    return E3IPMResponseCurveResult(not failed, tuple(curves), tuple(failed))


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
