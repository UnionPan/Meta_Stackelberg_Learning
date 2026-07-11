"""Deterministic finite-candidate follower best response."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np

from meta_stackelberg.agents import IPMScalePolicy, IPMScalePolicySnapshot
from meta_stackelberg.feedback import SupportEpisodeFeedback
from meta_stackelberg.stackelberg.commitment import DefenderCommitment


class SupportFactory(Protocol):
    def __call__(
        self,
        commitment: DefenderCommitment,
        scale: float,
        seed: int,
    ) -> SupportEpisodeFeedback: ...


@dataclass(frozen=True)
class CandidateSupportRecord:
    scale: float
    seed_scalars: tuple[tuple[int, float], ...]
    mean_scalar: float


@dataclass(frozen=True)
class BestResponseResult:
    initial_follower_snapshot: IPMScalePolicySnapshot
    adapted_follower_snapshot: IPMScalePolicySnapshot
    candidate_records: tuple[CandidateSupportRecord, ...]
    training_steps: int
    leader_fingerprint_before: str
    leader_fingerprint_after: str
    protocol: str = 'finite-candidate-ipm-response-v1'


class CandidateIPMBestResponseSolver:
    def __init__(self, candidates) -> None:
        checked = tuple(sorted(_positive_finite(value) for value in candidates))
        if not checked:
            raise ValueError('candidates must not be empty')
        if len(checked) != len(set(checked)):
            raise ValueError('candidates must be unique')
        self.candidates = checked

    def solve(
        self,
        commitment: DefenderCommitment,
        follower_initialization: IPMScalePolicy,
        support_seeds: tuple[int, ...],
        support_factory: SupportFactory,
    ) -> BestResponseResult:
        seeds = _checked_seeds(support_seeds)
        commitment.verify()
        before = commitment.policy_fingerprint
        initial = follower_initialization.snapshot()
        records = []
        for scale in self.candidates:
            values = []
            for seed in seeds:
                commitment.verify()
                feedback = support_factory(commitment, scale, seed)
                if not isinstance(feedback, SupportEpisodeFeedback):
                    raise TypeError('support factory must return support feedback')
                if not math.isfinite(feedback.scalar):
                    raise ValueError('support feedback scalar must be finite')
                values.append((seed, feedback.scalar))
                commitment.verify()
            records.append(CandidateSupportRecord(
                scale,
                tuple(values),
                float(np.mean([value for _, value in values])),
            ))
        selected = max(records, key=lambda record: (record.mean_scalar, -record.scale))
        adapted = follower_initialization.clone()
        adapted.restore(IPMScalePolicySnapshot(selected.scale))
        commitment.verify()
        return BestResponseResult(
            initial,
            adapted.snapshot(),
            tuple(records),
            len(self.candidates) * len(seeds),
            before,
            commitment.policy_fingerprint,
        )


def _positive_finite(value: float) -> float:
    if isinstance(value, bool):
        raise TypeError('candidate scale must be a real number')
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError('candidate scale must be finite and positive')
    return result


def _checked_seeds(values: tuple[int, ...]) -> tuple[int, ...]:
    seeds = tuple(sorted(values))
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError('support seeds must be non-empty integers')
    if len(seeds) != len(set(seeds)):
        raise ValueError('support seeds must be unique')
    return seeds
