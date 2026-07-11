"""Policy-level TD3 best response against a fully frozen Defender."""

from __future__ import annotations

from dataclasses import dataclass
import math

from meta_stackelberg.agents.td3.agent import (
    TD3Agent,
    TD3Snapshot,
    TD3UpdateStats,
)
from meta_stackelberg.agents.td3.replay import TD3ReplayBuffer


@dataclass(frozen=True)
class PolicyBestResponseResult:
    initial_attacker_fingerprint: str
    adapted_attacker_fingerprint: str
    initial_independent_objective: float
    adapted_independent_objective: float
    update_stats: tuple[TD3UpdateStats, ...]
    approximate_best_response: TD3Snapshot
    kappa_A: float
    protocol: str = 'policy-td3-best-response-v1'


class PolicyBestResponseTrainer:
    def __init__(self, *, N_A: int, batch_size: int, kappa_A: float) -> None:
        for value, name in ((N_A, 'N_A'), (batch_size, 'batch_size')):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        self.N_A = N_A
        self.batch_size = batch_size
        if not math.isfinite(kappa_A) or kappa_A <= 0:
            raise ValueError('kappa_A must be positive and finite')
        self.kappa_A = float(kappa_A)

    def train(
        self,
        *,
        defender: TD3Agent,
        attacker: TD3Agent,
        replay: TD3ReplayBuffer,
        collect_fresh,
        independent_objective,
    ) -> PolicyBestResponseResult:
        if defender.role != 'defender' or attacker.role != 'attacker':
            raise ValueError('policy roles do not match best-response protocol')
        if replay.role != 'attacker':
            raise ValueError('best-response replay must have attacker role')
        defender_guard = defender.freeze_guard()
        initial_fingerprint = attacker.fingerprint()
        initial_objective = float(independent_objective(attacker))
        attacker.set_learning_rate(self.kappa_A)
        stats = []
        for step in range(self.N_A):
            collect_fresh(step)
            defender_guard.verify()
            stats.append(attacker.update(replay.sample(self.batch_size)))
            defender_guard.verify()
        adapted_objective = float(independent_objective(attacker))
        return PolicyBestResponseResult(
            initial_fingerprint,
            attacker.fingerprint(),
            initial_objective,
            adapted_objective,
            tuple(stats),
            attacker.snapshot(),
            self.kappa_A,
        )
