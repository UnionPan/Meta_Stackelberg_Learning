"""Task adaptation of a Defender policy against a frozen attack policy."""

from __future__ import annotations

from dataclasses import dataclass
import math

from meta_stackelberg.agents.td3.agent import TD3Agent, TD3UpdateStats
from meta_stackelberg.agents.td3.replay import TD3ReplayBuffer


@dataclass(frozen=True)
class PolicyDefenderAdaptationResult:
    initial_defender_fingerprint: str
    adapted_defender_fingerprint: str
    attacker_fingerprint: str
    adapted_defender: TD3Agent
    update_stats: tuple[TD3UpdateStats, ...]
    eta: float
    protocol: str = 'frozen-attacker-defender-adaptation-v1'


class PolicyDefenderAdapter:
    def __init__(self, *, l: int, batch_size: int, eta: float) -> None:
        for value, name in ((l, 'l'), (batch_size, 'batch_size')):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if not math.isfinite(eta) or eta <= 0:
            raise ValueError('eta must be positive and finite')
        self.l = l
        self.batch_size = batch_size
        self.eta = float(eta)

    def adapt(
        self,
        *,
        defender: TD3Agent,
        attacker: TD3Agent,
        replay: TD3ReplayBuffer,
        collect_fresh,
    ) -> PolicyDefenderAdaptationResult:
        if defender.role != 'defender' or attacker.role != 'attacker':
            raise ValueError('adaptation policy roles do not match protocol')
        if replay.role != 'defender':
            raise ValueError('adaptation replay must have defender role')
        initial = defender.fingerprint()
        adapted = defender.clone()
        adapted.set_learning_rate(self.eta)
        attacker_guard = attacker.freeze_guard()
        stats = []
        for _ in range(self.l):
            collect_fresh(adapted, attacker, replay)
            attacker_guard.verify()
            stats.append(adapted.update(replay.sample(self.batch_size)))
            attacker_guard.verify()
        return PolicyDefenderAdaptationResult(
            initial,
            adapted.fingerprint(),
            attacker_guard.fingerprint,
            adapted,
            tuple(stats),
            self.eta,
        )
