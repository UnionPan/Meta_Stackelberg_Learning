"""Sequential online TD3 adaptation of a frozen Meta-SG initialization."""

from __future__ import annotations

from dataclasses import dataclass
import math

from meta_stackelberg.agents.td3.agent import TD3Agent, TD3UpdateStats
from meta_stackelberg.agents.td3.replay import TD3ReplayBuffer


@dataclass(frozen=True)
class PolicyOnlineIterationTrace:
    online_iteration: int
    updates: tuple[TD3UpdateStats, ...]
    defender_fingerprint: str


@dataclass(frozen=True)
class PolicyOnlineAdaptationResult:
    initial_defender_fingerprint: str
    adapted_defender_fingerprint: str
    attacker_fingerprint: str
    adapted_defender: TD3Agent
    iterations: tuple[PolicyOnlineIterationTrace, ...]
    total_updates: int
    protocol: str = 'paper-meta-sg-online-adaptation-v1'


class PolicyOnlineAdaptationRunner:
    """Execute online_T × online_l consecutive updates on one policy copy."""

    def __init__(
        self,
        *,
        online_T: int,
        online_l: int,
        online_steps: int,
        batch_size: int,
        adaptation_step: float,
    ) -> None:
        for value, name in (
            (online_T, 'online_T'), (online_l, 'online_l'),
            (online_steps, 'online_steps'), (batch_size, 'batch_size'),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if online_T * online_l != online_steps:
            raise ValueError('online_steps must equal online_T * online_l')
        if not math.isfinite(adaptation_step) or adaptation_step <= 0:
            raise ValueError('adaptation_step must be positive and finite')
        self.online_T = online_T
        self.online_l = online_l
        self.online_steps = online_steps
        self.batch_size = batch_size
        self.adaptation_step = float(adaptation_step)

    def run(
        self,
        *,
        meta_defender: TD3Agent,
        attacker: TD3Agent,
        replay: TD3ReplayBuffer,
        collect_fresh,
        start_iteration: int = 0,
        resumed_defender: TD3Agent | None = None,
        iteration_history: tuple[PolicyOnlineIterationTrace, ...] = (),
        iteration_callback=None,
    ) -> PolicyOnlineAdaptationResult:
        if meta_defender.role != 'defender' or attacker.role != 'attacker':
            raise ValueError('online adaptation policy roles do not match protocol')
        if replay.role != 'defender':
            raise ValueError('online adaptation replay must have defender role')
        if (
            isinstance(start_iteration, bool)
            or not isinstance(start_iteration, int)
            or not 0 <= start_iteration <= self.online_T
        ):
            raise ValueError('start_iteration must be within [0, online_T]')
        history = tuple(iteration_history)
        if len(history) != start_iteration:
            raise ValueError('iteration_history must end at start_iteration')
        if start_iteration and resumed_defender is None:
            raise ValueError('resumed_defender is required after iteration zero')
        if not start_iteration and resumed_defender is not None:
            raise ValueError('resumed_defender is invalid at iteration zero')
        initial_fingerprint = meta_defender.fingerprint()
        adapted = (
            resumed_defender.clone()
            if resumed_defender is not None
            else meta_defender.clone()
        )
        if adapted.role != 'defender':
            raise ValueError('resumed online policy must have defender role')
        adapted.set_learning_rate(self.adaptation_step)
        attacker_guard = attacker.freeze_guard()
        iterations = list(history)
        global_step = start_iteration * self.online_l
        for online_iteration in range(start_iteration, self.online_T):
            stats = []
            for local_step in range(self.online_l):
                collect_fresh(
                    adapted, attacker, replay,
                    online_iteration, local_step, global_step,
                )
                attacker_guard.verify()
                stats.append(adapted.update(replay.sample(self.batch_size)))
                attacker_guard.verify()
                global_step += 1
            trace = PolicyOnlineIterationTrace(
                online_iteration,
                tuple(stats),
                adapted.fingerprint(),
            )
            iterations.append(trace)
            if iteration_callback is not None:
                iteration_callback(trace, adapted, replay)
        if global_step != self.online_steps:
            raise RuntimeError('online adaptation update count drifted')
        return PolicyOnlineAdaptationResult(
            initial_fingerprint,
            adapted.fingerprint(),
            attacker_guard.fingerprint,
            adapted,
            tuple(iterations),
            global_step,
        )
