"""Diagnostics for RL training and sim2real deployment guard."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RLSim2RealDiagnostics:
    simulated_reward: float = 0.0
    real_reward: float = 0.0
    gap: float = 0.0
    deploy_guard_blocked: bool = False
    trainer: dict[str, float] = field(default_factory=dict)
    window: int = 20
    max_gap: float = 5.0
    deploy_guard_blocks_total: int = 0
    _gaps: deque[float] = field(default_factory=deque, init=False, repr=False)
    _component_gaps: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(deque), init=False, repr=False)

    def __post_init__(self) -> None:
        self._gaps = deque(maxlen=max(1, int(self.window)))
        self._component_gaps = defaultdict(lambda: deque(maxlen=max(1, int(self.window))))

    @property
    def gap_mean(self) -> float:
        return float(np.mean(self._gaps)) if self._gaps else float(self.gap)

    @property
    def gap_std(self) -> float:
        return float(np.std(self._gaps)) if self._gaps else 0.0

    def record_gap(self, *, real_reward: float, simulated_reward: float, components: dict[str, float] | None = None) -> None:
        self.real_reward = float(real_reward)
        self.simulated_reward = float(simulated_reward)
        self.gap = float(real_reward) - float(simulated_reward)
        self._gaps.append(self.gap)
        for key, value in (components or {}).items():
            self._component_gaps[key].append(float(value))
        self.deploy_guard_blocked = abs(self.gap_mean) > float(self.max_gap)
        if self.deploy_guard_blocked:
            self.deploy_guard_blocks_total += 1

    def as_dict(self) -> dict[str, float | bool]:
        payload = {
            "rl_simulated_reward": float(self.simulated_reward),
            "rl_real_reward": float(self.real_reward),
            "rl_sim2real_gap": float(self.gap),
            "rl_sim2real_gap_mean": float(self.gap_mean),
            "rl_sim2real_gap_std": float(self.gap_std),
            "rl_deploy_guard_blocked": bool(self.deploy_guard_blocked),
            "rl_deploy_guard_blocks_total": float(self.deploy_guard_blocks_total),
            **{f"rl_{key}": value for key, value in self.trainer.items()},
        }
        for key, values in self._component_gaps.items():
            payload[f"rl_gap_{key}_mean"] = float(np.mean(values)) if values else 0.0
        return payload


def deploy_guard_allows(*, proxy_samples: int, sim2real_gap: float, min_proxy_samples: int, max_gap: float) -> bool:
    return int(proxy_samples) >= int(min_proxy_samples) and abs(float(sim2real_gap)) <= float(max_gap)
