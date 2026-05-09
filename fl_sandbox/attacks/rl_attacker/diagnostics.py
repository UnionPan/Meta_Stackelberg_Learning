"""Diagnostics for RL training and sim2real deployment guard."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RLSim2RealDiagnostics:
    simulated_reward: float = 0.0
    real_reward: float = 0.0
    gap: float = 0.0
    deploy_guard_blocked: bool = False
    trainer: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "rl_simulated_reward": float(self.simulated_reward),
            "rl_real_reward": float(self.real_reward),
            "rl_sim2real_gap": float(self.gap),
            "rl_deploy_guard_blocked": bool(self.deploy_guard_blocked),
            **{f"rl_{key}": value for key, value in self.trainer.items()},
        }


def deploy_guard_allows(*, proxy_samples: int, sim2real_gap: float, min_proxy_samples: int, max_gap: float) -> bool:
    return int(proxy_samples) >= int(min_proxy_samples) and abs(float(sim2real_gap)) <= float(max_gap)
