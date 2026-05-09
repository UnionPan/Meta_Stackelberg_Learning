"""Public RL attacker wrapper for the attacker package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fl_sandbox.attacks.base import SandboxAttack


@dataclass
class RLAttack(SandboxAttack):
    """Paper-style RL attacker with online distribution and policy learning."""

    default_action: tuple[float, float, float] = (0.0, 0.0, 0.0)
    config: Optional[object] = None
    name: str = "RL"
    attack_type: str = "rl"

    def __post_init__(self) -> None:
        from fl_sandbox.attacks.rl_attacker.legacy_td3 import PaperRLAttacker

        self._attacker = PaperRLAttacker(self.config)

    def observe_round(self, ctx) -> None:
        self._attacker.observe_round(ctx)

    def execute(self, ctx, attacker_action=None):
        return self._attacker.execute(ctx, attacker_action=attacker_action)

    def after_round(self, **kwargs):
        if hasattr(self._attacker, "after_round"):
            return self._attacker.after_round(**kwargs)
        return {}
