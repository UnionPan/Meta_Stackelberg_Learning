"""RL-based sandbox attacker implementations.

Main APIs in this file:
- ``RLAttack``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .base import SandboxAttack


@dataclass
class RLAttack(SandboxAttack):
    """Paper-style RL attacker with online distribution and policy learning."""

    default_action: tuple[float, float, float] = (0.0, 0.0, 0.0)
    config: Optional[object] = None
    name: str = "RL"
    attack_type: str = "rl"

    def __post_init__(self) -> None:
        from fl_sandbox.core.rl.attacker import PaperRLAttacker

        self._attacker = PaperRLAttacker(self.config)

    def observe_round(self, ctx) -> None:
        self._attacker.observe_round(ctx)

    def execute(self, ctx, attacker_action=None):
        action = self.resolve_action(ctx, attacker_action, default_action=self.default_action)
        return self._attacker.execute(ctx, attacker_action=action)
