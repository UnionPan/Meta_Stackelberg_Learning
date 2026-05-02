"""Adaptive RL-based attacks: RLAttack (TD3 policy) and RLAttackV2 (stealth-craft)."""

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
        from fl_sandbox.attacks.adaptive.td3_attacker import PaperRLAttacker
        self._attacker = PaperRLAttacker(self.config)

    def observe_round(self, ctx) -> None:
        self._attacker.observe_round(ctx)

    def execute(self, ctx, attacker_action=None):
        return self._attacker.execute(ctx, attacker_action=attacker_action)


@dataclass
class RLAttackV2(SandboxAttack):
    """V2 RL attacker: stealth-craft + bypass reward + faster policy learning."""

    default_action: tuple[float, float, float] = (0.0, 0.0, 0.0)
    config: Optional[object] = None
    name: str = "RLv2"
    attack_type: str = "rl2"

    def __post_init__(self) -> None:
        from fl_sandbox.attacks.adaptive.td3_attacker_v2 import PaperRLAttackerV2
        self._attacker = PaperRLAttackerV2(self.config)

    def observe_round(self, ctx) -> None:
        self._attacker.observe_round(ctx)

    def execute(self, ctx, attacker_action=None):
        return self._attacker.execute(ctx, attacker_action=attacker_action)


__all__ = ["RLAttack", "RLAttackV2"]
