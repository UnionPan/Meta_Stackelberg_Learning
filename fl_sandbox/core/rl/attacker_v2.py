"""Compatibility shim — implementation lives in fl_sandbox.attacks.adaptive.td3_attacker_v2."""

from fl_sandbox.attacks.adaptive.td3_attacker_v2 import (
    PaperRLAttackerV2,
    RLAttackerConfigV2,
    SimulatedFLEnvV2,
)

__all__ = ["PaperRLAttackerV2", "RLAttackerConfigV2", "SimulatedFLEnvV2"]
