"""Compatibility shim — implementation lives in fl_sandbox.attacks.backdoor."""

from fl_sandbox.attacks.backdoor import BFLAttack, BRLAttack, DBAAttack, SelfGuidedBRLAttack

__all__ = ["BFLAttack", "BRLAttack", "DBAAttack", "SelfGuidedBRLAttack"]
