"""Compatibility shim for backdoor attacks."""

from fl_sandbox.attacks import BFLAttack, BRLAttack, DBAAttack, SelfGuidedBRLAttack

__all__ = ["BFLAttack", "BRLAttack", "DBAAttack", "SelfGuidedBRLAttack"]
