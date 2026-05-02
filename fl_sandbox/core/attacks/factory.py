"""Compatibility shim — implementation lives in fl_sandbox.attacks.registry."""

from fl_sandbox.attacks.registry import ATTACK_CHOICES, create_attack, supported_attack_types

__all__ = ["ATTACK_CHOICES", "create_attack", "supported_attack_types"]
