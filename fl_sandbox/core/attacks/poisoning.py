"""Compatibility shim — implementation lives in fl_sandbox.attacks.vector."""

from fl_sandbox.attacks.vector import (
    ALIEAttack,
    GaussianAttack,
    IPMAttack,
    LMPAttack,
    SignFlipAttack,
    craft_alie,
    craft_ipm,
    craft_lmp,
)

__all__ = [
    "ALIEAttack",
    "GaussianAttack",
    "IPMAttack",
    "LMPAttack",
    "SignFlipAttack",
    "craft_alie",
    "craft_ipm",
    "craft_lmp",
]
