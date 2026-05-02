"""Sandbox attack implementations and factories."""

from fl_sandbox.attacks.base import (
    SandboxAttack,
    bounded_boost,
    bounded_local_epochs,
    bounded_local_lr,
    get_model_weights,
    set_model_weights,
    train_on_loader,
)
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
from fl_sandbox.attacks.backdoor import BFLAttack, BRLAttack, DBAAttack, SelfGuidedBRLAttack
from fl_sandbox.attacks.adaptive import RLAttack, RLAttackV2
from fl_sandbox.attacks.registry import ATTACK_CHOICES, create_attack, supported_attack_types
from fl_sandbox.core.runtime import Weights

__all__ = [
    "ATTACK_CHOICES",
    "ALIEAttack",
    "BFLAttack",
    "BRLAttack",
    "DBAAttack",
    "GaussianAttack",
    "IPMAttack",
    "LMPAttack",
    "RLAttack",
    "RLAttackV2",
    "SandboxAttack",
    "SelfGuidedBRLAttack",
    "SignFlipAttack",
    "Weights",
    "bounded_boost",
    "bounded_local_epochs",
    "bounded_local_lr",
    "craft_alie",
    "craft_ipm",
    "craft_lmp",
    "create_attack",
    "get_model_weights",
    "set_model_weights",
    "supported_attack_types",
    "train_on_loader",
]
