"""Sandbox attack implementations and public factory API."""

from fl_sandbox.attacks.alie import ALIEAttack, craft_alie
from fl_sandbox.attacks.base import (
    SandboxAttack,
    bounded_boost,
    bounded_local_epochs,
    bounded_local_lr,
    get_model_weights,
    set_model_weights,
    train_on_loader,
)
from fl_sandbox.attacks.bfl import BFLAttack
from fl_sandbox.attacks.brl import BRLAttack, SelfGuidedBRLAttack
from fl_sandbox.attacks.clipped_median_geometry_search import ClippedMedianGeometrySearchAttack
from fl_sandbox.attacks.dba import DBAAttack
from fl_sandbox.attacks.gaussian import GaussianAttack
from fl_sandbox.attacks.ipm import IPMAttack, craft_ipm
from fl_sandbox.attacks.krum_geometry_search import KrumGeometrySearchAttack
from fl_sandbox.attacks.lmp import LMPAttack, craft_lmp
from fl_sandbox.attacks.registry import ATTACK_CHOICES, create_attack, supported_attack_types
from fl_sandbox.attacks.rl_attacker import RLAttack
from fl_sandbox.attacks.signflip import SignFlipAttack
from fl_sandbox.core.runtime import Weights

__all__ = [
    "ATTACK_CHOICES",
    "ALIEAttack",
    "BFLAttack",
    "BRLAttack",
    "ClippedMedianGeometrySearchAttack",
    "DBAAttack",
    "GaussianAttack",
    "IPMAttack",
    "KrumGeometrySearchAttack",
    "LMPAttack",
    "RLAttack",
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
