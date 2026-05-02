"""Compatibility shim — implementation lives in fl_sandbox.attacks.base."""

from fl_sandbox.attacks.base import (
    SandboxAttack,
    bounded_boost,
    bounded_local_epochs,
    bounded_local_lr,
    get_model_weights,
    set_model_weights,
    train_on_loader,
)
from fl_sandbox.core.runtime import Weights

__all__ = [
    "SandboxAttack",
    "Weights",
    "bounded_boost",
    "bounded_local_epochs",
    "bounded_local_lr",
    "get_model_weights",
    "set_model_weights",
    "train_on_loader",
]
