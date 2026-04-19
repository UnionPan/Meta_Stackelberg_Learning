"""Factory helpers for constructing sandbox attackers.

Main APIs in this file:
- ``ATTACK_CHOICES``
- ``supported_attack_types``
- ``create_attack``
"""

from __future__ import annotations

from typing import Optional

from .backdoor import BFLAttack, BRLAttack, DBAAttack
from .base import SandboxAttack
from .poisoning import IPMAttack, LMPAttack
from .rl import RLAttack


ATTACK_CHOICES = ("clean", "ipm", "lmp", "bfl", "dba", "rl", "brl")


def supported_attack_types() -> tuple[str, ...]:
    return ATTACK_CHOICES


def create_attack(attacker_config) -> Optional[SandboxAttack]:
    attack_type = getattr(attacker_config, "type", None)
    if attack_type is None:
        raise ValueError("Attacker config is missing required field: type")

    attack_type = str(attack_type)
    if attack_type not in ATTACK_CHOICES:
        supported = ", ".join(supported_attack_types())
        raise ValueError(f"Unsupported attack type: {attack_type}. Supported attack types: {supported}")
    if attack_type == "clean":
        return None
    if attack_type == "ipm":
        return IPMAttack(scale=attacker_config.ipm_scaling)
    if attack_type == "lmp":
        return LMPAttack(scale=attacker_config.lmp_scale)
    if attack_type == "bfl":
        return BFLAttack(poison_frac=attacker_config.bfl_poison_frac)
    if attack_type == "dba":
        return DBAAttack(
            num_sub_triggers=attacker_config.dba_num_sub_triggers,
            poison_frac=attacker_config.dba_poison_frac,
        )
    if attack_type == "rl":
        from fl_sandbox.core.rl.attacker import RLAttackerConfig

        return RLAttack(
            default_action=tuple(attacker_config.attacker_action),
            config=RLAttackerConfig(
                distribution_steps=attacker_config.rl_distribution_steps,
                attack_start_round=attacker_config.rl_attack_start_round,
                policy_train_end_round=attacker_config.rl_policy_train_end_round,
                inversion_steps=attacker_config.rl_inversion_steps,
                reconstruction_batch_size=attacker_config.rl_reconstruction_batch_size,
                episodes_per_observation=attacker_config.rl_policy_train_episodes_per_round,
                simulator_horizon=attacker_config.rl_simulator_horizon,
            ),
        )
    if attack_type == "brl":
        return BRLAttack(default_action=tuple(attacker_config.attacker_action))
    raise AssertionError(f"Unreachable attack type branch: {attack_type}")
