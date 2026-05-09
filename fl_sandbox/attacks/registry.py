"""Attack factory — map config to concrete SandboxAttack instances."""

from __future__ import annotations

from typing import Optional

from fl_sandbox.attacks.alie import ALIEAttack
from fl_sandbox.attacks.base import SandboxAttack
from fl_sandbox.attacks.bfl import BFLAttack
from fl_sandbox.attacks.brl import BRLAttack, SelfGuidedBRLAttack
from fl_sandbox.attacks.clipped_median_geometry_search import ClippedMedianGeometrySearchAttack
from fl_sandbox.attacks.dba import DBAAttack
from fl_sandbox.attacks.gaussian import GaussianAttack
from fl_sandbox.attacks.ipm import IPMAttack
from fl_sandbox.attacks.krum_geometry_search import KrumGeometrySearchAttack
from fl_sandbox.attacks.lmp import LMPAttack
from fl_sandbox.attacks.rl_attacker import RLAttack
from fl_sandbox.attacks.signflip import SignFlipAttack


ATTACK_CHOICES = (
    "clean",
    "ipm",
    "lmp",
    "alie",
    "signflip",
    "gaussian",
    "bfl",
    "dba",
    "rl",
    "krum_geometry_search",
    "clipped_median_geometry_search",
    "brl",
    "sgbrl",
)


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
    if attack_type == "alie":
        return ALIEAttack(tau=attacker_config.alie_tau)
    if attack_type == "signflip":
        return SignFlipAttack()
    if attack_type == "gaussian":
        return GaussianAttack(sigma=attacker_config.gaussian_sigma)
    if attack_type == "bfl":
        return BFLAttack(poison_frac=attacker_config.bfl_poison_frac)
    if attack_type == "dba":
        return DBAAttack(
            num_sub_triggers=attacker_config.dba_num_sub_triggers,
            poison_frac=attacker_config.dba_poison_frac,
        )
    if attack_type == "rl":
        from fl_sandbox.attacks.rl_attacker.config import RLAttackerConfig

        return RLAttack(
            default_action=tuple(attacker_config.attacker_action),
            config=RLAttackerConfig(
                algorithm=getattr(attacker_config, "rl_algorithm", "td3"),
                distribution_steps=attacker_config.rl_distribution_steps,
                attack_start_round=attacker_config.rl_attack_start_round,
                policy_train_end_round=attacker_config.rl_policy_train_end_round,
                inversion_steps=attacker_config.rl_inversion_steps,
                reconstruction_batch_size=attacker_config.rl_reconstruction_batch_size,
                episodes_per_observation=max(2, attacker_config.rl_policy_train_episodes_per_round),
                simulator_horizon=max(12, attacker_config.rl_simulator_horizon),
            ),
        )
    if attack_type == "krum_geometry_search":
        from fl_sandbox.attacks.krum_geometry_search import KrumGeometrySearchConfig

        return KrumGeometrySearchAttack(config=KrumGeometrySearchConfig())
    if attack_type == "clipped_median_geometry_search":
        from fl_sandbox.attacks.clipped_median_geometry_search import ClippedMedianGeometrySearchConfig

        return ClippedMedianGeometrySearchAttack(config=ClippedMedianGeometrySearchConfig())
    if attack_type == "brl":
        return BRLAttack(default_action=tuple(attacker_config.attacker_action))
    if attack_type == "sgbrl":
        return SelfGuidedBRLAttack()
    raise AssertionError(f"Unreachable attack type branch: {attack_type}")


__all__ = ["ATTACK_CHOICES", "create_attack", "supported_attack_types"]
