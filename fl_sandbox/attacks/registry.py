"""Attack factory — map config to concrete SandboxAttack instances."""

from __future__ import annotations

from typing import Optional

from fl_sandbox.attacks.alie import ALIEAttack
from fl_sandbox.attacks.base import SandboxAttack
from fl_sandbox.attacks.bfl import BFLAttack
from fl_sandbox.attacks.dba import DBAAttack
from fl_sandbox.attacks.gaussian import GaussianAttack
from fl_sandbox.attacks.ipm import IPMAttack
from fl_sandbox.attacks.lmp import LMPAttack
from fl_sandbox.attacks.rl_attacker import RLAttack
from fl_sandbox.attacks.rl_backdoor import RLBackdoorAttack
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
    "rl_backdoor",
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

        distribution_dir = getattr(attacker_config, "rl_distribution_dir", "")
        if not distribution_dir:
            raise ValueError("attack_type='rl' requires rl_distribution_dir / --distribution_dir")
        return RLAttack(
            default_action=tuple(getattr(attacker_config, "attacker_action", (0.0, 0.0))[:2]),
            config=RLAttackerConfig(
                algorithm="td3",
                attacker_semantics="paper_clipped_median",
                distribution_dir=distribution_dir,
                distribution_split=getattr(attacker_config, "rl_distribution_split", "train"),
                policy_warmup_steps=getattr(attacker_config, "rl_policy_warmup_steps", 80_000),
                policy_warmup_random_steps=getattr(attacker_config, "rl_policy_warmup_random_steps", 100),
                policy_warmup_checkpoint_interval=getattr(
                    attacker_config,
                    "rl_policy_warmup_checkpoint_interval",
                    0,
                ),
                policy_warmup_checkpoint_dir=getattr(attacker_config, "rl_policy_warmup_checkpoint_dir", ""),
                distribution_growth_mode=getattr(attacker_config, "rl_distribution_growth_mode", "paper_growth"),
                policy_lr=getattr(attacker_config, "rl_policy_lr", 1e-7),
                critic_lr=getattr(attacker_config, "rl_critic_lr", 1e-7),
                gamma=getattr(attacker_config, "rl_gamma", 1.0),
                replay_capacity=getattr(attacker_config, "rl_replay_capacity", 100_000),
                batch_size=getattr(attacker_config, "rl_batch_size", 256),
                hidden_sizes=tuple(getattr(attacker_config, "rl_hidden_sizes", (256, 128))),
                exploration_noise=getattr(attacker_config, "rl_exploration_noise", 0.1),
                train_freq_steps=getattr(attacker_config, "rl_train_freq_steps", 5),
                reward_transform=getattr(attacker_config, "rl_reward_transform", "raw"),
                reward_scale=getattr(attacker_config, "rl_reward_scale", 10.0),
                policy_train_steps_per_round=getattr(attacker_config, "rl_policy_train_steps_per_round", 0),
                policy_checkpoint_path=getattr(attacker_config, "rl_policy_checkpoint_path", ""),
                policy_checkpoint_dir=getattr(attacker_config, "rl_policy_checkpoint_dir", ""),
                freeze_policy=True,
                strict_reproduction_initial_samples=getattr(
                    attacker_config,
                    "rl_strict_reproduction_initial_samples",
                    200,
                ),
                strict_reproduction_samples_per_epoch=getattr(
                    attacker_config,
                    "rl_strict_reproduction_samples_per_epoch",
                    80,
                ),
                distribution_steps=attacker_config.rl_distribution_steps,
                attack_start_round=attacker_config.rl_attack_start_round,
                policy_train_end_round=attacker_config.rl_policy_train_end_round,
                reconstruction_batch_size=attacker_config.rl_reconstruction_batch_size,
                episodes_per_observation=max(2, attacker_config.rl_policy_train_episodes_per_round),
                simulator_horizon=max(1, attacker_config.rl_simulator_horizon),
            ),
        )
    if attack_type == "rl_backdoor":
        from fl_sandbox.attacks.rl_backdoor.config import BackdoorRLConfig

        # ``rl_policy_train_steps_per_round=0`` in the schema means "not set"
        # (same convention as the ``rl`` attacker), so fall back to the
        # dataclass default rather than passing 0 — otherwise training would
        # collapse to a single gradient step per round.
        steps_per_round = (
            int(getattr(attacker_config, "rl_policy_train_steps_per_round", 0)) or 50
        )
        return RLBackdoorAttack(
            default_action=tuple(getattr(attacker_config, "rl_backdoor_default_action", (1.0, 0.0, -1.0, 0.0))),
            stealth_norm_cap=bool(getattr(attacker_config, "rl_backdoor_stealth_norm_cap", False)),
            config=BackdoorRLConfig.from_attacker_config(attacker_config),
            attack_start_round=int(getattr(attacker_config, "rl_attack_start_round", 10) or 10),
            policy_train_end_round=int(getattr(attacker_config, "rl_policy_train_end_round", 30) or 30),
            policy_train_steps_per_round=steps_per_round,
        )
    raise AssertionError(f"Unreachable attack type branch: {attack_type}")


__all__ = ["ATTACK_CHOICES", "create_attack", "supported_attack_types"]
