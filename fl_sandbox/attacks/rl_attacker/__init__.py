"""Adaptive RL attacker package.

Phase 1a keeps the existing TD3 implementation in ``legacy_td3`` while the
public package path moves to ``fl_sandbox.attacks.rl_attacker``.
"""

from fl_sandbox.attacks.rl_attacker.attack import RLAttack
from fl_sandbox.attacks.rl_attacker.env import AttackerRLEnv
from fl_sandbox.attacks.rl_attacker.legacy_td3 import (
    ConvDenoiser,
    DecodedAction,
    GradientDistributionLearner,
    PaperRLAttacker,
    ProxyDatasetBuffer,
    ReplayBuffer,
    RLAttackerConfig,
    SimulatedFLEnv,
    TD3Agent,
    local_search_update,
)
from fl_sandbox.attacks.rl_attacker.pz_env import AttackerPolicyGymEnv, AttackerPolicyParallelEnv

__all__ = [
    "AttackerPolicyGymEnv",
    "AttackerPolicyParallelEnv",
    "AttackerRLEnv",
    "ConvDenoiser",
    "DecodedAction",
    "GradientDistributionLearner",
    "PaperRLAttacker",
    "ProxyDatasetBuffer",
    "RLAttack",
    "RLAttackerConfig",
    "ReplayBuffer",
    "SimulatedFLEnv",
    "TD3Agent",
    "local_search_update",
]
