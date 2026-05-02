"""Compatibility shim — implementation lives in fl_sandbox.attacks.adaptive.td3_attacker."""

from fl_sandbox.attacks.adaptive.td3_attacker import (
    Actor,
    Critic,
    ConvDenoiser,
    DecodedAction,
    GradientDistributionLearner,
    PaperRLAttacker,
    ProxyDatasetBuffer,
    ReplayBuffer,
    RLAttackerConfig,
    SimulatedFLEnv,
    TD3Agent,
    _train_proxy_model,
    local_search_update,
)

__all__ = [
    "Actor",
    "Critic",
    "ConvDenoiser",
    "DecodedAction",
    "GradientDistributionLearner",
    "PaperRLAttacker",
    "ProxyDatasetBuffer",
    "ReplayBuffer",
    "RLAttackerConfig",
    "SimulatedFLEnv",
    "TD3Agent",
    "_train_proxy_model",
    "local_search_update",
]
