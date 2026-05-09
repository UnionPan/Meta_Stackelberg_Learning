"""Compatibility shim for legacy TD3 attacker exports."""

from fl_sandbox.attacks.rl_attacker.legacy_td3 import (
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
