"""Adaptive RL attacker package.

Phase 1a keeps the existing TD3 implementation in ``legacy_td3`` while the
public package path moves to ``fl_sandbox.attacks.rl_attacker``.
"""

from importlib import import_module

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

_LEGACY_TD3_MODULE = "fl_sandbox.attacks.rl_attacker.legacy_td3"

_EXPORTS = {
    "RLAttack": ("fl_sandbox.attacks.rl_attacker.attack", "RLAttack"),
    "AttackerRLEnv": ("fl_sandbox.attacks.rl_attacker.env", "AttackerRLEnv"),
    "AttackerPolicyGymEnv": ("fl_sandbox.attacks.rl_attacker.pz_env", "AttackerPolicyGymEnv"),
    "AttackerPolicyParallelEnv": ("fl_sandbox.attacks.rl_attacker.pz_env", "AttackerPolicyParallelEnv"),
    "ConvDenoiser": (_LEGACY_TD3_MODULE, "ConvDenoiser"),
    "DecodedAction": (_LEGACY_TD3_MODULE, "DecodedAction"),
    "GradientDistributionLearner": (_LEGACY_TD3_MODULE, "GradientDistributionLearner"),
    "PaperRLAttacker": (_LEGACY_TD3_MODULE, "PaperRLAttacker"),
    "ProxyDatasetBuffer": (_LEGACY_TD3_MODULE, "ProxyDatasetBuffer"),
    "ReplayBuffer": (_LEGACY_TD3_MODULE, "ReplayBuffer"),
    "RLAttackerConfig": (_LEGACY_TD3_MODULE, "RLAttackerConfig"),
    "SimulatedFLEnv": (_LEGACY_TD3_MODULE, "SimulatedFLEnv"),
    "TD3Agent": (_LEGACY_TD3_MODULE, "TD3Agent"),
    "local_search_update": (_LEGACY_TD3_MODULE, "local_search_update"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
