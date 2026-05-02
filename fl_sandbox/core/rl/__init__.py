"""RL-specific sandbox components."""

from __future__ import annotations

from importlib import import_module


__all__ = [
    'AttackerPolicyParallelEnv',
    'AttackerPolicyGymEnv',
    'AttackerRLEnv',
    'DecodedAction',
    'GradientDistributionLearner',
    'PaperRLAttacker',
    'ProxyDatasetBuffer',
    'ReplayBuffer',
    'RLAttackerConfig',
    'SimulatedFLEnv',
    'TD3Agent',
    'local_search_update',
]


_EXPORTS = {
    'AttackerPolicyParallelEnv': ('.pz_env', 'AttackerPolicyParallelEnv'),
    'AttackerPolicyGymEnv': ('.pz_env', 'AttackerPolicyGymEnv'),
    'AttackerRLEnv': ('.env', 'AttackerRLEnv'),
    'DecodedAction': ('.attacker', 'DecodedAction'),
    'GradientDistributionLearner': ('.attacker', 'GradientDistributionLearner'),
    'PaperRLAttacker': ('.attacker', 'PaperRLAttacker'),
    'ProxyDatasetBuffer': ('.attacker', 'ProxyDatasetBuffer'),
    'ReplayBuffer': ('.attacker', 'ReplayBuffer'),
    'RLAttackerConfig': ('.attacker', 'RLAttackerConfig'),
    'SimulatedFLEnv': ('.attacker', 'SimulatedFLEnv'),
    'TD3Agent': ('.attacker', 'TD3Agent'),
    'local_search_update': ('.attacker', 'local_search_update'),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
