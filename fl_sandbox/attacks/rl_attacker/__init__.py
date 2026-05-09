"""Adaptive RL attacker package."""

from importlib import import_module

__all__ = [
    "AttackerPolicyGymEnv",
    "AttackerRLEnv",
    "AttackParameters",
    "ConvDenoiser",
    "DecodedAction",
    "GradientDistributionLearner",
    "ProxyDatasetBuffer",
    "RLAttack",
    "RLAttackerConfig",
    "RLSim2RealDiagnostics",
    "SimulatedFLEnv",
    "TianshouSACTrainer",
    "TianshouTD3Trainer",
    "build_trainer",
    "decode_action",
    "deploy_guard_allows",
    "local_search_update",
]

_EXPORTS = {
    "RLAttack": ("fl_sandbox.attacks.rl_attacker.attack", "RLAttack"),
    "AttackerRLEnv": ("fl_sandbox.attacks.rl_attacker.simulator", "AttackerRLEnv"),
    "AttackerPolicyGymEnv": ("fl_sandbox.attacks.rl_attacker.simulator", "AttackerPolicyGymEnv"),
    "AttackParameters": ("fl_sandbox.attacks.rl_attacker.action_decoder", "AttackParameters"),
    "ConvDenoiser": ("fl_sandbox.attacks.rl_attacker.proxy", "ConvDenoiser"),
    "DecodedAction": ("fl_sandbox.attacks.rl_attacker.action_decoder", "DecodedAction"),
    "GradientDistributionLearner": ("fl_sandbox.attacks.rl_attacker.proxy", "GradientDistributionLearner"),
    "ProxyDatasetBuffer": ("fl_sandbox.attacks.rl_attacker.proxy", "ProxyDatasetBuffer"),
    "RLAttackerConfig": ("fl_sandbox.attacks.rl_attacker.config", "RLAttackerConfig"),
    "RLSim2RealDiagnostics": ("fl_sandbox.attacks.rl_attacker.diagnostics", "RLSim2RealDiagnostics"),
    "SimulatedFLEnv": ("fl_sandbox.attacks.rl_attacker.simulator", "SimulatedFLEnv"),
    "TianshouSACTrainer": ("fl_sandbox.attacks.rl_attacker.tianshou_backend", "TianshouSACTrainer"),
    "TianshouTD3Trainer": ("fl_sandbox.attacks.rl_attacker.tianshou_backend", "TianshouTD3Trainer"),
    "build_trainer": ("fl_sandbox.attacks.rl_attacker.trainer", "build_trainer"),
    "decode_action": ("fl_sandbox.attacks.rl_attacker.action_decoder", "decode_action"),
    "deploy_guard_allows": ("fl_sandbox.attacks.rl_attacker.diagnostics", "deploy_guard_allows"),
    "local_search_update": ("fl_sandbox.attacks.rl_attacker.simulator", "local_search_update"),
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
