"""Paper-aligned RL attacker package."""

from importlib import import_module

__all__ = [
    "AttackerRLEnv",
    "PaperAttackerPolicyGymEnv",
    "PaperDistributionDataset",
    "PaperDistributionSampler",
    "PaperFLSimulator",
    "PaperRLAttack",
    "RLAttack",
    "RLAttackerConfig",
    "TianshouTD3Trainer",
    "build_trainer",
    "craft_paper_malicious_update",
    "decode_paper_action",
]

_EXPORTS = {
    "RLAttack": ("fl_sandbox.attacks.rl_attacker.paper_attack", "PaperRLAttack"),
    "PaperRLAttack": ("fl_sandbox.attacks.rl_attacker.paper_attack", "PaperRLAttack"),
    "AttackerRLEnv": ("fl_sandbox.attacks.rl_attacker.simulator", "AttackerRLEnv"),
    "PaperAttackerPolicyGymEnv": ("fl_sandbox.attacks.rl_attacker.simulator", "PaperAttackerPolicyGymEnv"),
    "PaperFLSimulator": ("fl_sandbox.attacks.rl_attacker.simulator", "PaperFLSimulator"),
    "PaperDistributionDataset": ("fl_sandbox.attacks.rl_attacker.proxy.paper_dataset", "PaperDistributionDataset"),
    "PaperDistributionSampler": ("fl_sandbox.attacks.rl_attacker.proxy.paper_dataset", "PaperDistributionSampler"),
    "RLAttackerConfig": ("fl_sandbox.attacks.rl_attacker.config", "RLAttackerConfig"),
    "TianshouTD3Trainer": ("fl_sandbox.attacks.rl_attacker.tianshou_backend", "TianshouTD3Trainer"),
    "build_trainer": ("fl_sandbox.attacks.rl_attacker.trainer", "build_trainer"),
    "craft_paper_malicious_update": (
        "fl_sandbox.attacks.rl_attacker.simulator.paper_env",
        "craft_paper_malicious_update",
    ),
    "decode_paper_action": ("fl_sandbox.attacks.rl_attacker.simulator.paper_env", "decode_paper_action"),
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
