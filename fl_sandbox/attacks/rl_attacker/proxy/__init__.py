"""Proxy-distribution helpers for the paper RL attacker."""

from importlib import import_module

__all__ = [
    "PaperDistributionDataset",
    "PaperDistributionSampler",
]

_EXPORTS = {
    "PaperDistributionDataset": ("fl_sandbox.attacks.rl_attacker.proxy.paper_dataset", "PaperDistributionDataset"),
    "PaperDistributionSampler": ("fl_sandbox.attacks.rl_attacker.proxy.paper_dataset", "PaperDistributionSampler"),
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
