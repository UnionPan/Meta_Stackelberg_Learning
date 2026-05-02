"""Federated learning orchestration components."""

from __future__ import annotations

from importlib import import_module

__all__ = ["ClientTrainer", "DataPartitioner", "FederatedCoordinator", "PoisonedDataFactory"]

_EXPORTS = {
    "ClientTrainer": (".client", "ClientTrainer"),
    "DataPartitioner": (".partitioning", "DataPartitioner"),
    "FederatedCoordinator": (".coordinator", "FederatedCoordinator"),
    "PoisonedDataFactory": (".poisoning", "PoisonedDataFactory"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
