"""Simulation layer interfaces and adapters."""

from importlib import import_module

from meta_sg.simulation.interface import FLCoordinator
from meta_sg.simulation.stub import StubCoordinator
from meta_sg.simulation.types import InitialState, RoundSummary, SimulationSnapshot, SimulationSpec, Weights

__all__ = [
    "FLCoordinator",
    "FLSandboxCoordinatorAdapter",
    "InitialState",
    "RoundSummary",
    "SimulationSnapshot",
    "SimulationSpec",
    "StubCoordinator",
    "Weights",
]


def __getattr__(name: str):
    if name == "FLSandboxCoordinatorAdapter":
        module = import_module("meta_sg.simulation.fl_sandbox_adapter")
        value = module.FLSandboxCoordinatorAdapter
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
