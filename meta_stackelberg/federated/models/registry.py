"""Explicit construction registry for replaceable federated models."""

from __future__ import annotations

from collections.abc import Callable
import weakref

import torch


class ModelRegistry:
    """Map stable experiment names to factories without global mutable state."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], torch.nn.Module]] = {}
        self._live_models: dict[str, weakref.WeakSet[torch.nn.Module]] = {}

    def register(self, name: str, factory: Callable[[], torch.nn.Module]) -> None:
        if not name:
            raise ValueError('model name must not be empty')
        if name in self._factories:
            raise KeyError(f'model {name!r} is already registered')
        self._factories[name] = factory
        self._live_models[name] = weakref.WeakSet()

    def create(self, name: str) -> torch.nn.Module:
        try:
            factory = self._factories[name]
        except KeyError as error:
            raise KeyError(f'unknown model {name!r}') from error
        model = factory()
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f'model factory {name!r} did not return torch.nn.Module')
        if model in self._live_models[name]:
            raise RuntimeError(f'model factory {name!r} must return a fresh model instance')
        self._live_models[name].add(model)
        return model
