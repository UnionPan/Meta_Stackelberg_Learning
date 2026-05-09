"""Base classes for sandbox defender config adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SandboxDefender(ABC):
    """Shared base class for defender-side orchestration adapters."""

    name: str = "base"
    defense_type: str = "base"

    @abstractmethod
    def build_config_kwargs(self) -> dict[str, Any]:
        """Translate the defender into ``SandboxConfig`` keyword arguments."""
