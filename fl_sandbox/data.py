"""Compatibility data helpers for the FL sandbox runtime.

The sandbox runner expects a local ``fl_sandbox.data`` module, while the
canonical dataset utilities currently live under ``src.utils.data_loader``.
This shim keeps the sandbox entry points working without duplicating logic.
"""

from src.utils.data_loader import DatasetSplit, add_pattern_bd, get_datasets, poison_dataset

__all__ = ["DatasetSplit", "add_pattern_bd", "get_datasets", "poison_dataset"]
