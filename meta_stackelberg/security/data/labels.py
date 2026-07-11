"""Validation helpers for classification attack labels."""

from numbers import Integral


def class_id(value: int, *, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f'{name} must be a non-boolean integer')
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f'{name} must be non-negative')
    return normalized
