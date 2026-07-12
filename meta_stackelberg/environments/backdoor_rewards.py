"""Typed joint rewards for the MNIST white-box backdoor game."""

from __future__ import annotations

from dataclasses import dataclass
import math


_SOURCE = 'mnist-whitebox-real-data-v1'


@dataclass(frozen=True)
class WhiteBoxDefenderReward:
    scalar: float
    clean_loss: float
    safe_loss: float
    source: str = _SOURCE

    def __post_init__(self) -> None:
        object.__setattr__(self, 'scalar', _finite(self.scalar, 'scalar'))
        object.__setattr__(self, 'clean_loss', _loss(self.clean_loss, 'clean_loss'))
        object.__setattr__(self, 'safe_loss', _loss(self.safe_loss, 'safe_loss'))
        if self.source != _SOURCE:
            raise ValueError(f'source must equal {_SOURCE!r}')


@dataclass(frozen=True)
class WhiteBoxAttackerReward:
    scalar: float
    target_loss: float
    clean_damage: float
    source: str = _SOURCE

    def __post_init__(self) -> None:
        object.__setattr__(self, 'scalar', _finite(self.scalar, 'scalar'))
        object.__setattr__(self, 'target_loss', _loss(self.target_loss, 'target_loss'))
        object.__setattr__(self, 'clean_damage', _loss(self.clean_damage, 'clean_damage'))
        if self.source != _SOURCE:
            raise ValueError(f'source must equal {_SOURCE!r}')


def evaluate_whitebox_backdoor_rewards(
    *,
    clean_loss: float,
    safe_loss: float,
    target_loss: float,
    clean_damage: float,
    defender_lambda: float,
    attacker_lambda: float,
) -> tuple[WhiteBoxDefenderReward, WhiteBoxAttackerReward]:
    clean = _loss(clean_loss, 'clean_loss')
    safe = _loss(safe_loss, 'safe_loss')
    target = _loss(target_loss, 'target_loss')
    damage = _loss(clean_damage, 'clean_damage')
    defender_weight = _unit_interval(defender_lambda, 'defender_lambda')
    attacker_weight = _unit_interval(attacker_lambda, 'attacker_lambda')
    defender_scalar = -(
        (1.0 - defender_weight) * clean + defender_weight * safe
    )
    attacker_scalar = -(
        (1.0 - attacker_weight) * target + attacker_weight * damage
    )
    return (
        WhiteBoxDefenderReward(defender_scalar, clean, safe),
        WhiteBoxAttackerReward(attacker_scalar, target, damage),
    )


def _loss(value: float, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f'{name} must be non-negative')
    return result


def _unit_interval(value: float, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0 or result > 1.0:
        raise ValueError(f'{name} must be within [0, 1]')
    return result


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f'{name} must be a real number')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite')
    return result
