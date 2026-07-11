import dataclasses
import math

import numpy as np
import pytest

from meta_stackelberg.security.defenses.action_codec import ClipRadiusActionCodec
from meta_stackelberg.security.defenses.actions import DefenseAction


def test_defense_action_is_frozen_and_accepts_a_positive_finite_radius() -> None:
    action = DefenseAction(clip_radius=0.5)

    assert action.clip_radius == 0.5
    with pytest.raises(dataclasses.FrozenInstanceError):
        action.clip_radius = 1.0  # type: ignore[misc]


@pytest.mark.parametrize('radius', [0.0, -1.0, math.inf, -math.inf, math.nan, True, 'bad'])
def test_defense_action_rejects_invalid_radius(radius: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        DefenseAction(clip_radius=radius)  # type: ignore[arg-type]


def test_codec_maps_endpoints_and_round_trips() -> None:
    codec = ClipRadiusActionCodec(min_radius=0.1, max_radius=1.1)

    assert codec.decode(np.array([-1.0])).clip_radius == 0.1
    assert codec.decode(np.array([1.0])).clip_radius == 1.1
    middle = codec.decode(np.array([0.0]))

    assert middle.clip_radius == pytest.approx(0.6)
    np.testing.assert_allclose(codec.encode(middle), np.array([0.0]))


@pytest.mark.parametrize(
    'value',
    [
        np.array(0.0),
        np.array([0.0, 1.0]),
        np.array([1.01]),
        np.array([-1.01]),
        np.array([np.nan]),
        np.array([np.inf]),
        np.array([0], dtype=np.int64),
    ],
)
def test_codec_rejects_invalid_normalized_actions(value: np.ndarray) -> None:
    codec = ClipRadiusActionCodec(0.1, 1.1)

    with pytest.raises((TypeError, ValueError)):
        codec.decode(value)


@pytest.mark.parametrize(
    ('minimum', 'maximum'),
    [(0.0, 1.0), (-1.0, 1.0), (1.0, 1.0), (2.0, 1.0), (math.nan, 1.0)],
)
def test_codec_rejects_invalid_bounds(minimum: float, maximum: float) -> None:
    with pytest.raises(ValueError):
        ClipRadiusActionCodec(minimum, maximum)


def test_codec_rejects_physical_action_outside_bounds() -> None:
    codec = ClipRadiusActionCodec(0.1, 1.1)

    with pytest.raises(ValueError, match='outside codec bounds'):
        codec.encode(DefenseAction(1.2))


def test_codec_returns_an_independent_float_array() -> None:
    codec = ClipRadiusActionCodec(0.1, 1.1)

    encoded = codec.encode(DefenseAction(0.6))
    encoded[0] = 1.0

    np.testing.assert_array_equal(codec.encode(DefenseAction(0.6)), np.array([0.0]))
    assert encoded.dtype == np.float64
