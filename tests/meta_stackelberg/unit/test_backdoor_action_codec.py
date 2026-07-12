from __future__ import annotations

import numpy as np
import pytest

from meta_stackelberg.security.attacks.backdoor_action import (
    BackdoorAction,
    BackdoorActionCodec,
)


def test_backdoor_action_codec_decodes_paper_endpoints() -> None:
    codec = BackdoorActionCodec()

    low = codec.decode(np.array([-1.0, -1.0, -1.0], dtype=np.float64))
    high = codec.decode(np.array([1.0, 1.0, 1.0], dtype=np.float64))

    assert low == BackdoorAction(0.0, 0.0, 1)
    assert high == BackdoorAction(1.0, 0.1, 10)
    assert low.as_tuple() == (0.0, 0.0, 1)
    assert high.as_tuple() == (1.0, 0.1, 10)


def test_backdoor_action_codec_decodes_midpoint() -> None:
    action = BackdoorActionCodec().decode(np.zeros(3, dtype=np.float32))

    assert action == BackdoorAction(0.5, 0.05, 6)


@pytest.mark.parametrize(
    'raw_action',
    [
        np.zeros(2, dtype=np.float64),
        np.array([0.0, 0.0, np.nan], dtype=np.float64),
        np.array([0.0, 0.0, 1.1], dtype=np.float64),
    ],
)
def test_backdoor_action_codec_rejects_invalid_raw_values(
    raw_action: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        BackdoorActionCodec().decode(raw_action)


def test_backdoor_action_codec_rejects_integer_raw_action() -> None:
    with pytest.raises(TypeError, match='floating dtype'):
        BackdoorActionCodec().decode(np.zeros(3, dtype=np.int64))


@pytest.mark.parametrize(
    ('values', 'message'),
    [
        ((-0.1, 0.05, 1), 'poison_fraction'),
        ((0.5, 0.11, 1), 'learning_rate'),
        ((0.5, 0.05, 0), 'local_epochs'),
    ],
)
def test_backdoor_action_rejects_values_outside_contract(
    values: tuple[float, float, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        BackdoorAction(*values)
