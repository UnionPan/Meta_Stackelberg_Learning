import dataclasses
import math

import numpy as np
import pytest

from meta_stackelberg.security.attacks.rl_action import RLAttackAction, RLAttackActionCodec
from meta_stackelberg.security.defenses.paper_action import (
    PaperDefenderAction,
    PaperDefenderActionCodec,
)


def test_defender_codec_maps_3d_endpoints_and_round_trips() -> None:
    codec = PaperDefenderActionCodec()
    low = codec.decode(np.array([-1.0, -1.0, -1.0]), observed_max_norm=5.0)
    high = codec.decode(np.array([1.0, 1.0, 1.0]), observed_max_norm=5.0)
    assert low == PaperDefenderAction(1e-6, 0.0, 0.1)
    assert high == PaperDefenderAction(5.0, 0.45, 10.0)
    action = codec.decode(np.array([0.0, 0.0, 0.0]), observed_max_norm=5.0)
    np.testing.assert_allclose(codec.encode(action, observed_max_norm=5.0), np.zeros(3))
    with pytest.raises(dataclasses.FrozenInstanceError):
        action.alpha = 1.0  # type: ignore[misc]


def test_attacker_codec_maps_gamma_steps_lambda_and_round_trips_integer_steps() -> None:
    codec = RLAttackActionCodec()
    assert codec.decode(np.array([-1.0, -1.0, -1.0])) == RLAttackAction(0.1, 1, 0.05)
    assert codec.decode(np.array([1.0, 1.0, 1.0])) == RLAttackAction(2.9, 19, 0.95)
    for steps in (1, 2, 10, 18, 19):
        action = RLAttackAction(1.5, steps, 0.5)
        assert codec.decode(codec.encode(action)) == action


@pytest.mark.parametrize(
    'raw',
    [
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0, 0, 0]),
        np.array([math.nan, 0.0, 0.0]),
        np.array([1.01, 0.0, 0.0]),
    ],
)
def test_codecs_reject_invalid_raw_actions(raw: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        RLAttackActionCodec().decode(raw)
    with pytest.raises((TypeError, ValueError)):
        PaperDefenderActionCodec().decode(raw, observed_max_norm=5.0)


def test_action_records_and_dynamic_alpha_bound_fail_fast() -> None:
    with pytest.raises(ValueError, match='alpha'):
        PaperDefenderAction(0.0, 0.2, 1.0)
    with pytest.raises(ValueError, match='beta'):
        PaperDefenderAction(1.0, 0.5, 1.0)
    with pytest.raises(ValueError, match='local_steps'):
        RLAttackAction(1.0, 0, 0.5)
    with pytest.raises(ValueError, match='observed_max_norm'):
        PaperDefenderActionCodec().decode(np.zeros(3), observed_max_norm=1e-7)
