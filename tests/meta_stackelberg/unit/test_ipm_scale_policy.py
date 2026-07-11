import dataclasses
import math

import numpy as np
import pytest

from meta_stackelberg.agents import IPMScalePolicy, IPMScalePolicySnapshot


def test_policy_clone_and_restore_do_not_mutate_initialization() -> None:
    initial = IPMScalePolicy(1.0)
    clone = initial.clone()
    clone.restore(IPMScalePolicySnapshot(scale=3.0, schema_version=1))

    assert initial.scale == 1.0
    assert clone.scale == 3.0


def test_snapshot_round_trip_and_unknown_schema_rejection() -> None:
    policy = IPMScalePolicy(2.0)

    assert IPMScalePolicy.from_snapshot(policy.snapshot()).snapshot() == policy.snapshot()
    with pytest.raises(ValueError, match='schema'):
        policy.restore(IPMScalePolicySnapshot(2.0, schema_version=99))


@pytest.mark.parametrize('scale', [0.0, -1.0, math.inf, math.nan, True, 'bad'])
def test_policy_rejects_invalid_scale(scale: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        IPMScalePolicy(scale)  # type: ignore[arg-type]


def test_snapshot_is_frozen_and_act_is_exact_and_observation_independent() -> None:
    policy = IPMScalePolicy(3.0)
    snapshot = policy.snapshot()

    assert policy.act() == 3.0
    assert policy.act(np.array([math.nan])) == 3.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.scale = 4.0  # type: ignore[misc]
