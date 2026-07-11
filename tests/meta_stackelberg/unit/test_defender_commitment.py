import dataclasses
from dataclasses import replace
import math

import pytest

from meta_stackelberg.security.defenses.actions import DefenseAction
from meta_stackelberg.stackelberg import DefenderCommitment


def test_commitment_has_stable_canonical_fingerprint_and_is_frozen() -> None:
    first = DefenderCommitment.create('weak', DefenseAction(10.0, 0.0))
    second = DefenderCommitment.create('weak', DefenseAction(10.0, 0.0))

    assert first == second
    assert len(first.policy_fingerprint) == 64
    assert first.verify() is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.commitment_id = 'changed'  # type: ignore[misc]


def test_commitment_detects_tampered_fingerprint() -> None:
    commitment = DefenderCommitment.create('weak', DefenseAction(10.0, 0.0))

    with pytest.raises(ValueError, match='fingerprint'):
        replace(commitment, policy_fingerprint='0' * 64)


@pytest.mark.parametrize('commitment_id', ['', ' ', 1, True])
def test_commitment_rejects_invalid_id(commitment_id: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        DefenderCommitment.create(commitment_id, DefenseAction(1.0))  # type: ignore[arg-type]


def test_commitment_rejects_unknown_execution_semantics() -> None:
    commitment = DefenderCommitment.create('weak', DefenseAction(1.0))

    with pytest.raises(ValueError, match='aggregation_family'):
        replace(commitment, aggregation_family='unknown')
    with pytest.raises(ValueError, match='execution_protocol'):
        replace(commitment, execution_protocol='unknown')


@pytest.mark.parametrize('radius', [True, math.inf, math.nan])
def test_commitment_delegates_invalid_action_validation(radius: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        DefenderCommitment.create('weak', DefenseAction(radius))  # type: ignore[arg-type]


def test_fingerprint_distinguishes_exact_float_representations() -> None:
    lower = DefenderCommitment.create('lower', DefenseAction(1.0))
    adjacent = DefenderCommitment.create(
        'lower',
        DefenseAction(math.nextafter(1.0, math.inf)),
    )

    assert lower.policy_fingerprint != adjacent.policy_fingerprint
