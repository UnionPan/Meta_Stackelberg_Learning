from __future__ import annotations

import pytest

from meta_stackelberg.environments.backdoor_rewards import (
    evaluate_whitebox_backdoor_rewards,
)


def test_whitebox_rewards_keep_defender_and_attacker_objectives_distinct() -> None:
    defender, attacker = evaluate_whitebox_backdoor_rewards(
        clean_loss=0.2,
        safe_loss=0.4,
        target_loss=0.1,
        clean_damage=0.05,
        defender_lambda=0.5,
        attacker_lambda=0.5,
    )

    assert defender.scalar == pytest.approx(-0.3)
    assert defender.clean_loss == 0.2
    assert defender.safe_loss == 0.4
    assert attacker.scalar == pytest.approx(-0.075)
    assert attacker.target_loss == 0.1
    assert attacker.clean_damage == 0.05
    assert defender.source == 'mnist-whitebox-real-data-v1'
    assert attacker.source == 'mnist-whitebox-real-data-v1'


def test_whitebox_reward_weights_have_unambiguous_endpoints() -> None:
    defender, attacker = evaluate_whitebox_backdoor_rewards(
        clean_loss=0.2,
        safe_loss=0.4,
        target_loss=0.1,
        clean_damage=0.05,
        defender_lambda=1.0,
        attacker_lambda=0.0,
    )

    assert defender.scalar == pytest.approx(-0.4)
    assert attacker.scalar == pytest.approx(-0.1)


@pytest.mark.parametrize(
    'overrides',
    [
        {'clean_loss': -0.1},
        {'safe_loss': float('nan')},
        {'target_loss': float('inf')},
        {'clean_damage': -0.1},
        {'defender_lambda': 1.1},
        {'attacker_lambda': -0.1},
    ],
)
def test_whitebox_rewards_reject_invalid_components(overrides: dict[str, float]) -> None:
    values = {
        'clean_loss': 0.2,
        'safe_loss': 0.4,
        'target_loss': 0.1,
        'clean_damage': 0.05,
        'defender_lambda': 0.5,
        'attacker_lambda': 0.5,
    }
    values.update(overrides)

    with pytest.raises(ValueError):
        evaluate_whitebox_backdoor_rewards(**values)
