import dataclasses
import math

import pytest

from meta_stackelberg.environments.rewards import evaluate_paper_untargeted_rewards


def test_paper_rewards_use_same_post_defense_loss_objective() -> None:
    defender, attacker = evaluate_paper_untargeted_rewards(
        post_loss_before=1.5,
        post_loss_after=2.0,
    )
    assert defender.scalar == -2.0
    assert defender.post_defense_loss == 2.0
    assert attacker.scalar == 0.5
    assert attacker.loss_increase == 0.5
    assert defender.source == 'paper-untargeted-post-defense-v1'
    with pytest.raises(dataclasses.FrozenInstanceError):
        attacker.scalar = 0.0  # type: ignore[misc]


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf, True])
def test_paper_rewards_reject_nonfinite_or_bool_losses(value) -> None:
    with pytest.raises((TypeError, ValueError)):
        evaluate_paper_untargeted_rewards(post_loss_before=value, post_loss_after=1.0)
    with pytest.raises((TypeError, ValueError)):
        evaluate_paper_untargeted_rewards(post_loss_before=1.0, post_loss_after=value)
