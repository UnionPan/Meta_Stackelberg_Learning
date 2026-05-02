"""
Reward functions for the BSMG game (paper §II-A, Remark 1).

Defender reward:  r_D = -F(ŵ_g^t)  where ŵ = h(w_{t+1})
Attacker reward:  r_A = -F'(ŵ_g^t)

In practice we approximate using test accuracy:
  r_D = clean_acc(ŵ) - λ_bd * backdoor_acc(ŵ)
  r_A = backdoor_acc(ŵ)   [targeted]
       -clean_acc(ŵ)       [untargeted]

The post-training weights ŵ are passed directly — this module is stateless.
"""
from __future__ import annotations

from meta_sg.simulation.types import RoundSummary
from meta_sg.strategies.types import AttackType


def defender_reward(
    summary: RoundSummary,
    lambda_bd: float = 1.0,
) -> float:
    """
    r_D = clean_acc(ŵ) - lambda_bd * backdoor_acc(ŵ)

    lambda_bd=1.0 matches the paper's linear Acc-Bac objective.
    """
    return float(summary.clean_acc - lambda_bd * summary.backdoor_acc)


def attacker_reward(
    summary: RoundSummary,
    attack_type: AttackType,
) -> float:
    """
    Targeted (backdoor):   r_A = backdoor_acc(ŵ)
    Untargeted (poisoning): r_A = -clean_acc(ŵ)

    Both are negated from the paper's min-loss formulation since
    we maximise rewards.
    """
    if attack_type.objective == "targeted":
        return float(summary.backdoor_acc)
    return float(-summary.clean_acc)
