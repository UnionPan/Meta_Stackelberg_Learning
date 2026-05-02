"""
Attacker best-response inner loop (meta-SG mode).
For each adaptive attack type ξ, update φ_ξ for N_A steps.

Algorithm 1, lines 12-15:
  φ_ξ(k+1) = φ_ξ(k) + κ_A * ∇̂_φ J_A(θ, φ_ξ(k), ξ)
"""
from __future__ import annotations

import math
from typing import Dict, Optional
from meta_sg.learning.replay_buffer import ReplayBuffer
from meta_sg.learning.td3 import TD3Agent
from meta_sg.strategies.types import AttackType


class AttackerBestResponse:
    """
    Update the attacker's policy φ_ξ toward the best response
    against the current defender θ.

    For meta-SG: called inside MetaSGTrainer after collecting trajectory.
    For meta-RL: not used (non-adaptive attackers).
    """

    def __init__(
        self,
        attacker_agents: Dict[str, TD3Agent],
        attacker_buffers: Dict[str, ReplayBuffer],
        n_a: int = 10,
    ) -> None:
        self.attacker_agents = attacker_agents
        self.attacker_buffers = attacker_buffers
        self.n_a = n_a

    def update(
        self,
        attack_type: AttackType,
        steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Update attacker policy for `steps` TD3 gradient steps.
        Returns mean losses; actor_loss excludes NaN steps (delayed actor update).
        """
        key = attack_type.name
        if key not in self.attacker_agents:
            return {}

        agent = self.attacker_agents[key]
        buffer = self.attacker_buffers[key]
        n_steps = steps or self.n_a

        critic_sum = 0.0
        q_sum = 0.0
        actor_sum = 0.0
        actor_count = 0

        for _ in range(n_steps):
            step_loss = agent.update(buffer)
            if not step_loss:
                continue
            critic_sum += step_loss.get("critic_loss", 0.0)
            q_sum += step_loss.get("q_mean", 0.0)
            al = step_loss.get("actor_loss", float("nan"))
            if not math.isnan(al):
                actor_sum += al
                actor_count += 1

        return {
            "critic_loss": critic_sum / n_steps,
            "actor_loss": actor_sum / actor_count if actor_count > 0 else float("nan"),
            "q_mean": q_sum / n_steps,
        }
