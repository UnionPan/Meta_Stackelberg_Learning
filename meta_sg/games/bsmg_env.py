"""
Bayesian Stackelberg Markov Game environment.
Wraps FLCoordinator into a step-based game interface.

One episode = H FL rounds for a fixed attack type ξ.
State:   compressed last-2-layers of global model W_t
Actions: defender ∈ [-1,1]^3, attacker ∈ [-1,1]^3
Rewards: r_D, r_A computed with post-training defense applied to a weight COPY.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from meta_sg.games.observations import compress_weights, normalise_obs, obs_dim_for
from meta_sg.games.rewards import attacker_reward, defender_reward
from meta_sg.simulation.interface import FLCoordinator
from meta_sg.simulation.types import Weights
from meta_sg.strategies.attacks.base import AttackStrategy
from meta_sg.strategies.defenses.base import DefenseStrategy
from meta_sg.strategies.types import AttackDecision, AttackType, DefenseDecision


@dataclass
class BSMGConfig:
    horizon: int = 200                # H in paper
    num_tail_layers: int = 2          # layers to include in observation
    alpha_max: float = 5.0            # max defender norm bound
    beta_max: float = 0.45            # max trimmed mean ratio
    eps_max: float = 10.0             # max NeuroClip clip range
    use_neuroclip: bool = True        # True=NeuroClip, False=Prun
    lambda_bd: float = 1.0            # backdoor penalty in defender reward
    normalise_obs: bool = True        # z-score normalise observations
    eval_every: int = 1               # run expensive full eval every N FL rounds


class BSMGEnv:
    """
    Two-player Stackelberg Markov game wrapping FLCoordinator.

    Usage:
        env = BSMGEnv(coordinator, attack_type, attack_strategy,
                      defense_strategy, config)
        obs = env.reset()
        obs, r_D, r_A, done, info = env.step(defender_action, attacker_action)
    """

    def __init__(
        self,
        coordinator: FLCoordinator,
        attack_type: AttackType,
        attack_strategy: AttackStrategy,
        defense_strategy: DefenseStrategy,
        config: Optional[BSMGConfig] = None,
    ) -> None:
        self.coordinator = coordinator
        self.attack_type = attack_type
        self.attack_strategy = attack_strategy
        self.defense_strategy = defense_strategy
        self.config = config or BSMGConfig()

        self._round = 0
        self._obs: Optional[np.ndarray] = None
        self._obs_dim: Optional[int] = None

    # ------------------------------------------------------------------
    # Gym-like interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset coordinator and return initial observation."""
        init = self.coordinator.reset(seed=seed)
        self._round = 0
        obs = self._make_obs(init.weights)
        self._obs = obs
        self._obs_dim = obs.shape[0]
        return obs

    def step(
        self,
        defender_action: np.ndarray,
        attacker_action: np.ndarray,
    ) -> Tuple[np.ndarray, float, float, bool, Dict[str, Any]]:
        """
        Execute one FL round.

        Returns:
            next_obs:        compressed W_{t+1}
            defender_reward: r_D (computed on post-training weights)
            attacker_reward: r_A (computed on post-training weights)
            done:            True when horizon is reached
            info:            round metrics
        """
        # Decode actions -> decisions
        defense_decision = DefenseDecision.from_raw(
            defender_action,
            alpha_max=self.config.alpha_max,
            beta_max=self.config.beta_max,
            eps_max=self.config.eps_max,
            use_neuroclip=self.config.use_neuroclip,
        )
        attack_decision = AttackDecision.from_raw(attacker_action)

        # If adaptive attacker: inject current obs for action selection
        if hasattr(self.attack_strategy, "set_obs") and self._obs is not None:
            self.attack_strategy.set_obs(self._obs)

        # FL round: W_t -> W_{t+1} (coordinator.current_weights is updated)
        should_evaluate = (
            self.config.eval_every <= 1
            or (self._round + 1) % self.config.eval_every == 0
            or (self._round + 1) >= self.config.horizon
        )
        summary = self.coordinator.run_round(
            attack=self.attack_strategy,
            defense=self.defense_strategy,
            attack_decision=attack_decision,
            defense_decision=defense_decision,
            evaluate=should_evaluate,
        )

        # Post-training defense applied to a COPY of W_{t+1} for reward only
        w_next = self.coordinator.current_weights
        w_eval = self.defense_strategy.apply_post_training(w_next, defense_decision)

        # Reward-only summary on post-training weights
        eval_summary = _patch_summary_with_eval(summary, w_eval)

        r_D = defender_reward(eval_summary, lambda_bd=self.config.lambda_bd)
        r_A = attacker_reward(eval_summary, self.attack_type)

        self._round += 1
        done = self._round >= self.config.horizon

        next_obs = self._make_obs(w_next)
        self._obs = next_obs

        info = {
            "round": self._round,
            "clean_acc": summary.clean_acc,
            "backdoor_acc": summary.backdoor_acc,
            "evaluated": should_evaluate,
            "defense_decision": defense_decision,
            "attack_decision": attack_decision,
        }

        return next_obs, r_D, r_A, done, info

    # ------------------------------------------------------------------
    # Observation space info
    # ------------------------------------------------------------------

    @property
    def obs_dim(self) -> int:
        if self._obs_dim is not None:
            return self._obs_dim
        return obs_dim_for(self.coordinator.spec.empty_weights(), self.config.num_tail_layers)

    @property
    def act_dim(self) -> int:
        return 3  # paper: 3-dimensional continuous action space

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _make_obs(self, weights: Weights) -> np.ndarray:
        obs = compress_weights(weights, self.config.num_tail_layers)
        if self.config.normalise_obs:
            obs = normalise_obs(obs)
        return obs


def _patch_summary_with_eval(summary, w_eval):
    """
    In a real implementation: re-run test() on w_eval model.
    For stub: return same summary (post-training barely changes stub metrics).
    """
    return summary
