"""PettingZoo-style wrappers for attacker sandbox RL training.

This keeps the paper-style attacker simulator aligned with the rest of the
repo, which already uses PettingZoo `ParallelEnv` interfaces for meta-training.
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

import numpy as np

try:
    from gymnasium.spaces import Box, Dict as DictSpace, Discrete
except ImportError:  # pragma: no cover - lightweight fallback
    class Box:  # type: ignore[override]
        def __init__(self, low, high, shape=None, dtype=np.float32):
            if shape is None:
                low_arr = np.asarray(low, dtype=dtype)
                high_arr = np.asarray(high, dtype=dtype)
            else:
                low_arr = np.full(shape, low, dtype=dtype)
                high_arr = np.full(shape, high, dtype=dtype)
            self.low = low_arr
            self.high = high_arr
            self.shape = self.low.shape
            self.dtype = dtype

    class Discrete:  # type: ignore[override]
        def __init__(self, n):
            self.n = n

    class DictSpace(dict):  # type: ignore[override]
        def __init__(self, spaces):
            super().__init__(spaces)

try:
    from pettingzoo import ParallelEnv
except ImportError:  # pragma: no cover - lightweight fallback
    class ParallelEnv:  # type: ignore[override]
        metadata: dict[str, Any] = {}

if TYPE_CHECKING:
    from .attacker import RLAttackerConfig, SimulatedFLEnv


class AttackerPolicyParallelEnv(ParallelEnv):
    """Single-agent PettingZoo wrapper around the attacker simulator."""

    metadata = {"name": "attacker_policy_parallel_v0"}

    def __init__(
        self,
        simulator: "SimulatedFLEnv",
        rl_config: "RLAttackerConfig",
        defense_type: str,
        initial_weights,
    ) -> None:
        super().__init__()
        self.simulator = simulator
        self.rl_config = rl_config
        self.defense_type = defense_type
        self.initial_weights = initial_weights
        self.possible_agents = ["attacker"]
        self.agents = []
        low, high = rl_config.action_bounds(defense_type)
        self._action_space = Box(low=low, high=high, dtype=np.float32)
        flat_state = simulator.reset(initial_weights)
        pram_dim = flat_state.shape[0] - (1 if rl_config.state_include_num_attacker else 0)
        self._obs_space = DictSpace(
            {
                "pram": Box(low=-1.0, high=1.0, shape=(pram_dim,), dtype=np.float32),
                "num_attacker": Discrete(simulator.max_attackers_sampled + 1),
            }
        )
        self.agents = []

    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._action_space

    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        if seed is not None:
            np.random.seed(seed)
        self.agents = self.possible_agents[:]
        self.simulator.reset(self.initial_weights)
        obs = self.simulator._get_state_dict()
        info = {"defense_type": self.defense_type}
        return {"attacker": obs}, {"attacker": info}

    def step(self, actions):
        if not self.agents:
            raise RuntimeError("step() called on terminated AttackerPolicyParallelEnv")
        action = np.asarray(actions["attacker"], dtype=np.float32)
        _, reward, done = self.simulator.step(action)
        next_obs = self.simulator._get_state_dict()
        rewards = {"attacker": float(reward)}
        terminations = {"attacker": bool(done)}
        truncations = {"attacker": False}
        infos = {"attacker": {"defense_type": self.defense_type}}
        observations = {"attacker": next_obs}
        if done:
            self.agents = []
        return observations, rewards, terminations, truncations, infos
