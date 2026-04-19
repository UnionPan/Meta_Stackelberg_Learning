"""Gymnasium wrapper aligned with the paper's SB3 training style."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium.spaces import Box, Dict as DictSpace, Discrete
except ImportError:  # pragma: no cover - lightweight fallback
    class _FallbackEnv:
        metadata: dict[str, Any] = {}

        def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
            return None, {}

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

    class _FallbackGym:
        Env = _FallbackEnv

    gym = _FallbackGym()


class AttackerPolicyGymEnv(gym.Env):
    """Single-agent Gym wrapper over the sandbox RL attacker simulator."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, simulator, rl_config, defense_type: str, initial_weights) -> None:
        super().__init__()
        self.simulator = simulator
        self.rl_config = rl_config
        self.defense_type = defense_type
        self.initial_weights = initial_weights
        low, high = rl_config.action_bounds(defense_type)
        self.action_space = Box(low=low, high=high, dtype=np.float32)
        flat_state = simulator.reset(initial_weights)
        pram_dim = flat_state.shape[0] - (1 if rl_config.state_include_num_attacker else 0)
        self.observation_space = DictSpace(
            {
                "pram": Box(low=-1.0, high=1.0, shape=(pram_dim,), dtype=np.float32),
                "num_attacker": Discrete(simulator.max_attackers_sampled + 1),
            }
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        if seed is not None:
            np.random.seed(seed)
        self.simulator.reset(self.initial_weights)
        return self.simulator._get_state_dict(), {"defense_type": self.defense_type}

    def step(self, action):
        _, reward, done = self.simulator.step(np.asarray(action, dtype=np.float32))
        return self.simulator._get_state_dict(), float(reward), bool(done), False, {"defense_type": self.defense_type}
