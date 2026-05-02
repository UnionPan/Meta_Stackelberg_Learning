"""
Trajectory collector — bridges BSMGEnv and replay buffers.
"""
from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

from meta_sg.games.bsmg_env import BSMGEnv
from meta_sg.games.trajectory import Trajectory
from meta_sg.learning.replay_buffer import ReplayBuffer


class ActionPolicy(Protocol):
    def get_action(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        ...


class RandomPolicy:
    """Uniform random policy for warmup data collection."""
    def __init__(self, act_dim: int = 3) -> None:
        self.act_dim = act_dim

    def get_action(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        return np.random.uniform(-1, 1, size=self.act_dim).astype(np.float32)


class TrajectoryCollector:
    """
    Collect H-step trajectories from BSMGEnv using given policies.
    Fills separate replay buffers for defender and attacker.
    """

    def __init__(
        self,
        env: BSMGEnv,
        defender: ActionPolicy,
        attacker: ActionPolicy,
        defender_buffer: ReplayBuffer,
        attacker_buffer: Optional[ReplayBuffer],
        exploration_noise: float = 0.1,
        store_attacker: bool = True,
    ) -> None:
        self.env = env
        self.defender = defender
        self.attacker = attacker
        self.defender_buffer = defender_buffer
        self.attacker_buffer = attacker_buffer
        self.exploration_noise = exploration_noise
        self.store_attacker = store_attacker

    def collect(
        self,
        horizon: int,
        seed: Optional[int] = None,
        use_random: bool = False,
    ) -> Trajectory:
        """
        Collect one trajectory of `horizon` steps.
        Returns Trajectory and also fills both replay buffers.
        """
        obs = self.env.reset(seed=seed)
        traj = Trajectory(attack_type=self.env.attack_type)

        d_policy = RandomPolicy(self.env.act_dim) if use_random else self.defender
        a_policy = RandomPolicy(self.env.act_dim) if use_random else self.attacker

        for _ in range(horizon):
            a_D = d_policy.get_action(obs, noise=self.exploration_noise)
            a_A = a_policy.get_action(obs, noise=self.exploration_noise)

            next_obs, r_D, r_A, done, info = self.env.step(a_D, a_A)

            traj.transitions.append(
                _make_transition(obs, a_D, a_A, r_D, r_A, next_obs, done, info)
            )

            self.defender_buffer.add(obs, a_D, r_D, next_obs, done)
            if self.store_attacker and self.attacker_buffer is not None:
                self.attacker_buffer.add(obs, a_A, r_A, next_obs, done)

            obs = next_obs
            if done:
                break

        return traj

    def warmup(self, steps: int) -> None:
        """Fill buffers with random actions to bootstrap training."""
        obs = self.env.reset()
        collected = 0
        while collected < steps:
            a_D = np.random.uniform(-1, 1, self.env.act_dim).astype(np.float32)
            a_A = np.random.uniform(-1, 1, self.env.act_dim).astype(np.float32)
            next_obs, r_D, r_A, done, _ = self.env.step(a_D, a_A)
            self.defender_buffer.add(obs, a_D, r_D, next_obs, done)
            if self.store_attacker and self.attacker_buffer is not None:
                self.attacker_buffer.add(obs, a_A, r_A, next_obs, done)
            obs = next_obs if not done else self.env.reset()
            collected += 1


def _make_transition(obs, a_D, a_A, r_D, r_A, next_obs, done, info):
    from meta_sg.games.trajectory import Transition
    return Transition(
        state=obs.copy(),
        defender_action=a_D.copy(),
        attacker_action=a_A.copy(),
        defender_reward=r_D,
        attacker_reward=r_A,
        next_state=next_obs.copy(),
        done=done,
        info=info,
    )
