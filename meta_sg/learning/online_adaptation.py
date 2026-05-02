"""
Online adaptation stage (paper §II-C).

Load theta_meta, interact with the real FL environment (100 clients),
update defender policy for limited steps using online samples.

Paper: 100 adaptation steps, T=10, H=100(MNIST)/200(CIFAR), l=10.
"""
from __future__ import annotations

import os
from typing import Callable, List, Optional

import numpy as np

from meta_sg.games.bsmg_env import BSMGEnv, BSMGConfig
from meta_sg.learning.config import MetaSGConfig, TD3Config
from meta_sg.learning.replay_buffer import ReplayBuffer
from meta_sg.learning.td3 import TD3Agent
from meta_sg.simulation.interface import FLCoordinator
from meta_sg.simulation.types import RoundSummary
from meta_sg.strategies.attacks.base import AttackStrategy
from meta_sg.strategies.defenses.paper import PaperDefenseStrategy
from meta_sg.strategies.types import AttackType


class OnlineAdaptation:
    """
    Adapts the pre-trained meta-policy to the actual FL environment.

    The defender sees real model updates and adjusts its policy using
    gradient steps from online experience. The attacker type may be
    unknown — the defender must generalise from pre-training.
    """

    def __init__(
        self,
        coordinator_factory: Callable[[], FLCoordinator],
        attack_type: AttackType,
        attack_strategy: AttackStrategy,
        obs_dim: int,
        meta_config: MetaSGConfig,
        td3_config: TD3Config,
        checkpoint_dir: Optional[str] = None,
        device=None,
    ) -> None:
        import torch
        self.device = device or torch.device("cpu")
        self.coordinator_factory = coordinator_factory
        self.attack_type = attack_type
        self.attack_strategy = attack_strategy
        self.meta_cfg = meta_config
        self.td3_cfg = td3_config

        self.defender = TD3Agent(obs_dim, 3, td3_config, self.device)
        self.buffer = ReplayBuffer(td3_config.buffer_capacity, obs_dim, 3)

        if checkpoint_dir is not None:
            self.load(checkpoint_dir)

    def load(self, directory: str) -> None:
        path = os.path.join(directory, "defender_meta.pt")
        if os.path.exists(path):
            self.defender.load(path)
            print(f"[OnlineAdapt] Loaded meta-policy from {path}")
        else:
            print(f"[OnlineAdapt] Warning: no checkpoint at {path}, using random init")

    def run(
        self,
        total_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[RoundSummary]:
        """
        Run online adaptation.
        Collect experience, update defender policy for l steps per episode.
        """
        cfg = self.meta_cfg
        H = cfg.online_H
        l = cfg.online_l
        steps = total_steps or cfg.online_steps

        coordinator = self.coordinator_factory()
        defense_strategy = PaperDefenseStrategy()
        env = BSMGEnv(
            coordinator=coordinator,
            attack_type=self.attack_type,
            attack_strategy=self.attack_strategy,
            defense_strategy=defense_strategy,
            config=BSMGConfig(horizon=H),
        )

        all_summaries: List[RoundSummary] = []
        step_count = 0
        episode = 0

        while step_count < steps:
            obs = env.reset(seed=seed)
            episode_done = False
            episode_summaries = []

            while not episode_done and step_count < steps:
                a_D = self.defender.get_action(obs, noise=self.td3_cfg.exploration_noise)
                a_A = np.random.uniform(-1, 1, 3).astype(np.float32)  # unknown attacker

                next_obs, r_D, r_A, done, info = env.step(a_D, a_A)
                self.buffer.add(obs, a_D, r_D, next_obs, done)
                obs = next_obs
                step_count += 1
                episode_done = done

                episode_summaries.append(info.get("clean_acc", 0.0))

            # Update defender for l steps on online data
            if self.buffer.ready:
                for _ in range(l):
                    self.defender.update(self.buffer)

            episode += 1
            mean_acc = np.mean(episode_summaries) if episode_summaries else 0.0
            print(
                f"[OnlineAdapt] episode {episode}  "
                f"steps={step_count}/{steps}  "
                f"mean_clean_acc={mean_acc:.4f}"
            )

        return all_summaries

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self.defender.save(os.path.join(directory, "defender_adapted.pt"))
        print(f"[OnlineAdapt] Saved adapted policy to {directory}")
