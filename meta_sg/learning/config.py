"""
Hyperparameter config for Meta-SG learning.
Default values from paper §Appendix C-A (meta-learning setting).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TD3Config:
    """TD3 algorithm hyperparameters (paper: Stable-Baselines3 defaults)."""
    hidden_dim: int = 256
    policy_lr: float = 1e-3
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005              # Polyak averaging coefficient
    policy_delay: int = 2           # delayed policy update (every N critic steps)
    target_noise: float = 0.2       # target policy smoothing noise std
    noise_clip: float = 0.5         # clamp range for target noise
    exploration_noise: float = 0.1  # Gaussian noise added during data collection
    batch_size: int = 256
    buffer_capacity: int = 100_000
    warmup_steps: int = 1_000       # random actions before policy-guided collection


@dataclass
class MetaSGConfig:
    """
    Meta-Stackelberg Learning hyperparameters.
    Paper §Appendix C: pre-training and online-adaptation settings.
    """
    # Pre-training (Algorithm 2)
    T: int = 100          # outer loop iterations
    K: int = 10           # attack types sampled per iteration
    H_mnist: int = 200    # trajectory horizon for MNIST
    H_cifar: int = 500    # trajectory horizon for CIFAR-10
    l: int = 10           # inner TD3 update steps (= N_D)
    N_A: int = 10         # attacker best-response update steps
    post_br_defender_updates: int = 1  # extra defender updates after adaptive attacker BR
    kappa_D: float = 0.001         # meta-optimisation step (Reptile)
    kappa_A: float = 0.001         # attacker step size
    eta: float = 0.01              # one-step adaptation step size
    meta_update_step: float = 1.0  # Reptile meta step (κ_D=1 in paper)
    gamma: float = 0.99            # discount factor for returns
    eval_every: int = 1            # full evaluation cadence inside env rollouts
    warmup_steps: int | None = 0     # optional random rollout before task adaptation

    # Online adaptation
    online_T: int = 10
    online_H_mnist: int = 100
    online_H_cifar: int = 200
    online_l: int = 10
    online_steps: int = 100        # total online adaptation steps

    # Dataset
    dataset: str = "mnist"         # "mnist" | "cifar10"

    @property
    def H(self) -> int:
        return self.H_mnist if self.dataset == "mnist" else self.H_cifar

    @property
    def online_H(self) -> int:
        return self.online_H_mnist if self.dataset == "mnist" else self.online_H_cifar
