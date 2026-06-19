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
    eta: float = 0.01              # one-step adaptation step size
    meta_update_step: float = 1.0  # Reptile outer step (paper Appendix C-A)
    task_sampler: str = "iid"      # "iid" paper sampling | "stratified" coverage sampling
    gamma: float = 0.99            # discount factor for returns
    eval_every: int = 1            # full evaluation cadence inside env rollouts
    warmup_steps: int | None = 0     # optional random rollout before task adaptation
    history_len: int = 0           # append last-k action/reward/metric features to obs
    lambda_bd: float = 0.0         # model-poisoning default: no backdoor penalty
    reward_mode: str = "accuracy"  # BSMG defender reward mode: "accuracy" | "loss"
    defender_third_action: str = "neuroclip"  # "neuroclip" | "server_lr" | "both"
    server_lr_min: float = 0.0      # transition server-lr action lower bound
    server_lr_max: float = 1.0      # transition server-lr action upper bound
    server_lr_penalty_weight: float = 0.0  # reward penalty for shrinking transition server-lr
    native_sandbox_attacks: bool = False  # use fl_sandbox native attacks instead of meta_sg attack stubs

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
