"""Immutable parameter ledger for the paper-aligned Meta-SG implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


@dataclass(frozen=True)
class PaperMetaSGConfig:
    T: int = 100
    K: int = 10
    H_mnist: int = 200
    H_cifar: int = 500
    l: int = 10
    N_A: int = 10
    N_D: int = 10
    online_T: int = 10
    online_H_mnist: int = 100
    online_H_cifar: int = 200
    online_l: int = 10
    online_steps: int = 100
    rl_training_rounds: int = 300
    full_fl_rounds_mnist: int = 500
    full_fl_rounds_cifar: int = 1000
    policy_learning_rate: float = 0.001
    td3_batch_size: int = 256
    gamma: float = 0.99
    fl_batch_size: int = 128
    local_iterations: int = 1
    client_learning_rate: float = 0.05
    workers: int = 100
    untargeted_attackers: int = 20
    backdoor_attackers: int = 5
    subsampling_rate: float = 0.1
    root_samples_mnist: int = 100
    root_samples_cifar: int = 200
    non_iid_q: float = 0.5
    generated_seed_samples: int = 200
    generated_seed_q: float = 0.1
    default_backdoor_reward_lambda: float = 0.5
    kappa: float = 0.001
    kappa_attacker: float = 0.001
    kappa_defender: float = 0.001
    meta_update_step: float = 1.0
    adaptation_step: float = 0.01
    defender_action_dim: int = 3
    attacker_action_dim: int = 3
    state_encoder: str = 'last-two-learnable-blocks-v1'
    tau: float = 0.005
    policy_delay: int = 2
    target_policy_noise: float = 0.2
    noise_clip: float = 0.5
    replay_capacity: int = 1_000_000
    learning_starts: int = 100
    train_freq: int = 1
    gradient_steps: int = 1
    hidden_sizes: tuple[int, ...] = (256, 256)

    def __post_init__(self) -> None:
        for name in (
            'T', 'K', 'H_mnist', 'H_cifar', 'l', 'N_A', 'N_D', 'online_T',
            'online_H_mnist', 'online_H_cifar', 'online_l', 'online_steps',
            'rl_training_rounds', 'full_fl_rounds_mnist',
            'full_fl_rounds_cifar',
            'td3_batch_size', 'fl_batch_size', 'local_iterations', 'workers',
            'untargeted_attackers', 'backdoor_attackers',
            'root_samples_mnist', 'root_samples_cifar',
            'generated_seed_samples', 'policy_delay', 'replay_capacity',
            'learning_starts', 'train_freq', 'gradient_steps',
        ):
            _positive_integer(getattr(self, name), name)
        if self.defender_action_dim != 3 or self.attacker_action_dim != 3:
            raise ValueError('paper action dimensions must both equal 3')
        if self.state_encoder != 'last-two-learnable-blocks-v1':
            raise ValueError('paper state encoder semantics cannot be changed')
        for name in (
            'policy_learning_rate', 'client_learning_rate', 'kappa',
            'kappa_attacker', 'kappa_defender', 'meta_update_step',
            'adaptation_step', 'tau', 'target_policy_noise', 'noise_clip',
        ):
            _positive_finite(getattr(self, name), name)
        for name in (
            'gamma', 'subsampling_rate', 'non_iid_q', 'generated_seed_q',
            'default_backdoor_reward_lambda',
        ):
            value = _positive_finite(getattr(self, name), name)
            if value > 1.0:
                raise ValueError(f'{name} must be at most one')
        if not self.hidden_sizes or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.hidden_sizes
        ):
            raise ValueError('hidden_sizes must contain positive integers')
        if self.untargeted_attackers >= self.workers:
            raise ValueError('untargeted_attackers must be less than workers')
        if self.backdoor_attackers >= self.workers:
            raise ValueError('backdoor_attackers must be less than workers')

    def parameter_source(self, name: str) -> str:
        if name in {
            'T', 'K', 'H_mnist', 'H_cifar', 'l', 'N_A', 'N_D',
            'online_T', 'online_H_mnist', 'online_H_cifar', 'online_l',
            'online_steps',
            'policy_learning_rate', 'td3_batch_size', 'gamma', 'fl_batch_size',
            'local_iterations', 'client_learning_rate', 'workers',
            'untargeted_attackers', 'subsampling_rate', 'kappa',
            'kappa_attacker', 'kappa_defender', 'meta_update_step',
            'adaptation_step',
            'rl_training_rounds', 'full_fl_rounds_mnist',
            'full_fl_rounds_cifar', 'backdoor_attackers',
            'root_samples_mnist', 'root_samples_cifar', 'non_iid_q',
            'generated_seed_samples', 'generated_seed_q',
            'default_backdoor_reward_lambda',
        }:
            return 'paper-explicit'
        if name in {
            'tau', 'policy_delay', 'target_policy_noise', 'noise_clip',
            'replay_capacity', 'learning_starts', 'train_freq',
            'gradient_steps', 'hidden_sizes',
        }:
            return 'sb3-compatible-declared'
        if name in {
            'defender_action_dim', 'attacker_action_dim', 'state_encoder',
        }:
            return 'paper-semantic-contract'
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> PaperMetaSGConfig:
        normalized = dict(values)
        if 'hidden_sizes' in normalized:
            normalized['hidden_sizes'] = tuple(normalized['hidden_sizes'])
        return cls(**normalized)

    def scaled(
        self,
        *,
        T: int,
        K: int,
        H: int,
        l: int,
        N_A: int,
        N_D: int,
        workers: int,
        untargeted_attackers: int,
        sample_size: int,
        td3_batch_size: int,
        learning_starts: int,
        hidden_sizes: tuple[int, ...],
        replay_capacity: int,
    ) -> ScaledMetaSGConfig:
        return ScaledMetaSGConfig(
            self, T, K, H, l, N_A, N_D, workers, untargeted_attackers,
            sample_size, td3_batch_size, learning_starts, hidden_sizes,
            replay_capacity,
        )


@dataclass(frozen=True)
class ScaledMetaSGConfig:
    paper_reference: PaperMetaSGConfig
    T: int
    K: int
    H: int
    l: int
    N_A: int
    N_D: int
    workers: int
    untargeted_attackers: int
    sample_size: int
    td3_batch_size: int
    learning_starts: int
    hidden_sizes: tuple[int, ...]
    replay_capacity: int
    scale_provenance: str = 'scaled-conformance-only-v1'

    def __post_init__(self) -> None:
        if not isinstance(self.paper_reference, PaperMetaSGConfig):
            raise TypeError('paper_reference must be PaperMetaSGConfig')
        for name in (
            'T', 'K', 'H', 'l', 'N_A', 'N_D', 'workers',
            'untargeted_attackers', 'sample_size', 'td3_batch_size',
            'learning_starts', 'replay_capacity',
        ):
            _positive_integer(getattr(self, name), name)
        if self.untargeted_attackers >= self.workers:
            raise ValueError('untargeted_attackers must be less than workers')
        if self.sample_size > self.workers:
            raise ValueError('sample_size must not exceed workers')
        if not self.hidden_sizes or any(value <= 0 for value in self.hidden_sizes):
            raise ValueError('hidden_sizes must contain positive values')

    @property
    def defender_action_dim(self) -> int:
        return self.paper_reference.defender_action_dim

    @property
    def attacker_action_dim(self) -> int:
        return self.paper_reference.attacker_action_dim

    @property
    def state_encoder(self) -> str:
        return self.paper_reference.state_encoder


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f'{name} must be an integer')
    if value <= 0:
        raise ValueError(f'{name} must be positive')
    return value


def _positive_finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f'{name} must be a real number')
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f'{name} must be finite and positive')
    return result
