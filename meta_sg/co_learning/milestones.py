"""Small M1/M2 building blocks for Meta-SG co-learning experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from meta_sg.games.bsmg_env import BSMGConfig, BSMGEnv
from meta_sg.games.observations import obs_dim_for
from meta_sg.games.trajectory import Trajectory
from meta_sg.learning.collector import TrajectoryCollector
from meta_sg.learning.config import TD3Config
from meta_sg.learning.replay_buffer import ReplayBuffer
from meta_sg.learning.td3 import TD3Agent
from meta_sg.simulation.interface import FLCoordinator
from meta_sg.strategies.attacks.adaptive import AdaptiveAttackStrategy
from meta_sg.strategies.defenses.paper import PaperDefenseStrategy
from meta_sg.strategies.types import AttackType


@dataclass(frozen=True)
class HengerAttackerAction:
    """Decoded Henger-style attacker action."""

    raw: np.ndarray
    epsilon: float
    local_steps: int


@dataclass(frozen=True)
class M1DefenderAction:
    """Decoded clipped-median defender action used by M1/M2."""

    raw: np.ndarray
    clip_radius: float
    trim_ratio: float


@dataclass
class M1EpisodeResult:
    """Summary from one frozen-defender M1 attacker-response rollout."""

    transitions_collected: int
    attacker_buffer_size: int
    mean_epsilon: float
    mean_local_steps: float
    mean_attacker_reward: float
    clip_radius: float
    trim_ratio: float
    mean_survival: float
    trajectory: Trajectory
    attacker_updates: int = 0
    attacker_update_losses: list[dict[str, float]] | None = None


class FixedClippedMedianPolicy:
    """Policy object that always returns one raw clipped-median action."""

    def __init__(self, *, clip_radius: float = 2.5, trim_ratio: float = 0.2) -> None:
        self.clip_radius = float(clip_radius)
        self.trim_ratio = float(trim_ratio)

    def get_action(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        del obs, kwargs
        clip_raw = (np.clip(self.clip_radius, 0.5, 4.5) - 2.5) / 2.0
        trim_raw = (np.clip(self.trim_ratio, 0.0, 0.4) / 0.4) * 2.0 - 1.0
        return np.asarray([clip_raw, trim_raw], dtype=np.float32)


def decode_henger_attacker_action(raw: np.ndarray) -> HengerAttackerAction:
    """Decode raw [-1, 1]^2 action to Henger attacker parameters."""
    action = _fixed_len_action(raw, 2)
    epsilon = float(action[0]) * 14.9 + 15.0
    local_steps = max(1, int(round(float(action[1]) * 24.0 + 25.0)))
    return HengerAttackerAction(
        raw=action,
        epsilon=float(np.clip(epsilon, 0.1, 29.9)),
        local_steps=int(np.clip(local_steps, 1, 49)),
    )


def decode_m1_defender_action(raw: np.ndarray) -> M1DefenderAction:
    """Decode raw [-1, 1]^2 action to clipped-median M1/M2 parameters."""
    action = _fixed_len_action(raw, 2)
    clip_radius = float(action[0]) * 2.0 + 2.5
    trim_ratio = (float(action[1]) + 1.0) / 2.0 * 0.4
    return M1DefenderAction(
        raw=action,
        clip_radius=float(np.clip(clip_radius, 0.5, 4.5)),
        trim_ratio=float(np.clip(trim_ratio, 0.0, 0.4)),
    )


def poison_survival_cosine(malicious_update: np.ndarray, aggregate_update: np.ndarray) -> float:
    """Cosine alignment between malicious direction and final aggregate update."""
    malicious = np.asarray(malicious_update, dtype=np.float32).reshape(-1)
    aggregate = np.asarray(aggregate_update, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(malicious) * np.linalg.norm(aggregate))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(malicious, aggregate) / denom)


def run_m1_episode(
    *,
    coordinator: FLCoordinator,
    attacker: TD3Agent,
    defender: FixedClippedMedianPolicy,
    horizon: int,
    seed: int | None = None,
    td3_config: TD3Config | None = None,
    br_updates: int = 0,
) -> M1EpisodeResult:
    """Collect one M1 rollout against a frozen clipped-median defender."""
    obs_dim = obs_dim_for(coordinator.spec.empty_weights())
    cfg = td3_config or attacker.cfg
    attacker_buffer = ReplayBuffer(cfg.buffer_capacity, obs_dim, attacker.act_dim)
    defender_buffer = ReplayBuffer(cfg.buffer_capacity, obs_dim, 3)
    attack_type = AttackType(name="rl", objective="untargeted", adaptive=True)
    env = BSMGEnv(
        coordinator=coordinator,
        attack_type=attack_type,
        attack_strategy=AdaptiveAttackStrategy(attack_type, attacker, noise=0.0),
        defense_strategy=PaperDefenseStrategy(),
        config=BSMGConfig(horizon=max(1, int(horizon)), eval_every=1),
    )
    collector = TrajectoryCollector(
        env=env,
        defender=_EnvFixedDefenderPolicy(defender),
        attacker=attacker,
        defender_buffer=defender_buffer,
        attacker_buffer=attacker_buffer,
        exploration_noise=0.0,
        store_attacker=True,
    )
    trajectory = collector.collect(max(1, int(horizon)), seed=seed)
    update_losses: list[dict[str, float]] = []
    for _ in range(max(0, int(br_updates))):
        losses = attacker.update(attacker_buffer)
        if losses:
            update_losses.append(losses)
    attacker_actions = [
        decode_henger_attacker_action(transition.attacker_action)
        for transition in trajectory.transitions
    ]
    defender_action = decode_m1_defender_action(defender.get_action(np.zeros(obs_dim, dtype=np.float32)))
    rewards = [transition.attacker_reward for transition in trajectory.transitions]
    survivals = [
        _transition_survival(transition.info)
        for transition in trajectory.transitions
    ]
    return M1EpisodeResult(
        transitions_collected=len(trajectory.transitions),
        attacker_buffer_size=len(attacker_buffer),
        mean_epsilon=float(np.mean([action.epsilon for action in attacker_actions])) if attacker_actions else 0.0,
        mean_local_steps=float(np.mean([action.local_steps for action in attacker_actions])) if attacker_actions else 0.0,
        mean_attacker_reward=float(np.mean(rewards)) if rewards else 0.0,
        clip_radius=defender_action.clip_radius,
        trim_ratio=defender_action.trim_ratio,
        mean_survival=float(np.mean(survivals)) if survivals else 0.0,
        trajectory=trajectory,
        attacker_updates=max(0, int(br_updates)),
        attacker_update_losses=update_losses,
    )


def sweep_fixed_defenders(
    *,
    coordinator_factory: Callable[[], FLCoordinator],
    radii: Sequence[float],
    trim_ratio: float,
    horizon: int,
    td3_config: TD3Config | None = None,
    seed: int | None = None,
    br_updates: int = 0,
) -> list[M1EpisodeResult]:
    """Run one M1 rollout for each frozen defender clipping radius."""
    cfg = td3_config or TD3Config()
    results: list[M1EpisodeResult] = []
    for idx, radius in enumerate(radii):
        coordinator = coordinator_factory()
        obs_dim = obs_dim_for(coordinator.spec.empty_weights())
        attacker = TD3Agent(obs_dim=obs_dim, act_dim=2, config=cfg)
        result = run_m1_episode(
            coordinator=coordinator,
            attacker=attacker,
            defender=FixedClippedMedianPolicy(clip_radius=float(radius), trim_ratio=trim_ratio),
            horizon=horizon,
            seed=None if seed is None else int(seed) + idx,
            td3_config=cfg,
            br_updates=br_updates,
        )
        results.append(result)
    return results


class _EnvFixedDefenderPolicy:
    """Adapter that pads the 2D M1 defender action for current 3D BSMGEnv."""

    def __init__(self, policy: FixedClippedMedianPolicy) -> None:
        self.policy = policy

    def get_action(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        del kwargs
        action = self.policy.get_action(obs)
        return np.asarray([action[0], action[1], 0.0], dtype=np.float32)


def _transition_survival(info: dict) -> float:
    values = info.get("malicious_cosines_to_benign", [])
    if values:
        return float(np.mean(values))
    return 0.0


def _fixed_len_action(raw: np.ndarray, length: int) -> np.ndarray:
    action = np.asarray(raw, dtype=np.float32).reshape(-1)
    if action.shape[0] < length:
        action = np.pad(action, (0, length - action.shape[0]))
    return np.clip(action[:length], -1.0, 1.0).astype(np.float32)
