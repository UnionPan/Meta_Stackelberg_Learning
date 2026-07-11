"""Trajectory collection primitives for paper-aligned per-round TD3 policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.agents.td3.replay import TD3ReplayBuffer, flatten_observation
from meta_stackelberg.agents.td3.config import ScaledMetaSGConfig
from meta_stackelberg.environments.paper_bsmg import PaperBSMGEnv, PaperRoundStep
from meta_stackelberg.stackelberg.algorithm1 import Algorithm1Result
from meta_stackelberg.stackelberg.algorithm2 import Algorithm2Result


DEFENDER_OBSERVATION_KEYS = ('model_tail', 'round_progress')
ATTACKER_OBSERVATION_KEYS = (
    'model_tail', 'round_progress', 'malicious_count', 'defender_action',
)


@dataclass(frozen=True)
class PaperTrajectory:
    generation: int
    steps: tuple[PaperRoundStep, ...]
    defender_return: float
    attacker_return: float


@dataclass(frozen=True)
class ScaledConformanceResult:
    passed: bool
    checks: tuple[tuple[str, bool, int, int], ...]
    scale_provenance: str


def evaluate_scaled_conformance(
    *,
    config: ScaledMetaSGConfig,
    algorithm1: Algorithm1Result,
    algorithm2: Algorithm2Result,
    trajectory: PaperTrajectory,
) -> ScaledConformanceResult:
    """Check paper parameter meanings against immutable execution traces."""
    counts = {
        'algorithm1_leader_iterations': len(algorithm1.iterations),
        'algorithm1_attacker_updates': sum(
            event.kind == 'attacker_update' for event in algorithm1.events
        ),
        'algorithm1_leader_updates': sum(
            event.kind == 'leader_update' for event in algorithm1.events
        ),
        'algorithm2_meta_iterations': len(algorithm2.iterations),
        'algorithm2_task_adaptations': sum(
            event.kind == 'adapt_task' for event in algorithm2.events
        ),
        'algorithm2_meta_updates': sum(
            event.kind == 'meta_update' for event in algorithm2.events
        ),
        'trajectory_fl_rounds': len(trajectory.steps),
        'defender_3d_actions': sum(
            step.defender_raw_action.shape == (3,) for step in trajectory.steps
        ),
        'attacker_3d_actions': sum(
            step.attacker_raw_action.shape == (3,) for step in trajectory.steps
        ),
    }
    expected = {
        'algorithm1_leader_iterations': config.N_D,
        'algorithm1_attacker_updates': config.N_D * config.K * config.N_A,
        'algorithm1_leader_updates': config.N_D,
        'algorithm2_meta_iterations': config.T,
        'algorithm2_task_adaptations': config.T * config.K * config.l,
        'algorithm2_meta_updates': config.T,
        'trajectory_fl_rounds': config.H,
        'defender_3d_actions': config.H,
        'attacker_3d_actions': config.H,
    }
    checks = tuple(
        (name, counts[name] == wanted, counts[name], wanted)
        for name, wanted in expected.items()
    )
    return ScaledConformanceResult(
        all(check[1] for check in checks), checks, config.scale_provenance,
    )


class PaperTD3TrajectoryCollector:
    """Collect one complete H-round trajectory with both policies acting each round."""

    def collect(
        self,
        *,
        env: PaperBSMGEnv,
        defender: TD3Agent,
        attacker: TD3Agent,
        defender_replay: TD3ReplayBuffer,
        attacker_replay: TD3ReplayBuffer,
        generation: int,
        deterministic: bool = False,
    ) -> PaperTrajectory:
        if defender.role != 'defender' or attacker.role != 'attacker':
            raise ValueError('trajectory policy roles do not match protocol')
        if defender_replay.role != 'defender' or attacker_replay.role != 'attacker':
            raise ValueError('trajectory replay roles do not match protocol')
        steps = []
        pending_attacker = None
        while env.state.round_index < env.horizon:
            defender_obs = flatten_observation(
                env.defender_observation(), DEFENDER_OBSERVATION_KEYS,
            )
            defender_action = defender.act(defender_obs, deterministic=deterministic)
            pending = env.begin_round(defender_action)
            attacker_obs = flatten_observation(
                pending.attacker_observation, ATTACKER_OBSERVATION_KEYS,
            )
            if pending_attacker is not None:
                old_obs, old_action, old_reward = pending_attacker
                attacker_replay.add(
                    old_obs, old_action, old_reward, attacker_obs, False,
                    generation=generation, role='attacker',
                )
            attacker_action = attacker.act(attacker_obs, deterministic=deterministic)
            step = env.finish_round(attacker_action)
            next_defender_obs = flatten_observation(
                env.defender_observation(), DEFENDER_OBSERVATION_KEYS,
            )
            defender_replay.add(
                defender_obs, defender_action, step.defender_reward.scalar,
                next_defender_obs, step.done,
                generation=generation, role='defender',
            )
            pending_attacker = (
                attacker_obs, attacker_action, step.attacker_reward.scalar,
            )
            steps.append(step)
        if pending_attacker is not None:
            obs, action, reward = pending_attacker
            attacker_replay.add(
                obs, action, reward, obs, True,
                generation=generation, role='attacker',
            )
        return PaperTrajectory(
            generation,
            tuple(steps),
            float(sum(step.defender_reward.scalar for step in steps)),
            float(sum(step.attacker_reward.scalar for step in steps)),
        )
