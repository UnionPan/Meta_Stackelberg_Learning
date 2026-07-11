"""Trajectory collection primitives for paper-aligned per-round TD3 policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.agents.td3.replay import TD3ReplayBuffer, flatten_observation
from meta_stackelberg.environments.paper_bsmg import PaperBSMGEnv, PaperRoundStep


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
