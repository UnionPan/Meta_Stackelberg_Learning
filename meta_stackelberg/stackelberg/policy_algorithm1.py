"""Concrete paper-order composition of policy-level Meta-SG Algorithm 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.stackelberg.policy_adaptation import (
    PolicyDefenderAdaptationResult,
    PolicyDefenderAdapter,
)
from meta_stackelberg.stackelberg.policy_leader import (
    PolicyLeaderResult,
    PolicyLeaderTask,
    PolicyLeaderTrainer,
)
from meta_stackelberg.stackelberg.policy_response import (
    PolicyBestResponseResult,
    PolicyBestResponseTrainer,
)


@dataclass(frozen=True)
class PolicyAlgorithm1TaskTrace:
    task: object
    adaptation: PolicyDefenderAdaptationResult
    response: PolicyBestResponseResult


@dataclass(frozen=True)
class PolicyAlgorithm1IterationTrace:
    leader_iteration: int
    meta_defender_before: str
    meta_defender_after: str
    tasks: tuple[PolicyAlgorithm1TaskTrace, ...]
    leader: PolicyLeaderResult


@dataclass(frozen=True)
class PolicyAlgorithm1Result:
    iterations: tuple[PolicyAlgorithm1IterationTrace, ...]
    protocol: str = 'paper-meta-sg-algorithm1-policy-v1'


class PolicyMetaSGAlgorithm1:
    """Adapt θξ, find BR against θt, then evaluate leader gradient at θξ."""

    def __init__(
        self,
        *,
        N_D: int,
        K: int,
        N_A: int,
        batch_size: int,
        eta: float,
        kappa_A: float,
        kappa_D: float,
    ) -> None:
        for value, name in (
            (N_D, 'N_D'), (K, 'K'), (N_A, 'N_A'),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        self.N_D = N_D
        self.K = K
        self.adapter = PolicyDefenderAdapter(
            l=1, batch_size=batch_size, eta=eta,
        )
        self.response_trainer = PolicyBestResponseTrainer(
            N_A=N_A, batch_size=batch_size, kappa_A=kappa_A,
        )
        self.leader_trainer = PolicyLeaderTrainer(
            defender_updates=1, batch_size=batch_size, kappa_D=kappa_D,
        )

    def run(
        self,
        *,
        defender: TD3Agent,
        attackers: Mapping[object, TD3Agent],
        sample_tasks,
        replay_factory,
        collect_adaptation,
        collect_response,
        collect_leader,
        independent_attacker_objective,
        start_iteration: int = 0,
        iteration_callback=None,
    ) -> PolicyAlgorithm1Result:
        if (
            isinstance(start_iteration, bool)
            or not isinstance(start_iteration, int)
            or start_iteration < 0
            or start_iteration > self.N_D
        ):
            raise ValueError('start_iteration must be within [0, N_D]')
        iterations = []
        for leader_iteration in range(start_iteration, self.N_D):
            tasks = tuple(sample_tasks(leader_iteration, self.K))
            if len(tasks) != self.K:
                raise ValueError(f'task sampler must return exactly K={self.K} tasks')
            meta_before = defender.fingerprint()
            task_traces = []
            leader_tasks = []
            for task in tasks:
                if task not in attackers:
                    raise KeyError(f'missing persistent attacker for task {task!r}')
                attacker = attackers[task]
                adaptation_replay = replay_factory(
                    task, 'defender', 'adaptation', leader_iteration,
                )
                adaptation = self.adapter.adapt(
                    defender=defender,
                    attacker=attacker,
                    replay=adaptation_replay,
                    collect_fresh=lambda adapted, frozen, replay, task=task: collect_adaptation(
                        task, adapted, frozen, replay, leader_iteration,
                    ),
                )
                response_replay = replay_factory(
                    task, 'attacker', 'best_response', leader_iteration,
                )
                response = self.response_trainer.train(
                    defender=defender,
                    attacker=attacker,
                    replay=response_replay,
                    collect_fresh=lambda step, task=task: collect_response(
                        task, step, defender, attacker,
                        response_replay,
                        leader_iteration,
                    ),
                    independent_objective=lambda policy, task=task: independent_attacker_objective(
                        task, policy,
                    ),
                )
                task_traces.append(PolicyAlgorithm1TaskTrace(
                    task, adaptation, response,
                ))
                leader_tasks.append(PolicyLeaderTask(
                    task,
                    adaptation.adapted_defender,
                    attacker,
                    replay_factory(task, 'defender', 'leader', leader_iteration),
                ))
            leader = self.leader_trainer.train(
                defender=defender,
                tasks=tuple(leader_tasks),
                collect_fresh=lambda task, adapted, frozen, replay: collect_leader(
                    task, adapted, frozen, replay, leader_iteration,
                ),
            )
            trace = PolicyAlgorithm1IterationTrace(
                leader_iteration,
                meta_before,
                defender.fingerprint(),
                tuple(task_traces),
                leader,
            )
            iterations.append(trace)
            if iteration_callback is not None:
                iteration_callback(
                    trace, defender, attackers,
                )
        return PolicyAlgorithm1Result(tuple(iterations))
