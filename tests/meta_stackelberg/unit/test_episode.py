from dataclasses import dataclass, field

import numpy as np
import pytest

from meta_stackelberg.core.model_state import ModelState, apply_delta
from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.types import (
    ClientUpdate,
    RoundRequest,
    RoundState,
    RoundTransition,
)


def _state(seed: int = 3, value: float = 0.0) -> RoundState:
    source = RandomSource(seed)
    return RoundState(
        round_index=0,
        global_model=ModelState.from_tensors((np.array([value], dtype=np.float32),)),
        random_snapshot=source.capture(),
    )


@dataclass
class IncrementingExecutor:
    task_override: str | None = None
    unrelated_state: bool = False
    requests: list[RoundRequest] = field(default_factory=list)

    def run_round(self, request: RoundRequest, rng: RandomSource) -> RoundTransition:
        self.requests.append(request)
        rng.restore(request.state.random_snapshot)
        rng.spawn()
        before = request.state
        if self.unrelated_state:
            before = RoundState(
                round_index=request.state.round_index,
                global_model=ModelState.from_tensors((np.array([99.0], dtype=np.float32),)),
                random_snapshot=request.state.random_snapshot,
            )
        delta = ModelState.from_tensors((np.array([1.0], dtype=np.float32),))
        update = ClientUpdate(client_id=0, delta=delta, num_examples=1)
        after = RoundState(
            round_index=before.round_index + 1,
            global_model=apply_delta(before.global_model, delta, request.server_lr),
            random_snapshot=rng.capture(),
        )
        return RoundTransition(
            task_id=self.task_override or request.task_id,
            state_before=before,
            sampled_clients=(0,),
            benign_updates=(update,),
            malicious_updates=(),
            aggregate_delta=delta,
            state_after=after,
        )


def test_episode_runner_executes_exact_horizon_and_returns_trajectory() -> None:
    from meta_stackelberg.federated.episode import EpisodeRunner, EpisodeSpec

    executor = IncrementingExecutor()
    initial = _state()
    spec = EpisodeSpec(
        task_id='clean-task',
        horizon=3,
        sample_size=1,
        server_lr=0.5,
        initial_state=initial,
    )

    trajectory = EpisodeRunner(executor).run(spec, RandomSource(999))

    assert trajectory.task_id == 'clean-task'
    assert trajectory.initial_state is initial
    assert len(trajectory.transitions) == 3
    assert trajectory.final_state is trajectory.transitions[-1].state_after
    np.testing.assert_array_equal(trajectory.final_state.global_model.vector(), [1.5])
    assert [request.state.round_index for request in executor.requests] == [0, 1, 2]


@pytest.mark.parametrize(
    'kwargs',
    [
        {'task_id': ''},
        {'horizon': 0},
        {'sample_size': 0},
        {'server_lr': float('nan')},
    ],
)
def test_episode_spec_rejects_invalid_contract(kwargs) -> None:
    from meta_stackelberg.federated.episode import EpisodeSpec

    values = {
        'task_id': 'clean-task',
        'horizon': 2,
        'sample_size': 1,
        'server_lr': 1.0,
        'initial_state': _state(),
    }
    values.update(kwargs)

    with pytest.raises(ValueError):
        EpisodeSpec(**values)


def test_episode_runner_satisfies_round_executor_plugin_contract() -> None:
    from meta_stackelberg.federated.protocols import RoundExecutor

    assert isinstance(IncrementingExecutor(), RoundExecutor)


@pytest.mark.parametrize(
    'executor',
    [
        IncrementingExecutor(task_override='wrong-task'),
        IncrementingExecutor(unrelated_state=True),
    ],
)
def test_episode_runner_rejects_invalid_transition_boundary(executor) -> None:
    from meta_stackelberg.federated.episode import EpisodeRunner, EpisodeSpec

    spec = EpisodeSpec(
        task_id='clean-task',
        horizon=1,
        sample_size=1,
        server_lr=1.0,
        initial_state=_state(),
    )

    with pytest.raises(ValueError):
        EpisodeRunner(executor).run(spec, RandomSource(5))


def test_episode_runner_matches_direct_round_loop() -> None:
    from meta_stackelberg.federated.episode import EpisodeRunner, EpisodeSpec

    initial = _state(seed=17)
    spec = EpisodeSpec(
        task_id='manual-equivalence',
        horizon=4,
        sample_size=1,
        server_lr=0.25,
        initial_state=initial,
    )
    trajectory = EpisodeRunner(IncrementingExecutor()).run(spec, RandomSource(1))

    executor = IncrementingExecutor()
    source = RandomSource(1)
    state = initial
    direct = []
    for _ in range(4):
        transition = executor.run_round(
            RoundRequest(
                task_id='manual-equivalence',
                state=state,
                sample_size=1,
                server_lr=0.25,
            ),
            source,
        )
        direct.append(transition)
        state = transition.state_after

    assert [step.sampled_clients for step in trajectory.transitions] == [
        step.sampled_clients for step in direct
    ]
    np.testing.assert_array_equal(
        trajectory.final_state.global_model.vector(),
        state.global_model.vector(),
    )


def test_federated_trajectory_normalizes_transition_list_and_is_immutable() -> None:
    from meta_stackelberg.federated.episode import FederatedTrajectory

    executor = IncrementingExecutor()
    initial = _state()
    transition = executor.run_round(
        RoundRequest(
            task_id='trajectory-contract',
            state=initial,
            sample_size=1,
            server_lr=1.0,
        ),
        RandomSource(2),
    )
    mutable_transitions = [transition]

    trajectory = FederatedTrajectory(
        task_id='trajectory-contract',
        initial_state=initial,
        transitions=mutable_transitions,
        final_state=transition.state_after,
    )
    mutable_transitions.append(transition)

    assert isinstance(trajectory.transitions, tuple)
    assert trajectory.transitions == (transition,)


@pytest.mark.parametrize('invalid_kind', ['empty-task', 'empty-chain', 'wrong-final'])
def test_federated_trajectory_rejects_invalid_boundary(invalid_kind: str) -> None:
    from meta_stackelberg.federated.episode import FederatedTrajectory

    initial = _state()
    transition = IncrementingExecutor().run_round(
        RoundRequest(
            task_id='trajectory-contract',
            state=initial,
            sample_size=1,
            server_lr=1.0,
        ),
        RandomSource(2),
    )
    task_id = '' if invalid_kind == 'empty-task' else 'trajectory-contract'
    transitions = () if invalid_kind == 'empty-chain' else (transition,)
    final_state = initial if invalid_kind == 'wrong-final' else transition.state_after

    with pytest.raises(ValueError):
        FederatedTrajectory(
            task_id=task_id,
            initial_state=initial,
            transitions=transitions,
            final_state=final_state,
        )
