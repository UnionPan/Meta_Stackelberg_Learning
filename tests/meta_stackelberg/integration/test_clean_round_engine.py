from dataclasses import dataclass

import numpy as np
import pytest

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.aggregation.fedavg import FedAvg
from meta_stackelberg.federated.engine.round_engine import RoundEngine
from meta_stackelberg.federated.engine.server_optimizer import ServerSGD
from meta_stackelberg.federated.types import ClientUpdate, RoundRequest, RoundState


@dataclass
class FixedSampler:
    clients: tuple[int, ...]

    def sample(self, request: RoundRequest, rng: RandomSource) -> tuple[int, ...]:
        del request, rng
        return self.clients


class RandomSampler:
    def sample(self, request: RoundRequest, rng: RandomSource) -> tuple[int, ...]:
        values = rng.numpy.choice(4, size=request.sample_size, replace=False)
        return tuple(int(value) for value in values)


class ScriptedTrainer:
    DELTAS = {
        0: ([1.0, 1.0], 1),
        1: ([3.0, 5.0], 3),
        2: ([2.0, -1.0], 2),
        3: ([-2.0, 4.0], 4),
    }

    def train(self, client_id: int, state: RoundState, rng: RandomSource) -> ClientUpdate:
        del state, rng
        values, examples = self.DELTAS[client_id]
        return ClientUpdate(
            client_id=client_id,
            delta=ModelState.from_tensors([np.asarray(values, dtype=np.float32)]),
            num_examples=examples,
        )


class FirstUpdateAggregator:
    def aggregate(self, updates) -> ModelState:
        return updates[0].delta.clone()


def _initial_state(source: RandomSource) -> RoundState:
    return RoundState(
        round_index=0,
        global_model=ModelState.from_tensors([np.asarray([0.0, 0.0], dtype=np.float32)]),
        random_snapshot=source.capture(),
    )


def test_clean_round_exposes_every_hand_computable_intermediate() -> None:
    source = RandomSource(seed=7)
    request = RoundRequest(
        task_id='synthetic-clean',
        state=_initial_state(source),
        sample_size=2,
        server_lr=0.5,
    )
    engine = RoundEngine(
        sampler=FixedSampler((0, 1)),
        trainer=ScriptedTrainer(),
        aggregator=FedAvg(),
        server_optimizer=ServerSGD(),
    )

    transition = engine.run_round(request, source)

    assert transition.sampled_clients == (0, 1)
    assert [update.client_id for update in transition.benign_updates] == [0, 1]
    np.testing.assert_allclose(transition.benign_updates[0].delta.vector(), [1.0, 1.0])
    np.testing.assert_allclose(transition.benign_updates[1].delta.vector(), [3.0, 5.0])
    np.testing.assert_allclose(transition.aggregate_delta.vector(), [2.5, 4.0])
    np.testing.assert_allclose(transition.state_after.global_model.vector(), [1.25, 2.0])
    assert transition.state_after.round_index == 1
    assert transition.public_signals['sampled_client_count'] == 2
    assert transition.public_signals['aggregate_norm'] == np.linalg.norm([2.5, 4.0])
    assert transition.private_diagnostics == {}


def test_round_replays_from_the_same_state_snapshot() -> None:
    source = RandomSource(seed=13)
    request = RoundRequest(
        task_id='synthetic-clean',
        state=_initial_state(source),
        sample_size=2,
        server_lr=1.0,
    )
    engine = RoundEngine(
        sampler=RandomSampler(),
        trainer=ScriptedTrainer(),
        aggregator=FedAvg(),
        server_optimizer=ServerSGD(),
    )

    first = engine.run_round(request, source)
    second = engine.run_round(request, source)

    assert first.sampled_clients == second.sampled_clients
    np.testing.assert_array_equal(first.aggregate_delta.vector(), second.aggregate_delta.vector())
    np.testing.assert_array_equal(
        first.state_after.global_model.vector(),
        second.state_after.global_model.vector(),
    )


def test_round_engine_accepts_an_alternative_aggregator_without_changes() -> None:
    source = RandomSource(seed=17)
    request = RoundRequest(
        task_id='synthetic-clean',
        state=_initial_state(source),
        sample_size=2,
        server_lr=0.5,
    )
    engine = RoundEngine(
        sampler=FixedSampler((0, 1)),
        trainer=ScriptedTrainer(),
        aggregator=FirstUpdateAggregator(),
        server_optimizer=ServerSGD(),
    )

    transition = engine.run_round(request, source)

    np.testing.assert_allclose(transition.aggregate_delta.vector(), [1.0, 1.0])
    np.testing.assert_allclose(transition.state_after.global_model.vector(), [0.5, 0.5])


@pytest.mark.parametrize('invalid_client_id', [1.9, True])
def test_round_engine_rejects_non_integer_sampled_client_ids(invalid_client_id) -> None:
    source = RandomSource(seed=23)
    engine = RoundEngine(
        sampler=FixedSampler((invalid_client_id,)),
        trainer=ScriptedTrainer(),
        aggregator=FedAvg(),
        server_optimizer=ServerSGD(),
    )

    with pytest.raises(ValueError, match='integers'):
        engine.run_round(
            RoundRequest(
                task_id='invalid-client-id',
                state=_initial_state(source),
                sample_size=1,
            ),
            source,
        )
