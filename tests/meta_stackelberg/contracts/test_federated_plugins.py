from dataclasses import dataclass

import numpy as np

from meta_stackelberg.core.model_state import ModelState, apply_delta
from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.protocols import Aggregator, ClientSampler, LocalTrainer, ServerOptimizer
from meta_stackelberg.federated.types import ClientUpdate, RoundRequest, RoundState


@dataclass
class FixedSampler:
    client_ids: tuple[int, ...]

    def sample(self, request: RoundRequest, rng: RandomSource) -> tuple[int, ...]:
        del request, rng
        return self.client_ids


class ScriptedTrainer:
    def train(self, client_id: int, state: RoundState, rng: RandomSource) -> ClientUpdate:
        del rng
        return ClientUpdate(
            client_id=client_id,
            delta=ModelState.from_tensors([
                np.full_like(state.global_model.tensors[0], float(client_id + 1)),
            ]),
            num_examples=1,
        )


class MeanAggregator:
    def aggregate(self, updates) -> ModelState:
        vectors = np.stack([update.delta.vector() for update in updates])
        return ModelState.from_tensors([np.mean(vectors, axis=0).astype(np.float32)])


class SimpleServerOptimizer:
    def step(self, model: ModelState, aggregate_delta: ModelState, learning_rate: float) -> ModelState:
        return apply_delta(model, aggregate_delta, scale=learning_rate)


def test_structural_plugins_satisfy_runtime_protocols() -> None:
    assert isinstance(FixedSampler((0, 1)), ClientSampler)
    assert isinstance(ScriptedTrainer(), LocalTrainer)
    assert isinstance(MeanAggregator(), Aggregator)
    assert isinstance(SimpleServerOptimizer(), ServerOptimizer)


def test_objects_missing_required_methods_do_not_satisfy_protocols() -> None:
    assert not isinstance(object(), ClientSampler)
    assert not isinstance(object(), LocalTrainer)
    assert not isinstance(object(), Aggregator)
    assert not isinstance(object(), ServerOptimizer)
