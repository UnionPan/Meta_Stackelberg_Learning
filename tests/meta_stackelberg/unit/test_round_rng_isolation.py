from dataclasses import dataclass, field

import numpy as np
import torch

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.core.random_state import RandomSnapshot, RandomSource
from meta_stackelberg.federated.aggregation.fedavg import FedAvg
from meta_stackelberg.federated.clients.sampling import UniformClientSampler
from meta_stackelberg.federated.engine.round_engine import RoundEngine
from meta_stackelberg.federated.engine.server_optimizer import ServerSGD
from meta_stackelberg.federated.types import ClientUpdate, RoundRequest, RoundState


@dataclass
class RecordingTrainer:
    extra_draws: int
    first_draws: list[tuple[int, float, float, float]] = field(default_factory=list)

    def train(self, client_id: int, state: RoundState, rng: RandomSource) -> ClientUpdate:
        first = (
            client_id,
            rng.python.random(),
            float(rng.numpy.normal()),
            float(torch.rand((), generator=rng.torch).item()),
        )
        self.first_draws.append(first)
        for _ in range(self.extra_draws):
            rng.python.random()
            rng.numpy.normal()
            torch.rand((), generator=rng.torch)
        return ClientUpdate(
            client_id=client_id,
            delta=ModelState.from_tensors((np.zeros(2, dtype=np.float32),)),
            num_examples=1,
        )


def _initial_state(source: RandomSource) -> RoundState:
    return RoundState(
        round_index=0,
        global_model=ModelState.from_tensors((np.zeros(2, dtype=np.float32),)),
        random_snapshot=source.capture(),
    )


def _run(extra_draws: int):
    source = RandomSource(71)
    state = _initial_state(source)
    trainer = RecordingTrainer(extra_draws=extra_draws)
    engine = RoundEngine(
        sampler=UniformClientSampler(num_clients=5),
        trainer=trainer,
        aggregator=FedAvg(),
        server_optimizer=ServerSGD(),
    )
    sampled_history = []
    for _ in range(3):
        transition = engine.run_round(
            RoundRequest(
                task_id='rng-isolation',
                state=state,
                sample_size=2,
            ),
            source,
        )
        sampled_history.append(transition.sampled_clients)
        state = transition.state_after
    return tuple(sampled_history), tuple(trainer.first_draws), state.random_snapshot


def _next_draws(snapshot: RandomSnapshot):
    source = RandomSource(0)
    source.restore(snapshot)
    return (
        source.python.random(),
        source.numpy.normal(size=4),
        torch.rand(4, generator=source.torch),
    )


def test_client_random_consumption_does_not_shift_sampling_or_parent_state() -> None:
    light_history, light_first_draws, light_snapshot = _run(extra_draws=0)
    heavy_history, heavy_first_draws, heavy_snapshot = _run(extra_draws=100)

    assert light_history == heavy_history
    assert light_first_draws == heavy_first_draws
    light_next = _next_draws(light_snapshot)
    heavy_next = _next_draws(heavy_snapshot)
    assert light_next[0] == heavy_next[0]
    np.testing.assert_array_equal(light_next[1], heavy_next[1])
    torch.testing.assert_close(light_next[2], heavy_next[2], rtol=0.0, atol=0.0)
