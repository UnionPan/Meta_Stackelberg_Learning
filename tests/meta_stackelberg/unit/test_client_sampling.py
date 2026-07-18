import numpy as np
import pytest

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.clients.sampling import (
    BenignReferenceClientSampler,
    UniformClientSampler,
)
from meta_stackelberg.federated.protocols import ClientSampler
from meta_stackelberg.federated.types import RoundRequest, RoundState


def _request(sample_size: int) -> RoundRequest:
    source = RandomSource(0)
    return RoundRequest(
        task_id='clean',
        state=RoundState(
            round_index=0,
            global_model=ModelState.from_tensors([np.zeros(1, dtype=np.float32)]),
            random_snapshot=source.capture(),
        ),
        sample_size=sample_size,
    )


def test_uniform_sampler_is_reproducible_and_without_replacement() -> None:
    sampler = UniformClientSampler(num_clients=10)
    first = sampler.sample(_request(4), RandomSource(7))
    second = sampler.sample(_request(4), RandomSource(7))

    assert first == second
    assert len(first) == 4
    assert len(set(first)) == 4
    assert all(0 <= client_id < 10 for client_id in first)


def test_uniform_sampler_satisfies_client_sampler_protocol() -> None:
    assert isinstance(UniformClientSampler(num_clients=3), ClientSampler)


def test_uniform_sampler_rejects_invalid_population_or_request() -> None:
    with pytest.raises(ValueError, match='num_clients'):
        UniformClientSampler(num_clients=0)
    with pytest.raises(ValueError, match='sample_size'):
        UniformClientSampler(num_clients=2).sample(_request(3), RandomSource(1))


def test_reference_sampler_never_returns_an_all_malicious_subset() -> None:
    malicious = frozenset({0, 1, 2, 3})
    sampler = BenignReferenceClientSampler(20, malicious)
    source = RandomSource(31)

    samples = tuple(sampler.sample(_request(4), source) for _ in range(10_000))

    assert all(len(sample) == len(set(sample)) == 4 for sample in samples)
    assert all(set(sample) - malicious for sample in samples)
    assert isinstance(sampler, ClientSampler)


def test_reference_sampler_rejects_population_without_benign_client() -> None:
    with pytest.raises(ValueError, match='benign client'):
        BenignReferenceClientSampler(2, frozenset({0, 1}))
