import math

import numpy as np
import pytest

from meta_stackelberg.core.model_state import ModelState, state_l2_norm
from meta_stackelberg.federated.aggregation.fedavg import FedAvg
from meta_stackelberg.federated.protocols import Aggregator
from meta_stackelberg.federated.types import ClientUpdate
from meta_stackelberg.security.defenses.clipping import ClippedAggregator


def _update(
    client_id: int,
    tensors: list[np.ndarray],
    *,
    examples: int = 1,
    malicious: bool = False,
) -> ClientUpdate:
    return ClientUpdate(
        client_id=client_id,
        delta=ModelState.from_tensors(np.asarray(value, dtype=np.float32) for value in tensors),
        num_examples=examples,
        is_malicious=malicious,
        metadata={'source': 'fixture'},
    )


def test_clips_joint_multi_tensor_norm_before_weighted_fedavg() -> None:
    first = _update(0, [np.array([3.0]), np.array([4.0])], examples=1)
    second = _update(1, [np.array([0.0]), np.array([0.0])], examples=3)
    aggregator = ClippedAggregator(FedAvg(), clip_radius=2.5)

    actual = aggregator.aggregate((first, second))

    np.testing.assert_allclose(actual.vector(), np.array([0.375, 0.5]))
    assert state_l2_norm(first.delta) == 5.0


def test_no_clip_reference_is_exactly_fedavg() -> None:
    updates = (
        _update(0, [np.array([1.0, 2.0])]),
        _update(1, [np.array([-2.0, 1.0])], examples=2),
    )

    expected = FedAvg().aggregate(updates)
    actual = ClippedAggregator(FedAvg(), clip_radius=10.0).aggregate(updates)

    for left, right in zip(actual.tensors, expected.tensors):
        np.testing.assert_array_equal(left, right)


class RecordingAggregator:
    def __init__(self) -> None:
        self.updates: tuple[ClientUpdate, ...] = ()

    def aggregate(self, updates) -> ModelState:
        self.updates = tuple(updates)
        return self.updates[0].delta


def test_preserves_update_fields_dtype_and_input_immutability() -> None:
    original = _update(7, [np.array([6.0, 8.0])], examples=9, malicious=True)
    before = original.delta.vector()
    base = RecordingAggregator()

    ClippedAggregator(base, clip_radius=5.0).aggregate((original,))

    clipped = base.updates[0]
    assert clipped.client_id == 7
    assert clipped.num_examples == 9
    assert clipped.is_malicious
    assert clipped.metadata == original.metadata
    assert clipped.delta.tensors[0].dtype == np.float32
    np.testing.assert_allclose(clipped.delta.vector(), np.array([3.0, 4.0]))
    np.testing.assert_array_equal(original.delta.vector(), before)
    assert clipped is not original


def test_zero_and_exact_boundary_updates_are_not_scaled() -> None:
    updates = (
        _update(0, [np.array([0.0, 0.0])]),
        _update(1, [np.array([3.0, 4.0])]),
    )
    base = RecordingAggregator()

    ClippedAggregator(base, clip_radius=5.0).aggregate(updates)

    for actual, expected in zip(base.updates, updates):
        np.testing.assert_array_equal(actual.delta.vector(), expected.delta.vector())


def test_summary_is_pure_and_reports_pre_clip_norms() -> None:
    aggregator = ClippedAggregator(FedAvg(), clip_radius=2.0)
    first = (_update(0, [np.array([3.0, 4.0])]), _update(1, [np.array([0.0, 0.0])]))
    second = (_update(2, [np.array([1.0, 0.0])]),)

    summary = aggregator.summarize(first)
    aggregator.aggregate(second)

    assert summary.clip_radius == 2.0
    assert summary.client_count == 2
    assert summary.clipped_client_count == 1
    assert summary.clipped_client_fraction == 0.5
    assert summary.pre_clip_norm_min == 0.0
    assert summary.pre_clip_norm_median == 2.5
    assert summary.pre_clip_norm_max == 5.0
    assert aggregator.summarize(first) == summary


def test_permutation_does_not_change_symmetric_fedavg_result() -> None:
    updates = (
        _update(0, [np.array([3.0, 4.0])], examples=2),
        _update(1, [np.array([-4.0, 3.0])], examples=1),
    )
    aggregator = ClippedAggregator(FedAvg(), clip_radius=2.0)

    forward = aggregator.aggregate(updates)
    reverse = aggregator.aggregate(tuple(reversed(updates)))

    np.testing.assert_array_equal(forward.vector(), reverse.vector())


@pytest.mark.parametrize('radius', [0.0, -1.0, math.nan, math.inf, True])
def test_rejects_invalid_radius(radius: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ClippedAggregator(FedAvg(), radius)  # type: ignore[arg-type]


def test_rejects_empty_and_non_finite_updates() -> None:
    aggregator = ClippedAggregator(FedAvg(), 1.0)
    with pytest.raises(ValueError, match='at least one'):
        aggregator.aggregate(())
    invalid = _update(0, [np.array([np.nan])])
    with pytest.raises(ValueError, match='finite'):
        aggregator.aggregate((invalid,))


def test_satisfies_existing_aggregator_protocol() -> None:
    assert isinstance(ClippedAggregator(FedAvg(), 1.0), Aggregator)
