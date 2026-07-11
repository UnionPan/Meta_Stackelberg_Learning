import numpy as np

from meta_stackelberg.core.model_state import ModelState, state_l2_norm
from meta_stackelberg.federated.protocols import Aggregator
from meta_stackelberg.federated.types import ClientUpdate
from meta_stackelberg.security.defenses.clipped_trimmed_mean import ClippedTrimmedMean
from meta_stackelberg.security.defenses.clipping import (
    ClippedAggregator,
    _clip_client_updates,
)
from meta_stackelberg.security.defenses.trimmed_mean import CoordinateTrimmedMean


def _update(client_id: int, vector) -> ClientUpdate:
    return ClientUpdate(
        client_id=client_id,
        delta=ModelState.from_tensors((np.asarray(vector, dtype=np.float32),)),
        num_examples=client_id + 1,
    )


def test_composition_clips_each_client_before_coordinate_trimming() -> None:
    vectors = (
        [14.0, -10.0],
        [-16.0, -8.0],
        [-4.0, 13.0],
        [-2.0, -17.0],
        [-7.0, 4.0],
    )
    updates = tuple(_update(index, vector) for index, vector in enumerate(vectors))
    combined = ClippedTrimmedMean(clip_radius=5.0, trim_ratio=0.2)

    actual = combined.aggregate(updates)
    expected = CoordinateTrimmedMean(0.2).aggregate(_clip_client_updates(updates, 5.0))
    trim_then_clip = _clip_client_updates(
        (ClientUpdate(0, CoordinateTrimmedMean(0.2).aggregate(updates), 1),),
        5.0,
    )[0].delta

    np.testing.assert_array_equal(actual.vector(), expected.vector())
    assert not np.allclose(actual.vector(), trim_then_clip.vector())


def test_zero_trim_equals_clipped_equal_client_mean() -> None:
    updates = (_update(0, [10.0, 0.0]), _update(1, [0.0, 2.0]))

    combined = ClippedTrimmedMean(clip_radius=5.0, trim_ratio=0.0).aggregate(updates)
    decomposed = ClippedAggregator(
        CoordinateTrimmedMean(0.0),
        clip_radius=5.0,
    ).aggregate(updates)

    np.testing.assert_array_equal(combined.vector(), decomposed.vector())


def test_summary_is_pure_and_reports_both_stages() -> None:
    updates = tuple(_update(index, [value, 0.0]) for index, value in enumerate((1, 2, 3, 4, 10)))
    aggregator = ClippedTrimmedMean(clip_radius=5.0, trim_ratio=0.2)

    summary = aggregator.summarize(updates)
    aggregator.aggregate(tuple(reversed(updates)))

    assert summary.clipping.client_count == 5
    assert summary.clipping.clipped_client_count == 1
    assert summary.clipping.clipped_client_fraction == 0.2
    assert summary.trimming.per_tail_trim_count == 1
    assert summary.trimming.retained_count == 3
    assert aggregator.summarize(updates) == summary


def test_composition_does_not_mutate_inputs_and_enforces_clip_bound() -> None:
    updates = (_update(0, [6.0, 8.0]), _update(1, [0.0, 1.0]), _update(2, [1.0, 0.0]))
    before = tuple(update.delta.vector() for update in updates)

    clipped = _clip_client_updates(updates, 5.0)
    ClippedTrimmedMean(5.0, 0.2).aggregate(updates)

    assert all(state_l2_norm(update.delta) <= 5.0 for update in clipped)
    for update, expected in zip(updates, before):
        np.testing.assert_array_equal(update.delta.vector(), expected)


def test_satisfies_existing_aggregator_protocol() -> None:
    assert isinstance(ClippedTrimmedMean(1.0, 0.2), Aggregator)
