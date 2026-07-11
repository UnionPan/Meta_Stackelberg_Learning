import numpy as np
import torch

from meta_stackelberg.core.random_state import RandomSource


def _draw_all(source: RandomSource):
    return (
        source.python.random(),
        source.numpy.normal(size=4),
        torch.rand(4, generator=source.torch),
    )


def _assert_draws_equal(left, right) -> None:
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    torch.testing.assert_close(left[2], right[2], rtol=0.0, atol=0.0)


def test_capture_and_restore_replays_all_owned_generators() -> None:
    source = RandomSource(seed=7)
    snapshot = source.capture()

    first = _draw_all(source)
    source.restore(snapshot)
    second = _draw_all(source)

    _assert_draws_equal(first, second)


def test_equal_seeds_create_equal_independent_sources() -> None:
    first = RandomSource(seed=19)
    second = RandomSource(seed=19)

    _assert_draws_equal(_draw_all(first), _draw_all(second))

    _draw_all(first)
    first_next = _draw_all(first)
    second_next = _draw_all(second)
    assert first_next[0] != second_next[0]


def test_snapshot_is_not_changed_by_later_draws() -> None:
    source = RandomSource(seed=31)
    snapshot = source.capture()
    expected = _draw_all(source)

    for _ in range(5):
        _draw_all(source)
    source.restore(snapshot)

    _assert_draws_equal(expected, _draw_all(source))
