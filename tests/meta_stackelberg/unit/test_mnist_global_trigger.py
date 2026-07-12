from __future__ import annotations

import torch

from meta_stackelberg.security.data.mnist_global_trigger import mnist_global_trigger


def test_mnist_global_trigger_matches_pinned_rlbackdoorfl_square() -> None:
    fixture = mnist_global_trigger()

    assert fixture.identifier == 'mnist-global-1-to-7-v1'
    assert fixture.source_class == 1
    assert fixture.target_class == 7
    assert fixture.pixels == tuple(
        (row, column, 2.82148653034729)
        for row in range(5, 7)
        for column in range(6, 11)
    )
    assert fixture.sha256 == (
        'c5226726b8b70efa2f667d59784a81e6f3f7e1a359a53d067b2db086d570c93b'
    )
    assert fixture.compute_sha256() == fixture.sha256


def test_mnist_global_trigger_applies_without_mutating_source() -> None:
    fixture = mnist_global_trigger()
    image = torch.zeros(1, 28, 28, dtype=torch.float32)

    triggered = fixture.trigger.apply(image)

    assert torch.equal(image, torch.zeros_like(image))
    assert int(torch.count_nonzero(triggered)) == 10
    assert torch.all(triggered[:, 5:7, 6:11] == fixture.normalized_white)


def test_mnist_global_trigger_returns_one_immutable_fixture() -> None:
    assert mnist_global_trigger() is mnist_global_trigger()
