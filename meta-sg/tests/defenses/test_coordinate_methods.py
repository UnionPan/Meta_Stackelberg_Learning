"""Check mean, median, and clipping definitions using known coordinates and norms."""

import pytest
import torch

from meta_sg.defenses import clipmed, fedavg, median, trimmed_mean


def test_coordinate_methods_have_known_results_with_outliers(updates):
    clients = updates([[-100, 90], [1, 1], [2, 3], [4, 5], [6, 9], [100, -90]])
    median_update, _ = median(clients)
    trimmed_update, _ = trimmed_mean(clients, byzantine_bound=1)
    torch.testing.assert_close(median_update[0], torch.tensor([3., 4.], dtype=torch.float64))
    torch.testing.assert_close(trimmed_update[0], torch.tensor([3.25, 4.5], dtype=torch.float64))


def test_fedavg_uses_equal_client_weights(updates):
    update, _ = fedavg(updates([[1, 3], [3, 7]]))
    torch.testing.assert_close(update[0], torch.tensor([2., 5.], dtype=torch.float64))


def test_clipmed_clips_whole_updates_before_even_coordinate_median():
    clients = [
        (torch.tensor([x], dtype=torch.float64), torch.tensor([y], dtype=torch.float64))
        for x, y in [(6, 8), (0, 20), (2, 0), (0, 0)]
    ]
    before = [tuple(part.clone() for part in update) for update in clients]
    update, diagnostics = clipmed(clients, norm_bound=5)
    torch.testing.assert_close(torch.cat(update), torch.tensor([1., 2.], dtype=torch.float64))
    assert diagnostics['clipped_clients'] == 2
    assert diagnostics['minimum_clip_scale'] == .25
    assert all(
        torch.equal(a, b)
        for old, new in zip(before, clients, strict=True)
        for a, b in zip(old, new, strict=True)
    )


def test_clipmed_odd_median_and_zero_updates(updates):
    update, _ = clipmed(updates([[0, 0], [3, 4], [6, 8]]), norm_bound=5)
    torch.testing.assert_close(update[0], torch.tensor([3., 4.], dtype=torch.float64))
    zero, diagnostics = clipmed(updates([[0, 0]] * 3))
    assert torch.equal(zero[0], torch.zeros(2, dtype=torch.float64))
    assert diagnostics['clipped_clients'] == 0


def test_clipmed_large_finite_float32_update_retains_direction():
    update, _ = clipmed([(torch.tensor([3e30, 4e30]),)], norm_bound=5)
    torch.testing.assert_close(update[0], torch.tensor([3., 4.]))


def test_trimmed_mean_rejects_insufficient_clients(updates):
    with pytest.raises(ValueError, match='n > 2f'):
        trimmed_mean(updates([[1]] * 3))


def test_zero_trimming_matches_mean(updates):
    clients = updates([[1, 3], [3, 7]])
    trimmed, _ = trimmed_mean(clients, byzantine_bound=0)
    average, _ = fedavg(clients)
    torch.testing.assert_close(trimmed, average)
