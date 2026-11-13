"""Check root-based trust, norm scaling, and zero updates when total trust is zero."""

import pytest
import torch

from meta_sg.defenses import fltrust


def test_fltrust_rejects_opposite_and_normalizes_all_aligned_magnitudes(updates):
    update, diagnostics = fltrust(
        updates([[100, 0], [.01, 0], [-50, 0], [0, 0]]),
        root_update=updates([[2, 0]])[0],
    )
    torch.testing.assert_close(update[0], torch.tensor([2., 0.], dtype=torch.float64))
    assert diagnostics['trust_sum'] == 2
    assert not diagnostics['zero_trust']


@pytest.mark.parametrize('root', [[0, 0], [1, 0]])
def test_fltrust_zero_trust_returns_zero_update(updates, root):
    update, diagnostics = fltrust(updates([[-1, 0], [0, 0]]), root_update=updates([root])[0])
    assert torch.equal(update[0], torch.zeros(2, dtype=torch.float64))
    assert diagnostics['zero_trust']
