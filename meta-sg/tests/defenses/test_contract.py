"""Check imports, parameter shape restoration, and preservation of client inputs."""

from pathlib import Path

import pytest
import torch

import meta_sg
from meta_sg.defenses import clipmed, fedavg, fltrust, krum, median, multi_krum, rfa, trimmed_mean


@pytest.mark.parametrize('defense,parameters', [
    (fedavg, {}), (median, {}), (trimmed_mean, {'byzantine_bound': 1}),
    (krum, {'byzantine_bound': 1}), (multi_krum, {'byzantine_bound': 1, 'selected': 3}),
    (rfa, {}), (fltrust, {}), (clipmed, {'norm_bound': 10}),
])
def test_methods_preserve_identical_multitensor_updates_and_inputs(defense, parameters):
    clients = [(torch.tensor([[2., -3.]]), torch.tensor(1.)) for _ in range(7)]
    before = [tuple(part.clone() for part in update) for update in clients]
    if defense is fltrust:
        parameters = {'root_update': clients[0]}
    update, diagnostics = defense(clients, **parameters)
    torch.testing.assert_close(update, clients[0])
    assert diagnostics['aggregate_norm'] == pytest.approx(14 ** .5)
    for old, new in zip(before, clients, strict=True):
        assert all(torch.equal(a, b) for a, b in zip(old, new, strict=True))


def test_imports_use_migrated_package():
    expected = Path(__file__).resolve().parents[2] / 'src/meta_sg/__init__.py'
    assert Path(meta_sg.__file__).resolve() == expected
