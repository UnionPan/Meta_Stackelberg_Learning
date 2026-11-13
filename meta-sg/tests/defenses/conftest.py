"""Provide float64 client updates for numerical tests of the defense methods."""

import pytest
import torch


@pytest.fixture
def updates():
    def make(values):
        return [(torch.as_tensor(value, dtype=torch.float64),) for value in values]
    return make
