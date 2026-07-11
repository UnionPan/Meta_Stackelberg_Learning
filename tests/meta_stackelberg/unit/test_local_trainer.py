import math

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.clients.trainer import TorchLocalTrainer
from meta_stackelberg.federated.models.parameters import TorchParameterCodec
from meta_stackelberg.federated.protocols import LocalTrainer
from meta_stackelberg.federated.types import RoundState


def _model_factory() -> torch.nn.Module:
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    return model


def _dataset() -> TensorDataset:
    features = torch.tensor([
        [-2.0, -1.0],
        [-1.0, -2.0],
        [1.0, 2.0],
        [2.0, 1.0],
    ])
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    return TensorDataset(features, labels)


def _round_state(source: RandomSource) -> RoundState:
    codec = TorchParameterCodec()
    return RoundState(
        round_index=0,
        global_model=codec.capture(_model_factory()),
        random_snapshot=source.capture(),
    )


def test_local_trainer_returns_local_minus_global_delta_and_metrics() -> None:
    source = RandomSource(7)
    state = _round_state(source)
    original = state.global_model.vector().copy()
    trainer = TorchLocalTrainer(
        model_factory=_model_factory,
        client_datasets={0: _dataset()},
        codec=TorchParameterCodec(),
        learning_rate=0.1,
        local_epochs=2,
        batch_size=2,
    )

    update = trainer.train(0, state, source)

    assert isinstance(trainer, LocalTrainer)
    assert update.client_id == 0
    assert update.num_examples == 4
    assert update.is_malicious is False
    assert np.linalg.norm(update.delta.vector()) > 0.0
    assert math.isfinite(update.metadata['train_loss'])
    assert 0.0 <= update.metadata['train_accuracy'] <= 1.0
    np.testing.assert_array_equal(state.global_model.vector(), original)


def test_local_training_replays_from_same_random_snapshot() -> None:
    source = RandomSource(13)
    state = _round_state(source)
    trainer = TorchLocalTrainer(
        model_factory=_model_factory,
        client_datasets={0: _dataset()},
        codec=TorchParameterCodec(),
        learning_rate=0.1,
        local_epochs=3,
        batch_size=2,
    )
    snapshot = source.capture()

    first = trainer.train(0, state, source)
    source.restore(snapshot)
    second = trainer.train(0, state, source)

    np.testing.assert_array_equal(first.delta.vector(), second.delta.vector())
    assert first.metadata == second.metadata


def test_local_trainer_rejects_unknown_clients_and_invalid_config() -> None:
    source = RandomSource(3)
    state = _round_state(source)
    trainer = TorchLocalTrainer(
        model_factory=_model_factory,
        client_datasets={0: _dataset()},
        codec=TorchParameterCodec(),
        learning_rate=0.1,
        local_epochs=1,
        batch_size=2,
    )
    with pytest.raises(KeyError, match='client 1'):
        trainer.train(1, state, source)
    with pytest.raises(ValueError, match='learning_rate'):
        TorchLocalTrainer(
            model_factory=_model_factory,
            client_datasets={0: _dataset()},
            codec=TorchParameterCodec(),
            learning_rate=-0.1,
            local_epochs=1,
            batch_size=2,
        )
