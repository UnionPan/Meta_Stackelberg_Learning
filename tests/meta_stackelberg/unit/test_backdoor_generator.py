from dataclasses import dataclass

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.types import ClientUpdate, RoundState
from meta_stackelberg.federated.clients.trainer import TorchLocalTrainer
from meta_stackelberg.federated.models.parameters import TorchParameterCodec
from meta_stackelberg.security.protocols import MaliciousUpdateGenerator
from meta_stackelberg.security.training import ScopedLocalTrainer
from meta_stackelberg.security.types import AttackContext


@dataclass
class FixtureTrainer:
    def train(self, client_id: int, state: RoundState, rng: RandomSource) -> ClientUpdate:
        del state, rng
        return ClientUpdate(
            client_id=client_id,
            delta=ModelState.from_tensors((np.array([1.5, -0.5], dtype=np.float32),)),
            num_examples=9,
            metadata={'train_loss': 0.2},
        )


def _model() -> torch.nn.Module:
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].bias.zero_()
    return model


def _poisoned_trainer() -> ScopedLocalTrainer:
    from meta_stackelberg.security.data.poisoning import SourceTargetPoisonedDataset
    from meta_stackelberg.security.data.trigger import PatchTrigger

    clean = TensorDataset(
        torch.tensor([
            [[[-1.0, -1.0], [-1.0, -1.0]]],
            [[[-0.5, -0.5], [-0.5, -0.5]]],
            [[[0.5, 0.5], [0.5, 0.5]]],
            [[[1.0, 1.0], [1.0, 1.0]]],
        ]),
        torch.tensor([0, 0, 1, 1]),
    )
    poisoned = SourceTargetPoisonedDataset(
        dataset=clean,
        trigger=PatchTrigger(row=1, column=1, height=1, width=1, value=2.0),
        source_class=0,
        target_class=1,
        poison_fraction=0.5,
        rng=RandomSource(3),
    )
    return ScopedLocalTrainer(
        TorchLocalTrainer(
            model_factory=_model,
            client_datasets={2: poisoned},
            codec=TorchParameterCodec(),
            learning_rate=0.1,
            local_epochs=1,
            batch_size=2,
        ),
        {2},
    )


def _context() -> AttackContext:
    return AttackContext(
        client_id=2,
        round_index=5,
        global_model=TorchParameterCodec().capture(_model()),
    )


def test_backdoor_generator_wraps_poisoned_local_training_update() -> None:
    from meta_stackelberg.security.attacks.backdoor import BackdoorLocalUpdateGenerator

    generator = BackdoorLocalUpdateGenerator(
        trainer=_poisoned_trainer(),
        source_class=0,
        target_class=1,
        poison_fraction=0.5,
    )

    expected = _poisoned_trainer().train(2, RoundState(
        round_index=5,
        global_model=_context().global_model,
        random_snapshot=RandomSource(0).capture(),
    ), RandomSource(7))
    update = generator.craft(_context(), RandomSource(7))

    assert update.client_id == 2
    assert update.num_examples == 4
    assert update.is_malicious
    np.testing.assert_array_equal(update.delta.vector(), expected.delta.vector())
    assert update.metadata == {
        **expected.metadata,
        'attack_type': 'bfl_backdoor',
        'source_class': 0,
        'target_class': 1,
        'poison_fraction': 0.5,
    }
    assert generator.allowed_client_ids == frozenset({2})
    assert isinstance(generator, MaliciousUpdateGenerator)


@pytest.mark.parametrize(
    'kwargs',
    [
        {'source_class': 1, 'target_class': 1},
        {'poison_fraction': -0.1},
        {'poison_fraction': 1.1},
        {'poison_fraction': float('nan')},
    ],
)
def test_backdoor_generator_rejects_invalid_configuration(kwargs) -> None:
    from meta_stackelberg.security.attacks.backdoor import BackdoorLocalUpdateGenerator

    values = {
        'trainer': _poisoned_trainer(),
        'source_class': 0,
        'target_class': 1,
        'poison_fraction': 0.5,
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        BackdoorLocalUpdateGenerator(**values)


def test_backdoor_generator_requires_scoped_local_trainer() -> None:
    from meta_stackelberg.security.attacks.backdoor import BackdoorLocalUpdateGenerator

    with pytest.raises(TypeError, match='ScopedLocalTrainer'):
        BackdoorLocalUpdateGenerator(
            trainer=FixtureTrainer(),
            source_class=0,
            target_class=1,
            poison_fraction=0.5,
        )


def test_backdoor_generator_rejects_trainer_without_matching_poisoned_datasets() -> None:
    from meta_stackelberg.security.attacks.backdoor import BackdoorLocalUpdateGenerator

    with pytest.raises(ValueError, match='TorchLocalTrainer'):
        BackdoorLocalUpdateGenerator(
            trainer=ScopedLocalTrainer(FixtureTrainer(), {2}),
            source_class=0,
            target_class=1,
            poison_fraction=0.5,
        )
