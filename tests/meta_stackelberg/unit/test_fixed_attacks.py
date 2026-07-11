from dataclasses import FrozenInstanceError, dataclass, field

import numpy as np
import pytest

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.types import ClientUpdate, RoundState
from meta_stackelberg.security.types import AttackContext
from meta_stackelberg.security.training import ScopedLocalTrainer


@dataclass
class RecordingTrainer:
    delta_values: tuple[float, ...]
    client_override: int | None = None
    malicious: bool = False
    states: list[RoundState] = field(default_factory=list)

    def train(self, client_id: int, state: RoundState, rng: RandomSource) -> ClientUpdate:
        del rng
        self.states.append(state)
        return ClientUpdate(
            client_id=client_id if self.client_override is None else self.client_override,
            delta=ModelState.from_tensors((np.asarray(self.delta_values, dtype=np.float32),)),
            num_examples=7,
            is_malicious=self.malicious,
            metadata={'train_loss': 0.25},
        )


def _context(client_id: int = 2) -> AttackContext:
    return AttackContext(
        client_id=client_id,
        round_index=4,
        global_model=ModelState.from_tensors((np.array([3.0, 5.0], dtype=np.float32),)),
    )


def test_identity_generator_preserves_base_update_and_sanitizes_state() -> None:
    from meta_stackelberg.security.attacks.identity import IdentityMaliciousUpdateGenerator

    trainer = RecordingTrainer((1.0, -2.0))
    generator = IdentityMaliciousUpdateGenerator(ScopedLocalTrainer(trainer, {2}))
    rng = RandomSource(17)
    expected_snapshot = rng.capture()
    context = _context()

    update = generator.craft(context, rng)

    assert update.client_id == 2
    assert update.num_examples == 7
    assert update.is_malicious
    assert dict(update.metadata) == {'train_loss': 0.25}
    np.testing.assert_array_equal(update.delta.vector(), [1.0, -2.0])
    assert len(trainer.states) == 1
    sanitized = trainer.states[0]
    assert sanitized.round_index == 4
    assert sanitized.global_model is context.global_model
    assert dict(sanitized.component_states) == {}
    assert sanitized.random_snapshot.python_state == expected_snapshot.python_state


@pytest.mark.parametrize(
    ('budget', 'expected'),
    [
        (0.0, [1.0, -2.0]),
        (1.0, [0.0, 0.0]),
        (2.0, [-1.0, 2.0]),
    ],
)
def test_delta_reversal_has_hand_computable_budget_semantics(
    budget: float,
    expected: list[float],
) -> None:
    from meta_stackelberg.security.attacks.delta_reversal import DeltaReversalAttack

    update = DeltaReversalAttack(
        trainer=ScopedLocalTrainer(RecordingTrainer((1.0, -2.0)), {2}),
        budget=budget,
    ).craft(_context(), RandomSource(19))

    np.testing.assert_array_equal(update.delta.vector(), expected)
    assert update.is_malicious
    assert update.metadata['attack_type'] == 'delta_reversal'
    assert update.metadata['reversal_budget'] == budget
    assert update.metadata['displacement_ratio'] == pytest.approx(budget)


def test_delta_reversal_keeps_zero_base_update_zero() -> None:
    from meta_stackelberg.security.attacks.delta_reversal import DeltaReversalAttack

    update = DeltaReversalAttack(
        trainer=ScopedLocalTrainer(RecordingTrainer((0.0, 0.0)), {2}),
        budget=8.0,
    ).craft(_context(), RandomSource(23))

    np.testing.assert_array_equal(update.delta.vector(), [0.0, 0.0])
    assert update.metadata['displacement_ratio'] == 0.0


@pytest.mark.parametrize('budget', [-1.0, float('nan'), float('inf')])
def test_delta_reversal_rejects_invalid_budget(budget: float) -> None:
    from meta_stackelberg.security.attacks.delta_reversal import DeltaReversalAttack

    with pytest.raises(ValueError, match='budget'):
        DeltaReversalAttack(
            trainer=ScopedLocalTrainer(RecordingTrainer((1.0,)), {2}),
            budget=budget,
        )


@pytest.mark.parametrize(
    'generator_factory',
    [
        lambda trainer: __import__(
            'meta_stackelberg.security.attacks.identity',
            fromlist=['IdentityMaliciousUpdateGenerator'],
        ).IdentityMaliciousUpdateGenerator(ScopedLocalTrainer(trainer, {2})),
        lambda trainer: __import__(
            'meta_stackelberg.security.attacks.delta_reversal',
            fromlist=['DeltaReversalAttack'],
        ).DeltaReversalAttack(
            trainer=ScopedLocalTrainer(trainer, {2}),
            budget=2.0,
        ),
    ],
)
def test_fixed_generators_reject_invalid_base_update(generator_factory) -> None:
    wrong_client = generator_factory(RecordingTrainer((1.0,), client_override=9))
    already_malicious = generator_factory(RecordingTrainer((1.0,), malicious=True))

    with pytest.raises(ValueError, match='client 9'):
        wrong_client.craft(_context(client_id=2), RandomSource(1))
    with pytest.raises(ValueError, match='malicious'):
        already_malicious.craft(_context(client_id=2), RandomSource(1))


def test_scoped_local_trainer_rejects_clients_outside_declared_scope() -> None:
    trainer = ScopedLocalTrainer(RecordingTrainer((1.0,)), {2})

    with pytest.raises(ValueError, match='outside scoped local data'):
        trainer.train(
            3,
            RoundState(
                round_index=0,
                global_model=ModelState.from_tensors((np.zeros(1, dtype=np.float32),)),
                random_snapshot=RandomSource(1).capture(),
            ),
            RandomSource(1),
        )


def test_scoped_local_trainer_exposes_read_only_base_trainer() -> None:
    base = RecordingTrainer((1.0,))
    trainer = ScopedLocalTrainer(base, {2})

    assert trainer.base_trainer is base
    with pytest.raises(FrozenInstanceError):
        trainer.base_trainer = RecordingTrainer((2.0,))
