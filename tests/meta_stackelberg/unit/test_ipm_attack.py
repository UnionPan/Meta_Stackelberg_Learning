import numpy as np
import pytest

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.types import ClientUpdate
from meta_stackelberg.security.protocols import RoundMaliciousUpdateGenerator
from meta_stackelberg.security.types import RoundAttackContext


def _update(client_id: int, first, second, *, malicious: bool = False) -> ClientUpdate:
    return ClientUpdate(
        client_id=client_id,
        delta=ModelState.from_tensors((
            np.asarray(first, dtype=np.float32),
            np.asarray(second, dtype=np.float64),
        )),
        num_examples=client_id + 1,
        is_malicious=malicious,
    )


def _context() -> RoundAttackContext:
    return RoundAttackContext(
        round_index=4,
        global_model=ModelState.from_tensors((
            np.zeros(2, dtype=np.float32),
            np.zeros((1, 2), dtype=np.float64),
        )),
        malicious_client_ids=(1, 3),
        benign_updates=(
            _update(0, [1.0, 3.0], [[2.0, -2.0]]),
            _update(2, [3.0, 5.0], [[4.0, 2.0]]),
        ),
    )


def test_ipm_negates_and_scales_mean_benign_delta_per_layer() -> None:
    from meta_stackelberg.security.attacks.ipm import IPMAttack

    attack = IPMAttack(scale=2.0, num_examples_by_client={1: 7, 3: 9})
    updates = attack.craft_round(_context(), (RandomSource(1), RandomSource(2)))

    assert isinstance(attack, RoundMaliciousUpdateGenerator)
    assert attack.allowed_client_ids == frozenset({1, 3})
    assert [update.client_id for update in updates] == [1, 3]
    assert [update.num_examples for update in updates] == [7, 9]
    for update in updates:
        assert update.is_malicious
        np.testing.assert_array_equal(update.delta.tensors[0], [-4.0, -8.0])
        np.testing.assert_array_equal(update.delta.tensors[1], [[-6.0, 0.0]])
        assert update.delta.tensors[0].dtype == np.float32
        assert update.delta.tensors[1].dtype == np.float64
        assert update.metadata == {
            'attack_type': 'ipm',
            'scale': 2.0,
            'reference_count': 2,
        }
    assert updates[0].delta.tensors[0] is not updates[1].delta.tensors[0]


@pytest.mark.parametrize('scale', [0.0, -1.0, float('nan'), float('inf')])
def test_ipm_rejects_invalid_scale(scale: float) -> None:
    from meta_stackelberg.security.attacks.ipm import IPMAttack

    with pytest.raises(ValueError, match='scale'):
        IPMAttack(scale=scale, num_examples_by_client={1: 1})


def test_ipm_rejects_missing_references_rng_or_sample_count() -> None:
    from meta_stackelberg.security.attacks.ipm import IPMAttack

    attack = IPMAttack(scale=2.0, num_examples_by_client={1: 3, 3: 4})
    empty = RoundAttackContext(
        round_index=0,
        global_model=_context().global_model,
        malicious_client_ids=(1,),
        benign_updates=(),
    )
    with pytest.raises(ValueError, match='benign reference'):
        attack.craft_round(empty, (RandomSource(1),))
    with pytest.raises(ValueError, match='RNG'):
        attack.craft_round(_context(), (RandomSource(1),))

    missing = IPMAttack(scale=2.0, num_examples_by_client={1: 3})
    with pytest.raises(ValueError, match='sample count'):
        missing.craft_round(_context(), (RandomSource(1), RandomSource(2)))


def test_ipm_rejects_reference_structure_or_dtype_mismatch() -> None:
    from meta_stackelberg.security.attacks.ipm import IPMAttack

    bad = ClientUpdate(
        client_id=5,
        delta=ModelState.from_tensors((
            np.zeros(3, dtype=np.float32),
            np.zeros((1, 2), dtype=np.float64),
        )),
        num_examples=1,
    )
    context = RoundAttackContext(
        round_index=0,
        global_model=_context().global_model,
        malicious_client_ids=(1,),
        benign_updates=(_context().benign_updates[0], bad),
    )
    with pytest.raises(ValueError, match='structure'):
        IPMAttack(scale=2.0, num_examples_by_client={1: 1}).craft_round(
            context, (RandomSource(1),)
        )
