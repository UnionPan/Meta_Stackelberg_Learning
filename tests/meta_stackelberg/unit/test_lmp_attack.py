import numpy as np
import pytest

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.types import ClientUpdate
from meta_stackelberg.security.protocols import RoundMaliciousUpdateGenerator
from meta_stackelberg.security.types import RoundAttackContext


def _context() -> RoundAttackContext:
    global_model = ModelState.from_tensors((
        np.array([1.0, -1.0, 2.0, 0.0], dtype=np.float32),
    ))
    references = (
        ClientUpdate(
            client_id=0,
            delta=ModelState.from_tensors((
                np.array([1.0, -2.0, 0.0, 1.0], dtype=np.float32),
            )),
            num_examples=5,
        ),
        ClientUpdate(
            client_id=2,
            delta=ModelState.from_tensors((
                np.array([2.0, -1.0, 0.0, 2.0], dtype=np.float32),
            )),
            num_examples=5,
        ),
    )
    return RoundAttackContext(
        round_index=2,
        global_model=global_model,
        malicious_client_ids=(1, 3),
        benign_updates=references,
    )


def test_lmp_median_craft_uses_coordinate_direction_and_reference_range() -> None:
    from meta_stackelberg.security.attacks.lmp import LMPAttack

    attack = LMPAttack(scale=2.0, num_examples_by_client={1: 7, 3: 9})
    updates = attack.craft_round(_context(), (RandomSource(11), RandomSource(12)))

    assert isinstance(attack, RoundMaliciousUpdateGenerator)
    assert attack.allowed_client_ids == frozenset({1, 3})
    assert [update.num_examples for update in updates] == [7, 9]
    for update in updates:
        crafted_model = _context().global_model.vector() + update.delta.vector()
        assert 1.0 <= crafted_model[0] <= 2.0
        assert -2.0 <= crafted_model[1] <= -1.0
        assert crafted_model[2] == 0.0
        assert 0.5 <= crafted_model[3] <= 1.0
        assert update.metadata == {
            'attack_type': 'lmp',
            'scale': 2.0,
            'reference_count': 2,
            'variant': 'median_craft_real',
        }
    assert updates[0].delta.tensors[0] is not updates[1].delta.tensors[0]


def test_lmp_replays_from_explicit_rngs() -> None:
    from meta_stackelberg.security.attacks.lmp import LMPAttack

    attack = LMPAttack(scale=3.0, num_examples_by_client={1: 7, 3: 9})
    first = attack.craft_round(_context(), (RandomSource(21), RandomSource(22)))
    second = attack.craft_round(_context(), (RandomSource(21), RandomSource(22)))

    for left, right in zip(first, second):
        np.testing.assert_array_equal(left.delta.vector(), right.delta.vector())
    assert not np.array_equal(first[0].delta.vector(), first[1].delta.vector())


def test_lmp_vectorizes_coordinate_statistics_and_random_draws(monkeypatch) -> None:
    from meta_stackelberg.security.attacks.lmp import LMPAttack

    median_calls = 0
    original_median = np.median

    def counted_median(*args, **kwargs):
        nonlocal median_calls
        median_calls += 1
        return original_median(*args, **kwargs)

    monkeypatch.setattr(np, 'median', counted_median)
    rngs = (RandomSource(21), RandomSource(22))
    for rng in rngs:
        monkeypatch.setattr(
            rng.python, 'uniform',
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError('LMP must not draw one Python random value per coordinate')
            ),
        )

    updates = LMPAttack(
        scale=3.0, num_examples_by_client={1: 7, 3: 9},
    ).craft_round(_context(), rngs)

    assert len(updates) == 2
    assert median_calls == len(_context().global_model.tensors)


def test_lmp_reconstructs_multiple_layers_without_coordinate_aliasing() -> None:
    from meta_stackelberg.security.attacks.lmp import LMPAttack

    base = _context()
    context = RoundAttackContext(
        round_index=base.round_index,
        global_model=ModelState.from_tensors((
            base.global_model.tensors[0][:2],
            base.global_model.tensors[0][2:],
        )),
        malicious_client_ids=(1,),
        benign_updates=tuple(ClientUpdate(
            client_id=update.client_id,
            delta=ModelState.from_tensors((
                update.delta.tensors[0][:2],
                update.delta.tensors[0][2:],
            )),
            num_examples=update.num_examples,
        ) for update in base.benign_updates),
    )
    crafted = LMPAttack(scale=2.0, num_examples_by_client={1: 7}).craft_round(
        context, (RandomSource(11),)
    )[0]

    assert len(crafted.delta.tensors) == 2
    assert crafted.delta.tensors[0].shape == (2,)
    assert crafted.delta.tensors[1].shape == (2,)


@pytest.mark.parametrize('scale', [0.0, 0.9, float('nan'), float('inf')])
def test_lmp_rejects_invalid_scale(scale: float) -> None:
    from meta_stackelberg.security.attacks.lmp import LMPAttack

    with pytest.raises(ValueError, match='scale'):
        LMPAttack(scale=scale, num_examples_by_client={1: 1})


def test_lmp_rejects_empty_references_or_wrong_rng_count() -> None:
    from meta_stackelberg.security.attacks.lmp import LMPAttack

    attack = LMPAttack(scale=2.0, num_examples_by_client={1: 1, 3: 1})
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
