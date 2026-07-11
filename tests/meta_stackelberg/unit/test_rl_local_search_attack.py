import numpy as np
import torch
from torch.utils.data import TensorDataset

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.models.parameters import TorchParameterCodec
from meta_stackelberg.federated.types import ClientUpdate
from meta_stackelberg.security.attacks.local_search import RLLocalSearchAttack
from meta_stackelberg.security.attacks.rl_action import RLAttackAction
from meta_stackelberg.security.types import RoundAttackContext


def _model_factory() -> torch.nn.Module:
    model = torch.nn.Linear(1, 2, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    return model


def _context() -> RoundAttackContext:
    benign_delta = ModelState.from_tensors((np.array([[0.1], [-0.1]], dtype=np.float32),))
    return RoundAttackContext(
        round_index=0,
        global_model=TorchParameterCodec().capture(_model_factory()),
        malicious_client_ids=(0, 1),
        benign_updates=(ClientUpdate(2, benign_delta, 2),),
    )


def _dataset() -> TensorDataset:
    return TensorDataset(
        torch.tensor([[1.0], [2.0], [-1.0], [-2.0]]),
        torch.tensor([0, 0, 1, 1]),
    )


def _attack(action: RLAttackAction) -> RLLocalSearchAttack:
    return RLLocalSearchAttack(
        action=action,
        model_factory=_model_factory,
        codec=TorchParameterCodec(),
        local_dataset=_dataset(),
        num_examples_by_client={0: 2, 1: 2},
        learning_rate=0.05,
        batch_size=4,
        trajectories=1,
    )


def test_local_search_replays_and_shares_one_action_across_malicious_clients() -> None:
    context = _context()
    attack = _attack(RLAttackAction(1.5, 3, 0.4))
    first = attack.craft_round(context, (RandomSource(7), RandomSource(8)))
    replay = attack.craft_round(context, (RandomSource(7), RandomSource(8)))

    assert len(first) == 2
    assert all(update.is_malicious for update in first)
    assert first[0].metadata['gamma'] == first[1].metadata['gamma'] == 1.5
    assert first[0].metadata['local_steps'] == 3
    np.testing.assert_array_equal(first[0].delta.vector(), first[1].delta.vector())
    np.testing.assert_array_equal(first[0].delta.vector(), replay[0].delta.vector())
    assert np.linalg.norm(first[0].delta.vector()) > 0.0


def test_gamma_scales_same_local_search_endpoint_linearly() -> None:
    context = _context()
    low = _attack(RLAttackAction(0.5, 2, 0.2)).craft_round(
        context, (RandomSource(3), RandomSource(4)),
    )[0]
    high = _attack(RLAttackAction(1.0, 2, 0.2)).craft_round(
        context, (RandomSource(3), RandomSource(4)),
    )[0]
    np.testing.assert_allclose(high.delta.vector(), 2.0 * low.delta.vector(), rtol=1e-6)


def test_local_search_declares_paper_capabilities_and_rejects_bad_rng_count() -> None:
    attack = _attack(RLAttackAction(1.0, 1, 0.5))
    assert attack.capabilities.needs_global_model
    assert attack.capabilities.needs_local_data
    assert attack.capabilities.observes_benign_updates
    try:
        attack.craft_round(_context(), (RandomSource(1),))
    except ValueError as error:
        assert 'RNG' in str(error)
    else:
        raise AssertionError('accepted mismatched RNG count')
