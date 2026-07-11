import math

import numpy as np
import pytest

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.types import ClientUpdate, RoundRequest, RoundState, RoundTransition


def _state(round_index: int, values=(0.0, 0.0)) -> RoundState:
    source = RandomSource(seed=round_index + 1)
    return RoundState(
        round_index=round_index,
        global_model=ModelState.from_tensors([np.asarray(values, dtype=np.float32)]),
        random_snapshot=source.capture(),
    )


def test_client_update_metadata_is_immutable() -> None:
    metadata = {'loss': 1.25}
    update = ClientUpdate(
        client_id=2,
        delta=ModelState.from_tensors([np.asarray([1.0], dtype=np.float32)]),
        num_examples=4,
        metadata=metadata,
    )
    metadata['loss'] = 9.0

    assert update.metadata['loss'] == 1.25
    with pytest.raises(TypeError):
        update.metadata['loss'] = 3.0


def test_round_request_validates_execution_parameters() -> None:
    state = _state(0)
    request = RoundRequest(task_id='clean', state=state, sample_size=2, server_lr=0.5)

    assert request.task_id == 'clean'
    assert request.sample_size == 2

    with pytest.raises(ValueError, match='task_id'):
        RoundRequest(task_id='', state=state, sample_size=2)
    with pytest.raises(ValueError, match='sample_size'):
        RoundRequest(task_id='clean', state=state, sample_size=0)
    with pytest.raises(ValueError, match='server_lr'):
        RoundRequest(task_id='clean', state=state, sample_size=1, server_lr=math.nan)


def test_records_reject_invalid_counts_and_rounds() -> None:
    delta = ModelState.from_tensors([np.asarray([1.0], dtype=np.float32)])
    with pytest.raises(ValueError, match='num_examples'):
        ClientUpdate(client_id=0, delta=delta, num_examples=0)
    with pytest.raises(ValueError, match='round_index'):
        RoundState(
            round_index=-1,
            global_model=delta,
            random_snapshot=RandomSource(0).capture(),
        )


def test_transition_separates_public_signals_from_private_diagnostics() -> None:
    before = _state(0)
    after = _state(1, values=(1.0, 2.0))
    update = ClientUpdate(
        client_id=0,
        delta=ModelState.from_tensors([np.asarray([1.0, 2.0], dtype=np.float32)]),
        num_examples=2,
    )
    public = {'aggregate_norm': 2.0}
    private = {'true_attacker_ids': (3,)}

    transition = RoundTransition(
        task_id='clean',
        state_before=before,
        sampled_clients=(0,),
        benign_updates=(update,),
        malicious_updates=(),
        aggregate_delta=update.delta,
        state_after=after,
        public_signals=public,
        private_diagnostics=private,
    )
    public['aggregate_norm'] = 99.0
    private['true_attacker_ids'] = ()

    assert transition.public_signals['aggregate_norm'] == 2.0
    assert transition.private_diagnostics['true_attacker_ids'] == (3,)
    assert 'true_attacker_ids' not in transition.public_signals
    with pytest.raises(TypeError):
        transition.public_signals['new'] = 1


def test_transition_rejects_duplicate_sampled_clients() -> None:
    before = _state(0)
    after = _state(1)
    delta = ModelState.from_tensors([np.asarray([0.0, 0.0], dtype=np.float32)])

    with pytest.raises(ValueError, match='duplicate'):
        RoundTransition(
            task_id='clean',
            state_before=before,
            sampled_clients=(0, 0),
            benign_updates=(),
            malicious_updates=(),
            aggregate_delta=delta,
            state_after=after,
            public_signals={},
            private_diagnostics={},
        )
