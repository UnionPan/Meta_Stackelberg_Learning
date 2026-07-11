import json
from io import BytesIO
import zipfile

import numpy as np
import pytest
import torch

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.types import RoundState


def test_round_checkpoint_restores_model_metadata_and_all_rng_streams(tmp_path) -> None:
    from meta_stackelberg.federated.checkpointing.round_state import (
        load_round_state,
        save_round_state,
    )

    source = RandomSource(29)
    source.python.random()
    source.numpy.normal(size=3)
    torch.rand(4, generator=source.torch)
    state = RoundState(
        round_index=7,
        global_model=ModelState.from_tensors((
            np.array([[1.0, 2.0]], dtype=np.float32),
            np.array([-3.0], dtype=np.float64),
        )),
        random_snapshot=source.capture(),
        component_states={'scheduler': {'step': 7}, 'enabled': True},
    )
    path = tmp_path / 'round-7.msr'

    save_round_state(path, state)
    restored = load_round_state(path)

    assert restored.round_index == 7
    assert dict(restored.component_states) == {'scheduler': {'step': 7}, 'enabled': True}
    for expected, actual in zip(state.global_model.tensors, restored.global_model.tensors):
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype

    expected_rng = RandomSource(0)
    expected_rng.restore(state.random_snapshot)
    actual_rng = RandomSource(0)
    actual_rng.restore(restored.random_snapshot)
    assert actual_rng.python.random() == expected_rng.python.random()
    np.testing.assert_array_equal(actual_rng.numpy.integers(0, 100, size=8), expected_rng.numpy.integers(0, 100, size=8))
    assert torch.equal(
        torch.rand(8, generator=actual_rng.torch),
        torch.rand(8, generator=expected_rng.torch),
    )


def test_round_checkpoint_rejects_non_json_component_state(tmp_path) -> None:
    from meta_stackelberg.federated.checkpointing.round_state import save_round_state

    source = RandomSource(3)
    state = RoundState(
        round_index=0,
        global_model=ModelState.from_tensors((np.zeros(1, dtype=np.float32),)),
        random_snapshot=source.capture(),
        component_states={'invalid': object()},
    )

    with pytest.raises(TypeError, match='JSON-compatible'):
        save_round_state(tmp_path / 'invalid.msr', state)


def test_round_checkpoint_rejects_unknown_schema_version(tmp_path) -> None:
    from meta_stackelberg.federated.checkpointing.round_state import load_round_state

    path = tmp_path / 'future.msr'
    with zipfile.ZipFile(path, mode='w') as archive:
        archive.writestr('metadata.json', json.dumps({'schema': 'meta-stackelberg-round-state', 'version': 99}))

    with pytest.raises(ValueError, match='version'):
        load_round_state(path)


def test_round_checkpoint_rejects_invalid_rng_payload_at_load_time(tmp_path) -> None:
    from meta_stackelberg.federated.checkpointing.round_state import (
        load_round_state,
        save_round_state,
    )

    source = RandomSource(5)
    state = RoundState(
        round_index=1,
        global_model=ModelState.from_tensors((np.zeros(1, dtype=np.float32),)),
        random_snapshot=source.capture(),
    )
    path = tmp_path / 'corrupt-rng.msr'
    save_round_state(path, state)
    with zipfile.ZipFile(path, mode='r') as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    contents = BytesIO()
    np.save(contents, np.array([1, 2], dtype=np.int64), allow_pickle=False)
    entries['rng/torch_cpu.npy'] = contents.getvalue()
    with zipfile.ZipFile(path, mode='w') as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)

    with pytest.raises(ValueError, match='random state'):
        load_round_state(path)


def test_failed_checkpoint_write_preserves_destination_and_cleans_temporary_file(
    tmp_path,
    monkeypatch,
) -> None:
    from meta_stackelberg.federated.checkpointing import round_state as checkpointing

    source = RandomSource(7)
    original = RoundState(
        round_index=2,
        global_model=ModelState.from_tensors((np.ones(1, dtype=np.float32),)),
        random_snapshot=source.capture(),
    )
    path = tmp_path / 'round.msr'
    checkpointing.save_round_state(path, original)

    def fail_to_encode(_array):
        raise RuntimeError('injected write failure')

    monkeypatch.setattr(checkpointing, '_array_bytes', fail_to_encode)
    with pytest.raises(RuntimeError, match='injected write failure'):
        checkpointing.save_round_state(path, original)

    assert checkpointing.load_round_state(path).round_index == 2
    assert list(tmp_path.glob('*.tmp')) == []
