"""Portable, pickle-free serialization of clean federated round state."""

from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import zipfile

import numpy as np

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.core.random_state import RandomSnapshot, RandomSource
from meta_stackelberg.federated.types import RoundState

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


_SCHEMA = 'meta-stackelberg-round-state'
_VERSION = 1


def save_round_state(path: str | Path, state: RoundState) -> None:
    """Atomically save a round boundary, including all explicit RNG streams."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    components = dict(state.component_states)
    try:
        json.dumps(components, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise TypeError('component_states must be JSON-compatible') from error

    tensor_specs = [
        {'name': f'model/{index}.npy', 'shape': list(tensor.shape), 'dtype': tensor.dtype.str}
        for index, tensor in enumerate(state.global_model.tensors)
    ]
    has_torch_state = state.random_snapshot.torch_cpu_state is not None
    metadata = {
        'schema': _SCHEMA,
        'version': _VERSION,
        'round_index': state.round_index,
        'component_states': components,
        'model_tensors': tensor_specs,
        'python_random_state': _tuples_to_lists(state.random_snapshot.python_state),
        'numpy_random_state': state.random_snapshot.numpy_state,
        'has_torch_cpu_state': has_torch_state,
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + '.',
        suffix='.tmp',
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, mode='w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('metadata.json', json.dumps(metadata, allow_nan=False))
            for spec, tensor in zip(tensor_specs, state.global_model.tensors):
                archive.writestr(spec['name'], _array_bytes(tensor))
            if has_torch_state:
                torch_state = state.random_snapshot.torch_cpu_state
                if torch is None or not isinstance(torch_state, torch.Tensor):
                    raise TypeError('torch_cpu_state must be a Torch tensor')
                archive.writestr('rng/torch_cpu.npy', _array_bytes(torch_state.cpu().numpy()))
        with temporary.open('rb') as checkpoint_file:
            os.fsync(checkpoint_file.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_round_state(path: str | Path) -> RoundState:
    """Load and validate a version-one round checkpoint without unpickling."""

    with zipfile.ZipFile(Path(path), mode='r') as archive:
        metadata = json.loads(archive.read('metadata.json'))
        if metadata.get('schema') != _SCHEMA:
            raise ValueError('unknown round checkpoint schema')
        if metadata.get('version') != _VERSION:
            raise ValueError(f'unsupported round checkpoint version {metadata.get("version")!r}')

        tensors = []
        for spec in metadata['model_tensors']:
            tensor = _load_array(archive.read(spec['name']))
            if list(tensor.shape) != spec['shape'] or tensor.dtype.str != spec['dtype']:
                raise ValueError(f'model tensor {spec["name"]!r} does not match metadata')
            tensors.append(tensor)

        torch_state = None
        if metadata['has_torch_cpu_state']:
            if torch is None:
                raise RuntimeError('Torch is required to restore this checkpoint')
            torch_state = torch.from_numpy(_load_array(archive.read('rng/torch_cpu.npy')).copy())

    snapshot = RandomSnapshot(
        python_state=_lists_to_tuples(metadata['python_random_state']),
        numpy_state=metadata['numpy_random_state'],
        torch_cpu_state=torch_state,
    )
    try:
        RandomSource(0).restore(snapshot)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError('checkpoint contains an invalid random state') from error
    return RoundState(
        round_index=int(metadata['round_index']),
        global_model=ModelState.from_tensors(tensors),
        random_snapshot=snapshot,
        component_states=metadata['component_states'],
    )


def _array_bytes(array: np.ndarray) -> bytes:
    output = BytesIO()
    np.save(output, np.asarray(array), allow_pickle=False)
    return output.getvalue()


def _load_array(contents: bytes) -> np.ndarray:
    return np.load(BytesIO(contents), allow_pickle=False)


def _tuples_to_lists(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_tuples_to_lists(item) for item in value]
    return value


def _lists_to_tuples(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_lists_to_tuples(item) for item in value)
    return value
