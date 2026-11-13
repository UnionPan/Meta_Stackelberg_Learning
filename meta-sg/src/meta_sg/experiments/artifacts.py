"""Design: persist completed-round state with ordinary files and dictionaries.
Data flow: model, RNGs, loaders, and round -> atomic checkpoint; saved state ->
reconstructed model/loaders -> continuation without repeating initial evaluation.
Local SGD optimizers are recreated per call and have no cross-round state.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import RandomSampler

from meta_sg.experiments.config import resolve_config


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def save_tensor_file(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def file_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dataset_signature(arguments):
    train = arguments['client_loaders'][0].dataset.dataset
    test = arguments['test_loader'].dataset
    result = {}
    for name, dataset in [('train', train), ('test', test)]:
        images = np.ascontiguousarray(dataset.data)
        labels = np.asarray(dataset.targets, dtype=np.int64)
        result[name] = {
            'shape': list(images.shape), 'dtype': str(images.dtype),
            'images_sha256': hashlib.sha256(memoryview(images).cast('B')).hexdigest(),
            'labels_sha256': hashlib.sha256(labels.tobytes()).hexdigest(),
        }
    return result


def partition_state(arguments):
    clients = [torch.tensor(loader.dataset.indices, dtype=torch.long) for loader in arguments['client_loaders']]
    if arguments['root_loader'] is not None:
        root = torch.tensor(arguments['root_loader'].dataset.indices, dtype=torch.long)
    else:
        remaining = torch.ones(len(arguments['client_loaders'][0].dataset.dataset), dtype=torch.bool)
        remaining[torch.cat(clients)] = False
        root = torch.where(remaining)[0]
    return {'root': root, 'clients': clients}


def _loader_generators(loader):
    generators = [loader.generator]
    if isinstance(loader.sampler, RandomSampler):
        generators.append(loader.sampler.generator)
    return generators


def _loaders(arguments):
    loaders = list(arguments['client_loaders']) + [arguments['test_loader']]
    if arguments['root_loader'] is not None:
        loaders.append(arguments['root_loader'])
    return loaders


def random_state(device):
    numpy_state = np.random.get_state()
    device = torch.device(device)
    return {'python': random.getstate(), 'numpy': [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
            'mps': torch.mps.get_rng_state() if device.type == 'mps' else None}


def restore_random_state(state):
    random.setstate(state['python'])
    np.random.set_state((state['numpy'][0], np.asarray(state['numpy'][1], dtype=np.uint32), *state['numpy'][2:]))
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])
    if state['mps'] is not None:
        torch.mps.set_rng_state(state['mps'])


def checkpoint_state(model, arguments, config, round_index, metrics_offset, metadata):
    return {
        'round': round_index, 'config': config, 'metrics_offset': metrics_offset, 'metadata': metadata,
        'model': {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        'participants': deepcopy(arguments['participant_generator'].bit_generator.state),
        'defender': arguments['defender'].state_dict() if arguments['defender'] is not None else None,
        'loaders': [[generator.get_state() for generator in _loader_generators(loader)]
                    for loader in _loaders(arguments)],
        'random': random_state(config['training']['device']),
    }


def restore_checkpoint(checkpoint, model, arguments):
    model.load_state_dict(checkpoint['model'])
    arguments['start_round'] = checkpoint['round']
    arguments['participant_generator'].bit_generator.state = checkpoint['participants']
    if arguments['defender'] is not None:
        arguments['defender'].load_state_dict(checkpoint['defender'])
    for loader, states in zip(_loaders(arguments), checkpoint['loaders'], strict=True):
        for generator, state in zip(_loader_generators(loader), states, strict=True):
            generator.set_state(state)
    restore_random_state(checkpoint['random'])


def check_resume_config(config, saved, round_index):
    current, previous = resolve_config(config), resolve_config(saved)
    for settings in [current, previous]:
        settings.pop('output')
        settings['training'].pop('rounds')
        settings['dataset'].pop('data_root')
        if settings['attack']['name'] == 'rl':
            settings['attack']['parameters'].pop('checkpoint')
        if settings['defense']['name'] == 'rl' and settings['rl']['mode'] == 'frozen':
            settings['rl'].pop('checkpoint')
    if current != previous:
        raise ValueError('resume requires unchanged experiment settings; only rounds and output settings may change, or input paths may relocate')
    if config['training']['rounds'] < round_index:
        raise ValueError('training.rounds cannot be less than the checkpoint round')
