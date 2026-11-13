"""Design: keep experiment settings as ordinary JSON-compatible dictionaries.
Data flow: JSON sections -> merge section defaults -> resolved settings.
Reject unknown section fields at this boundary rather than silently ignoring typos.
"""

from copy import deepcopy
import json
from pathlib import Path

from meta_sg.rl.agent import DEFAULTS as TD3_DEFAULTS
from meta_sg.rl.actions import resolve_action
from meta_sg.rl.observations import resolve_observation


DEFAULTS = {
    'dataset': {'name': 'mnist', 'data_root': 'data', 'options': {}},
    'model': {'name': 'cnn'},
    'partition': {'method': 'iid', 'alpha': 0.5, 'root_samples_per_class': 20},
    'federation': {'num_clients': 100, 'clients_per_round': 10},
    'training': {'rounds': 5, 'local_steps': 1, 'root_steps': 5, 'batch_size': 128,
                 'local_epochs': None, 'learning_rate': 0.05, 'learning_rate_decay': 1.0,
                 'momentum': 0.0, 'weight_decay': 0.0, 'seed': 42, 'device': 'cpu', 'num_workers': 0},
    'defense': {'name': 'fedavg', 'parameters': {}},
    'attack': {'name': 'none', 'malicious_clients': [], 'start_round': 3, 'parameters': {}},
    'evaluation': {'batch_size': 256},
    'output': {'directory': None, 'checkpoint_interval': 50},
    'rl': {'mode': 'train', 'checkpoint': None, 'fixed_action': [1.0, 0.0, 1.0],
           'observation': {'name': 'compact8', 'parameters': {}},
           'action': {'name': 'mixed_norm', 'parameters': {}},
           'reward': {'name': 'root_loss_delta', 'parameters': {}}, 'agent': TD3_DEFAULTS},
    'meta': {'tasks': [], 'iterations': 10, 'tasks_per_iteration': 2, 'horizon': 20,
             'inner_steps': 2, 'response_steps': 2, 'post_critic_steps': 20,
             'outer_learning_rate': 1.0, 'follower_agent': {}, 'feedback_samples': 256,
             'checkpoint_interval': 1},
}


def load_config(path: str | Path) -> dict:
    return resolve_config(json.loads(Path(path).read_text()))


def resolve_config(supplied: dict) -> dict:
    config = deepcopy(DEFAULTS)
    for section, values in supplied.items():
        unknown = values.keys() - config[section].keys()
        if unknown:
            raise ValueError(f'unknown fields in {section}: {sorted(unknown)}')
        config[section].update(deepcopy(values))
    config['rl']['agent'] = {**TD3_DEFAULTS, **config['rl']['agent']}
    for component in ['observation', 'reward']:
        config['rl'][component].setdefault('parameters', {})
    config['rl']['observation'] = resolve_observation(config['rl']['observation'])
    legacy = config['defense']['parameters'].pop('action', {}) if config['defense']['name'] == 'rl' else {}
    config['rl']['action'] = resolve_action(config['rl']['action'], legacy)
    return config
