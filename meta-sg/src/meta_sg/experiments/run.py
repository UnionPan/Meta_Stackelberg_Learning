"""Design: assemble existing functions from one JSON configuration, without a runner class.
Data flow: settings -> datasets/model -> root and client splits -> seeded loaders ->
attack/defense callables -> federated round records, printed as JSON Lines by the CLI.
"""

import argparse
from copy import copy
from functools import partial
import json
import random
from pathlib import Path
import platform

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, RandomSampler, Subset

from meta_sg import defenses
from meta_sg.attacks import (
    FrozenLi2022Actor, bfl_round, dba_round, evaluate_asr, ipm_updates, lmp_updates,
    min_max_updates, model_poisoning_round, rl_round,
)
from meta_sg.attacks.model_poisoning.rl import archived_normalization
from meta_sg.data import partition_dirichlet_indices, partition_iid_indices, split_root_indices
from meta_sg.datasets import cifar10, mnist, tiny_imagenet
from meta_sg.experiments.config import load_config, resolve_config
from meta_sg.experiments.artifacts import (
    check_resume_config, checkpoint_state, dataset_signature, file_digest, partition_state,
    restore_checkpoint, save_tensor_file, write_json,
)
from meta_sg.federated import run_federated
from meta_sg.rl.defender import RLDefender
from meta_sg.rl.observations import OBSERVATIONS
from meta_sg.rl.rewards import REWARDS
from meta_sg.rl.actions import resolve_action
from meta_sg.rl.distributions import COMPONENTS, DistributionTransform, distribution_features


DATASETS = {
    'mnist': (mnist.load_mnist, mnist.dba_trigger),
    'cifar10': (cifar10.load_cifar10, cifar10.dba_trigger),
    'tiny_imagenet': (tiny_imagenet.load_tiny_imagenet, tiny_imagenet.dba_trigger),
}
MODELS = {
    'mnist': {'cnn': mnist.MNISTCNN},
    'cifar10': {'cnn': cifar10.CIFAR10CNN, 'resnet18': cifar10.cifar10_resnet18},
    'tiny_imagenet': {'cnn': tiny_imagenet.TinyImageNetCNN, 'resnet18': tiny_imagenet.tiny_imagenet_resnet18},
}
DEFENSES = {name: getattr(defenses, name) for name in (
    'fedavg', 'median', 'trimmed_mean', 'krum', 'multi_krum', 'rfa', 'fltrust', 'clipmed',
)}
MODEL_ATTACKS = {'ipm': ipm_updates, 'lmp': lmp_updates, 'min_max': min_max_updates}
BACKDOORS = {'dba': dba_round, 'bfl': bfl_round}


def describe_experiment(config):
    dataset = config['dataset']['name']
    loader, _ = DATASETS[dataset]
    model = MODELS[dataset][config['model']['name']]
    defense = config['defense']['name']
    if defense == 'rl':
        action = resolve_action(config['rl']['action'], config['defense']['parameters'].get('action', {}))
        OBSERVATIONS[config['rl']['observation']['name']]
        REWARDS[config['rl']['reward']['name']]
        if config['rl']['mode'] not in {'train', 'frozen', 'fixed'}:
            raise ValueError('rl.mode must be train, frozen, or fixed')
        if config['rl']['mode'] == 'frozen' and config['rl']['checkpoint'] is None:
            raise ValueError('frozen RL defense requires rl.checkpoint')
    else:
        DEFENSES[defense]
    attack = config['attack']['name']
    if attack not in {'none', 'rl', *MODEL_ATTACKS, *BACKDOORS}:
        raise ValueError(f'unknown attack: {attack}')
    if config['partition']['method'] not in {'iid', 'dirichlet'}:
        raise ValueError('partition method must be iid or dirichlet')
    if attack == 'rl' and dataset != 'mnist':
        raise ValueError('the archived RL attacker is configured for MNIST only')
    if attack != 'none' and not config['attack']['malicious_clients']:
        raise ValueError('an active attack needs explicit malicious_clients')
    if defense in {'fltrust', 'rl'} and config['partition']['root_samples_per_class'] == 0:
        raise ValueError('FLTrust and RL defense require a nonempty trusted root set')
    resolved = {'model': model.__name__, 'loader': loader.__name__}
    if defense == 'rl':
        resolved.update(action_names=action['controls'], action_dimensions=len(action['controls']))
        observation_name = config['rl']['observation']['name']
        if observation_name in COMPONENTS:
            resolved['observation_dimensions'] = 40
    return {'config': config, 'resolved': resolved}


def _training_loader(dataset, indices, *, batch_size, steps, seed, num_workers, epochs=None):
    subset = Subset(dataset, indices.tolist())
    generator = torch.Generator().manual_seed(seed)
    if epochs is not None:
        sampler = RandomSampler(subset, generator=generator)
    elif steps:
        sampler = RandomSampler(subset, replacement=True, num_samples=steps * batch_size, generator=generator)
    else:
        sampler = None
    return DataLoader(
        subset, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed + 1),
    )


def _attack(config, train, trigger):
    settings = config['attack']
    name = settings['name']
    if name == 'none':
        return None, None
    parameters = dict(settings['parameters'])
    common = {'malicious_clients': settings['malicious_clients'], 'attack_start': settings['start_round']}
    if name in MODEL_ATTACKS:
        return partial(model_poisoning_round, craft=partial(MODEL_ATTACKS[name], **parameters), **common), None
    if name in BACKDOORS:
        parameters.setdefault('target_label', 2 if config['dataset']['name'] == 'mnist' else 9)
        parameters.setdefault('source_label', 0 if name == 'bfl' and config['dataset']['name'] != 'tiny_imagenet' else None)
        attack = partial(BACKDOORS[name], trigger=trigger, **common, **parameters)
        evaluation = dict(trigger=trigger, target_label=parameters['target_label'],
                          source_label=parameters['source_label'])
        return attack, evaluation
    with torch.random.fork_rng(devices=[]):
        actor = FrozenLi2022Actor(parameters.pop('checkpoint'))
    known_indices = parameters.pop('known_indices')
    normalization = parameters.pop('known_normalization', 'defender')
    if normalization == 'defender':
        samples = [train[index] for index in known_indices]
        known_images = torch.stack([image for image, _ in samples])
        known_labels = torch.tensor([label for _, label in samples])
    elif normalization == 'archive':
        mean, std = archived_normalization(train.data)
        known_images = train.data[known_indices].float().div(255).sub(mean).div(std).unsqueeze(1)
        known_labels = train.targets[known_indices]
    else:
        raise ValueError('known_normalization must be defender or archive')
    return partial(rl_round, actor=actor, known_images=known_images,
                   known_labels=known_labels, **common, **parameters), None


def build_experiment(config, *, partitions=None):
    describe_experiment(config)
    settings = config['training']
    seed = settings['seed']
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)
    dataset = config['dataset']
    load, trigger = DATASETS[dataset['name']]
    model = MODELS[dataset['name']][config['model']['name']]
    train, test = load(dataset['data_root'], **dataset['options'])
    if dataset['name'] == 'tiny_imagenet':
        trigger = partial(trigger, normalization=dataset['options'].get('normalization', 'legacy'))
    count = config['federation']['num_clients']
    if partitions is not None:
        root, clients = partitions['root'], partitions['clients']
    else:
        root, remaining = split_root_indices(train.targets, config['partition']['root_samples_per_class'], seed)
        if config['partition']['method'] == 'iid':
            clients = partition_iid_indices(remaining, count, seed)
        else:
            clients = partition_dirichlet_indices(train.targets, remaining, count, config['partition']['alpha'], seed)
    loader_arguments = {'batch_size': settings['batch_size'], 'num_workers': settings['num_workers']}
    loaders = [_training_loader(train, indices, steps=settings['local_steps'], epochs=settings['local_epochs'], seed=seed + client,
                                 **loader_arguments) for client, indices in enumerate(clients)]
    test_loader = DataLoader(test, batch_size=config['evaluation']['batch_size'],
                             num_workers=settings['num_workers'],
                             generator=torch.Generator().manual_seed(seed + count + 1))
    root_loader = None
    if config['defense']['name'] in {'fltrust', 'rl'}:
        root_dataset = copy(train)
        root_dataset.transform = test.transform
        root_loader = _training_loader(root_dataset, root, steps=settings['root_steps'],
                                        seed=seed + count, **loader_arguments)
    global_model = model()
    defender = None
    if config['defense']['name'] == 'rl':
        probe_loader = DataLoader(Subset(root_dataset, root.tolist()), batch_size=config['evaluation']['batch_size'])
        defender = RLDefender(config['rl'], config['defense']['parameters'], probe_loader, settings['device'], seed)
    attack, backdoor = _attack(config, train, trigger)
    if backdoor is not None:
        backdoor = partial(evaluate_asr, loader=test_loader, device=settings['device'], **backdoor)
    arguments = {
        'client_loaders': loaders, 'test_loader': test_loader,
        'aggregate': None if defender is not None else partial(DEFENSES[config['defense']['name']], **config['defense']['parameters']),
        'rounds': settings['rounds'], 'clients_per_round': config['federation']['clients_per_round'],
        'local_steps': settings['local_steps'], 'learning_rate': settings['learning_rate'],
        'local_epochs': settings['local_epochs'], 'learning_rate_decay': settings['learning_rate_decay'],
        'momentum': settings['momentum'], 'weight_decay': settings['weight_decay'],
        'device': settings['device'], 'seed': seed, 'root_loader': root_loader,
        'root_steps': settings['root_steps'], 'attack': attack, 'backdoor_evaluation': backdoor,
        'participant_generator': np.random.default_rng(seed),
        'defender': defender,
    }
    return global_model, arguments


def run_experiment(config, *, resume=None):
    config = resolve_config(config)
    if resume is not None:
        resume = Path(resume).expanduser().resolve()
    config['dataset']['data_root'] = str(Path(config['dataset']['data_root']).expanduser().resolve())
    if config['attack']['name'] == 'rl':
        parameters = config['attack']['parameters']
        parameters['checkpoint'] = str(Path(parameters['checkpoint']).expanduser().resolve())
    if config['defense']['name'] == 'rl' and config['rl']['mode'] == 'frozen':
        config['rl']['checkpoint'] = str(Path(config['rl']['checkpoint']).expanduser().resolve())
    if config['output']['directory'] is None and resume is None:
        model, arguments = build_experiment(config)
        yield from run_federated(model, **arguments)
        return
    output = Path(resume).resolve().parent if resume is not None else Path(config['output']['directory']).expanduser().resolve()
    config['output']['directory'] = str(output)
    interval = config['output']['checkpoint_interval']
    if interval < 1:
        raise ValueError('checkpoint_interval must be positive')
    saved = None
    if resume is not None:
        saved = torch.load(resume, map_location='cpu', weights_only=True)
        check_resume_config(config, saved['config'], saved['round'])
        metadata = saved['metadata']
        if file_digest(output / 'partitions.pt') != metadata['partitions_sha256']:
            raise ValueError('saved partitions differ from the checkpoint')
        partitions = torch.load(output / 'partitions.pt', map_location='cpu', weights_only=True)
        model, arguments = build_experiment(config, partitions=partitions)
        if dataset_signature(arguments) != metadata['dataset']:
            raise ValueError('dataset contents differ from the checkpoint')
        if config['attack']['name'] == 'rl' and file_digest(config['attack']['parameters']['checkpoint']) != metadata['attacker_sha256']:
            raise ValueError('frozen attacker checkpoint contents changed')
        if config['defense']['name'] == 'rl' and config['rl']['mode'] == 'frozen' and file_digest(config['rl']['checkpoint']) != metadata['defender_sha256']:
            raise ValueError('frozen defender checkpoint contents changed')
        restore_checkpoint(saved, model, arguments)
    else:
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise FileExistsError(f'fresh experiment requires an empty output directory: {output}')
        model, arguments = build_experiment(config)
        save_tensor_file(output / 'partitions.pt', partition_state(arguments))
        metadata = {
            'dataset': dataset_signature(arguments), 'partitions_sha256': file_digest(output / 'partitions.pt'),
            'python': platform.python_version(), 'torch': str(torch.__version__),
            'torchvision': str(torchvision.__version__), 'numpy': np.__version__,
        }
        if config['attack']['name'] == 'rl':
            metadata['attacker_sha256'] = file_digest(config['attack']['parameters']['checkpoint'])
        if config['defense']['name'] == 'rl' and config['rl']['mode'] == 'frozen':
            metadata['defender_sha256'] = file_digest(config['rl']['checkpoint'])
        write_json(output / 'metadata.json', metadata)
    metrics_path = output / 'metrics.jsonl'
    if saved is not None and metrics_path.stat().st_size < saved['metrics_offset']:
        raise ValueError('metrics file is shorter than the saved checkpoint prefix')
    write_json(output / 'config.json', config)
    completed = saved['round'] if saved is not None else None
    checkpoint_round = completed
    status = {'phase': 'running', 'round': completed, 'checkpoint_round': checkpoint_round,
              'target_round': config['training']['rounds']}
    with metrics_path.open('r+' if saved is not None else 'w', encoding='utf-8') as metrics:
        if saved is not None:
            metrics.seek(saved['metrics_offset'])
            metrics.truncate()
        write_json(output / 'status.json', status)
        try:
            for record in run_federated(model, **arguments):
                metrics.write(json.dumps(record, allow_nan=False) + '\n')
                metrics.flush()
                completed = record['round']
                if completed % interval == 0 or completed == config['training']['rounds']:
                    state = checkpoint_state(model, arguments, config, completed, metrics.tell(), metadata)
                    save_tensor_file(output / 'latest.pt', state)
                    if arguments['defender'] is not None:
                        arguments['defender'].save_policy(output, completed)
                    checkpoint_round = completed
                status.update(round=completed, checkpoint_round=checkpoint_round,
                              phase='complete' if completed == config['training']['rounds'] else 'running')
                write_json(output / 'status.json', status)
                yield record
            status['phase'] = 'complete'
            write_json(output / 'status.json', status)
        except BaseException as error:
            if status['phase'] != 'complete':
                status['phase'] = 'interrupted' if isinstance(error, (KeyboardInterrupt, GeneratorExit)) else 'failed'
                status['error'] = type(error).__name__ + ': ' + str(error)
                write_json(output / 'status.json', status)
            raise


def main():
    parser = argparse.ArgumentParser(description='Run a configured federated experiment')
    parser.add_argument('--config')
    parser.add_argument('--output', help='Override output.directory')
    parser.add_argument('--resume', help='Resume a run from its latest.pt checkpoint')
    parser.add_argument('--fit-observation', choices=list(COMPONENTS), help='Fit a frozen transform from training contexts only')
    parser.add_argument('--calibration', help='Training-run metrics.jsonl or NPZ containing training_features [rounds, clients, 4]')
    parser.add_argument('--calibration-key', help='Array key in the training NPZ; default training_features')
    parser.add_argument('--no-standardize', action='store_true', help='Fit the transform without mean/std standardization')
    parser.add_argument('--signed-log-norm', action='store_true', help='For metrics.jsonl calibration, replace N with log(client_norm)')
    parser.add_argument('--dry-run', action='store_true', help='Print resolved choices without loading data or training')
    args = parser.parse_args()
    if args.fit_observation:
        if not args.calibration or not args.output or args.config or args.resume or args.dry_run:
            parser.error('--fit-observation requires --calibration and --output, without --config, --resume, or --dry-run')
        if Path(args.calibration).suffix == '.jsonl':
            if args.calibration_key:
                parser.error('--calibration-key applies only to NPZ archives')
            rows = [json.loads(line) for line in Path(args.calibration).read_text(encoding='utf-8').splitlines()]
            calibration_key = 'aggregation.client_features'
            training = np.asarray([
                distribution_features({'features': row['aggregation']['client_features'], 'norms': row['aggregation']['client_norms']},
                                      signed_log_norm=args.signed_log_norm)
                for row in rows if row['round'] > 0
            ])
            feature_norm = 'signed_log_norm' if args.signed_log_norm else 'absolute_log_norm_ratio'
        else:
            if args.signed_log_norm:
                parser.error('--signed-log-norm conversion requires metrics.jsonl; NPZ arrays must contain the intended N channel')
            with np.load(args.calibration, allow_pickle=False) as archive:
                calibration_key = args.calibration_key or 'training_features'
                training = archive[calibration_key]
            feature_norm = 'as_supplied'
        if training.ndim != 3 or training.shape[2] != 4:
            raise ValueError('training_features must have shape [rounds, clients, 4]')
        transform = DistributionTransform.fit(training, COMPONENTS[args.fit_observation], standardized=not args.no_standardize)
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f'transform output already exists: {output}')
        write_json(output, {**transform.to_dict(), 'source_sha256': file_digest(args.calibration),
                            'source_key': calibration_key, 'feature_norm': feature_norm,
                            'training_rounds': len(training), 'training_clients': int(np.prod(training.shape[:2]))})
        print(json.dumps({'transform': str(output), 'variant': transform.variant, 'dimensions': len(transform.mean)}))
        return
    if args.calibration or args.calibration_key or args.no_standardize or args.signed_log_norm:
        parser.error('calibration options and --no-standardize require --fit-observation')
    if args.config:
        config = load_config(args.config)
    elif args.resume:
        config = torch.load(args.resume, map_location='cpu', weights_only=True)['config']
    else:
        parser.error('provide --config or --resume')
    if args.output:
        if args.resume:
            parser.error('--resume continues in its existing directory; omit --output')
        config['output']['directory'] = args.output
    if args.dry_run:
        print(json.dumps(describe_experiment(config), indent=2))
    else:
        for record in run_experiment(config, resume=args.resume):
            print(json.dumps(record, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
