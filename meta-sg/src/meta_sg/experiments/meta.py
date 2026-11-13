"""Design: configure and persist first-order TD3 Meta-SG without a second simulator.
Data flow: JSON tasks and shared FL components -> finite-response meta iterations ->
per-round/phase metrics, full outer-boundary checkpoint, frozen defender export.
Resume rolls back incomplete outer work and replays it from saved episode RNGs.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform

import numpy as np
import torch
import torchvision

from meta_sg.experiments.artifacts import random_state, restore_random_state, save_tensor_file, write_json
from meta_sg.experiments.config import load_config, resolve_config
from meta_sg.experiments.meta_game import FederatedMetaGame
from meta_sg.experiments.run import describe_experiment
from meta_sg.rl.meta import MetaSGLearner


def describe_meta(config):
    if config['defense']['name'] != 'rl' or config['rl']['mode'] != 'train':
        raise ValueError('Meta-SG requires defense.name=rl and rl.mode=train')
    if config['attack']['name'] != 'none':
        raise ValueError('configure meta attacks in meta.tasks and set attack.name=none')
    report = describe_experiment(config)
    settings, tasks = config['meta'], config['meta']['tasks']
    names = [task['name'] for task in tasks]
    if len(tasks) < 2 or len(set(names)) != len(names):
        raise ValueError('Meta-SG requires at least two uniquely named training tasks')
    if not 1 <= settings['tasks_per_iteration'] <= len(tasks):
        raise ValueError('tasks_per_iteration must select a nonempty subset without replacement')
    if min(settings[key] for key in ['iterations', 'horizon', 'inner_steps', 'post_critic_steps', 'checkpoint_interval', 'feedback_samples']) < 1:
        raise ValueError('meta iteration, rollout, update, checkpoint, and feedback budgets must be positive')
    if settings['response_steps'] < 0 or settings['outer_learning_rate'] <= 0:
        raise ValueError('response_steps must be nonnegative and outer_learning_rate positive')
    if config['output']['directory'] is None:
        raise ValueError('Meta-SG requires an output.directory')
    for task in tasks:
        if task['kind'] not in {'model_poisoning', 'backdoor'}:
            raise ValueError('meta task kind must be model_poisoning or backdoor')
        ids = task['malicious_clients']
        if not ids or len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= config['federation']['num_clients']:
            raise ValueError('each meta task needs valid distinct malicious population IDs')
        if task.get('start_round', 3) > settings['horizon']:
            raise ValueError('meta horizon must include the task attack onset')
    report['resolved'].update(algorithm='first_order_meta_sg_td3', tasks=names, follower_observation_dimensions=8,
                              follower_action_dimensions=3, maximum_fl_rounds=settings['iterations'] * settings['tasks_per_iteration']
                              * (settings['inner_steps'] + settings['response_steps'] + 1) * settings['horizon'])
    return report


def _check_resume(config, saved, iteration):
    current, previous = resolve_config(config), resolve_config(saved)
    for settings in [current, previous]:
        settings.pop('output')
        settings['meta'].pop('iterations')
        settings['meta'].pop('checkpoint_interval')
        settings['dataset'].pop('data_root')
    if current != previous or config['meta']['iterations'] < iteration:
        raise ValueError('meta resume requires unchanged experiment settings and a total budget at least the saved iteration')


def run_meta(config, *, resume=None):
    config = resolve_config(config)
    config['dataset']['data_root'] = str(Path(config['dataset']['data_root']).expanduser().resolve())
    resume = None if resume is None else Path(resume).expanduser().resolve()
    if resume is not None:
        config['output']['directory'] = str(resume.parent)
    describe_meta(config)
    output = Path(config['output']['directory']).expanduser().resolve()
    config['output']['directory'] = str(output)
    saved = None
    if resume is not None:
        saved = torch.load(resume, map_location='cpu', weights_only=True)
        _check_resume(config, saved['config'], saved['learner']['iteration'])
    else:
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise FileExistsError(f'fresh meta experiment requires an empty output directory: {output}')
    game = FederatedMetaGame(config, partitions=None if saved is None else saved['partitions'],
                             initial_model=None if saved is None else saved['initial_model'],
                             observation_state=None if saved is None else saved['observation_state'], output=output)
    if saved is not None and game.signature != saved['dataset']:
        raise ValueError('meta dataset contents differ from the checkpoint')
    width = game.observation_width()
    learner = MetaSGLearner(config, width, len(game.template.action_names), config['training']['seed'], config['training']['device'])
    if saved is not None:
        learner.load_state_dict(saved['learner'])
        restore_random_state(saved['random'])
    write_json(output / 'config.json', config)
    write_json(output / 'protocol.json', {
        'algorithm': 'first_order_meta_sg_td3_finite_response',
        'sequence': ['task-local defender adaptation', 'persistent follower response with defender fixed',
                     'fresh post-response replay and critic-only fitting', 'actor gradient at adapted parameters',
                     'outer actor descent, averaged critics, synchronized targets'],
        'follower_feedback': 'attacker-owned training clean accuracy; backdoor target accuracy minus clean accuracy',
        'defender_feedback': config['rl']['reward'], 'tasks': config['meta']['tasks'],
        'follower_clock': 'one transition per physical round; controls execute only when an attacker is active',
        'test_metrics_used_for_learning': False, 'exact_best_response': False, 'exact_paper_reproduction': False,
        'defender_agent': learner.defender.config, 'follower_agent': learner.follower_parameters,
        'defender_observation_dimensions': width, 'defender_action_names': list(game.template.action_names),
        'follower_observation_dimensions': 8, 'follower_action_dimensions': 3,
        'dataset': game.signature, 'python': platform.python_version(), 'torch': str(torch.__version__),
        'torchvision': str(torchvision.__version__), 'numpy': np.__version__,
    })
    phase_path, round_path = output / 'metrics.jsonl', output / 'rounds.jsonl'
    if saved is not None:
        for path, offset in [(phase_path, saved['metrics_offset']), (round_path, saved['rounds_offset'])]:
            if path.stat().st_size < offset:
                raise ValueError('meta log is shorter than the checkpoint prefix')
    with phase_path.open('r+' if saved else 'w', encoding='utf-8') as phases, round_path.open('r+' if saved else 'w', encoding='utf-8') as rounds:
        if saved is not None:
            phases.seek(saved['metrics_offset'])
            phases.truncate()
            rounds.seek(saved['rounds_offset'])
            rounds.truncate()

        def log_round(record):
            rounds.write(json.dumps(record, allow_nan=False) + '\n')
            rounds.flush()

        game.log = log_round

        def checkpoint():
            save_tensor_file(output / 'latest.pt', {
                'config': config, 'learner': learner.state_dict(), 'partitions': game.partitions,
                'initial_model': game.initial, 'observation_state': game.observation_state(), 'dataset': game.signature,
                'random': random_state(config['training']['device']),
                'metrics_offset': phases.tell(), 'rounds_offset': rounds.tell(),
            })
            observation_state = game.observation_state()
            if observation_state is not None:
                write_json(output / 'observation.json', observation_state)
            learner.defender.save_policy(output / 'policy.zip')
            write_json(output / 'policy.json', {
                'meta_iteration': learner.iteration, 'observation': game.template.settings['observation'],
                'observation_state': observation_state, 'observation_dim': width,
                'reward': game.template.settings['reward'], 'action_parameters': game.template.parameters,
                'action': game.template.settings['action'], 'actions': list(game.template.action_names),
            })

        status = {'phase': 'running', 'iteration': learner.iteration, 'episodes': learner.episodes,
                  'fl_rounds': learner.fl_rounds, 'target_iterations': config['meta']['iterations']}
        write_json(output / 'status.json', status)
        try:
            if saved is None:
                checkpoint()
            while learner.iteration < config['meta']['iterations']:
                records, summary = learner.step(game)
                records.append({'phase': 'outer', **summary})
                for record in records:
                    phases.write(json.dumps(record, allow_nan=False) + '\n')
                phases.flush()
                if learner.iteration % config['meta']['checkpoint_interval'] == 0 or learner.iteration == config['meta']['iterations']:
                    checkpoint()
                status.update(summary, phase='complete' if learner.iteration == config['meta']['iterations'] else 'running')
                write_json(output / 'status.json', status)
                yield summary
            status['phase'] = 'complete'
            write_json(output / 'status.json', status)
        except BaseException as error:
            if status['phase'] != 'complete':
                status.update(phase='interrupted' if isinstance(error, (KeyboardInterrupt, GeneratorExit)) else 'failed',
                              error=type(error).__name__ + ': ' + str(error))
                write_json(output / 'status.json', status)
            raise


def main():
    parser = argparse.ArgumentParser(description='Train first-order TD3 Meta-SG from the shared JSON configuration')
    parser.add_argument('--config')
    parser.add_argument('--resume')
    parser.add_argument('--output')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.config:
        config = load_config(args.config)
    elif args.resume:
        config = torch.load(args.resume, map_location='cpu', weights_only=True)['config']
    else:
        parser.error('provide --config or --resume')
    if args.output:
        if args.resume:
            parser.error('meta resume continues in its checkpoint directory; omit --output')
        config['output']['directory'] = args.output
    if args.dry_run:
        print(json.dumps(describe_meta(config), indent=2))
    else:
        for record in run_meta(config, resume=args.resume):
            print(json.dumps(record, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
