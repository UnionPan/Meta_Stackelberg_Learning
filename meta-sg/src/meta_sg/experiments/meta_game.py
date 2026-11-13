"""Design: supply actual-FL episodes to the Meta-SG learner using shared components.
Data flow: reset one initial classifier/partition -> fresh seeded client streams ->
follower submissions -> configurable defender -> root/owned feedback -> transitions.
Test evaluation is disabled; the follower receives only the previous Compact8.
"""

from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import RandomSampler

from meta_sg.attacks.follower import FollowerRound
from meta_sg.datasets import cifar10, mnist, tiny_imagenet
from meta_sg.experiments.artifacts import dataset_signature, partition_state, save_tensor_file
from meta_sg.experiments.run import build_experiment
from meta_sg.federated import run_federated
from meta_sg.rl.defender import RLDefender
from meta_sg.rl.observations import compact8


class EpisodeDefender(RLDefender):
    def __init__(self, template, policy, follower, *, explore):
        super().__init__(template.settings, template.parameters, template.root_loader,
                         template.device, template.seed)
        if self.distribution is not None:
            self.distribution.load_state_dict(template.distribution.state_dict())
        self.policy, self.follower, self.explore = policy, follower, explore
        self.transitions, self.following = [], None

    def select_action(self, observation):
        return self.policy.action(observation) if self.explore else self.policy.model.predict(observation, deterministic=True)[0]

    def aggregate(self, model, updates, root_update):
        update, diagnostics = super().aggregate(model, updates, root_update)
        diagnostics['mode'] = 'meta'
        self.follower.following = compact8({'features': np.asarray(diagnostics['client_features'], dtype=np.float32)})
        return update, diagnostics

    def feedback(self, model, *, terminal):
        observation, action, _, _ = self.current
        diagnostics = super().feedback(model, terminal=terminal)
        self.transitions.append((observation.copy(), action.copy(), diagnostics['reward'], None, terminal))
        self.follower.feedback(model, terminal=terminal)
        return diagnostics

    def completed_transitions(self):
        return [(observation, action, reward, self.transitions[index + 1][0].copy() if index + 1 < len(self.transitions)
                 else np.zeros_like(observation), terminal)
                for index, (observation, action, reward, _, terminal) in enumerate(self.transitions)]


class FederatedMetaGame:
    def __init__(self, config, *, partitions=None, initial_model=None, observation_state=None, output=None, log=None):
        self.config = deepcopy(config)
        setup = deepcopy(config)
        setup['defense']['name'] = 'rl'
        setup['rl']['mode'] = 'fixed'
        setup['attack']['name'] = 'none'
        self.model, self.arguments = build_experiment(setup, partitions=partitions)
        if initial_model is not None:
            self.model.load_state_dict(initial_model)
        self.initial = {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()}
        self.signature = dataset_signature(self.arguments)
        self.partitions = partition_state(self.arguments)
        self.template = self.arguments['defender']
        if self.template.distribution is not None and observation_state is not None:
            self.template.distribution.load_state_dict(observation_state)
        train = self.arguments['client_loaders'][0].dataset.dataset
        self.train = train
        self.evaluation_transform = self.template.root_loader.dataset.dataset.transform
        trigger = {'mnist': mnist.dba_trigger, 'cifar10': cifar10.dba_trigger, 'tiny_imagenet': tiny_imagenet.dba_trigger}[config['dataset']['name']]
        if config['dataset']['name'] == 'tiny_imagenet':
            from functools import partial
            trigger = partial(trigger, normalization=config['dataset']['options'].get('normalization', 'legacy'))
        self.trigger, self.output, self.log = trigger, output, log
        self.arguments.update(test_loader=None, backdoor_evaluation=None, attack=None)

    def observation_width(self):
        count = self.config['federation']['clients_per_round']
        context = {'features': np.zeros((count, 4)), 'norms': np.ones(count), 'root_norm': 1.,
                   'risks': np.zeros(count), 'before': {'loss': 0., 'accuracy': 0.}, 'score_limit': 8.}
        return len(self.template.observation(context))

    def observation_state(self):
        return self.template.distribution.state_dict() if self.template.distribution is not None else None

    def __call__(self, task, defender, follower, horizon, seed, phase, episode, defender_explore, follower_explore):
        import random

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.model.load_state_dict(self.initial)
        loaders = list(self.arguments['client_loaders']) + [self.arguments['root_loader']]
        for index, loader in enumerate(loaders):
            loader.generator.manual_seed(seed + 2 * index)
            if isinstance(loader.sampler, RandomSampler):
                loader.sampler.generator.manual_seed(seed + 2 * index + 1)
        attacker = FollowerRound(task, follower, self.train, self.arguments['client_loaders'], self.evaluation_transform,
                                 self.trigger, self.config['training']['batch_size'], self.config['meta']['feedback_samples'],
                                 self.config['training']['device'], seed, explore=follower_explore)
        controller = EpisodeDefender(self.template, defender, attacker, explore=defender_explore)
        arguments = {**self.arguments, 'seed': seed, 'rounds': horizon, 'participant_generator': np.random.default_rng(seed),
                     'attack': attacker, 'defender': controller, 'start_round': 0}
        active_rounds = 0
        for record in run_federated(self.model, **arguments):
            active_rounds += int(record['attack']['follower_active'])
            if self.log is not None:
                self.log({'episode': episode, 'task': task['name'], 'phase': phase, **record})
            if self.output is not None and (record['round'] % self.config['output']['checkpoint_interval'] == 0 or record['round'] == horizon):
                save_tensor_file(self.output / 'latest_fl.pt', {'episode': episode, 'round': record['round'],
                                  'model': {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()}})
        defender_transitions = controller.completed_transitions()
        return {'defender': defender_transitions, 'follower': attacker.transitions,
                'metrics': {'defender_reward_mean': float(np.mean([row[2] for row in defender_transitions])),
                            'follower_reward_mean': float(np.mean([row[2] for row in attacker.transitions])),
                            'follower_active_rounds': active_rounds}}
