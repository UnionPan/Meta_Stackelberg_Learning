"""Design: compose replaceable observation/action/reward functions with TD3.
Data flow: trusted-root client features -> observation -> selected controls -> delta;
updated global model -> root feedback -> pending transition -> next-round replay.
Each run trains one attack-specific policy. No attack identity or test feedback enters it.
"""

from copy import deepcopy
from functools import partial
import json
from pathlib import Path

import numpy as np

from meta_sg.rl.actions import build_action, resolve_action
from meta_sg.experiments.artifacts import write_json
from meta_sg.rl.agent import TD3Agent, isolated_random_state
from meta_sg.rl.features import root_metrics, round_context
from meta_sg.rl.observations import OBSERVATIONS, resolve_observation
from meta_sg.rl.distributions import COMPONENTS, FrozenDistributionObservation
from meta_sg.rl.rewards import REWARDS


class RLDefender:
    def __init__(self, settings, parameters, root_loader, device, seed, *, observation=None, reward=None):
        self.settings, self.parameters = deepcopy(settings), deepcopy(parameters)
        self.settings['action'] = resolve_action(settings['action'], self.parameters.pop('action', {}))
        self.action_names, self.apply_action = build_action(self.settings['action'])
        if not self.action_names and settings['mode'] != 'fixed':
            raise ValueError('train and frozen modes require at least one policy action')
        self.root_loader, self.device, self.seed = root_loader, device, seed
        spec = resolve_observation(settings['observation'])
        self.settings['observation'] = spec
        self.distribution = (FrozenDistributionObservation(spec['name'], **spec['parameters'])
                             if observation is None and spec['name'] in COMPONENTS else None)
        self.observation = (observation if observation is not None else self.distribution if self.distribution is not None
                            else partial(OBSERVATIONS[spec['name']], **spec['parameters']))
        spec = settings['reward']
        self.reward = reward if reward is not None else partial(REWARDS[spec['name']], **spec.get('parameters', {}))
        self.agent, self.pending, self.current = None, None, None
        self.frozen = None
        if settings['mode'] == 'frozen':
            from stable_baselines3 import TD3
            path = Path(settings['checkpoint'])
            metadata = json.loads(path.with_suffix('.json').read_text())
            saved_parameters = deepcopy(metadata['action_parameters'])
            saved_action = resolve_action(metadata.get('action', {'name': 'mixed_norm'}), saved_parameters.pop('action', {}))
            if (resolve_observation(metadata['observation']) != self.settings['observation'] or saved_parameters != self.parameters
                    or saved_action != self.settings['action']):
                raise ValueError('frozen policy observation and action configuration must match its export')
            if self.distribution is not None:
                self.distribution.load_state_dict(metadata['observation_state'])
            with isolated_random_state(device):
                self.frozen = TD3.load(path, device='cpu')
            self.frozen.policy.set_training_mode(False)
            self.frozen.policy.requires_grad_(False)

    def aggregate(self, model, updates, root_update):
        context = round_context(model, updates, root_update, self.root_loader, self.device,
                                **self.parameters.get('features', {}))
        observation = np.asarray(self.observation(context), dtype=np.float32)
        action = self.select_action(observation)
        update, diagnostics = self.apply_action(updates, context, action)
        self.current = (observation, action, context['before'], diagnostics['retained_risk'])
        return update, {**diagnostics, 'observation': observation.tolist(), 'action': action.tolist(),
                        'risks': context['risks'].tolist(), 'mode': self.settings['mode'],
                        'client_features': np.asarray(context['features'], dtype=np.float32).tolist(),
                        'client_norms': context['norms'].tolist(),
                        'action_names': list(self.action_names)}

    def select_action(self, observation):
        if self.settings['mode'] == 'train':
            if self.agent is None:
                self.agent = TD3Agent(len(observation), self.settings['agent'], self.seed, self.device,
                                      action_dimensions=len(self.action_names))
            if self.pending is not None:
                previous, action, reward = self.pending
                self.agent.observe(np.asarray(previous, dtype=np.float32), np.asarray(action, dtype=np.float32),
                                   reward, observation, False)
                self.pending = None
            action = self.agent.action(observation)
        elif self.settings['mode'] == 'frozen':
            if self.frozen.observation_space.shape != observation.shape or self.frozen.action_space.shape != (len(self.action_names),):
                raise ValueError('frozen policy dimensions do not match the selected components')
            action = self.frozen.predict(observation, deterministic=True)[0]
        else:
            action = np.asarray(self.settings['fixed_action'], dtype=np.float32)
        return action

    def feedback(self, model, *, terminal):
        observation, action, before, retained_risk = self.current
        after, _ = root_metrics(model, self.root_loader, self.device)
        reward = float(self.reward({'before': before, 'after': after, 'retained_risk': retained_risk}))
        if self.settings['mode'] == 'train':
            if terminal:
                self.agent.observe(observation, action, reward, np.zeros_like(observation), True)
            else:
                self.pending = (observation.tolist(), action.tolist(), reward)
        self.current = None
        return {'reward': reward, 'root_before': before, 'root_after': after,
                'td3_updates': self.agent.model._n_updates if self.agent is not None else 0}

    def state_dict(self):
        return {'pending': self.pending, 'agent': self.agent.state_dict() if self.agent is not None else None,
                'observation': self.distribution.state_dict() if self.distribution is not None else None}

    def load_state_dict(self, state):
        self.pending = state['pending']
        if self.distribution is not None:
            self.distribution.load_state_dict(state['observation'])
        if state['agent'] is not None:
            self.agent = TD3Agent(state['agent']['width'], self.settings['agent'], self.seed, self.device,
                                  action_dimensions=len(self.action_names))
            self.agent.load_state_dict(state['agent'])

    def save_policy(self, output, round_index):
        observation_state = self.distribution.state_dict() if self.distribution is not None else None
        if observation_state is not None:
            write_json(output / 'observation.json', observation_state)
        if self.agent is not None:
            self.agent.save_policy(output / 'policy.zip')
            write_json(output / 'policy.json', {
                'round': round_index, 'observation': self.settings['observation'], 'reward': self.settings['reward'],
                'action_parameters': self.parameters, 'observation_dim': self.agent.width,
                'action': self.settings['action'], 'actions': list(self.action_names),
                'observation_state': observation_state,
            })
