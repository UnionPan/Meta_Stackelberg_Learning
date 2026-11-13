"""Design: reuse the existing SB3 TD3 adapter for explicitly supplied FL transitions.
Data flow: observation -> warmup/noisy action -> replay transition -> TD3 updates.
Policy optimization RNGs are isolated from classifier training and checkpointed.
"""

from contextlib import contextmanager
from copy import deepcopy
import random

import numpy as np
import torch


DEFAULTS = {'learning_rate': .001, 'buffer_size': 10000, 'learning_starts': 32,
            'batch_size': 32, 'gamma': .99, 'tau': .005, 'policy_delay': 2,
            'exploration_noise': .15, 'target_policy_noise': .2, 'target_noise_clip': .5,
            'hidden_units': [64, 64], 'gradient_steps': 1}


@contextmanager
def isolated_random_state(device='cpu'):
    numpy_state, python_state = np.random.get_state(), random.getstate()
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    mps_state = torch.mps.get_rng_state() if torch.device(device).type == 'mps' else None
    with torch.random.fork_rng(devices=cuda_devices):
        try:
            yield
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            if mps_state is not None:
                torch.mps.set_rng_state(mps_state)


class TD3Agent:
    def __init__(self, width, parameters, seed, environment_device='cpu', *, action_dimensions=3):
        import gymnasium as gym
        from stable_baselines3 import TD3
        from stable_baselines3.common.logger import configure

        class Interface(gym.Env):
            observation_space = gym.spaces.Box(-np.inf, np.inf, (width,), dtype=np.float32)
            action_space = gym.spaces.Box(-1., 1., (action_dimensions,), dtype=np.float32)

        self.width = width
        self.action_dimensions = action_dimensions
        self.config = {**DEFAULTS, **parameters}
        options = {key: value for key, value in self.config.items()
                   if key not in {'exploration_noise', 'hidden_units', 'gradient_steps'}}
        with isolated_random_state(environment_device):
            self.model = TD3('MlpPolicy', Interface(), **options, seed=seed, device='cpu', verbose=0,
                             policy_kwargs={'net_arch': self.config['hidden_units']})
            self.numpy_state = np.random.get_state()
            self.torch_state = torch.get_rng_state().clone()
        self.model.set_logger(configure(folder=None, format_strings=[]))
        self.rng = np.random.default_rng(seed)
        self.samples, self.actions = 0, 0

    def action(self, observation):
        if self.actions < self.config['learning_starts']:
            action = self.rng.uniform(-1, 1, self.action_dimensions)
        else:
            action = self.model.predict(observation, deterministic=True)[0]
            action = action + self.rng.normal(0, self.config['exploration_noise'], self.action_dimensions)
        self.actions += 1
        return np.clip(action, -1, 1).astype(np.float32)

    def observe(self, observation, action, reward, following, terminal):
        self.add_transition(observation, action, reward, following, terminal)
        if self.samples < max(self.config['learning_starts'], self.config['batch_size']):
            return
        self.update(self.config['gradient_steps'])

    def add_transition(self, observation, action, reward, following, terminal):
        self.model.replay_buffer.add(observation[None], following[None], action[None],
                                     np.array([reward]), np.array([terminal]), [{}])
        self.samples += 1

    def update(self, steps, *, critic_only=False):
        previous, delay = self.model._n_updates, self.model.policy_delay
        try:
            if critic_only:
                self.model.policy_delay = previous + steps + 1
            with isolated_random_state():
                np.random.set_state(self.numpy_state)
                torch.set_rng_state(self.torch_state)
                self.model.train(gradient_steps=steps, batch_size=self.config['batch_size'])
                self.numpy_state = np.random.get_state()
                self.torch_state = torch.get_rng_state().clone()
        finally:
            self.model.policy_delay = delay
        values = self.model.logger.name_to_value
        actor_updates = 0 if critic_only else self.model._n_updates // delay - previous // delay
        return {'critic_updates': steps, 'actor_updates': actor_updates,
                'critic_loss': float(values['train/critic_loss']),
                'actor_loss': float(values['train/actor_loss']) if actor_updates else None, 'critic_only': critic_only}

    def snapshot(self):
        return deepcopy(self.model.policy.state_dict())

    def actor_loss_gradient(self):
        with isolated_random_state():
            np.random.set_state(self.numpy_state)
            torch.set_rng_state(self.torch_state)
            data = self.model.replay_buffer.sample(self.config['batch_size'])
            loss = -self.model.critic.q1_forward(data.observations, self.model.actor(data.observations)).mean()
            gradients = torch.autograd.grad(loss, tuple(self.model.actor.parameters()))
            self.numpy_state = np.random.get_state()
            self.torch_state = torch.get_rng_state().clone()
        return {name: value.detach().clone() for (name, _), value in zip(self.model.actor.named_parameters(), gradients, strict=True)}

    def state_dict(self):
        replay = self.model.replay_buffer
        count = replay.buffer_size if replay.full else replay.pos
        numpy_state = self.numpy_state
        return {
            'width': self.width, 'action_dimensions': self.action_dimensions,
            'parameters': deepcopy(self.model.get_parameters()),
            'updates': self.model._n_updates, 'samples': self.samples, 'actions': self.actions,
            'replay': {key: torch.from_numpy(getattr(replay, key)[:count].copy()) for key in
                       ('observations', 'next_observations', 'actions', 'rewards', 'dones', 'timeouts')},
            'position': replay.pos, 'full': replay.full, 'rng': deepcopy(self.rng.bit_generator.state),
            'numpy': [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]], 'torch': self.torch_state,
        }

    def load_state_dict(self, state):
        self.model.set_parameters(state['parameters'])
        self.model._n_updates = state['updates']
        self.samples, self.actions = state['samples'], state['actions']
        replay = self.model.replay_buffer
        for key, values in state['replay'].items():
            getattr(replay, key)[:len(values)] = values.numpy()
        replay.pos, replay.full = state['position'], state['full']
        self.rng.bit_generator.state = state['rng']
        self.numpy_state = (state['numpy'][0], np.asarray(state['numpy'][1], dtype=np.uint32), *state['numpy'][2:])
        self.torch_state = state['torch']

    def save_policy(self, path):
        temporary = path.with_name(path.stem + '.tmp.zip')
        self.model.save(temporary)
        temporary.replace(path)
