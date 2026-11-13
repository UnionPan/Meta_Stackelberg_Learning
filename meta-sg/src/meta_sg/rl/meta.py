"""Design: first-order TD3 Meta-SG with persistent, task-specific followers.
Data flow: fresh task-local defender -> inner episodes -> frozen-defender follower
response -> fresh post-response replay/critic fitting -> actor loss gradient ->
shared outer actor descent and averaged critic initialization, with synced targets.
Follower responses are finite optimizer steps, not exact best-response equilibria.
"""

from copy import deepcopy

import numpy as np
import torch

from meta_sg.rl.agent import TD3Agent


def first_order_update(initial, adapted, gradients, rate):
    result = deepcopy(initial)
    for key in initial:
        if key.startswith('critic.'):
            result[key] = torch.stack([state[key] for state in adapted]).mean(0)
        elif key.startswith('actor.'):
            name = key[len('actor.'):]
            result[key] = initial[key] - rate * torch.stack([gradient[name] for gradient in gradients]).mean(0)
    for key in result:
        if key.startswith('actor_target.'):
            result[key] = result['actor.' + key[len('actor_target.'):]].clone()
        elif key.startswith('critic_target.'):
            result[key] = result['critic.' + key[len('critic_target.'):]].clone()
    return result


def add_episode(agent, transitions):
    for observation, action, reward, following, terminal in transitions:
        agent.add_transition(observation, action, reward, following, terminal)


class MetaSGLearner:
    def __init__(self, config, width, action_dimensions, seed, environment_device='cpu'):
        self.settings, self.seed, self.device = deepcopy(config['meta']), seed, environment_device
        self.agent_parameters = {**config['rl']['agent'], 'learning_starts': 0}
        self.follower_parameters = {**self.agent_parameters, **self.settings['follower_agent'], 'learning_starts': 0}
        self.width, self.action_dimensions = width, action_dimensions
        self.defender = self.fresh_defender()
        self.followers = {task['name']: TD3Agent(8, self.follower_parameters, seed, environment_device)
                          for task in self.settings['tasks']}
        self.rng = np.random.default_rng(seed)
        self.iteration, self.episodes, self.fl_rounds = 0, 0, 0

    def fresh_defender(self, state=None):
        agent = TD3Agent(self.width, self.agent_parameters, self.seed, self.device, action_dimensions=self.action_dimensions)
        if state is not None:
            agent.model.policy.load_state_dict(state)
        return agent

    def step(self, rollout):
        settings = self.settings
        tasks = [settings['tasks'][index] for index in self.rng.choice(len(settings['tasks']), settings['tasks_per_iteration'], replace=False)]
        initial, states, gradients, records = self.defender.snapshot(), [], [], []

        def collect(task, defender, follower, phase, defender_explore, follower_explore):
            self.episodes += 1
            seed = int(self.rng.integers(1, 2**31 - 1))
            result = rollout(task, defender, follower, settings['horizon'], seed,
                             phase, self.episodes, defender_explore, follower_explore)
            if len(result['defender']) != settings['horizon'] or len(result['follower']) != settings['horizon']:
                raise ValueError('meta rollout must supply one transition per physical FL round for each policy')
            self.fl_rounds += settings['horizon']
            records.append({'iteration': self.iteration + 1, 'task': task['name'], 'phase': phase,
                            'episode': self.episodes, 'episode_seed': seed, 'horizon': settings['horizon'], **result['metrics']})
            return result

        for task in tasks:
            follower = self.followers[task['name']]
            adapted = self.fresh_defender(initial)
            for index in range(settings['inner_steps']):
                samples = collect(task, adapted, follower, f'inner_{index + 1}', True, False)
                add_episode(adapted, samples['defender'])
                records[-1].update(adapted.update(1))
            for index in range(settings['response_steps']):
                samples = collect(task, adapted, follower, f'response_{index + 1}', False, True)
                add_episode(follower, samples['follower'])
                records[-1].update(follower.update(1))
            post = self.fresh_defender(adapted.snapshot())
            samples = collect(task, post, follower, 'post_response', True, False)
            add_episode(post, samples['defender'])
            records[-1].update(post.update(settings['post_critic_steps'], critic_only=True))
            gradient = post.actor_loss_gradient()
            records[-1]['outer_gradient_norm'] = float(torch.stack([value.square().sum() for value in gradient.values()]).sum().sqrt())
            states.append(post.snapshot())
            gradients.append(gradient)
        updated = first_order_update(initial, states, gradients, settings['outer_learning_rate'])
        self.defender.model.policy.load_state_dict(updated)
        self.iteration += 1
        delta = float(torch.stack([(updated[key] - value).square().sum() for key, value in initial.items()
                                   if key.startswith('actor.')]).sum().sqrt())
        return records, {'iteration': self.iteration, 'episodes': self.episodes, 'fl_rounds': self.fl_rounds,
                         'actor_delta_norm': delta}

    def state_dict(self):
        return {'defender': self.defender.state_dict(), 'followers': {name: agent.state_dict() for name, agent in self.followers.items()},
                'iteration': self.iteration, 'episodes': self.episodes, 'fl_rounds': self.fl_rounds,
                'rng': deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state):
        self.defender.load_state_dict(state['defender'])
        for name, agent in self.followers.items():
            agent.load_state_dict(state['followers'][name])
        self.iteration, self.episodes, self.fl_rounds = state['iteration'], state['episodes'], state['fl_rounds']
        self.rng.bit_generator.state = state['rng']
