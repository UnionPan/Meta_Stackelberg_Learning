"""Design: persistent task-specific RL/BRL followers in the Meta-SG game.
Data flow: previous compact observation -> bounded SGD controls -> owned-data
poison update -> compromised submissions; owned clean/triggered accuracy -> reward.
The defender never receives these identities, triggers, feedback labels, or rewards.
"""

from copy import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset

from meta_sg.attacks._common import active_clients
from meta_sg.attacks.backdoor.training import train_backdoor_update
from meta_sg.federated.buffers import model_buffers
from meta_sg.federated.client import train_clients, train_local_update


def decode_follower_action(action, *, maximum_scale=10., maximum_learning_rate=.1, maximum_steps=11):
    unit = (np.clip(action, -1, 1) + 1) / 2
    return float(unit[0] * maximum_scale), float(unit[1] * maximum_learning_rate), min(maximum_steps, max(1, int(1 + unit[2] * (maximum_steps - 1))))


class FollowerRound:
    def __init__(self, task, policy, train, client_loaders, evaluation_transform, trigger,
                 batch_size, feedback_samples, device, seed, *, explore):
        self.task, self.policy, self.trigger = task, policy, trigger
        self.batch_size, self.device, self.seed, self.explore = batch_size, device, seed, explore
        self.parameters = task.get('parameters', {})
        owned = sorted({index for client in task['malicious_clients'] for index in client_loaders[client].dataset.indices})
        self.owned_indices = torch.tensor(owned, dtype=torch.long)
        self.owned = Subset(train, owned)
        feedback = copy(train)
        feedback.transform = evaluation_transform
        generator = torch.Generator().manual_seed(seed)
        clean = self.owned_indices[torch.randperm(len(owned), generator=generator)[:feedback_samples]]
        self.clean_loader = DataLoader(Subset(feedback, clean.tolist()), batch_size=batch_size)
        self.target_label = self.parameters.get('target_label', 9)
        self.source_label = self.parameters.get('source_label', 0)
        self.target_loader = None
        if task['kind'] == 'backdoor':
            labels = torch.as_tensor(train.targets)[self.owned_indices]
            eligible = labels.ne(self.target_label)
            if self.source_label is not None:
                eligible &= labels.eq(self.source_label)
            targeted = self.owned_indices[eligible]
            if not len(targeted):
                raise ValueError('backdoor follower needs eligible attacker-owned source samples')
            targeted = targeted[torch.randperm(len(targeted), generator=generator)[:feedback_samples]]
            self.target_loader = DataLoader(Subset(feedback, targeted.tolist()), batch_size=batch_size)
        self.previous = np.zeros(8, dtype=np.float32)
        self.following = None
        self.transitions, self.current, self.diagnostics = [], None, None

    def __call__(self, model, base_state, client_loaders, selected_clients, *,
                 round_index, steps, learning_rate, device, seed, base_buffers=None,
                 client_buffers=None, epochs=None, momentum=0., weight_decay=0.):
        action = self.policy.action(self.previous) if self.explore else self.policy.model.predict(self.previous, deterministic=True)[0]
        self.current = (self.previous.copy(), np.asarray(action, dtype=np.float32).copy())
        malicious = active_clients(selected_clients, self.task['malicious_clients'], round_index, self.task.get('start_round', 3))
        honest = [client for client in selected_clients if client not in malicious]
        updates, metrics = train_clients(model, base_state, client_loaders, honest, steps=steps,
                                         learning_rate=learning_rate, device=device, base_buffers=base_buffers,
                                         client_buffers=client_buffers, epochs=epochs, momentum=momentum, weight_decay=weight_decay)
        scale, rate, attack_steps = decode_follower_action(action, **self.parameters.get('decoder', {}))
        if malicious:
            sampler = RandomSampler(self.owned, replacement=True, num_samples=attack_steps * self.batch_size,
                                    generator=torch.Generator().manual_seed(seed + round_index))
            loader = DataLoader(self.owned, batch_size=self.batch_size, sampler=sampler,
                                generator=torch.Generator().manual_seed(seed + round_index + 1))
            if self.task['kind'] == 'backdoor':
                update, row = train_backdoor_update(model, base_state, loader, trigger=self.trigger,
                    target_label=self.target_label, source_label=self.source_label,
                    poison_fraction=self.parameters.get('poison_fraction', 1.), model_replacement_scale=1.,
                    steps=attack_steps, learning_rate=rate, device=device, seed=seed + round_index,
                    base_buffers=base_buffers)
            else:
                update, row = train_local_update(model, base_state, loader, steps=attack_steps,
                                                 learning_rate=rate, device=device, base_buffers=base_buffers)
                update = tuple(-scale * part for part in update)
            buffers = model_buffers(model) if client_buffers is not None else None
            for client in malicious:
                updates[client] = tuple(part.clone() for part in update)
                metrics.append({'client_id': client, 'shared_attacker_update': True, **row})
                if client_buffers is not None:
                    client_buffers[client] = buffers
        self.diagnostics = {'malicious_clients': malicious, 'follower_active': bool(malicious),
                            'follower_action': action.tolist(), 'attacker_steps': attack_steps,
                            'attacker_learning_rate': rate, 'attacker_scale': scale if self.task['kind'] == 'model_poisoning' else 1.}
        return updates, metrics, self.diagnostics

    @torch.no_grad()
    def feedback(self, model, *, terminal):
        model.eval()
        correct, count = 0, 0
        for images, labels in self.clean_loader:
            correct += int(model(images.to(self.device)).argmax(1).cpu().eq(labels).sum())
            count += len(labels)
        clean_accuracy = correct / count
        targeted_accuracy = 0.
        if self.target_loader is not None:
            hits, targeted_count = 0, 0
            for images, labels in self.target_loader:
                hits += int(model(self.trigger(images).to(self.device)).argmax(1).eq(self.target_label).sum())
                targeted_count += len(labels)
            targeted_accuracy = hits / targeted_count
        reward = targeted_accuracy - clean_accuracy
        observation, action = self.current
        self.transitions.append((observation, action, reward, np.zeros_like(self.following) if terminal else self.following.copy(), terminal))
        self.previous = self.following.copy()
        self.diagnostics.update(follower_reward=reward, owned_clean_accuracy=clean_accuracy,
                                owned_target_accuracy=targeted_accuracy if self.target_loader is not None else None)
