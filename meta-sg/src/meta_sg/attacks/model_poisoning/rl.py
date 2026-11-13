"""Design: run the archived frozen Li2022 SB3 attacker without fitting a policy.
Data flow: normalized classifier parameters and selected attacker count -> actor
action -> gamma/SGD steps -> sign-flipped known-data update -> colluding copies.
The policy requires the MNIST 128-to-10 classifier and externally supplied data.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from meta_sg.attacks._common import active_clients
from meta_sg.attacks.model_poisoning._common import repeat_update
from meta_sg.federated.client import train_clients
from meta_sg.federated.state import load_model_state, state_update


def attacker_observation(state, num_attackers):
    if not 0 <= num_attackers <= 10:
        raise ValueError('the archived actor supports 0 through 10 selected attackers')
    if tuple(state[-2].shape) != (10, 128) or tuple(state[-1].shape) != (10,):
        raise ValueError('the archived actor requires the MNIST 128-to-10 classifier')
    values = np.concatenate([part.detach().cpu().numpy().ravel() for part in state[-2:]])
    span = values.max() - values.min()
    scaled = 2 * (values - values.min()) / span - 1 if span else np.zeros_like(values)
    return {'pram': scaled.astype(np.float32), 'num_attacker': num_attackers}


def decode_action(action):
    values = np.asarray(action, dtype=np.float32)
    return float((values[0] + 1) * 5), int(np.rint((values[1] + 1) * 25))


class FrozenLi2022Actor:
    def __init__(self, checkpoint: str | Path):
        from stable_baselines3 import TD3

        self.policy = TD3.load(checkpoint, device='cpu', custom_objects={
            'lr_schedule': lambda _: 0.0, 'learning_rate': 0.0,
        })
        spaces = self.policy.observation_space.spaces
        if spaces['pram'].shape != (1290,) or spaces['num_attacker'].n != 11 or self.policy.action_space.shape != (2,):
            raise ValueError('checkpoint does not match the archived MNIST attacker interface')
        self.policy.policy.set_training_mode(False)
        self.policy.policy.requires_grad_(False)

    def __call__(self, state, num_attackers):
        observation = attacker_observation(state, num_attackers)
        action, _ = self.policy.predict(observation, deterministic=True)
        return action, observation


def archived_normalization(raw_images):
    """Compute the archive's replacement normalization using training pixels only."""
    count, total, squares = 0, 0.0, 0.0
    for batch in raw_images.split(1000):
        values = batch.float().div(255).sub(0.1307).div(0.3081).double()
        count += values.numel()
        total += float(values.sum())
        squares += float(values.square().sum())
    mean = total / count
    return mean, ((squares - count * mean ** 2) / (count - 1)) ** 0.5


def malicious_update(model, state, images, labels, *, gamma, steps, device, learning_rate=0.01):
    """Use eval-mode SGD with configurable proxy learning rate and negative gamma."""
    model.to(device)
    load_model_state(model, state)
    model.eval()
    loader = DataLoader(TensorDataset(images, labels), batch_size=200, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    total_loss = 0.0
    examples_seen = 0
    for _ in range(steps):
        batch, targets = next(iter(loader))
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(batch.to(device)), targets.to(device))
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * len(targets)
        examples_seen += len(targets)
    update = state_update(state, tuple(parameter.detach() for parameter in model.parameters()))
    return tuple(-gamma * part for part in update), {
        'attacker_loss': total_loss / examples_seen if examples_seen else None,
        'attacker_examples_seen': examples_seen,
    }


def rl_round(
    model, base_state, client_loaders, selected_clients, *,
    round_index, steps, learning_rate, device, seed,
    malicious_clients, actor, known_images, known_labels, attack_start=1,
    attacker_learning_rate=0.01,
    epochs=None, momentum=0.0, weight_decay=0.0,
):
    malicious = active_clients(selected_clients, malicious_clients, round_index, attack_start)
    honest = [client for client in selected_clients if client not in malicious]
    updates, metrics = train_clients(
        model, base_state, client_loaders, honest, steps=steps,
        learning_rate=learning_rate, device=device,
        epochs=epochs, momentum=momentum, weight_decay=weight_decay,
    )
    diagnostics = {'malicious_clients': malicious, 'attacker_learning_rate': attacker_learning_rate}
    if malicious:
        action, observation = actor(base_state, len(malicious))
        gamma, local_steps = decode_action(action)
        target_device = torch.device(device)
        devices = [target_device.index or 0] if target_device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed + round_index)
            update, row = malicious_update(
                model, base_state, known_images, known_labels,
                gamma=gamma, steps=local_steps, device=device,
                learning_rate=attacker_learning_rate,
            )
        updates.update(zip(malicious, repeat_update(update, len(malicious)), strict=True))
        diagnostics.update(
            action=np.asarray(action).tolist(), gamma=gamma, local_steps=local_steps,
            observation_dim=len(observation['pram']), known_samples=len(known_labels), **row,
        )
    return updates, metrics, diagnostics
