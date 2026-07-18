"""Actor and twin critics matching TD3 equations."""

from __future__ import annotations

import torch


def _mlp(input_dim: int, output_dim: int, hidden_sizes: tuple[int, ...]) -> torch.nn.Sequential:
    layers = []
    current = input_dim
    for size in hidden_sizes:
        layers.extend((torch.nn.Linear(current, size), torch.nn.ReLU()))
        current = size
    layers.append(torch.nn.Linear(current, output_dim))
    return torch.nn.Sequential(*layers)


class TD3Actor(torch.nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()
        self.model = _mlp(obs_dim, action_dim, hidden_sizes)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.logits(observation))

    def logits(self, observation: torch.Tensor) -> torch.Tensor:
        """Return pre-tanh actions for saturation-aware optimization."""
        return self.model(observation)


class TD3Critic(torch.nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()
        self.model = _mlp(obs_dim + action_dim, 1, hidden_sizes)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat((observation, action), dim=-1))
