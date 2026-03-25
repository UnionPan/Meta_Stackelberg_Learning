import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class PolicyAgent(nn.Module):
    """
    A simple actor-critic agent for policy gradient methods.
    Used as both defender and attacker in meta-Stackelberg learning.
    """
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        logits = self.actor(obs)
        value = self.critic(obs)
        return logits, value

    def get_action(self, obs):
        """Sample an action and return (action, log_prob, value)."""
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1)

    def evaluate_action(self, obs, action):
        """Evaluate a given action: return (log_prob, entropy, value)."""
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy(), value.squeeze(-1)

    def clone_params(self):
        """Return a deep copy of current parameters as a flat dict."""
        return {k: v.clone() for k, v in self.state_dict().items()}

    def load_params(self, params):
        """Load parameters from a dict (e.g., from clone_params)."""
        self.load_state_dict(params)

    def get_adapted(self, grad_dict, step_size):
        """
        Return a new PolicyAgent with parameters adapted by one gradient step.
        θ_new = θ + step_size * grad
        Does NOT modify self.
        """
        adapted = copy.deepcopy(self)
        with torch.no_grad():
            for name, param in adapted.named_parameters():
                if name in grad_dict and grad_dict[name] is not None:
                    param.add_(step_size * grad_dict[name])
        return adapted


class ContinuousPolicyAgent(nn.Module):
    """
    Actor-critic agent for continuous action spaces (e.g., FL defender/attacker).
    Uses diagonal Gaussian policy with learnable log_std.
    Actions are squashed to [-1, 1] via tanh.
    """
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super().__init__()
        self.act_dim = act_dim
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        mean = self.actor_mean(obs)
        value = self.critic(obs)
        return mean, value

    def get_action(self, obs):
        """Sample continuous action, return (action, log_prob, value)."""
        mean, value = self.forward(obs)
        std = self.actor_log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)

        # Log prob with tanh correction
        log_prob = dist.log_prob(raw_action).sum(-1)
        log_prob -= (2 * (torch.log(torch.tensor(2.0)) - raw_action
                          - F.softplus(-2 * raw_action))).sum(-1)

        return action.squeeze(0), log_prob, value.squeeze(-1)

    def evaluate_action(self, obs, action):
        """Evaluate log_prob, entropy, value for given actions in [-1,1]."""
        mean, value = self.forward(obs)
        std = self.actor_log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)

        # Inverse tanh to get raw action
        action_clamped = action.clamp(-0.999, 0.999)
        raw_action = torch.atanh(action_clamped)

        log_prob = dist.log_prob(raw_action).sum(-1)
        log_prob -= (2 * (torch.log(torch.tensor(2.0)) - raw_action
                          - F.softplus(-2 * raw_action))).sum(-1)
        entropy = dist.entropy().sum(-1)

        return log_prob, entropy, value.squeeze(-1)

    def clone_params(self):
        return {k: v.clone() for k, v in self.state_dict().items()}

    def load_params(self, params):
        self.load_state_dict(params)

    def get_adapted(self, grad_dict, step_size):
        adapted = copy.deepcopy(self)
        with torch.no_grad():
            for name, param in adapted.named_parameters():
                if name in grad_dict and grad_dict[name] is not None:
                    param.add_(step_size * grad_dict[name])
        return adapted


class FixedPolicyAgent:
    """
    Wraps a non-adaptive (fixed) attacker strategy.
    Returns a fixed action regardless of observation.
    The actual attack logic is handled by the environment.
    """
    def __init__(self, act_dim, name="fixed"):
        self.act_dim = act_dim
        self.name = name

    def get_action(self, obs):
        """Return zero action, zero log_prob, zero value."""
        action = torch.zeros(self.act_dim)
        log_prob = torch.tensor(0.0)
        value = torch.tensor(0.0)
        return action, log_prob, value

    def parameters(self):
        """No learnable parameters."""
        return iter([])
