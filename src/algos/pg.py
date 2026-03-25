import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict

from .trajectory import Trajectory


def compute_returns_and_advantages(trajectory: Trajectory, gamma: float, gae_lambda: float):
    """
    Compute discounted returns and GAE advantages from a trajectory.

    Args:
        trajectory: collected trajectory data
        gamma: discount factor
        gae_lambda: GAE lambda

    Returns:
        (returns, advantages) as tensors
    """
    data = trajectory.to_tensors()
    rewards = data["rewards"]
    values = data["values"].squeeze()  # ensure 1D
    dones = data["dones"]
    T = len(rewards)

    advantages = torch.zeros(T)
    last_advantage = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0.0  # bootstrap with 0 at end of collection
        else:
            next_value = values[t + 1].item()

        non_terminal = 1.0 - dones[t].item()
        delta = rewards[t] + gamma * next_value * non_terminal - values[t].item()
        last_advantage = delta + gamma * gae_lambda * non_terminal * last_advantage
        advantages[t] = last_advantage

    returns = advantages + values.detach()
    return returns.detach(), advantages.detach()


def compute_policy_gradient(
    policy,
    trajectory: Trajectory,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    entropy_coef: float = 0.01,
    value_loss_coef: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    Compute policy gradient (REINFORCE with GAE baseline) for a given policy
    and trajectory. Returns gradient dict {param_name: gradient_tensor}.

    This performs a forward pass through the policy to get fresh log_probs
    (needed for gradient computation), computes the surrogate loss, and
    backpropagates to get gradients.

    Args:
        policy: PolicyAgent with evaluate_action method
        trajectory: collected trajectory
        gamma: discount factor
        gae_lambda: GAE lambda for advantage estimation
        entropy_coef: entropy bonus coefficient
        value_loss_coef: value loss coefficient

    Returns:
        dict mapping parameter names to their gradients (for ascent on J)
    """
    data = trajectory.to_tensors()
    obs = data["obs"]
    actions = data["actions"]
    returns, advantages = compute_returns_and_advantages(trajectory, gamma, gae_lambda)

    # Normalize advantages
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Forward pass through policy to get differentiable log_probs
    log_probs, entropy, values = policy.evaluate_action(obs, actions)

    # Policy loss: -E[log_prob * advantage] (negative because we want ascent)
    policy_loss = -(log_probs * advantages.detach()).mean()

    # Value loss
    value_loss = F.mse_loss(values, returns.detach())

    # Total loss (for gradient descent, but we negate for ascent later)
    loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy.mean()

    # Compute gradients
    policy.zero_grad()
    loss.backward()

    # Extract gradients (negate because loss = -J, so -grad(loss) = grad(J))
    grad_dict = {}
    for name, param in policy.named_parameters():
        if param.grad is not None:
            grad_dict[name] = -param.grad.clone()  # ascent direction
        else:
            grad_dict[name] = torch.zeros_like(param)

    return grad_dict


class MetaGradientScheme(ABC):
    """Base class for meta-gradient computation strategies."""

    @abstractmethod
    def compute_meta_gradient(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Compute the meta-gradient for the defender."""
        pass


class ReptileScheme(MetaGradientScheme):
    """
    Reptile meta-gradient: simply evaluate the policy gradient at the
    adapted parameters theta_xi. No Hessian needed.

    meta_grad = nabla_theta J_D(theta_xi, phi_xi(N_A), xi) |_{theta=theta_xi}

    This is a first-order approximation that still points in an
    ascent direction (Nichol et al., 2018).
    """
    def compute_meta_gradient(
        self,
        adapted_defender,
        post_traj: Trajectory,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            adapted_defender: PolicyAgent with adapted params theta_xi
            post_traj: trajectory collected under (theta_xi, phi_xi(N_A))
        """
        return compute_policy_gradient(
            adapted_defender,
            post_traj,
            gamma=gamma,
            gae_lambda=gae_lambda,
            entropy_coef=entropy_coef,
            value_loss_coef=value_loss_coef,
        )


class DebiasedScheme(MetaGradientScheme):
    """
    Debiased meta-gradient using the chain rule with Hessian estimation.

    meta_grad = nabla J_D(theta') * (I + eta * H) + score_term

    where theta' = theta + eta * nabla J_D(theta), and H is the
    Hessian estimate from the pre-adaptation trajectory.

    NOTE: This is significantly more expensive and complex.
    For now, we provide the interface; full implementation deferred.
    """
    def compute_meta_gradient(
        self,
        adapted_defender,
        post_traj: Trajectory,
        pre_traj: Trajectory = None,
        original_defender=None,
        eta: float = 0.01,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # For now, fall back to Reptile (first-order approximation)
        # Full Hessian estimation can be added later
        return compute_policy_gradient(
            adapted_defender,
            post_traj,
            gamma=kwargs.get("gamma", 0.99),
            gae_lambda=kwargs.get("gae_lambda", 0.95),
        )
