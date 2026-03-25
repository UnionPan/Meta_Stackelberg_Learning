import torch
import copy
import random
import os
from typing import Dict, List, Optional, Callable
from torch.utils.tensorboard import SummaryWriter

from .trajectory import TrajectoryCollector
from .pg import (
    compute_policy_gradient,
    ReptileScheme,
    DebiasedScheme,
    MetaGradientScheme,
)
from ..agents.meta_agents import PolicyAgent, FixedPolicyAgent


class AttackerDomain:
    """
    Registry of attacker types for meta-training.
    Each entry maps a type name to an attacker policy (PolicyAgent or FixedPolicyAgent).
    """
    def __init__(self):
        self.attackers: Dict[str, object] = {}

    def register(self, type_name: str, attacker):
        self.attackers[type_name] = attacker

    def sample(self, K: int) -> List[str]:
        """Sample K attack types uniformly from the domain."""
        types = list(self.attackers.keys())
        return random.choices(types, k=K) if K >= len(types) else random.sample(types, K)

    def get(self, type_name: str):
        return self.attackers[type_name]

    def __len__(self):
        return len(self.attackers)

    def __iter__(self):
        return iter(self.attackers)


class MetaSGTrainer:
    """
    Meta-Stackelberg Learning trainer (Algorithm 2 from the paper).

    Implements the nested two-timescale loop:
      Outer loop: update defender's meta-policy (Reptile or Debiased)
      Inner loop: update attacker's best response per sampled type

    Supports two modes:
      - meta-RL: attacker is non-adaptive (FixedPolicyAgent), no inner loop
      - meta-SG: attacker is adaptive (PolicyAgent), inner loop updates phi_xi
    """

    DEFAULT_CONFIG = {
        # Meta-learning
        "N_D": 100,            # outer loop iterations
        "K": 3,                # attack types sampled per iteration
        "eta": 0.01,           # adaptation step size
        "kappa_D": 0.001,      # meta-optimization step size (Reptile)

        # Inner loop (attacker best response)
        "N_A": 10,             # inner loop iterations for attacker
        "kappa_A": 0.001,      # attacker learning rate

        # Trajectory collection
        "H": 50,               # trajectory horizon (steps per rollout)
        "gamma": 0.99,         # discount factor
        "gae_lambda": 0.95,    # GAE lambda

        # Policy gradient
        "entropy_coef": 0.01,
        "value_loss_coef": 0.5,

        # Meta-gradient scheme
        "scheme": "reptile",   # "reptile" or "debiased"

        # Logging
        "log_dir": "runs/meta_sg",
        "log_interval": 1,
        "save_interval": 50,   # save checkpoints every N iterations
        "save_dir": "checkpoints",
    }

    def __init__(
        self,
        env,
        defender: PolicyAgent,
        attacker_domain: AttackerDomain,
        defender_id: str = "player_1",
        attacker_id: str = "player_2",
        config: Optional[Dict] = None,
        exploitability_fn: Optional[Callable] = None,
        obs_dim: Optional[int] = None,
    ):
        self.env = env
        self.defender = defender
        self.attacker_domain = attacker_domain
        self.defender_id = defender_id
        self.attacker_id = attacker_id
        self.exploitability_fn = exploitability_fn
        self.obs_dim = obs_dim

        # Merge config with defaults
        self.config = {**self.DEFAULT_CONFIG}
        if config:
            self.config.update(config)

        self.collector = TrajectoryCollector(defender_id, attacker_id)

        # Select meta-gradient scheme
        if self.config["scheme"] == "reptile":
            self.gradient_scheme = ReptileScheme()
        elif self.config["scheme"] == "debiased":
            self.gradient_scheme = DebiasedScheme()
        else:
            raise ValueError(f"Unknown scheme: {self.config['scheme']}")

        self.writer = SummaryWriter(self.config["log_dir"])

        # Policy history for visualization
        self.policy_history = {
            "defender": [],
        }
        for type_name in self.attacker_domain:
            self.policy_history[f"attacker_{type_name}"] = []

    def _get_policy_probs(self, agent, obs_index=-1):
        """Get policy distribution from an agent at a given observation."""
        if not isinstance(agent, PolicyAgent) or self.obs_dim is None:
            return None
        with torch.no_grad():
            obs = torch.zeros(1, self.obs_dim)
            if obs_index >= 0:
                obs[0, obs_index] = 1.0
            else:
                obs[0, self.obs_dim - 1] = 1.0
            logits, _ = agent(obs)
            return torch.softmax(logits, dim=-1).squeeze().numpy()

    def _log_metrics(self, t, iter_rewards_def, iter_rewards_att,
                     type_rewards, meta_grad_norms):
        """Log comprehensive metrics to TensorBoard."""
        # Rewards
        avg_def_reward = sum(iter_rewards_def) / len(iter_rewards_def)
        avg_att_reward = sum(iter_rewards_att) / len(iter_rewards_att)
        self.writer.add_scalar("reward/defender_avg", avg_def_reward, t)
        self.writer.add_scalar("reward/attacker_avg", avg_att_reward, t)

        # Per-type rewards
        for type_name, (def_r, att_r) in type_rewards.items():
            self.writer.add_scalar(f"reward_per_type/{type_name}_defender", def_r, t)
            self.writer.add_scalar(f"reward_per_type/{type_name}_attacker", att_r, t)

        # Meta-gradient norm
        if meta_grad_norms:
            avg_grad_norm = sum(meta_grad_norms) / len(meta_grad_norms)
            self.writer.add_scalar("gradient/meta_grad_norm", avg_grad_norm, t)

        # Defender policy distribution
        def_probs = self._get_policy_probs(self.defender)
        if def_probs is not None:
            self.policy_history["defender"].append(def_probs.copy())
            for i, name in enumerate(["action_0", "action_1", "action_2"]):
                self.writer.add_scalar(f"policy/defender_{name}", def_probs[i], t)

            # Exploitability
            if self.exploitability_fn:
                exploit = self.exploitability_fn(def_probs)
                self.writer.add_scalar("meta/exploitability", exploit, t)

        # Attacker policy distributions
        for type_name in self.attacker_domain:
            attacker = self.attacker_domain.get(type_name)
            att_probs = self._get_policy_probs(attacker)
            if att_probs is not None:
                self.policy_history[f"attacker_{type_name}"].append(att_probs.copy())
                for i, name in enumerate(["action_0", "action_1", "action_2"]):
                    self.writer.add_scalar(
                        f"policy/attacker_{type_name}_{name}", att_probs[i], t
                    )

        # Console output
        exploit_str = ""
        if self.exploitability_fn and def_probs is not None:
            exploit_str = f" exploit={self.exploitability_fn(def_probs):.4f}"
        print(
            f"[Meta-SG iter {t+1}] "
            f"def_r={avg_def_reward:.3f} "
            f"att_r={avg_att_reward:.3f}"
            f"{exploit_str}"
        )

    def _save_checkpoint(self, t):
        """Save defender and attacker checkpoints."""
        save_dir = self.config["save_dir"]
        os.makedirs(save_dir, exist_ok=True)

        # Save defender
        torch.save(
            self.defender.state_dict(),
            os.path.join(save_dir, f"defender_iter{t}.pt")
        )

        # Save all attackers
        for type_name in self.attacker_domain:
            attacker = self.attacker_domain.get(type_name)
            if isinstance(attacker, PolicyAgent):
                torch.save(
                    attacker.state_dict(),
                    os.path.join(save_dir, f"attacker_{type_name}_iter{t}.pt")
                )

    def train(self, num_iterations: Optional[int] = None):
        """Run the meta-Stackelberg learning algorithm."""
        N_D = num_iterations or self.config["N_D"]
        K = self.config["K"]
        eta = self.config["eta"]
        kappa_D = self.config["kappa_D"]
        N_A = self.config["N_A"]
        kappa_A = self.config["kappa_A"]
        H = self.config["H"]
        gamma = self.config["gamma"]
        gae_lambda = self.config["gae_lambda"]
        entropy_coef = self.config["entropy_coef"]
        value_loss_coef = self.config["value_loss_coef"]
        save_interval = self.config["save_interval"]

        pg_kwargs = dict(
            gamma=gamma,
            gae_lambda=gae_lambda,
            entropy_coef=entropy_coef,
            value_loss_coef=value_loss_coef,
        )

        for t in range(N_D):
            # 1. Sample K attack types
            sampled_types = self.attacker_domain.sample(K)

            meta_grads: List[Dict[str, torch.Tensor]] = []
            iter_rewards_def = []
            iter_rewards_att = []
            type_rewards: Dict[str, tuple] = {}
            meta_grad_norms = []

            for xi in sampled_types:
                attacker = self.attacker_domain.get(xi)
                is_adaptive = isinstance(attacker, PolicyAgent)

                # --- PRE-ADAPTATION ---
                pre_traj = self.collector.collect(
                    self.env, self.defender, attacker, H, attack_type=xi
                )
                def_pre_traj = pre_traj[self.defender_id]

                pre_grad = compute_policy_gradient(
                    self.defender, def_pre_traj, **pg_kwargs
                )

                adapted_defender = self.defender.get_adapted(pre_grad, eta)

                # --- INNER LOOP (attacker best response) ---
                if is_adaptive:
                    working_attacker = copy.deepcopy(attacker)

                    for k in range(N_A):
                        inner_traj = self.collector.collect(
                            self.env, adapted_defender, working_attacker, H,
                            attack_type=xi
                        )
                        att_inner_traj = inner_traj[self.attacker_id]

                        att_grad = compute_policy_gradient(
                            working_attacker, att_inner_traj, **pg_kwargs
                        )

                        with torch.no_grad():
                            for name, param in working_attacker.named_parameters():
                                if name in att_grad and att_grad[name] is not None:
                                    param.add_(kappa_A * att_grad[name])

                    # Persist updated attacker back to domain
                    attacker.load_state_dict(working_attacker.state_dict())
                    post_attacker = attacker
                else:
                    post_attacker = attacker

                # --- POST-ADAPTATION ---
                post_traj = self.collector.collect(
                    self.env, adapted_defender, post_attacker, H,
                    attack_type=xi
                )
                def_post_traj = post_traj[self.defender_id]

                meta_grad = self.gradient_scheme.compute_meta_gradient(
                    adapted_defender=adapted_defender,
                    post_traj=def_post_traj,
                    pre_traj=def_pre_traj,
                    original_defender=self.defender,
                    eta=eta,
                    **pg_kwargs,
                )
                meta_grads.append(meta_grad)

                # Track metrics
                def_reward = sum(def_post_traj.rewards)
                att_reward = sum(post_traj[self.attacker_id].rewards)
                iter_rewards_def.append(def_reward)
                iter_rewards_att.append(att_reward)
                type_rewards[xi] = (def_reward, att_reward)

                # Gradient norm
                grad_norm = sum(
                    g.norm().item() ** 2 for g in meta_grad.values()
                ) ** 0.5
                meta_grad_norms.append(grad_norm)

            # --- META-UPDATE ---
            with torch.no_grad():
                for name, param in self.defender.named_parameters():
                    grad_sum = sum(
                        mg[name] for mg in meta_grads if name in mg
                    )
                    param.add_((kappa_D / K) * grad_sum)

            # --- LOGGING ---
            if t % self.config["log_interval"] == 0:
                self._log_metrics(
                    t, iter_rewards_def, iter_rewards_att,
                    type_rewards, meta_grad_norms
                )

            # --- CHECKPOINTING ---
            if save_interval > 0 and (t + 1) % save_interval == 0:
                self._save_checkpoint(t + 1)

        # Final save
        self._save_checkpoint(N_D)
        self.writer.close()
        return self.defender

    def get_policy_histories(self):
        """Return collected policy histories for visualization."""
        return self.policy_history
