"""
Meta-Stackelberg Learning (Algorithm 2, Reptile variant).

Outer loop: meta-update defender θ via Reptile over K attack types.
Inner loop: per-ξ TD3 adaptation of defender (l steps) +
            attacker best-response update (N_A steps, meta-SG mode only).

Paper §III-C and Appendix C-A:
  T=100, K=10, H=200(MNIST)/500(CIFAR), l=N_A=N_D=10
  κ_D = κ_A = 0.001, meta-step = 1.0
"""
from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from meta_sg.learning.best_response import AttackerBestResponse
from meta_sg.learning.config import MetaSGConfig, TD3Config
from meta_sg.learning.replay_buffer import ReplayBuffer
from meta_sg.learning.task_runner import AttackTaskRunner, TaskResult
from meta_sg.learning.td3 import TD3Agent
from meta_sg.simulation.interface import FLCoordinator
from meta_sg.strategies.types import AttackType


@dataclass
class MetaSGResult:
    defender_rewards: List[float] = field(default_factory=list)
    attacker_rewards: Dict[str, List[float]] = field(default_factory=dict)
    per_attack_rewards: Dict[str, List[float]] = field(default_factory=dict)
    defender_critic_losses: List[float] = field(default_factory=list)
    defender_actor_losses: List[float] = field(default_factory=list)
    reptile_delta_norms: List[float] = field(default_factory=list)
    meta_iterations: int = 0


class MetaSGTrainer:
    """
    Meta-Stackelberg Learning trainer (Algorithm 2, Reptile).

    TensorBoard metric groups (x-axis = outer iteration t):

      train/           — aggregate training health
      reward_per_attack/ — per-attack-type mean r_D
      td3/defender/    — defender TD3 loss + Q statistics
      td3/attacker/    — per-adaptive-attacker BR losses
      reptile/         — meta-update magnitude breakdown
      policy/defender/ — decoded defense parameter means + action diversity
      buffers/         — replay buffer occupancy
    """

    def __init__(
        self,
        coordinator_factory: Callable[[], FLCoordinator],
        attack_domain: Sequence[AttackType],
        meta_config: MetaSGConfig,
        td3_config: TD3Config,
        obs_dim: int,
        act_dim: int = 3,
        device: Optional[torch.device] = None,
        log_interval: int = 10,
        writer=None,
    ) -> None:
        self.coordinator_factory = coordinator_factory
        self.attack_domain = list(attack_domain)
        self.meta_cfg = meta_config
        self.td3_cfg = td3_config
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device or torch.device("cpu")
        self.log_interval = log_interval
        self.writer = writer

        self.defender = TD3Agent(obs_dim, act_dim, td3_config, device)

        self.attacker_agents: Dict[str, TD3Agent] = {}
        self.attacker_buffers: Dict[str, ReplayBuffer] = {}
        for at in self.attack_domain:
            self.attacker_agents[at.name] = TD3Agent(obs_dim, act_dim, td3_config, device)
            self.attacker_buffers[at.name] = ReplayBuffer(
                td3_config.buffer_capacity, obs_dim, act_dim
            )

        self.best_response = AttackerBestResponse(
            self.attacker_agents, self.attacker_buffers, n_a=meta_config.N_A
        )
        self.task_runner = AttackTaskRunner(
            coordinator_factory=coordinator_factory,
            td3_config=td3_config,
            meta_config=meta_config,
            obs_dim=obs_dim,
            act_dim=act_dim,
            attacker_agents=self.attacker_agents,
            attacker_buffers=self.attacker_buffers,
            best_response=self.best_response,
        )
        self._result = MetaSGResult()

    # ------------------------------------------------------------------
    # Pre-training (Algorithm 2)
    # ------------------------------------------------------------------

    def train(self) -> MetaSGResult:
        cfg = self.meta_cfg
        print(f"[MetaSG] Pre-training: T={cfg.T}, K={cfg.K}, H={cfg.H}, l={cfg.l}")

        for t in range(cfg.T):
            batch_types = self._sample_attack_types(cfg.K)
            adapted_params: List[Dict] = []
            task_results: List[TaskResult] = []

            for xi in batch_types:
                task_result = self.task_runner.run(
                    attack_type=xi,
                    meta_defender=self.defender,
                    seed_base=t * 100_000 + len(adapted_params) * 10_000,
                )
                adapted_params.append(task_result.adapted_params)
                task_results.append(task_result)

            # Reptile meta-update θ ← θ + (1/K) Σ (θ_ξ - θ)
            reptile_norms = self._reptile_update(adapted_params, step_size=cfg.meta_update_step)

            # --- Aggregate metrics ---
            d_rewards  = [r.mean_defender_reward for r in task_results]
            mean_d_rew = float(np.mean(d_rewards))

            self._result.defender_rewards.append(mean_d_rew)
            self._result.reptile_delta_norms.append(reptile_norms["full"])
            self._result.meta_iterations = t + 1

            if (t + 1) % self.log_interval == 0:
                print(
                    f"[MetaSG] iter {t + 1}/{cfg.T}  "
                    f"r_D={mean_d_rew:.4f}  "
                    f"reptile_δ={reptile_norms['actor']:.5f}  "
                    f"batch={[xi.name for xi in batch_types]}"
                )

            if self.writer is not None:
                self._log_all(t, batch_types, task_results, d_rewards, reptile_norms)

        return self._result

    # ------------------------------------------------------------------
    # TensorBoard logging
    # ------------------------------------------------------------------

    def _log_all(
        self,
        t: int,
        batch_types: List[AttackType],
        task_results: List[TaskResult],
        d_rewards: List[float],
        reptile_norms: Dict[str, float],
    ) -> None:
        w = self.writer

        # ── train/ ────────────────────────────────────────────────────
        w.add_scalar("train/reward_mean",      float(np.mean(d_rewards)),  t)
        w.add_scalar("train/reward_std",       float(np.std(d_rewards)),   t)
        w.add_scalar("train/reward_min",       float(np.min(d_rewards)),   t)
        w.add_scalar("train/reward_max",       float(np.max(d_rewards)),   t)
        w.add_scalar("train/adaptive_fraction",
                     float(np.mean([xi.adaptive for xi in batch_types])), t)

        env_diag = _mean_diag_values(task_results, "clean_acc", "backdoor_acc")
        if "clean_acc" in env_diag:
            w.add_scalar("train/clean_acc",    env_diag["clean_acc"],    t)
        if "backdoor_acc" in env_diag:
            w.add_scalar("train/backdoor_acc", env_diag["backdoor_acc"], t)

        # ── reward_per_attack/ ────────────────────────────────────────
        # Bucket rewards by attack type; multiple tasks of the same type are averaged.
        per_attack: Dict[str, List[float]] = defaultdict(list)
        for result in task_results:
            per_attack[result.attack_type.name].append(result.mean_defender_reward)
        for name, vals in per_attack.items():
            w.add_scalar(f"reward_per_attack/{name}", float(np.mean(vals)), t)
            self._result.per_attack_rewards.setdefault(name, []).append(float(np.mean(vals)))

        # ── td3/defender/ ─────────────────────────────────────────────
        def_losses = _aggregate_task_losses([r.defender_losses for r in task_results])
        if "critic_loss" in def_losses:
            w.add_scalar("td3/defender/critic_loss", def_losses["critic_loss"], t)
        if "actor_loss" in def_losses:
            w.add_scalar("td3/defender/actor_loss",  def_losses["actor_loss"],  t)
        if "q_mean" in def_losses:
            w.add_scalar("td3/defender/q_mean",      def_losses["q_mean"],      t)

        inner_deltas = [r.inner_delta_norm for r in task_results]
        w.add_scalar("td3/defender/inner_adapt_norm", float(np.mean(inner_deltas)), t)

        # ── td3/attacker/ ─────────────────────────────────────────────
        # Only log when the attacker was actually updated this iteration.
        for result in task_results:
            if result.attacker_br_losses:
                name = result.attack_type.name
                bl = result.attacker_br_losses
                if "critic_loss" in bl:
                    w.add_scalar(f"td3/attacker/{name}_critic_loss", bl["critic_loss"], t)
                al = bl.get("actor_loss", float("nan"))
                if not math.isnan(al):
                    w.add_scalar(f"td3/attacker/{name}_actor_loss", al, t)
                if "q_mean" in bl:
                    w.add_scalar(f"td3/attacker/{name}_q_mean", bl["q_mean"], t)

        # ── reptile/ ──────────────────────────────────────────────────
        w.add_scalar("reptile/delta_norm",        reptile_norms["full"],   t)
        w.add_scalar("reptile/actor_delta_norm",  reptile_norms["actor"],  t)
        w.add_scalar("reptile/critic_delta_norm", reptile_norms["critic"], t)

        # ── policy/defender/ ──────────────────────────────────────────
        action_diag = _mean_diag_values(
            task_results,
            "defender_alpha", "defender_beta", "defender_post_param",
            "defender_action_std_0", "defender_action_std_1", "defender_action_std_2",
        )
        for diag_key, tb_key in [
            ("defender_alpha",        "policy/defender/alpha_mean"),
            ("defender_beta",         "policy/defender/beta_mean"),
            ("defender_post_param",   "policy/defender/post_param_mean"),
            ("defender_action_std_0", "policy/defender/action_std_0"),
            ("defender_action_std_1", "policy/defender/action_std_1"),
            ("defender_action_std_2", "policy/defender/action_std_2"),
        ]:
            if diag_key in action_diag:
                w.add_scalar(tb_key, action_diag[diag_key], t)

        # ── buffers/ ──────────────────────────────────────────────────
        for name, buf in self.attacker_buffers.items():
            w.add_scalar(f"buffers/attacker_{name}", len(buf), t)
        local_buf_sizes = [r.diagnostics.get("buffer_size", 0) for r in task_results]
        w.add_scalar("buffers/local_defender_mean", float(np.mean(local_buf_sizes)), t)

    # ------------------------------------------------------------------
    # Reptile meta-update
    # ------------------------------------------------------------------

    def _reptile_update(
        self, adapted_params: List[Dict], step_size: float = 1.0
    ) -> Dict[str, float]:
        """
        θ_{t+1} = θ_t + step_size * mean_ξ(θ_ξ(l) - θ_t)

        Returns L2 norms of the applied update split by actor / critic / full.
        """
        if not adapted_params:
            return {"full": 0.0, "actor": 0.0, "critic": 0.0}

        meta_params = self.defender.get_params()
        K = len(adapted_params)
        updated: Dict = {}
        actor_sq = critic_sq = full_sq = 0.0

        for key in meta_params:
            delta = sum(
                ap[key] - meta_params[key]
                for ap in adapted_params
                if key in ap
            ) / K
            step = step_size * delta
            updated[key] = meta_params[key] + step

            sq = float(step.norm() ** 2)
            full_sq += sq
            if key.startswith("actor."):
                actor_sq += sq
            elif key.startswith("critic."):
                critic_sq += sq

        self.defender.set_params(updated)
        return {
            "full":   math.sqrt(full_sq),
            "actor":  math.sqrt(actor_sq),
            "critic": math.sqrt(critic_sq),
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self.defender.save(os.path.join(directory, "defender_meta.pt"))
        for name, agent in self.attacker_agents.items():
            agent.save(os.path.join(directory, f"attacker_{name}.pt"))
        print(f"[MetaSG] Saved checkpoint to {directory}")

    def load(self, directory: str) -> None:
        self.defender.load(os.path.join(directory, "defender_meta.pt"))
        for name, agent in self.attacker_agents.items():
            path = os.path.join(directory, f"attacker_{name}.pt")
            if os.path.exists(path):
                agent.load(path)
        print(f"[MetaSG] Loaded checkpoint from {directory}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_attack_types(self, k: int) -> List[AttackType]:
        indices = np.random.choice(len(self.attack_domain), size=k, replace=True)
        return [self.attack_domain[i] for i in indices]


# ── Module-level helpers ──────────────────────────────────────────────────────

def _mean_diag_values(task_results: List[TaskResult], *keys: str) -> Dict[str, float]:
    """Mean of diagnostics[key] over all task results (ignores NaN)."""
    buckets: Dict[str, List[float]] = defaultdict(list)
    for result in task_results:
        for key in keys:
            val = result.diagnostics.get(key)
            if val is not None and not math.isnan(val):
                buckets[key].append(float(val))
    return {key: float(np.mean(vals)) for key, vals in buckets.items() if vals}


def _aggregate_task_losses(loss_dicts: List[Dict]) -> Dict[str, float]:
    """Mean losses over K tasks, excluding NaN actor_loss entries."""
    critic_vals: List[float] = []
    actor_vals:  List[float] = []
    q_vals:      List[float] = []
    for d in loss_dicts:
        if not d:
            continue
        if "critic_loss" in d:
            critic_vals.append(d["critic_loss"])
        al = d.get("actor_loss", float("nan"))
        if not math.isnan(al):
            actor_vals.append(al)
        if "q_mean" in d:
            q_vals.append(d["q_mean"])
    result: Dict[str, float] = {}
    if critic_vals:
        result["critic_loss"] = float(np.mean(critic_vals))
    if actor_vals:
        result["actor_loss"]  = float(np.mean(actor_vals))
    if q_vals:
        result["q_mean"]      = float(np.mean(q_vals))
    return result
