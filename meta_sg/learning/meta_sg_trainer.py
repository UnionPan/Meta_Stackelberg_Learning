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
import json
import shutil
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from meta_sg.learning.best_response import AttackerBestResponse
from meta_sg.learning.config import MetaSGConfig, TD3Config
from meta_sg.learning.replay_buffer import ReplayBuffer
from meta_sg.learning.task_runner import (
    AttackTaskRunner,
    TaskResult,
    _query_backdoor_constraints_accept,
    _query_backdoor_reduction_accept,
)
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
        coordinator_factory: Callable[..., FLCoordinator],
        attack_domain: Sequence[AttackType],
        meta_config: MetaSGConfig,
        td3_config: TD3Config,
        obs_dim: int,
        act_dim: int = 3,
        device: Optional[torch.device] = None,
        log_interval: int = 10,
        writer=None,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 0,
        checkpoint_latest_only: bool = False,
        checkpoint_master_seed: Optional[int] = None,
        metrics_jsonl_path: Optional[str] = None,
        start_iteration: int = 0,
        total_iterations: Optional[int] = None,
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
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = max(0, int(checkpoint_interval))
        self.checkpoint_latest_only = bool(checkpoint_latest_only)
        self.checkpoint_master_seed = (
            None if checkpoint_master_seed is None else int(checkpoint_master_seed)
        )
        self.metrics_jsonl_path = metrics_jsonl_path
        self.start_iteration = max(0, int(start_iteration))
        self.total_iterations = int(total_iterations or (self.start_iteration + meta_config.T))
        if self.metrics_jsonl_path:
            Path(self.metrics_jsonl_path).parent.mkdir(parents=True, exist_ok=True)

        self.defender = TD3Agent(obs_dim, act_dim, td3_config, device)
        self.attacker_act_dim = 3

        self.attacker_agents: Dict[str, TD3Agent] = {}
        self.attacker_buffers: Dict[str, ReplayBuffer] = {}
        for at in self.attack_domain:
            self.attacker_agents[at.name] = TD3Agent(obs_dim, self.attacker_act_dim, td3_config, device)
            self.attacker_buffers[at.name] = ReplayBuffer(
                td3_config.buffer_capacity, obs_dim, self.attacker_act_dim
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
            attacker_act_dim=self.attacker_act_dim,
        )
        self._result = MetaSGResult()

    # ------------------------------------------------------------------
    # Pre-training (Algorithm 2)
    # ------------------------------------------------------------------

    def train(self) -> MetaSGResult:
        cfg = self.meta_cfg
        if self.start_iteration:
            print(
                "[MetaSG] Pre-training: "
                f"start={self.start_iteration + 1}, "
                f"end={self.start_iteration + cfg.T}, "
                f"total={self.total_iterations}, K={cfg.K}, H={cfg.H}, l={cfg.l}"
            )
        else:
            print(f"[MetaSG] Pre-training: T={cfg.T}, K={cfg.K}, H={cfg.H}, l={cfg.l}")

        for local_t in range(cfg.T):
            t = self.start_iteration + local_t
            batch_types = self._sample_attack_types(cfg.K)
            adapted_params: List[Dict] = []
            task_results: List[TaskResult] = []
            task_elapsed_seconds: List[float] = []
            iter_started_at = time.perf_counter()

            for task_index, xi in enumerate(batch_types, start=1):
                task_started_at = time.perf_counter()
                if self.log_interval <= 1:
                    print(
                        f"[MetaSG] iter {t + 1}/{self.total_iterations} "
                        f"task {task_index}/{len(batch_types)} start attack={xi.name}",
                        flush=True,
                    )
                task_result = self.task_runner.run(
                    attack_type=xi,
                    meta_defender=self.defender,
                    seed_base=t * 100_000 + len(adapted_params) * 10_000,
                )
                adapted_params.append(task_result.adapted_params)
                task_results.append(task_result)
                task_elapsed = time.perf_counter() - task_started_at
                task_elapsed_seconds.append(task_elapsed)
                if self.log_interval <= 1:
                    print(
                        f"[MetaSG] iter {t + 1}/{self.total_iterations} "
                        f"task {task_index}/{len(batch_types)} done attack={xi.name} "
                        f"elapsed={task_elapsed:.1f}s "
                        f"r_D={task_result.mean_defender_reward:.4f}",
                        flush=True,
                    )

            # Reptile meta-update θ ← θ + (1/K) Σ (θ_ξ - θ)
            meta_update_params = _meta_update_adapted_params(
                task_results,
                meta_objective=cfg.meta_objective,
                query_accept_margin=cfg.query_accept_margin,
                query_clean_floor=cfg.query_clean_floor,
                query_clean_drop_tolerance=cfg.query_clean_drop_tolerance,
                query_backdoor_ceiling=cfg.query_backdoor_ceiling,
                query_backdoor_increase_tolerance=cfg.query_backdoor_increase_tolerance,
                query_backdoor_improvement_margin=cfg.query_backdoor_improvement_margin,
                query_targeted_asr_reduction_margin=cfg.query_targeted_asr_reduction_margin,
                query_targeted_min_base_backdoor=cfg.query_targeted_min_base_backdoor,
            )
            reptile_norms = self._reptile_update(meta_update_params, step_size=cfg.meta_update_step)

            # --- Aggregate metrics ---
            d_rewards  = [r.mean_defender_reward for r in task_results]
            mean_d_rew = float(np.mean(d_rewards))

            self._result.defender_rewards.append(mean_d_rew)
            self._result.reptile_delta_norms.append(reptile_norms["full"])
            self._result.meta_iterations = t + 1
            iteration_elapsed_seconds = time.perf_counter() - iter_started_at

            if (t + 1) % self.log_interval == 0:
                query_metrics = _query_metric_values(task_results)
                query_suffix = ""
                if query_metrics:
                    query_suffix = (
                        f"  q_gain={query_metrics['query_gain_mean']:.5f}  "
                        f"q_accept={query_metrics['query_accept_rate']:.2f}"
                    )
                print(
                    f"[MetaSG] iter {t + 1}/{self.total_iterations}  "
                    f"r_D={mean_d_rew:.4f}  "
                    f"reptile_δ={reptile_norms['actor']:.5f}  "
                    f"{query_suffix}  "
                    f"elapsed={iteration_elapsed_seconds:.1f}s  "
                    f"batch={[xi.name for xi in batch_types]}",
                    flush=True,
                )

            if self.writer is not None:
                self._log_all(
                    t,
                    batch_types,
                    task_results,
                    d_rewards,
                    reptile_norms,
                    task_elapsed_seconds,
                    iteration_elapsed_seconds,
                )

            self._write_metrics_record(
                t,
                batch_types,
                task_results,
                d_rewards,
                reptile_norms,
                task_elapsed_seconds,
                iteration_elapsed_seconds,
            )

            if (
                self.checkpoint_dir
                and self.checkpoint_interval > 0
                and (t + 1) % self.checkpoint_interval == 0
            ):
                latest_path = os.path.join(self.checkpoint_dir, "latest")
                if self.checkpoint_latest_only:
                    self.save(latest_path, completed_iteration=t + 1, replace=True)
                else:
                    self.save(
                        os.path.join(self.checkpoint_dir, f"iter_{t + 1:04d}"),
                        completed_iteration=t + 1,
                    )
                    self.save(latest_path, completed_iteration=t + 1, replace=True)

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
        task_elapsed_seconds: Sequence[float],
        iteration_elapsed_seconds: float,
    ) -> None:
        w = self.writer

        # ── train/ ────────────────────────────────────────────────────
        w.add_scalar("train/reward_mean",      float(np.mean(d_rewards)),  t)
        w.add_scalar("train/reward_std",       float(np.std(d_rewards)),   t)
        w.add_scalar("train/reward_min",       float(np.min(d_rewards)),   t)
        w.add_scalar("train/reward_max",       float(np.max(d_rewards)),   t)
        a_rewards = [r.mean_attacker_reward for r in task_results]
        w.add_scalar("train/attacker_reward_mean", float(np.mean(a_rewards)), t)
        w.add_scalar("train/adaptive_fraction",
                     float(np.mean([xi.adaptive for xi in batch_types])), t)
        query_metrics = _query_metric_values(task_results)
        for key, value in query_metrics.items():
            w.add_scalar(f"query/{key}", value, t)

        env_diag = _mean_diag_values(task_results, "clean_acc", "backdoor_acc")
        if "clean_acc" in env_diag:
            w.add_scalar("train/clean_acc",    env_diag["clean_acc"],    t)
            w.add_scalar("train/attack_success", 1.0 - env_diag["clean_acc"], t)
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

        per_attack_diag: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for result in task_results:
            for key in ("clean_acc", "backdoor_acc"):
                value = result.diagnostics.get(key)
                if value is not None and math.isfinite(float(value)):
                    per_attack_diag[result.attack_type.name][key].append(float(value))
        for attack_name, diagnostic_buckets in per_attack_diag.items():
            for key, values in diagnostic_buckets.items():
                w.add_scalar(
                    f"metrics_per_attack/{attack_name}_{key}",
                    float(np.mean(values)),
                    t,
                )

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
            "defender_alpha", "defender_beta", "defender_post_param", "defender_server_lr",
            "server_lr_penalty",
            "defender_action_std_0", "defender_action_std_1", "defender_action_std_2",
            "defender_action_std_3",
        )
        for diag_key, tb_key in [
            ("defender_alpha",        "policy/defender/alpha_mean"),
            ("defender_beta",         "policy/defender/beta_mean"),
            ("defender_post_param",   "policy/defender/post_param_mean"),
            ("defender_server_lr",    "policy/defender/server_lr_mean"),
            ("server_lr_penalty",     "policy/defender/server_lr_penalty_mean"),
            ("defender_action_std_0", "policy/defender/action_std_0"),
            ("defender_action_std_1", "policy/defender/action_std_1"),
            ("defender_action_std_2", "policy/defender/action_std_2"),
            ("defender_action_std_3", "policy/defender/action_std_3"),
        ]:
            if diag_key in action_diag:
                w.add_scalar(tb_key, action_diag[diag_key], t)

        attacker_action_diag = _mean_diag_values(
            task_results,
            "attacker_gamma", "attacker_local_steps", "attacker_lambda_stealth",
            "attacker_action_std_0", "attacker_action_std_1", "attacker_action_std_2",
        )
        for diag_key, tb_key in [
            ("attacker_gamma",          "policy/attacker/gamma_mean"),
            ("attacker_local_steps",    "policy/attacker/local_steps_mean"),
            ("attacker_lambda_stealth", "policy/attacker/lambda_stealth_mean"),
            ("attacker_action_std_0",   "policy/attacker/action_std_0"),
            ("attacker_action_std_1",   "policy/attacker/action_std_1"),
            ("attacker_action_std_2",   "policy/attacker/action_std_2"),
        ]:
            if diag_key in attacker_action_diag:
                w.add_scalar(tb_key, attacker_action_diag[diag_key], t)

        # ── buffers/ ──────────────────────────────────────────────────
        for name, buf in self.attacker_buffers.items():
            w.add_scalar(f"buffers/attacker_{name}", len(buf), t)
        local_buf_sizes = [r.diagnostics.get("buffer_size", 0) for r in task_results]
        w.add_scalar("buffers/local_defender_mean", float(np.mean(local_buf_sizes)), t)

        # ── timing/ ───────────────────────────────────────────────────
        w.add_scalar("timing/iteration_elapsed_seconds", iteration_elapsed_seconds, t)
        per_attack_elapsed: Dict[str, List[float]] = defaultdict(list)
        for result, elapsed in zip(task_results, task_elapsed_seconds):
            per_attack_elapsed[result.attack_type.name].append(float(elapsed))
        for name, values in per_attack_elapsed.items():
            w.add_scalar(
                f"timing/task_{name}_elapsed_seconds",
                float(np.mean(values)),
                t,
            )

    def _write_metrics_record(
        self,
        t: int,
        batch_types: List[AttackType],
        task_results: List[TaskResult],
        d_rewards: List[float],
        reptile_norms: Dict[str, float],
        task_elapsed_seconds: Sequence[float],
        iteration_elapsed_seconds: float,
    ) -> None:
        if not self.metrics_jsonl_path:
            return

        a_rewards = [r.mean_attacker_reward for r in task_results]
        env_diag = _mean_diag_values(task_results, "clean_acc", "backdoor_acc")
        action_diag = _mean_diag_values(
            task_results,
            "defender_alpha",
            "defender_beta",
            "defender_post_param",
            "defender_server_lr",
            "server_lr_penalty",
            "defender_action_std_0",
            "defender_action_std_1",
            "defender_action_std_2",
            "defender_action_std_3",
            "attacker_gamma",
            "attacker_local_steps",
            "attacker_lambda_stealth",
            "attacker_action_std_0",
            "attacker_action_std_1",
            "attacker_action_std_2",
        )
        per_attack: Dict[str, List[float]] = defaultdict(list)
        for result in task_results:
            per_attack[result.attack_type.name].append(result.mean_defender_reward)

        defender_losses = _aggregate_task_losses(
            [result.defender_losses for result in task_results]
        )
        attacker_losses: Dict[str, Dict[str, float]] = {}
        attacker_loss_buckets: Dict[str, List[Dict]] = defaultdict(list)
        for result in task_results:
            if result.attacker_br_losses:
                attacker_loss_buckets[result.attack_type.name].append(
                    result.attacker_br_losses
                )
        for name, loss_dicts in attacker_loss_buckets.items():
            attacker_losses[name] = _aggregate_task_losses(loss_dicts)

        inner_delta_norms = [float(result.inner_delta_norm) for result in task_results]
        local_buffer_sizes = [
            int(result.diagnostics.get("buffer_size", 0)) for result in task_results
        ]

        record = {
            "iteration": int(t + 1),
            "local_iteration": int(t - self.start_iteration + 1),
            "batch_attack_types": [xi.name for xi in batch_types],
            "reward_mean": float(np.mean(d_rewards)) if d_rewards else float("nan"),
            "reward_std": float(np.std(d_rewards)) if d_rewards else float("nan"),
            "reward_min": float(np.min(d_rewards)) if d_rewards else float("nan"),
            "reward_max": float(np.max(d_rewards)) if d_rewards else float("nan"),
            "attacker_reward_mean": float(np.mean(a_rewards)) if a_rewards else float("nan"),
            "adaptive_fraction": float(np.mean([xi.adaptive for xi in batch_types])) if batch_types else 0.0,
            "clean_acc": env_diag.get("clean_acc"),
            "backdoor_acc": env_diag.get("backdoor_acc"),
            "reptile_delta_norm": float(reptile_norms["full"]),
            "reptile_actor_delta_norm": float(reptile_norms["actor"]),
            "reptile_critic_delta_norm": float(reptile_norms["critic"]),
            "per_attack_reward_mean": {
                name: float(np.mean(vals)) for name, vals in per_attack.items()
            },
            "defender_losses": defender_losses,
            "attacker_losses": attacker_losses,
            "inner_delta_norm_mean": (
                float(np.mean(inner_delta_norms)) if inner_delta_norms else float("nan")
            ),
            "inner_delta_norm_min": (
                float(np.min(inner_delta_norms)) if inner_delta_norms else float("nan")
            ),
            "inner_delta_norm_max": (
                float(np.max(inner_delta_norms)) if inner_delta_norms else float("nan")
            ),
            "trajectories_collected": int(
                sum(result.trajectories_collected for result in task_results)
            ),
            "transitions_collected": int(
                sum(result.transitions_collected for result in task_results)
            ),
            "buffer_sizes": {
                "local_defender_mean": (
                    float(np.mean(local_buffer_sizes)) if local_buffer_sizes else 0.0
                ),
                "local_defender_per_task": local_buffer_sizes,
                "attackers": {
                    name: int(len(buffer)) for name, buffer in self.attacker_buffers.items()
                },
            },
            "iteration_elapsed_seconds": float(iteration_elapsed_seconds),
            "task_records": [
                _task_metrics_record(result, elapsed)
                for result, elapsed in zip(task_results, task_elapsed_seconds)
            ],
        }
        record.update(_query_metric_values(task_results))
        record.update(action_diag)
        with open(self.metrics_jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(_json_safe(record), allow_nan=False, sort_keys=True) + "\n")

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
            elif key.startswith(("critic.", "critic1.", "critic2.")):
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

    def save(
        self,
        directory: str,
        *,
        completed_iteration: Optional[int] = None,
        replace: bool = False,
    ) -> None:
        if replace:
            self._replace_checkpoint_directory(
                directory,
                completed_iteration=completed_iteration,
            )
        else:
            self._write_checkpoint_directory(
                directory,
                completed_iteration=completed_iteration,
            )
        print(f"[MetaSG] Saved checkpoint to {directory}")

    def _write_checkpoint_directory(
        self,
        directory: str | os.PathLike[str],
        *,
        completed_iteration: Optional[int],
    ) -> None:
        directory = os.fspath(directory)
        os.makedirs(directory, exist_ok=True)
        self.defender.save(os.path.join(directory, "defender_meta.pt"))
        for name, agent in self.attacker_agents.items():
            agent.save(os.path.join(directory, f"attacker_{name}.pt"))
        if completed_iteration is not None:
            metadata = {
                "completed_iteration": int(completed_iteration),
                "master_seed": self.checkpoint_master_seed,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            Path(directory, "checkpoint.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    def _replace_checkpoint_directory(
        self,
        directory: str | os.PathLike[str],
        *,
        completed_iteration: Optional[int],
    ) -> None:
        destination = Path(directory)
        destination.parent.mkdir(parents=True, exist_ok=True)
        suffix = f"{os.getpid()}-{uuid.uuid4().hex}"
        staging = destination.parent / f".{destination.name}.staging-{suffix}"
        backup = destination.parent / f".{destination.name}.backup-{suffix}"
        moved_existing = False

        try:
            self._write_checkpoint_directory(
                staging,
                completed_iteration=completed_iteration,
            )
            if destination.exists():
                os.replace(destination, backup)
                moved_existing = True
            os.replace(staging, destination)
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            if moved_existing and backup.exists():
                if destination.exists():
                    shutil.rmtree(destination, ignore_errors=True)
                os.replace(backup, destination)
            raise
        else:
            if backup.exists():
                shutil.rmtree(backup)

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
        if not self.attack_domain or k <= 0:
            return []

        sampler = getattr(self.meta_cfg, "task_sampler", "iid")
        if sampler == "iid":
            indices = np.random.choice(len(self.attack_domain), size=k, replace=True)
            return [self.attack_domain[int(i)] for i in indices]

        if sampler != "stratified":
            raise ValueError(f"Unsupported task_sampler={sampler!r}")

        if k < len(self.attack_domain):
            indices = np.random.choice(len(self.attack_domain), size=k, replace=False)
            return [self.attack_domain[i] for i in indices]

        batch: List[AttackType] = []
        while len(batch) < k:
            indices = np.random.permutation(len(self.attack_domain))
            for idx in indices:
                batch.append(self.attack_domain[int(idx)])
                if len(batch) == k:
                    break
        return batch


# ── Module-level helpers ──────────────────────────────────────────────────────

def _json_safe(value):
    """Return bounded metrics data with standards-compliant JSON scalars."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _task_metrics_record(result: TaskResult, elapsed_seconds: float) -> Dict:
    return {
        "attack_type": str(result.attack_type.name),
        "attack_objective": str(result.attack_type.objective),
        "adaptive": bool(result.attack_type.adaptive),
        "mean_defender_reward": float(result.mean_defender_reward),
        "mean_attacker_reward": float(result.mean_attacker_reward),
        "defender_reward_sum": float(result.defender_reward_sum),
        "trajectories_collected": int(result.trajectories_collected),
        "transitions_collected": int(result.transitions_collected),
        "defender_losses": dict(result.defender_losses),
        "attacker_losses": dict(result.attacker_br_losses),
        "inner_delta_norm": float(result.inner_delta_norm),
        "query": {
            "base_reward": float(result.query_base_reward),
            "adapted_reward": float(result.query_adapted_reward),
            "gain": float(result.query_gain),
            "base_clean_acc": float(result.query_base_clean_acc),
            "adapted_clean_acc": float(result.query_adapted_clean_acc),
            "clean_drop": float(result.query_clean_drop),
            "base_backdoor_acc": float(result.query_base_backdoor_acc),
            "adapted_backdoor_acc": float(result.query_adapted_backdoor_acc),
            "gain_accepted": bool(result.query_gain_accepted),
            "clean_accepted": bool(result.query_clean_accepted),
            "backdoor_accepted": bool(result.query_backdoor_accepted),
            "accepted": bool(result.query_accepted),
        },
        "diagnostics": dict(result.diagnostics),
        "elapsed_seconds": float(elapsed_seconds),
    }


def _mean_diag_values(task_results: List[TaskResult], *keys: str) -> Dict[str, float]:
    """Mean of diagnostics[key] over all task results (ignores NaN)."""
    buckets: Dict[str, List[float]] = defaultdict(list)
    for result in task_results:
        for key in keys:
            val = result.diagnostics.get(key)
            if val is not None and not math.isnan(val):
                buckets[key].append(float(val))
    return {key: float(np.mean(vals)) for key, vals in buckets.items() if vals}


def _query_metric_values(task_results: List[TaskResult]) -> Dict[str, float]:
    gains = [r.query_gain for r in task_results if not math.isnan(r.query_gain)]
    if not gains:
        return {}
    base = [r.query_base_reward for r in task_results if not math.isnan(r.query_base_reward)]
    adapted = [r.query_adapted_reward for r in task_results if not math.isnan(r.query_adapted_reward)]
    accepted = [float(r.query_accepted) for r in task_results if not math.isnan(r.query_gain)]
    gain_accepted = [
        float(r.query_gain_accepted)
        for r in task_results
        if not math.isnan(r.query_gain)
    ]
    base_clean = [
        r.query_base_clean_acc
        for r in task_results
        if not math.isnan(r.query_base_clean_acc)
    ]
    adapted_clean = [
        r.query_adapted_clean_acc
        for r in task_results
        if not math.isnan(r.query_adapted_clean_acc)
    ]
    clean_drops = [r.query_clean_drop for r in task_results if not math.isnan(r.query_clean_drop)]
    clean_accepted = [
        float(r.query_clean_accepted)
        for r in task_results
        if not math.isnan(r.query_gain)
    ]
    base_bd = [
        r.query_base_backdoor_acc
        for r in task_results
        if not math.isnan(r.query_base_backdoor_acc)
    ]
    adapted_bd = [
        r.query_adapted_backdoor_acc
        for r in task_results
        if not math.isnan(r.query_adapted_backdoor_acc)
    ]
    backdoor_accepted = [
        float(r.query_backdoor_accepted)
        for r in task_results
        if not math.isnan(r.query_gain)
    ]
    backdoor_increases = [
        r.query_adapted_backdoor_acc - r.query_base_backdoor_acc
        for r in task_results
        if not math.isnan(r.query_base_backdoor_acc)
        and not math.isnan(r.query_adapted_backdoor_acc)
    ]
    targeted_reductions = [
        r.query_base_backdoor_acc - r.query_adapted_backdoor_acc
        for r in task_results
        if str(r.attack_type.objective) == "targeted"
        and not math.isnan(r.query_base_backdoor_acc)
        and not math.isnan(r.query_adapted_backdoor_acc)
    ]
    metrics = {
        "query_base_reward_mean": float(np.mean(base)) if base else float("nan"),
        "query_adapted_reward_mean": float(np.mean(adapted)) if adapted else float("nan"),
        "query_gain_mean": float(np.mean(gains)),
        "query_gain_min": float(np.min(gains)),
        "query_gain_median": float(np.median(gains)),
        "query_accept_rate": float(np.mean(accepted)) if accepted else float("nan"),
    }
    if gain_accepted:
        metrics["query_gain_accept_rate"] = float(np.mean(gain_accepted))
    if base_clean:
        metrics["query_base_clean_acc_mean"] = float(np.mean(base_clean))
    if adapted_clean:
        metrics["query_adapted_clean_acc_mean"] = float(np.mean(adapted_clean))
    if clean_drops:
        metrics["query_clean_drop_mean"] = float(np.mean(clean_drops))
        metrics["query_clean_drop_max"] = float(np.max(clean_drops))
    if clean_accepted:
        metrics["query_clean_accept_rate"] = float(np.mean(clean_accepted))
    if base_bd:
        metrics["query_base_backdoor_acc_mean"] = float(np.mean(base_bd))
    if adapted_bd:
        metrics["query_adapted_backdoor_acc_mean"] = float(np.mean(adapted_bd))
    if backdoor_accepted:
        metrics["query_backdoor_accept_rate"] = float(np.mean(backdoor_accepted))
    if backdoor_increases:
        metrics["query_backdoor_increase_mean"] = float(np.mean(backdoor_increases))
        metrics["query_backdoor_increase_max"] = float(np.max(backdoor_increases))
    if targeted_reductions:
        metrics["query_targeted_asr_reduction_mean"] = float(np.mean(targeted_reductions))
        metrics["query_targeted_asr_reduction_min"] = float(np.min(targeted_reductions))
        metrics["query_targeted_asr_reduction_max"] = float(np.max(targeted_reductions))
    for result in task_results:
        if math.isnan(result.query_gain):
            continue
        attack_name = str(result.attack_type.name)
        attack_prefix = f"query_attack_name__{attack_name}__"
        metrics[f"query_attack_{attack_name}_gain"] = float(result.query_gain)
        metrics[f"{attack_prefix}gain"] = float(result.query_gain)
        metrics[f"query_attack_{attack_name}_accepted"] = float(result.query_accepted)
        metrics[f"{attack_prefix}accepted"] = float(result.query_accepted)
        metrics[f"query_attack_{attack_name}_gain_accepted"] = float(result.query_gain_accepted)
        metrics[f"{attack_prefix}gain_accepted"] = float(result.query_gain_accepted)
        metrics[f"query_attack_{attack_name}_clean_accepted"] = float(result.query_clean_accepted)
        metrics[f"{attack_prefix}clean_accepted"] = float(result.query_clean_accepted)
        metrics[f"query_attack_{attack_name}_backdoor_accepted"] = float(result.query_backdoor_accepted)
        metrics[f"{attack_prefix}backdoor_accepted"] = float(result.query_backdoor_accepted)
        if not math.isnan(result.query_base_clean_acc):
            metrics[f"query_attack_{attack_name}_base_clean_acc"] = float(result.query_base_clean_acc)
            metrics[f"{attack_prefix}base_clean_acc"] = float(result.query_base_clean_acc)
        if not math.isnan(result.query_adapted_clean_acc):
            metrics[f"query_attack_{attack_name}_adapted_clean_acc"] = float(result.query_adapted_clean_acc)
            metrics[f"{attack_prefix}adapted_clean_acc"] = float(result.query_adapted_clean_acc)
        if not math.isnan(result.query_base_backdoor_acc):
            metrics[f"query_attack_{attack_name}_base_backdoor_acc"] = float(result.query_base_backdoor_acc)
            metrics[f"{attack_prefix}base_backdoor_acc"] = float(result.query_base_backdoor_acc)
        if not math.isnan(result.query_adapted_backdoor_acc):
            metrics[f"query_attack_{attack_name}_adapted_backdoor_acc"] = float(result.query_adapted_backdoor_acc)
            metrics[f"{attack_prefix}adapted_backdoor_acc"] = float(result.query_adapted_backdoor_acc)
        if (
            not math.isnan(result.query_base_backdoor_acc)
            and not math.isnan(result.query_adapted_backdoor_acc)
        ):
            metrics[f"query_attack_{attack_name}_backdoor_increase"] = float(
                result.query_adapted_backdoor_acc - result.query_base_backdoor_acc
            )
            metrics[f"{attack_prefix}backdoor_increase"] = float(
                result.query_adapted_backdoor_acc - result.query_base_backdoor_acc
            )
            metrics[f"query_attack_{attack_name}_backdoor_reduction"] = float(
                result.query_base_backdoor_acc - result.query_adapted_backdoor_acc
            )
            metrics[f"{attack_prefix}backdoor_reduction"] = float(
                result.query_base_backdoor_acc - result.query_adapted_backdoor_acc
            )
    return metrics


def _meta_update_adapted_params(
    task_results: List[TaskResult],
    *,
    meta_objective: str,
    query_accept_margin: float,
    query_clean_floor: float | None = None,
    query_clean_drop_tolerance: float | None = None,
    query_backdoor_ceiling: float | None = None,
    query_backdoor_increase_tolerance: float | None = None,
    query_backdoor_improvement_margin: float | None = None,
    query_targeted_asr_reduction_margin: float | None = None,
    query_targeted_min_base_backdoor: float | None = None,
) -> List[Dict]:
    if meta_objective == "reptile":
        return [result.adapted_params for result in task_results]
    if meta_objective == "query_gated_reptile":
        return [
            result.adapted_params
            for result in task_results
            if not math.isnan(result.query_gain)
            and result.query_gain >= query_accept_margin
            and _query_clean_constraints_accept(
                result,
                clean_floor=query_clean_floor,
                clean_drop_tolerance=query_clean_drop_tolerance,
            )
            and _query_backdoor_constraints_accept(
                base_backdoor=result.query_base_backdoor_acc,
                adapted_backdoor=result.query_adapted_backdoor_acc,
                backdoor_ceiling=query_backdoor_ceiling,
                backdoor_increase_tolerance=query_backdoor_increase_tolerance,
                backdoor_improvement_margin=query_backdoor_improvement_margin,
            )
        ]
    if meta_objective == "query_targeted_reptile":
        targeted = sorted(
            [
                result
                for result in task_results
                if not math.isnan(result.query_gain)
                and str(result.attack_type.objective) == "targeted"
                and _query_clean_constraints_accept(
                    result,
                    clean_floor=query_clean_floor,
                    clean_drop_tolerance=query_clean_drop_tolerance,
                )
                and _query_backdoor_reduction_accept(
                    base_backdoor=result.query_base_backdoor_acc,
                    adapted_backdoor=result.query_adapted_backdoor_acc,
                    reduction_margin=query_targeted_asr_reduction_margin,
                    min_base_backdoor=query_targeted_min_base_backdoor,
                )
            ],
            key=_targeted_query_rank_key,
            reverse=True,
        )
        non_targeted = [
            result
            for result in task_results
            if not math.isnan(result.query_gain)
            and str(result.attack_type.objective) != "targeted"
            and result.query_gain >= query_accept_margin
            and _query_clean_constraints_accept(
                result,
                clean_floor=query_clean_floor,
                clean_drop_tolerance=query_clean_drop_tolerance,
            )
        ]
        return [result.adapted_params for result in [*targeted, *non_targeted]]
    raise ValueError(f"Unsupported meta_objective={meta_objective!r}")


def _targeted_query_rank_key(result: TaskResult) -> tuple[float, float, float, float]:
    backdoor_reduction = result.query_base_backdoor_acc - result.query_adapted_backdoor_acc
    adapted_clean = result.query_adapted_clean_acc
    adapted_backdoor = result.query_adapted_backdoor_acc
    return (
        float(backdoor_reduction) if math.isfinite(backdoor_reduction) else float("-inf"),
        float(result.query_gain) if math.isfinite(result.query_gain) else float("-inf"),
        float(adapted_clean) if math.isfinite(adapted_clean) else float("-inf"),
        -float(adapted_backdoor) if math.isfinite(adapted_backdoor) else float("-inf"),
    )


def _query_clean_constraints_accept(
    result: TaskResult,
    *,
    clean_floor: float | None,
    clean_drop_tolerance: float | None,
) -> bool:
    if clean_floor is not None:
        if (
            math.isnan(result.query_adapted_clean_acc)
            or result.query_adapted_clean_acc < float(clean_floor)
        ):
            return False
    if clean_drop_tolerance is not None:
        if math.isnan(result.query_clean_drop):
            if math.isnan(result.query_base_clean_acc) or math.isnan(result.query_adapted_clean_acc):
                return False
            clean_drop = result.query_base_clean_acc - result.query_adapted_clean_acc
        else:
            clean_drop = result.query_clean_drop
        if clean_drop > float(clean_drop_tolerance):
            return False
    return True


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
