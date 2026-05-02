"""
Diagnostic Meta-SG training run.

Runs T=10 outer iterations with K=6 tasks, H=20, l=10 and prints
per-task and per-iteration metrics to expose training dynamics.

Checks:
  - Reward stability and range
  - TD3 critic/actor losses
  - Reptile delta norms (are meta-updates non-trivial?)
  - Defender action distribution (exploring or stuck?)
  - Adaptive vs non-adaptive task path differences
  - NaN/inf guards
"""
from __future__ import annotations

import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import torch
from collections import defaultdict
from typing import Dict, List

from meta_sg.simulation.stub import StubCoordinator
from meta_sg.games.bsmg_env import BSMGEnv, BSMGConfig
from meta_sg.games.trajectory import Trajectory
from meta_sg.learning.collector import TrajectoryCollector
from meta_sg.learning.config import MetaSGConfig, TD3Config
from meta_sg.learning.td3 import TD3Agent
from meta_sg.learning.replay_buffer import ReplayBuffer
from meta_sg.learning.best_response import AttackerBestResponse
from meta_sg.learning.policies import ConstantActionPolicy
from meta_sg.strategies.attacks.adaptive import AdaptiveAttackStrategy
from meta_sg.strategies.attacks.fixed import build_fixed_attack
from meta_sg.strategies.defenses.paper import PaperDefenseStrategy
from meta_sg.strategies.types import ATTACK_DOMAIN, AttackType


# ─── Config ─────────────────────────────────────────────────────────────────

T       = 10   # outer iterations
K       = 6    # attack types sampled per outer iteration
H       = 20   # trajectory horizon (FL rounds per episode)
L       = 10   # inner TD3 update steps
N_A     = 5    # attacker best-response steps

TD3_CFG = TD3Config(
    hidden_dim=128,
    policy_lr=1e-3,
    critic_lr=1e-3,
    gamma=0.99,
    tau=0.005,
    policy_delay=2,
    target_noise=0.2,
    noise_clip=0.5,
    exploration_noise=0.1,
    batch_size=64,
    buffer_capacity=10_000,
    warmup_steps=0,
)

META_CFG = MetaSGConfig(
    T=T, K=K, H_mnist=H, l=L, N_A=N_A,
    meta_update_step=1.0,
    warmup_steps=0,
    post_br_defender_updates=1,
    eval_every=1,
)

ATTACK_DOMAIN_KEYS = ["ipm", "lmp", "bfl", "dba", "rl", "brl"]
ATTACK_TYPES = [ATTACK_DOMAIN[k] for k in ATTACK_DOMAIN_KEYS]

OBS_DIM = 1290   # last-2-layers of DEFAULT_LAYER_SHAPES
ACT_DIM = 3

np.random.seed(0)
torch.manual_seed(0)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _reptile_update(meta_defender: TD3Agent, adapted_params: List[Dict], step_size: float) -> float:
    """Apply Reptile and return mean actor delta L2 norm."""
    if not adapted_params:
        return 0.0
    meta_p = meta_defender.get_params()
    K_ = len(adapted_params)
    updated = {}
    actor_delta_sq = 0.0
    actor_key_count = 0
    for key in meta_p:
        delta = sum(ap[key] - meta_p[key] for ap in adapted_params if key in ap) / K_
        updated[key] = meta_p[key] + step_size * delta
        if key.startswith("actor."):
            actor_delta_sq += float((step_size * delta).norm() ** 2)
            actor_key_count += 1
    meta_defender.set_params(updated)
    return math.sqrt(actor_delta_sq) if actor_delta_sq > 0 else 0.0


def _has_nan_inf(arr: np.ndarray, name: str) -> bool:
    if np.any(np.isnan(arr)):
        print(f"    !! NaN in {name}")
        return True
    if np.any(np.isinf(arr)):
        print(f"    !! Inf in {name}")
        return True
    return False


def _run_task(
    xi: AttackType,
    meta_defender: TD3Agent,
    attacker_agents: Dict[str, TD3Agent],
    attacker_buffers: Dict[str, ReplayBuffer],
    best_response: AttackerBestResponse,
    outer_idx: int,
    task_idx: int,
    seed_base: int,
) -> Dict:
    local_def = meta_defender.clone()
    local_buf = ReplayBuffer(TD3_CFG.buffer_capacity, OBS_DIM, ACT_DIM)

    if xi.adaptive:
        attack_strategy = AdaptiveAttackStrategy(xi, attacker_agents[xi.name])
        att_policy = attacker_agents[xi.name]
    else:
        attack_strategy = build_fixed_attack(xi)
        att_policy = ConstantActionPolicy(act_dim=ACT_DIM)

    env = BSMGEnv(
        coordinator=StubCoordinator(num_clients=6, num_attackers=2, seed=seed_base),
        attack_type=xi,
        attack_strategy=attack_strategy,
        defense_strategy=PaperDefenseStrategy(),
        config=BSMGConfig(horizon=H, eval_every=1),
    )
    collector = TrajectoryCollector(
        env=env,
        defender=local_def,
        attacker=att_policy,
        defender_buffer=local_buf,
        attacker_buffer=attacker_buffers[xi.name],
        exploration_noise=TD3_CFG.exploration_noise,
        store_attacker=xi.adaptive,
    )

    traj = collector.collect(H, seed=seed_base)

    d_rewards = [t.defender_reward for t in traj.transitions]
    a_rewards = [t.attacker_reward for t in traj.transitions]
    d_actions = np.array([t.defender_action for t in traj.transitions])
    states    = np.array([t.state for t in traj.transitions])

    nan_in_obs = _has_nan_inf(states, f"{xi.name}:obs")
    nan_in_rD  = _has_nan_inf(np.array(d_rewards), f"{xi.name}:r_D")
    nan_in_act = _has_nan_inf(d_actions, f"{xi.name}:actions")

    # TD3 inner updates
    critic_losses, actor_losses = [], []
    for _ in range(L):
        losses = local_def.update(local_buf)
        if losses:
            critic_losses.append(losses.get("critic_loss", float("nan")))
            al = losses.get("actor_loss", float("nan"))
            if not math.isnan(al):
                actor_losses.append(al)

    # Reptile delta before BR
    adapted_p = local_def.get_params()
    meta_p    = meta_defender.get_params()
    delta_norm = math.sqrt(sum(
        float((adapted_p[k] - meta_p[k]).norm() ** 2)
        for k in adapted_p
    ))

    # Adaptive attacker best-response
    br_losses = {}
    if xi.adaptive:
        br_losses = best_response.update(xi)
        if hasattr(attack_strategy, "agent"):
            attack_strategy.agent.set_params(attacker_agents[xi.name].get_params())
        collector.attacker = attacker_agents[xi.name]

        # Second trajectory after BR
        traj2 = collector.collect(H, seed=seed_base + 10_000)
        d_rewards += [t.defender_reward for t in traj2.transitions]

        for _ in range(META_CFG.post_br_defender_updates):
            losses = local_def.update(local_buf)
            if losses:
                critic_losses.append(losses.get("critic_loss", float("nan")))

    # Final adapted params
    adapted_p_final = local_def.get_params()

    print(
        f"  [{xi.name:4s}|{'adapt' if xi.adaptive else 'fixed'}]"
        f"  r_D={np.mean(d_rewards):+.4f}±{np.std(d_rewards):.4f}"
        f"  r_A={np.mean(a_rewards):+.4f}"
        f"  critic_loss={np.nanmean(critic_losses):.4f}" if critic_losses else
        f"  [{xi.name:4s}|{'adapt' if xi.adaptive else 'fixed'}]"
        f"  r_D={np.mean(d_rewards):+.4f}  r_A={np.mean(a_rewards):+.4f}"
        f"  [no losses]",
        end=""
    )
    if critic_losses:
        print(
            f"  actor_loss={np.mean(actor_losses):.4f}" if actor_losses else "  actor_loss=n/a",
            f"  Δθ_norm={delta_norm:.5f}",
            f"  buf={len(local_buf)}",
        )
    else:
        print()

    # Action saturation: fraction of |a| > 0.95 per dim
    sat_frac = (np.abs(d_actions) > 0.95).mean(axis=0)
    print(
        f"         act_D mean={np.mean(d_actions, axis=0).round(3)}"
        f"  std={np.std(d_actions, axis=0).round(3)}"
        f"  saturation(|a|>0.95)={sat_frac.round(3)}"
    )
    if xi.adaptive and br_losses:
        print(f"         BR attacker loss: critic={br_losses.get('critic_loss',0):.4f}")

    return {
        "adapted_params": adapted_p_final,
        "mean_defender_reward": float(np.mean(d_rewards)),
        "delta_norm": delta_norm,
        "critic_losses": critic_losses,
        "actor_losses": actor_losses,
        "has_nan": nan_in_obs or nan_in_rD or nan_in_act,
    }


# ─── Main training loop ───────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print(f"Meta-SG Diagnostic  T={T} K={K} H={H} l={L} N_A={N_A}")
    print(f"Attack domain: {ATTACK_DOMAIN_KEYS}")
    print(f"OBS_DIM={OBS_DIM}  ACT_DIM={ACT_DIM}")
    print("=" * 70)

    meta_def = TD3Agent(OBS_DIM, ACT_DIM, TD3_CFG)
    attacker_agents  = {k: TD3Agent(OBS_DIM, ACT_DIM, TD3_CFG) for k in ATTACK_DOMAIN_KEYS}
    attacker_buffers = {k: ReplayBuffer(TD3_CFG.buffer_capacity, OBS_DIM, ACT_DIM) for k in ATTACK_DOMAIN_KEYS}
    best_response    = AttackerBestResponse(attacker_agents, attacker_buffers, n_a=N_A)

    history = defaultdict(list)

    for outer in range(T):
        print(f"\n{'─'*70}")
        print(f"[Outer {outer+1}/{T}]")

        task_types = [ATTACK_TYPES[i % len(ATTACK_TYPES)] for i in range(K)]  # cycle for full coverage

        adapted_params = []
        batch_d_rewards = []
        batch_delta_norms = []
        any_nan = False

        for task_idx, xi in enumerate(task_types):
            result = _run_task(
                xi=xi,
                meta_defender=meta_def,
                attacker_agents=attacker_agents,
                attacker_buffers=attacker_buffers,
                best_response=best_response,
                outer_idx=outer,
                task_idx=task_idx,
                seed_base=outer * 100_000 + task_idx * 10_000,
            )
            adapted_params.append(result["adapted_params"])
            batch_d_rewards.append(result["mean_defender_reward"])
            batch_delta_norms.append(result["delta_norm"])
            if result["has_nan"]:
                any_nan = True

        # Reptile meta-update
        actor_delta_norm = _reptile_update(meta_def, adapted_params, step_size=1.0)

        print(f"\n  → Reptile update: actor_Δ_norm={actor_delta_norm:.6f}")
        print(f"  → batch r_D: mean={np.mean(batch_d_rewards):.4f}  "
              f"min={min(batch_d_rewards):.4f}  max={max(batch_d_rewards):.4f}")
        print(f"  → per-task Δθ norms: {[f'{v:.4f}' for v in batch_delta_norms]}")
        if any_nan:
            print("  !! NaN/Inf detected this iteration")

        history["outer_r_D"].append(float(np.mean(batch_d_rewards)))
        history["reptile_delta"].append(actor_delta_norm)
        history["any_nan"].append(any_nan)

        # Meta-policy action sample (sanity check)
        dummy_obs = np.zeros(OBS_DIM, dtype=np.float32)
        meta_action = meta_def.get_action(dummy_obs, noise=0.0)
        print(f"  → meta-policy action on zeros: {meta_action.round(4)}")

    # ─── Final summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Defender reward over time:  {[f'{v:.4f}' for v in history['outer_r_D']]}")
    print(f"Reptile actor delta norms:  {[f'{v:.6f}' for v in history['reptile_delta']]}")
    r_D = history["outer_r_D"]
    trend = "improving" if r_D[-1] > r_D[0] else "declining" if r_D[-1] < r_D[0] else "flat"
    print(f"Reward trend (first→last):  {r_D[0]:.4f} → {r_D[-1]:.4f}  [{trend}]")
    print(f"Any NaN iteration:          {any(history['any_nan'])}")

    # Attacker buffer sizes
    print("\nAttacker buffer sizes at end:")
    for k, buf in attacker_buffers.items():
        print(f"  {k}: {len(buf)} transitions")

    print("\nDiagnostic complete.")


if __name__ == "__main__":
    main()
