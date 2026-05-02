"""
Quick smoke test — verifies all 4 layers wire together correctly.
Run: python meta_sg/scripts/smoke_test.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import torch

# ── Layer 1: Stub coordinator ──────────────────────────────────────────────
from meta_sg.simulation.stub import StubCoordinator
from meta_sg.simulation.types import Weights

coord = StubCoordinator(num_clients=10, num_attackers=2, seed=42)
init = coord.reset(seed=42)
print(f"[L1] Weights layers: {len(init.weights)}, obs shapes: {[w.shape for w in init.weights[-2:]]}")

# ── Layer 2: Strategies ────────────────────────────────────────────────────
from meta_sg.strategies.types import AttackType, DefenseDecision, AttackDecision, ATTACK_DOMAIN
from meta_sg.strategies.attacks.fixed import IPMAttack, LMPAttack
from meta_sg.strategies.defenses.paper import PaperDefenseStrategy

ipm = IPMAttack(scaling=2.0)
defense = PaperDefenseStrategy()
raw_d = np.array([0.0, 0.0, 0.5], dtype=np.float32)
raw_a = np.array([0.2, -0.1, 0.3], dtype=np.float32)
d_decision = DefenseDecision.from_raw(raw_d)
a_decision = AttackDecision.from_raw(raw_a)
print(f"[L2] DefenseDecision: α={d_decision.norm_bound_alpha:.2f}, β={d_decision.trimmed_mean_beta:.3f}, ε={d_decision.neuroclip_epsilon:.2f}")
print(f"[L2] AttackDecision:  γ={a_decision.gamma_scale:.2f}, steps={a_decision.local_steps}, stealth={a_decision.lambda_stealth:.2f}")

# ── Layer 3: BSMG Env ─────────────────────────────────────────────────────
from meta_sg.games.bsmg_env import BSMGEnv, BSMGConfig
from meta_sg.games.observations import obs_dim_for

attack_type = ATTACK_DOMAIN["ipm"]
env = BSMGEnv(
    coordinator=StubCoordinator(num_clients=10, num_attackers=2),
    attack_type=attack_type,
    attack_strategy=ipm,
    defense_strategy=defense,
    config=BSMGConfig(horizon=5),
)
obs = env.reset(seed=0)
print(f"[L3] obs_dim={obs.shape[0]}, obs[:4]={obs[:4]}")

a_D = np.array([0.1, 0.2, 0.3], dtype=np.float32)
a_A = np.array([0.0, 0.0, 0.0], dtype=np.float32)
next_obs, r_D, r_A, done, info = env.step(a_D, a_A)
print(f"[L3] step: r_D={r_D:.4f}, r_A={r_A:.4f}, done={done}")
print(f"[L3]       clean_acc={info['clean_acc']:.4f}, backdoor_acc={info['backdoor_acc']:.4f}")

# ── Layer 4: TD3 + MetaSG ─────────────────────────────────────────────────
from meta_sg.learning.config import MetaSGConfig, TD3Config
from meta_sg.learning.td3 import TD3Agent
from meta_sg.learning.replay_buffer import ReplayBuffer

obs_dim = obs.shape[0]
td3_cfg = TD3Config(hidden_dim=64, batch_size=32, buffer_capacity=1000)
agent = TD3Agent(obs_dim, 3, td3_cfg)
action = agent.get_action(obs, noise=0.1)
print(f"[L4] TD3 action: {action}")

# Fill buffer with a few random samples
buf = ReplayBuffer(1000, obs_dim, 3)
for _ in range(50):
    s = np.random.randn(obs_dim).astype(np.float32)
    a = np.random.uniform(-1, 1, 3).astype(np.float32)
    r = float(np.random.randn())
    ns = np.random.randn(obs_dim).astype(np.float32)
    buf.add(s, a, r, ns, False)

losses = agent.update(buf)
print(f"[L4] TD3 update losses: {losses}")

# Clone for Reptile
cloned = agent.clone()
params = agent.get_params()
print(f"[L4] Reptile clone: {len(params)} parameter tensors")

# Mini MetaSGTrainer smoke test (T=1, K=2, H=3, l=2)
from meta_sg.learning.meta_sg_trainer import MetaSGTrainer
from meta_sg.strategies.types import ATTACK_DOMAIN

mini_meta_cfg = MetaSGConfig(T=1, K=2, H_mnist=3, l=2, N_A=2)
mini_td3_cfg = TD3Config(hidden_dim=32, batch_size=16, buffer_capacity=500)

attack_domain = [ATTACK_DOMAIN["ipm"], ATTACK_DOMAIN["lmp"]]

trainer = MetaSGTrainer(
    coordinator_factory=lambda: StubCoordinator(num_clients=5, num_attackers=1),
    attack_domain=attack_domain,
    meta_config=mini_meta_cfg,
    td3_config=mini_td3_cfg,
    obs_dim=obs_dim,
)
result = trainer.train()
print(f"[L4] MetaSG result: iterations={result.meta_iterations}, rewards={result.defender_rewards}")
print("\n✓ All layers smoke-tested successfully.")
