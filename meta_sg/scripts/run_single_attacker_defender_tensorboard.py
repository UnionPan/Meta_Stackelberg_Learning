"""Run a paper-aligned single-attacker defender experiment."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fl_sandbox.postprocess.tensorboard_utils import build_summary_writer
from meta_sg.games.bsmg_env import BSMGConfig, BSMGEnv
from meta_sg.learning.config import TD3Config
from meta_sg.learning.collector import TrajectoryCollector
from meta_sg.learning.replay_buffer import ReplayBuffer
from meta_sg.learning.td3 import TD3Agent
from meta_sg.simulation.fl_sandbox_adapter import FLSandboxCoordinatorAdapter, SandboxConfig
from meta_sg.simulation.stub import StubCoordinator
from meta_sg.strategies.attacks.adaptive import AdaptiveAttackStrategy
from meta_sg.strategies.defenses.paper import PaperDefenseStrategy
from meta_sg.strategies.types import AttackType


def single_attack_domain() -> list[AttackType]:
    return [AttackType(name="rl", objective="untargeted", adaptive=True)]


def build_bsmg_config(
    *,
    horizon: int,
    alpha_max: float,
    beta_max: float = 0.45,
    eps_min: float = 1.0,
    eps_max: float = 10.0,
    eval_every: int = 1,
) -> BSMGConfig:
    return BSMGConfig(
        horizon=int(horizon),
        num_tail_layers=2,
        alpha_min=0.0,
        alpha_max=float(alpha_max),
        beta_min=0.0,
        beta_max=float(beta_max),
        eps_min=float(eps_min),
        eps_max=float(eps_max),
        use_neuroclip=True,
        lambda_bd=0.0,
        action_prior_weight=0.0,
        relative_alpha=False,
        reward_mode="loss",
        normalise_obs=True,
        eval_every=int(eval_every),
        history_len=0,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["stub", "fl_sandbox"], default="stub")
    parser.add_argument("--run-name", default="single_attacker_defender")
    parser.add_argument("--output-root", default="runs/meta_sg_single_attacker_defender")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--alpha-max", type=float, default=5.0)
    parser.add_argument("--beta-max", type=float, default=0.45)
    parser.add_argument("--eps-min", type=float, default=1.0)
    parser.add_argument("--eps-max", type=float, default=10.0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--td3-iters", type=int, default=20)
    parser.add_argument("--td3-updates", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--buffer-capacity", type=int, default=20_000)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_root) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = build_summary_writer(output_dir)

    bsmg_config = build_bsmg_config(
        horizon=args.horizon,
        alpha_max=args.alpha_max,
        beta_max=args.beta_max,
        eps_min=args.eps_min,
        eps_max=args.eps_max,
        eval_every=args.eval_every,
    )
    attack_type = single_attack_domain()[0]
    seed_results: list[dict[str, Any]] = []

    try:
        for seed_index, seed in enumerate(args.seeds):
            result = _run_seed(args, bsmg_config, attack_type, int(seed))
            seed_results.append(result)
            writer.add_scalar("single_attacker/final_clean_acc", result["final_clean_acc"], seed_index)
            writer.add_scalar("single_attacker/min_clean_acc", result["min_clean_acc"], seed_index)
            writer.add_scalar("single_attacker/collapse_rate", result["collapse_rate"], seed_index)
            writer.add_scalar("single_attacker/reward_mean", result["reward_mean"], seed_index)
            for step, round_info in enumerate(result["rounds"]):
                prefix = f"seed_{seed}"
                writer.add_scalar(f"{prefix}/clean_acc", round_info["clean_acc"], step)
                writer.add_scalar(f"{prefix}/clean_loss", round_info["clean_loss"], step)
                writer.add_scalar(f"{prefix}/reward", round_info["defender_reward"], step)
                writer.add_scalar(f"{prefix}/defender_alpha", round_info["alpha"], step)
                writer.add_scalar(f"{prefix}/defender_beta", round_info["beta"], step)
                writer.add_scalar(f"{prefix}/defender_epsilon", round_info["epsilon"], step)
        payload = _summary_payload(args, bsmg_config, seed_results)
        (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    finally:
        writer.flush()
        writer.close()


def _run_seed(args, bsmg_config: BSMGConfig, attack_type: AttackType, seed: int) -> dict[str, Any]:
    np.random.seed(seed)
    torch.manual_seed(seed)

    coordinator = _build_coordinator(args.backend, seed)
    probe_env = BSMGEnv(
        coordinator=coordinator,
        attack_type=attack_type,
        attack_strategy=AdaptiveAttackStrategy(attack_type, _dummy_agent()),
        defense_strategy=PaperDefenseStrategy(),
        config=bsmg_config,
    )
    obs = probe_env.reset(seed=seed)
    obs_dim = int(obs.shape[0])
    td3_cfg = TD3Config(
        hidden_dim=int(args.hidden_dim),
        batch_size=int(args.batch_size),
        buffer_capacity=int(args.buffer_capacity),
        exploration_noise=float(args.exploration_noise),
        warmup_steps=0,
    )
    device = torch.device(args.device)
    defender = TD3Agent(obs_dim, 3, td3_cfg, device)
    attacker = TD3Agent(obs_dim, 3, td3_cfg, device)

    env = BSMGEnv(
        coordinator=_build_coordinator(args.backend, seed),
        attack_type=attack_type,
        attack_strategy=AdaptiveAttackStrategy(attack_type, attacker),
        defense_strategy=PaperDefenseStrategy(),
        config=bsmg_config,
    )
    defender_buffer = ReplayBuffer(td3_cfg.buffer_capacity, obs_dim, 3)
    attacker_buffer = ReplayBuffer(td3_cfg.buffer_capacity, obs_dim, 3)
    collector = TrajectoryCollector(
        env=env,
        defender=defender,
        attacker=attacker,
        defender_buffer=defender_buffer,
        attacker_buffer=attacker_buffer,
        exploration_noise=td3_cfg.exploration_noise,
        store_attacker=True,
    )

    rounds: list[dict[str, float]] = []
    rewards: list[float] = []
    for iteration in range(int(args.td3_iters)):
        traj = collector.collect(int(args.horizon), seed=seed + iteration)
        for transition in traj.transitions:
            info = transition.info
            decision = info["defense_decision"]
            rounds.append(
                {
                    "clean_acc": float(info["clean_acc"]),
                    "clean_loss": float(info["clean_loss"]),
                    "defender_reward": float(transition.defender_reward),
                    "alpha": float(decision.norm_bound_alpha),
                    "beta": float(decision.trimmed_mean_beta),
                    "epsilon": float(decision.neuroclip_epsilon or 0.0),
                }
            )
            rewards.append(float(transition.defender_reward))
        for _ in range(int(args.td3_updates)):
            defender.update(defender_buffer)
            attacker.update(attacker_buffer)

    clean_accs = [row["clean_acc"] for row in rounds] or [0.0]
    return {
        "seed": seed,
        "final_clean_acc": float(clean_accs[-1]),
        "min_clean_acc": float(min(clean_accs)),
        "collapse_rate": float(np.mean([acc < 0.2 for acc in clean_accs])),
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "rounds": rounds,
    }


def _build_coordinator(backend: str, seed: int):
    if backend == "stub":
        return StubCoordinator(num_clients=6, num_attackers=2, subsample_rate=1.0, seed=seed)
    return FLSandboxCoordinatorAdapter(
        SandboxConfig(
            attack_type="rl",
            defense_type="clipped_median",
            seed=seed,
            num_clients=10,
            num_attackers=2,
            subsample_rate=1.0,
            batch_size=16,
            eval_batch_size=128,
            num_workers=0,
        )
    )


def _dummy_agent():
    class _DummyAgent:
        def get_action(self, obs, noise: float = 0.0):
            del obs, noise
            return np.zeros(3, dtype=np.float32)

    return _DummyAgent()


def _summary_payload(args, bsmg_config: BSMGConfig, seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    final_accs = [item["final_clean_acc"] for item in seed_results]
    min_accs = [item["min_clean_acc"] for item in seed_results]
    return {
        "run_name": args.run_name,
        "backend": args.backend,
        "attack_domain": [asdict(attack) for attack in single_attack_domain()],
        "bsmg_config": asdict(bsmg_config),
        "seeds": list(args.seeds),
        "final_clean_acc_mean": float(np.mean(final_accs)) if final_accs else 0.0,
        "min_clean_acc_mean": float(np.mean(min_accs)) if min_accs else 0.0,
        "collapse_rate_mean": float(np.mean([item["collapse_rate"] for item in seed_results]))
        if seed_results
        else 0.0,
        "seed_results": seed_results,
    }


if __name__ == "__main__":
    main()
