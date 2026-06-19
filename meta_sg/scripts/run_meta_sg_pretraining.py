"""Meta-SG pretraining for model-poisoning or backdoor attack domains."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import torch

from meta_sg.games.bsmg_env import BSMGConfig, BSMGEnv
from meta_sg.learning.config import MetaSGConfig, TD3Config
from meta_sg.learning.meta_sg_trainer import MetaSGTrainer
from meta_sg.learning.task_runner import NativeSandboxAttackMarker
from meta_sg.simulation.fl_sandbox_adapter import FLSandboxCoordinatorAdapter, SandboxConfig
from meta_sg.simulation.stub import StubCoordinator
from meta_sg.strategies.attacks.fixed import build_fixed_attack
from meta_sg.strategies.defenses.paper import PaperDefenseStrategy
from meta_sg.strategies.types import ATTACK_DOMAIN, AttackType


def poisoning_attack_domain() -> list[AttackType]:
    """Return the model-poisoning-only task domain."""
    return [ATTACK_DOMAIN["ipm"], ATTACK_DOMAIN["lmp"], ATTACK_DOMAIN["rl"]]


def backdoor_attack_domain() -> list[AttackType]:
    """Return the targeted backdoor task domain using native fl_sandbox attacks."""
    return [
        ATTACK_DOMAIN["bfl"],
        ATTACK_DOMAIN["dba"],
        AttackType(name="rl_backdoor", objective="targeted", adaptive=False),
    ]


def mixed_attack_domain() -> list[AttackType]:
    """Return global model-poisoning plus targeted backdoor task domain."""
    return [*poisoning_attack_domain(), *backdoor_attack_domain()]


def attack_domain_from_name(name: str) -> list[AttackType]:
    if name == "model_poisoning":
        return poisoning_attack_domain()
    if name == "backdoor":
        return backdoor_attack_domain()
    if name == "mixed":
        return mixed_attack_domain()
    raise ValueError(f"Unsupported attack domain: {name}")


def defender_action_dim(args) -> int:
    return 4 if str(args.defender_third_action) == "both" else 3


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["stub", "fl_sandbox"], default="fl_sandbox")
    parser.add_argument("--output-dir", default="runs/meta_sg_pretraining")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    parser.add_argument(
        "--attack-domain",
        choices=["model_poisoning", "backdoor", "mixed"],
        default="model_poisoning",
        help="Meta-training attack task domain.",
    )
    parser.add_argument("--T", type=int, default=100, help="Outer Reptile iterations")
    parser.add_argument("--K", type=int, default=10, help="Attack tasks per outer iteration")
    parser.add_argument("--H", type=int, default=200, help="FL rollout horizon per task")
    parser.add_argument("--l", type=int, default=10, help="Defender inner TD3 updates")
    parser.add_argument("--N-A", dest="N_A", type=int, default=10, help="Adaptive attacker BR updates")
    parser.add_argument("--post-br-defender-updates", type=int, default=1)
    parser.add_argument("--meta-step", type=float, default=1.0)
    parser.add_argument("--lambda-bd", type=float, default=None, help="Backdoor ASR penalty in defender reward.")
    parser.add_argument(
        "--task-sampler",
        choices=["iid", "stratified"],
        default="iid",
        help="Attack task sampler: iid matches the paper; stratified guarantees coverage when possible.",
    )
    parser.add_argument(
        "--defender-third-action",
        choices=["neuroclip", "server_lr", "both"],
        default="neuroclip",
        help="Third defender action: neuroclip is reward-only post training; server_lr scales the FL transition.",
    )
    parser.add_argument("--server-lr-min", type=float, default=0.0)
    parser.add_argument("--server-lr-max", type=float, default=1.0)
    parser.add_argument("--server-lr-penalty-weight", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-capacity", type=int, default=100_000)
    parser.add_argument("--num-clients", type=int, default=100)
    parser.add_argument("--num-attackers", type=int, default=20)
    parser.add_argument("--subsample-rate", type=float, default=0.1)
    parser.add_argument("--client-samples", type=int, default=None)
    parser.add_argument("--eval-samples", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument(
        "--fl-parallel-clients",
        type=int,
        default=1,
        help="Number of benign FL clients to train concurrently inside each sandbox round.",
    )
    parser.add_argument(
        "--fl-num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes for fl_sandbox client/eval loaders.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, cuda:0")
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--resume-from", default="", help="Checkpoint directory to load before training.")
    parser.add_argument("--start-iteration", type=int, default=0, help="Completed outer iterations before this run.")
    parser.add_argument(
        "--total-iterations",
        type=int,
        default=None,
        help="Total planned global outer iterations for resumed logging.",
    )
    parser.add_argument("--tensorboard", action="store_true")
    return parser.parse_args(argv)


def build_meta_config(args) -> MetaSGConfig:
    lambda_bd = args.lambda_bd
    if lambda_bd is None:
        lambda_bd = 1.0 if args.attack_domain in {"backdoor", "mixed"} else 0.0
    return MetaSGConfig(
        T=args.T,
        K=args.K,
        H_mnist=args.H,
        H_cifar=args.H,
        l=args.l,
        N_A=args.N_A,
        post_br_defender_updates=args.post_br_defender_updates,
        meta_update_step=args.meta_step,
        task_sampler=args.task_sampler,
        eval_every=1,
        warmup_steps=0,
        history_len=0,
        lambda_bd=float(lambda_bd),
        reward_mode="accuracy",
        defender_third_action=args.defender_third_action,
        server_lr_min=float(args.server_lr_min),
        server_lr_max=float(args.server_lr_max),
        server_lr_penalty_weight=float(args.server_lr_penalty_weight),
        native_sandbox_attacks=(args.backend == "fl_sandbox" and args.attack_domain in {"backdoor", "mixed"}),
        dataset=args.dataset,
    )


def build_td3_config(args) -> TD3Config:
    return TD3Config(
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        warmup_steps=0,
    )


def resolve_torch_device(device: str) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_sandbox_config(args):
    resolved_device = str(resolve_torch_device(args.device))
    return SandboxConfig(
        dataset=args.dataset,
        attack_type="clean",
        defense_type="paper_norm_trimmed_mean",
        rounds=args.H,
        num_clients=args.num_clients,
        num_attackers=args.num_attackers,
        subsample_rate=args.subsample_rate,
        seed=args.seed,
        device=resolved_device,
        parallel_clients=args.fl_parallel_clients,
        num_workers=args.fl_num_workers,
        max_client_samples_per_client=args.client_samples,
        max_eval_samples=args.eval_samples,
        eval_batch_size=args.eval_batch_size,
        bfl_poison_frac=1.0,
        dba_poison_frac=0.5,
        dba_num_sub_triggers=4,
        rl_backdoor_default_action=(1.0, 0.0, -1.0, 0.0),
        rl_backdoor_stealth_norm_cap=True,
        rl_backdoor_freeze_boost=5.0,
        rl_backdoor_warmup_fixed_rollouts=0,
        rl_backdoor_simulator_shadow_clients=min(5, args.num_clients),
        rl_backdoor_simulator_shadow_samples_per_client=50,
        rl_backdoor_reward_mode="paper",
        rl_backdoor_reward_clean_lambda=0.375,
        rl_policy_train_steps_per_round=1,
        rl_attack_start_round=max(2, min(6, args.H)),
        rl_policy_train_end_round=max(2, args.H),
    )


def make_coordinator_factory(args):
    def factory():
        if args.backend == "stub":
            return StubCoordinator(
                num_clients=args.num_clients,
                num_attackers=args.num_attackers,
                subsample_rate=args.subsample_rate,
                seed=args.seed,
            )
        return FLSandboxCoordinatorAdapter(build_sandbox_config(args))

    return factory


def probe_obs_dim(args, meta_config: MetaSGConfig) -> int:
    coordinator = make_coordinator_factory(args)()
    attack_type = attack_domain_from_name(args.attack_domain)[0]
    attack_strategy = (
        NativeSandboxAttackMarker(attack_type)
        if meta_config.native_sandbox_attacks
        else build_fixed_attack(attack_type)
    )
    env = BSMGEnv(
        coordinator=coordinator,
        attack_type=attack_type,
        attack_strategy=attack_strategy,
        defense_strategy=PaperDefenseStrategy(),
        config=BSMGConfig(
            horizon=meta_config.H,
            eval_every=meta_config.eval_every,
            lambda_bd=meta_config.lambda_bd,
            reward_mode=meta_config.reward_mode,
            third_action=meta_config.defender_third_action,
            server_lr_min=meta_config.server_lr_min,
            server_lr_max=meta_config.server_lr_max,
            server_lr_penalty_weight=meta_config.server_lr_penalty_weight,
        ),
        evaluator=getattr(coordinator, "evaluate_weights", None),
    )
    obs = env.reset(seed=args.seed)
    return int(obs.shape[0])


def main(argv=None):
    args = parse_args(argv)
    device = resolve_torch_device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir) / time.strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    meta_config = build_meta_config(args)
    td3_config = build_td3_config(args)
    attack_domain = attack_domain_from_name(args.attack_domain)
    config_record = {
        "args": vars(args),
        "resolved_device": str(device),
        "resume_from": str(args.resume_from or ""),
        "start_iteration": int(args.start_iteration),
        "total_iterations": (
            int(args.total_iterations)
            if args.total_iterations is not None
            else int(args.start_iteration + args.T)
        ),
        "meta_config": asdict(meta_config),
        "td3_config": asdict(td3_config),
        "attack_domain": [attack.name for attack in attack_domain],
    }
    (output_dir / "config.json").write_text(
        json.dumps(config_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    obs_dim = probe_obs_dim(args, meta_config)
    print(
        f"[meta_sg] backend={args.backend} obs_dim={obs_dim} act_dim={defender_action_dim(args)} "
        f"device={device}"
    )

    trainer = MetaSGTrainer(
        coordinator_factory=make_coordinator_factory(args),
        attack_domain=attack_domain,
        meta_config=meta_config,
        td3_config=td3_config,
        obs_dim=obs_dim,
        act_dim=defender_action_dim(args),
        device=device,
        log_interval=args.log_interval,
        writer=writer,
        checkpoint_dir=str(output_dir / "checkpoints"),
        checkpoint_interval=args.checkpoint_interval,
        metrics_jsonl_path=str(output_dir / "metrics.jsonl"),
        start_iteration=args.start_iteration,
        total_iterations=args.total_iterations,
    )
    if args.resume_from:
        trainer.load(args.resume_from)
    result = trainer.train()
    trainer.save(str(output_dir / "final"))

    summary = {
        "meta_iterations": int(result.meta_iterations),
        "mean_r_D_last10": float(np.mean(result.defender_rewards[-10:])),
        "defender_rewards": [float(v) for v in result.defender_rewards],
        "reptile_delta_norms": [float(v) for v in result.reptile_delta_norms],
        "final_checkpoint": str(output_dir / "final"),
        "latest_checkpoint": str(output_dir / "checkpoints" / "latest"),
        "metrics_jsonl": str(output_dir / "metrics.jsonl"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        f"[meta_sg] done iters={result.meta_iterations} "
        f"mean_r_D_last10={summary['mean_r_D_last10']:.4f} "
        f"output={output_dir}"
    )
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
