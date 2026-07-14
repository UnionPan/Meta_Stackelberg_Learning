"""Meta-SG pretraining for model-poisoning or backdoor attack domains."""
from __future__ import annotations

import argparse
import json
import os
import random
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
from meta_sg.learning.memory_maintenance import perform_memory_maintenance
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


def clean_backdoor_attack_domain() -> list[AttackType]:
    """Return a clean task plus the targeted backdoor task domain."""
    return [ATTACK_DOMAIN["clean"], *backdoor_attack_domain()]


def clean_mixed_backdoor_attack_domain() -> list[AttackType]:
    """Return clean plus one task where BFL/DBA/RL-backdoor coexist among attackers."""
    return [
        ATTACK_DOMAIN["clean"],
        AttackType(name="mixed_backdoor", objective="targeted", adaptive=False),
    ]


def clean_backdoor_mixed_attack_domain() -> list[AttackType]:
    """Return clean, each single backdoor task, and the mixed-client backdoor task."""
    return [
        ATTACK_DOMAIN["clean"],
        *backdoor_attack_domain(),
        AttackType(name="mixed_backdoor", objective="targeted", adaptive=False),
    ]


def mixed_attack_domain() -> list[AttackType]:
    """Return global model-poisoning plus targeted backdoor task domain."""
    return [*poisoning_attack_domain(), *backdoor_attack_domain()]


def clean_mixed_attack_domain() -> list[AttackType]:
    """Return clean plus global model-poisoning and targeted backdoor tasks."""
    return [ATTACK_DOMAIN["clean"], *mixed_attack_domain()]


def clean_global_attack_domain() -> list[AttackType]:
    """Return clean plus global model-poisoning tasks."""
    return [ATTACK_DOMAIN["clean"], *poisoning_attack_domain()]


def clean_global_backdoor_mixed_attack_domain() -> list[AttackType]:
    """Return clean, global poisoning, single backdoors, and mixed-client backdoor."""
    return [
        ATTACK_DOMAIN["clean"],
        *poisoning_attack_domain(),
        *backdoor_attack_domain(),
        AttackType(name="mixed_backdoor", objective="targeted", adaptive=False),
    ]


NATIVE_BACKDOOR_ATTACK_DOMAINS = {
    "backdoor",
    "clean_backdoor",
    "clean_mixed_backdoor",
    "clean_backdoor_mixed",
    "mixed",
    "clean_mixed",
    "clean_global_backdoor_mixed",
}


def domain_uses_native_backdoor_attacks(name: str) -> bool:
    return str(name) in NATIVE_BACKDOOR_ATTACK_DOMAINS


def attack_domain_from_name(name: str) -> list[AttackType]:
    if name == "model_poisoning":
        return poisoning_attack_domain()
    if name == "backdoor":
        return backdoor_attack_domain()
    if name == "clean_backdoor":
        return clean_backdoor_attack_domain()
    if name == "clean_mixed_backdoor":
        return clean_mixed_backdoor_attack_domain()
    if name == "clean_backdoor_mixed":
        return clean_backdoor_mixed_attack_domain()
    if name == "mixed":
        return mixed_attack_domain()
    if name == "clean_mixed":
        return clean_mixed_attack_domain()
    if name == "clean_global":
        return clean_global_attack_domain()
    if name == "clean_global_backdoor_mixed":
        return clean_global_backdoor_mixed_attack_domain()
    raise ValueError(f"Unsupported attack domain: {name}")


def defender_action_dim(args) -> int:
    return 4 if str(args.defender_third_action) == "both" else 3


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["stub", "fl_sandbox"], default="fl_sandbox")
    parser.add_argument("--output-dir", default="runs/meta_sg_pretraining")
    parser.add_argument(
        "--run-name",
        default="",
        help="Exact child directory under --output-dir; timestamped when empty.",
    )
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    parser.add_argument(
        "--attack-domain",
        choices=[
            "model_poisoning",
            "backdoor",
            "clean_backdoor",
            "clean_mixed_backdoor",
            "clean_backdoor_mixed",
            "mixed",
            "clean_mixed",
            "clean_global",
            "clean_global_backdoor_mixed",
        ],
        default="model_poisoning",
        help="Meta-training attack task domain.",
    )
    parser.add_argument("--T", type=int, default=100, help="Outer Reptile iterations")
    parser.add_argument("--K", type=int, default=10, help="Attack tasks per outer iteration")
    parser.add_argument("--H", type=int, default=200, help="FL rollout horizon per task")
    parser.add_argument("--l", type=int, default=10, help="Defender inner TD3 updates")
    parser.add_argument(
        "--support-episodes",
        type=int,
        default=1,
        help=(
            "Number of support trajectories collected per attack task. "
            "Each episode collects H FL rounds and then runs l defender TD3 updates."
        ),
    )
    parser.add_argument("--N-A", dest="N_A", type=int, default=10, help="Adaptive attacker BR updates")
    parser.add_argument("--post-br-defender-updates", type=int, default=1)
    parser.add_argument("--meta-step", type=float, default=1.0)
    parser.add_argument(
        "--meta-objective",
        choices=["reptile", "query_gated_reptile", "query_targeted_reptile"],
        default="reptile",
        help=(
            "Outer objective: original Reptile, score-gated query Reptile, "
            "or targeted-ASR-reduction query Reptile."
        ),
    )
    parser.add_argument(
        "--query-horizon",
        type=int,
        default=None,
        help="Held-out query rollout horizon for query_gated_reptile. Defaults to H.",
    )
    parser.add_argument("--query-seed-offset", type=int, default=50_000)
    parser.add_argument(
        "--query-diagnostics-horizon",
        type=int,
        default=None,
        help="Held-out query horizon used only for diagnostics when the meta objective is vanilla Reptile.",
    )
    parser.add_argument("--query-accept-margin", type=float, default=0.0)
    parser.add_argument(
        "--query-clean-floor",
        type=float,
        default=None,
        help="For query_gated_reptile, reject adapted query policies whose clean accuracy falls below this floor.",
    )
    parser.add_argument(
        "--query-clean-drop-tolerance",
        type=float,
        default=None,
        help="For query_gated_reptile, reject adapted query policies that drop clean accuracy by more than this amount.",
    )
    parser.add_argument(
        "--query-backdoor-ceiling",
        type=float,
        default=None,
        help="For query_gated_reptile, reject adapted query policies whose backdoor accuracy exceeds this ceiling.",
    )
    parser.add_argument(
        "--query-backdoor-increase-tolerance",
        type=float,
        default=None,
        help="For query_gated_reptile, reject adapted query policies that increase backdoor accuracy by more than this amount.",
    )
    parser.add_argument(
        "--query-backdoor-improvement-margin",
        type=float,
        default=None,
        help=(
            "For query_gated_reptile, allow an adapted query policy above the backdoor ceiling "
            "only if it reduces backdoor accuracy by at least this margin."
        ),
    )
    parser.add_argument(
        "--query-targeted-asr-reduction-margin",
        type=float,
        default=None,
        help=(
            "For query_targeted_reptile, require targeted attacks to reduce query ASR "
            "by at least this amount."
        ),
    )
    parser.add_argument(
        "--query-targeted-min-base-backdoor",
        type=float,
        default=None,
        help=(
            "For query_targeted_reptile, only accept targeted updates when the base "
            "query ASR is at least this value."
        ),
    )
    parser.add_argument("--lambda-bd", type=float, default=None, help="Backdoor ASR penalty in defender reward.")
    parser.add_argument(
        "--task-sampler",
        choices=["iid", "stratified"],
        default="iid",
        help="Attack task sampler: iid matches the paper; stratified guarantees coverage when possible.",
    )
    parser.add_argument(
        "--task-warmup-steps",
        type=int,
        default=0,
        help="Random warmup transitions collected inside each task before support adaptation.",
    )
    parser.add_argument(
        "--attack-context",
        action="store_true",
        help="Append a one-hot attack-domain context vector to every Meta-SG observation.",
    )
    parser.add_argument(
        "--defender-third-action",
        choices=["neuroclip", "server_lr", "both"],
        default="neuroclip",
        help="Third defender action: neuroclip is reward-only post training; server_lr scales the FL transition.",
    )
    parser.add_argument(
        "--post-defense-mode",
        choices=["weight_copy", "model_aware_neuroclip"],
        default="weight_copy",
        help="Post-training reward/eval path used during Meta-SG rollouts.",
    )
    parser.add_argument("--neuroclip-eps-min", type=float, default=1.0)
    parser.add_argument("--neuroclip-eps-max", type=float, default=10.0)
    parser.add_argument(
        "--neuroclip-log-scale",
        action="store_true",
        help="Decode the NeuroClip epsilon action logarithmically.",
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
        "--fl-batch-size",
        type=int,
        default=None,
        help="Client DataLoader batch size inside fl_sandbox. Defaults to --batch-size for direct-eval parity.",
    )
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
    parser.add_argument(
        "--latest-checkpoint-only",
        action="store_true",
        help="Replace checkpoints/latest without retaining iter_NNNN history.",
    )
    parser.add_argument(
        "--memory-maintenance",
        choices=("off", "task"),
        default="off",
        help="Run Python GC and best-effort allocator trimming after each completed task.",
    )
    parser.add_argument("--resume-from", default="", help="Checkpoint directory to load before training.")
    parser.add_argument(
        "--allow-model-only-resume",
        action="store_true",
        help="Explicitly allow a discontinuous legacy resume without replay/RNG state.",
    )
    parser.add_argument("--start-iteration", type=int, default=0, help="Completed outer iterations before this run.")
    parser.add_argument(
        "--total-iterations",
        type=int,
        default=None,
        help="Total planned global outer iterations for resumed logging.",
    )
    parser.add_argument("--tensorboard", action="store_true")
    args = parser.parse_args(argv)
    if args.allow_model_only_resume and not args.resume_from:
        parser.error("--allow-model-only-resume requires --resume-from")
    return args


def build_meta_config(args) -> MetaSGConfig:
    lambda_bd = args.lambda_bd
    if lambda_bd is None:
        lambda_bd = 1.0 if domain_uses_native_backdoor_attacks(args.attack_domain) else 0.0
    attack_context_names = (
        tuple(attack.name for attack in attack_domain_from_name(args.attack_domain))
        if args.attack_context
        else ()
    )
    return MetaSGConfig(
        T=args.T,
        K=args.K,
        H_mnist=args.H,
        H_cifar=args.H,
        l=args.l,
        support_episodes=args.support_episodes,
        N_A=args.N_A,
        post_br_defender_updates=args.post_br_defender_updates,
        meta_update_step=args.meta_step,
        meta_objective=args.meta_objective,
        query_horizon=args.query_horizon,
        query_seed_offset=args.query_seed_offset,
        query_diagnostics_horizon=args.query_diagnostics_horizon,
        query_accept_margin=args.query_accept_margin,
        query_clean_floor=args.query_clean_floor,
        query_clean_drop_tolerance=args.query_clean_drop_tolerance,
        query_backdoor_ceiling=args.query_backdoor_ceiling,
        query_backdoor_increase_tolerance=args.query_backdoor_increase_tolerance,
        query_backdoor_improvement_margin=args.query_backdoor_improvement_margin,
        query_targeted_asr_reduction_margin=args.query_targeted_asr_reduction_margin,
        query_targeted_min_base_backdoor=args.query_targeted_min_base_backdoor,
        task_sampler=args.task_sampler,
        eval_every=1,
        warmup_steps=args.task_warmup_steps,
        history_len=0,
        lambda_bd=float(lambda_bd),
        reward_mode="accuracy",
        defender_third_action=args.defender_third_action,
        post_defense_mode=args.post_defense_mode,
        eps_min=float(args.neuroclip_eps_min),
        eps_max=float(args.neuroclip_eps_max),
        eps_log_scale=bool(args.neuroclip_log_scale),
        server_lr_min=float(args.server_lr_min),
        server_lr_max=float(args.server_lr_max),
        server_lr_penalty_weight=float(args.server_lr_penalty_weight),
        native_sandbox_attacks=(
            args.backend == "fl_sandbox" and domain_uses_native_backdoor_attacks(args.attack_domain)
        ),
        attack_context_names=attack_context_names,
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


def build_sandbox_config(
    args,
    *,
    attack_type: AttackType | str | None = None,
    horizon: int | None = None,
    seed: int | None = None,
):
    resolved_device = str(resolve_torch_device(args.device))
    attack_name = _attack_name_for_sandbox_config(attack_type)
    resolved_horizon = int(args.H if horizon is None else horizon)
    resolved_seed = int(args.seed if seed is None else seed)
    fl_batch_size = int(args.batch_size if args.fl_batch_size is None else args.fl_batch_size)
    return SandboxConfig(
        dataset=args.dataset,
        attack_type=attack_name,
        defense_type="paper_norm_trimmed_mean",
        rounds=resolved_horizon,
        num_clients=args.num_clients,
        num_attackers=0 if attack_name == "clean" else args.num_attackers,
        subsample_rate=args.subsample_rate,
        seed=resolved_seed,
        device=resolved_device,
        parallel_clients=args.fl_parallel_clients,
        num_workers=args.fl_num_workers,
        batch_size=fl_batch_size,
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
        rl_attack_start_round=max(2, min(6, resolved_horizon)),
        rl_policy_train_end_round=max(2, resolved_horizon),
    )


def _attack_name_for_sandbox_config(attack_type: AttackType | str | None) -> str:
    if attack_type is None:
        return "clean"
    return str(getattr(attack_type, "name", attack_type))


def make_coordinator_factory(args):
    def factory(
        *,
        attack_type: AttackType | str | None = None,
        horizon: int | None = None,
        seed: int | None = None,
    ):
        if args.backend == "stub":
            return StubCoordinator(
                num_clients=args.num_clients,
                num_attackers=args.num_attackers,
                subsample_rate=args.subsample_rate,
                seed=int(args.seed if seed is None else seed),
            )
        return FLSandboxCoordinatorAdapter(
            build_sandbox_config(
                args,
                attack_type=attack_type,
                horizon=horizon,
                seed=seed,
            )
        )

    return factory


def probe_obs_dim(args, meta_config: MetaSGConfig) -> int:
    attack_type = attack_domain_from_name(args.attack_domain)[0]
    coordinator = make_coordinator_factory(args)(
        attack_type=attack_type,
        horizon=meta_config.H,
        seed=args.seed,
    )
    if attack_type.name == "clean":
        attack_strategy = None
    elif meta_config.native_sandbox_attacks:
        attack_strategy = NativeSandboxAttackMarker(attack_type)
    else:
        attack_strategy = build_fixed_attack(attack_type)
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
            eps_min=meta_config.eps_min,
            eps_max=meta_config.eps_max,
            eps_log_scale=meta_config.eps_log_scale,
            server_lr_min=meta_config.server_lr_min,
            server_lr_max=meta_config.server_lr_max,
            server_lr_penalty_weight=meta_config.server_lr_penalty_weight,
            attack_context_names=meta_config.attack_context_names,
        ),
        evaluator=getattr(coordinator, "evaluate_weights", None),
    )
    obs = env.reset(seed=args.seed)
    return int(obs.shape[0])


def main(argv=None):
    args = parse_args(argv)
    device = resolve_torch_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    run_name = str(args.run_name).strip() or time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir) / run_name
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
        checkpoint_latest_only=args.latest_checkpoint_only,
        checkpoint_master_seed=args.seed,
        metrics_jsonl_path=str(output_dir / "metrics.jsonl"),
        start_iteration=args.start_iteration,
        total_iterations=args.total_iterations,
        memory_maintenance=(
            perform_memory_maintenance
            if args.memory_maintenance == "task"
            else None
        ),
    )
    if args.resume_from:
        trainer.load(
            args.resume_from,
            allow_model_only=args.allow_model_only_resume,
        )
        if (
            trainer.loaded_completed_iteration is not None
            and trainer.loaded_completed_iteration != args.start_iteration
        ):
            raise ValueError(
                "checkpoint completed_iteration does not match --start-iteration: "
                f"{trainer.loaded_completed_iteration} != {args.start_iteration}"
            )
    result = trainer.train()
    trainer.save(str(output_dir / "final"), completed_iteration=result.meta_iterations)

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
