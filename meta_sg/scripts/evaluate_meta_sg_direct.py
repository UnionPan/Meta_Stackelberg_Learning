"""Direct final-accuracy evaluation for a trained Meta-SG defender.

This script fixes a learned defender checkpoint and evaluates it in the same
H-step FL setting used by the fixed-defense baseline table. It intentionally
reports final clean accuracy, not the training-side rollout reward.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from meta_sg.games.bsmg_env import BSMGConfig, BSMGEnv
from meta_sg.learning.config import TD3Config
from meta_sg.learning.replay_buffer import ReplayBuffer
from meta_sg.learning.td3 import TD3Agent
from meta_sg.simulation.fl_sandbox_adapter import FLSandboxCoordinatorAdapter, SandboxConfig
from meta_sg.strategies.defenses.paper import PaperDefenseStrategy
from meta_sg.strategies.types import AttackType, DefenseDecision


@dataclass(frozen=True)
class Scenario:
    name: str
    attack_type: AttackType
    attack_name: str
    patch: dict
    seed: int


class SandboxAttackMarker:
    """Name-only marker that lets FLSandboxCoordinatorAdapter build native attacks."""

    def __init__(self, attack_type: AttackType) -> None:
        self.attack_type = attack_type
        self.name = attack_type.name


class ActionOffsetPolicy:
    """Policy adapter that adds a learned/search-selected raw action offset."""

    def __init__(self, base_policy, offset: np.ndarray) -> None:
        self.base_policy = base_policy
        self.offset = np.asarray(offset, dtype=np.float32)
        self.obs_dim = int(getattr(base_policy, "obs_dim"))
        self.act_dim = int(getattr(base_policy, "act_dim"))

    def get_action(self, obs, noise: float = 0.0) -> np.ndarray:
        base_action = np.asarray(self.base_policy.get_action(obs, noise=noise), dtype=np.float32)
        return np.clip(base_action + self.offset, -1.0, 1.0).astype(np.float32)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to defender_meta.pt or checkpoint directory.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    parser.add_argument(
        "--scenario-set",
        choices=["model_poisoning", "backdoor", "mixed"],
        default="model_poisoning",
    )
    parser.add_argument("--attack-context", action="store_true")
    parser.add_argument("--lambda-bd", type=float, default=0.0)
    parser.add_argument("--H", type=int, default=20)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--num-attackers", type=int, default=4)
    parser.add_argument("--subsample-rate", type=float, default=0.5)
    parser.add_argument("--client-samples", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=502)
    parser.add_argument("--rl-seed", type=int, default=506)
    parser.add_argument(
        "--defender-third-action",
        choices=["neuroclip", "server_lr", "both"],
        default="neuroclip",
    )
    parser.add_argument("--server-lr-min", type=float, default=0.0)
    parser.add_argument("--server-lr-max", type=float, default=1.0)
    parser.add_argument("--server-lr-penalty-weight", type=float, default=0.0)
    parser.add_argument("--few-shot", action="store_true", help="Run per-scenario adaptation from the loaded checkpoint.")
    parser.add_argument("--few-shot-method", choices=["td3", "action_offset"], default="td3")
    parser.add_argument("--adaptation-horizon", type=int, default=5)
    parser.add_argument("--adaptation-episodes", type=int, default=2)
    parser.add_argument("--adaptation-updates", type=int, default=10)
    parser.add_argument("--adaptation-noise", type=float, default=0.05)
    parser.add_argument("--adaptation-lr-scale", type=float, default=0.25)
    parser.add_argument("--offset-step", type=float, default=0.1)
    parser.add_argument(
        "--few-shot-selection",
        choices=["always", "guarded"],
        default="always",
        help="Whether to always deploy the adapted policy or accept it only after a validation rollout.",
    )
    parser.add_argument("--selection-margin", type=float, default=0.0)
    parser.add_argument("--selection-horizon", type=int, default=None)
    parser.add_argument("--selection-seed-offset", type=int, default=20_000)
    parser.add_argument(
        "--rl-policy-checkpoint",
        default=(
            "runs/meta_sg_goal/fl_medium_h20_rl_clipped_median_eval/"
            "outputs/mnist_rl_clipped_median_iid_20r/checkpoints/rl_policy_latest.pt"
        ),
    )
    parser.add_argument(
        "--rl-distribution-dir",
        default="fl_sandbox/outputs/rlfl_distribution_paper/mnist_clipping_median_q_0.1_init_pre_label",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    device = _resolve_device(args.device)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    probe = FLSandboxCoordinatorAdapter(_sandbox_config(args, "clean", seed=args.seed, patch={"num_attackers": 0}))
    obs_dim = BSMGEnv(
        coordinator=probe,
        attack_type=_attack_type("clean"),
        attack_strategy=None,
        defense_strategy=PaperDefenseStrategy(),
        config=_bsmg_config(args),
        evaluator=getattr(probe, "evaluate_weights", None),
    ).reset(seed=args.seed).shape[0]

    defender = TD3Agent(
        obs_dim,
        _defender_action_dim(args),
        TD3Config(hidden_dim=args.hidden_dim, batch_size=16, buffer_capacity=4096, warmup_steps=0),
        device=device,
    )
    defender.load(_checkpoint_path(args.checkpoint))

    scenarios = _scenarios(args)
    common_probe_obs = _make_common_probe_obs(args, scenarios[0])

    records = []
    for scenario in scenarios:
        direct_record = _evaluate_scenario(args, defender, scenario)
        record = dict(direct_record)
        if args.few_shot:
            few_shot = _few_shot_adapt_and_evaluate(args, defender, scenario, probe_obs=common_probe_obs)
            record["few_shot_adaptation"] = few_shot
            record["few_shot_gain_clean_acc"] = (
                float(few_shot["evaluation"]["final_clean_acc"]) - float(direct_record["final_clean_acc"])
            )
            record["few_shot_gain_backdoor_acc"] = (
                float(few_shot["evaluation"]["final_backdoor_acc"]) - float(direct_record["final_backdoor_acc"])
            )
            record["few_shot_gain_defense_score"] = (
                float(few_shot["evaluation"]["final_defense_score"]) - float(direct_record["final_defense_score"])
            )
        records.append(record)
        print(
            scenario.name,
            "final_clean_acc=",
            round(direct_record["final_clean_acc"], 4),
            "mean_reward=",
            round(direct_record["mean_defender_reward"], 4),
            "backdoor_acc=",
            round(direct_record["final_backdoor_acc"], 4),
            "defense_score=",
            round(direct_record["final_defense_score"], 4),
            "few_shot_final_clean_acc=",
            round(record["few_shot_adaptation"]["evaluation"]["final_clean_acc"], 4)
            if args.few_shot
            else "n/a",
        )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _evaluate_scenario(args, defender: TD3Agent, scenario: Scenario) -> dict:
    return _evaluate_scenario_at(args, defender, scenario, seed=int(scenario.seed), horizon=int(args.H))


def _evaluate_scenario_at(args, defender: TD3Agent, scenario: Scenario, *, seed: int, horizon: int) -> dict:
    env = _make_env(args, scenario, seed=seed, horizon=int(horizon))
    obs = env.reset(seed=seed)
    initial_action = _action_diagnostics(defender, obs, _bsmg_config(args, horizon=int(args.H)))
    rewards_d: list[float] = []
    rewards_a: list[float] = []
    last_info = {}
    for _ in range(int(args.H)):
        defender_action = defender.get_action(obs, noise=0.0)
        attacker_action = np.zeros(3, dtype=np.float32)
        obs, r_d, r_a, done, info = env.step(defender_action, attacker_action)
        rewards_d.append(float(r_d))
        rewards_a.append(float(r_a))
        last_info = dict(info)
        if done:
            break
    final_clean_acc = float(last_info.get("clean_acc", float("nan")))
    final_backdoor_acc = float(last_info.get("backdoor_acc", float("nan")))
    return {
        "scenario": scenario.name,
        "attack_type": scenario.attack_name,
        "seed": int(seed),
        "horizon": int(horizon),
        "final_clean_acc": final_clean_acc,
        "final_backdoor_acc": final_backdoor_acc,
        "final_defense_score": float(final_clean_acc - float(args.lambda_bd) * final_backdoor_acc),
        "final_defender_reward": float(rewards_d[-1]) if rewards_d else float("nan"),
        "mean_defender_reward": float(np.mean(rewards_d)) if rewards_d else float("nan"),
        "mean_attacker_reward": float(np.mean(rewards_a)) if rewards_a else float("nan"),
        "initial_action": initial_action,
        "final_defender_alpha": float(last_info.get("defense_decision").norm_bound_alpha)
        if last_info.get("defense_decision") is not None
        else float("nan"),
        "final_defender_beta": float(last_info.get("defense_decision").trimmed_mean_beta)
        if last_info.get("defense_decision") is not None
        else float("nan"),
        "final_defender_neuroclip": float(last_info.get("defense_decision").neuroclip_epsilon)
        if last_info.get("defense_decision") is not None
        and last_info.get("defense_decision").neuroclip_epsilon is not None
        else float("nan"),
        "final_defender_server_lr": float(last_info.get("defense_decision").server_lr)
        if last_info.get("defense_decision") is not None
        and last_info.get("defense_decision").server_lr is not None
        else float("nan"),
    }


def _few_shot_adapt_and_evaluate(
    args,
    defender: TD3Agent,
    scenario: Scenario,
    *,
    probe_obs: np.ndarray,
) -> dict:
    if str(args.few_shot_method) == "action_offset":
        return _action_offset_adapt_and_evaluate(args, defender, scenario, probe_obs=probe_obs)

    adapted = defender.clone()
    if float(args.adaptation_lr_scale) != 1.0:
        adapted.set_learning_rates(
            policy_lr=float(adapted.cfg.policy_lr) * float(args.adaptation_lr_scale),
            critic_lr=float(adapted.cfg.critic_lr) * float(args.adaptation_lr_scale),
        )

    bsmg_cfg = _bsmg_config(args, horizon=int(args.adaptation_horizon))
    start_action = _action_diagnostics(defender, probe_obs, bsmg_cfg)
    trace = [{"shot": 0, "updates": 0, "action": start_action}]

    buffer = ReplayBuffer(
        capacity=max(32, int(args.adaptation_horizon) * int(args.adaptation_episodes)),
        obs_dim=defender.obs_dim,
        act_dim=defender.act_dim,
    )
    update_losses: list[dict] = []
    for episode in range(int(args.adaptation_episodes)):
        seed = int(scenario.seed) + 10_000 + episode
        env = _make_env(args, scenario, seed=seed, horizon=int(args.adaptation_horizon))
        obs = env.reset(seed=seed)
        for _ in range(int(args.adaptation_horizon)):
            action = adapted.get_action(obs, noise=float(args.adaptation_noise))
            attacker_action = np.zeros(3, dtype=np.float32)
            next_obs, reward_d, _reward_a, done, _info = env.step(action, attacker_action)
            buffer.add(obs, action, reward_d, next_obs, done)
            obs = next_obs
            if done:
                break
        for _ in range(int(args.adaptation_updates)):
            losses = adapted.update(buffer)
            if losses:
                update_losses.append({k: float(v) for k, v in losses.items()})
        trace.append(
            {
                "shot": episode + 1,
                "updates": (episode + 1) * int(args.adaptation_updates),
                "action": _action_diagnostics(adapted, probe_obs, bsmg_cfg),
            }
        )

    adapted_evaluation = _evaluate_scenario(args, adapted, scenario)
    selection = {
        "mode": str(args.few_shot_selection),
        "accepted": True,
        "selected": "adapted",
        "margin": float(args.selection_margin),
    }
    evaluation = adapted_evaluation
    if str(args.few_shot_selection) == "guarded":
        validation_horizon = int(args.selection_horizon or args.adaptation_horizon)
        validation_seed = int(scenario.seed) + int(args.selection_seed_offset)
        base_validation = _evaluate_scenario_at(
            args,
            defender,
            scenario,
            seed=validation_seed,
            horizon=validation_horizon,
        )
        adapted_validation = _evaluate_scenario_at(
            args,
            adapted,
            scenario,
            seed=validation_seed,
            horizon=validation_horizon,
        )
        selection = _guarded_selection_decision(
            base_score=float(base_validation["final_defense_score"]),
            adapted_score=float(adapted_validation["final_defense_score"]),
            margin=float(args.selection_margin),
        )
        selection.update(
            {
                "mode": "guarded",
                "margin": float(args.selection_margin),
                "validation_horizon": validation_horizon,
                "validation_seed": validation_seed,
                "base_validation": base_validation,
                "adapted_validation": adapted_validation,
            }
        )
        if not selection["accepted"]:
            evaluation = _evaluate_scenario(args, defender, scenario)
    return {
        "episodes": int(args.adaptation_episodes),
        "horizon": int(args.adaptation_horizon),
        "updates_per_episode": int(args.adaptation_updates),
        "noise": float(args.adaptation_noise),
        "lr_scale": float(args.adaptation_lr_scale),
        "probe_observation": {
            "scenario": "clean",
            "seed": int(args.seed),
            "description": "shared clean initial observation used only for action-space diagnostics",
        },
        "transition": {
            "from": start_action,
            "to": trace[-1]["action"],
            "delta": _action_delta(start_action, trace[-1]["action"]),
        },
        "trace": trace,
        "num_transitions": len(buffer),
        "num_updates": len(update_losses),
        "last_update_loss": update_losses[-1] if update_losses else {},
        "selection": selection,
        "adapted_evaluation": adapted_evaluation,
        "evaluation": evaluation,
    }


def _guarded_selection_decision(*, base_score: float, adapted_score: float, margin: float) -> dict:
    score_gain = float(adapted_score) - float(base_score)
    accepted = bool(score_gain >= float(margin))
    return {
        "accepted": accepted,
        "selected": "adapted" if accepted else "base",
        "base_score": float(base_score),
        "adapted_score": float(adapted_score),
        "score_gain": score_gain,
    }


def _action_offset_adapt_and_evaluate(
    args,
    defender,
    scenario: Scenario,
    *,
    probe_obs: np.ndarray,
) -> dict:
    bsmg_cfg = _bsmg_config(args, horizon=int(args.adaptation_horizon))
    start_action = _action_diagnostics(defender, probe_obs, bsmg_cfg)
    candidates = _offset_candidates(act_dim=int(defender.act_dim), step=float(args.offset_step))
    candidate_records = []
    for offset in candidates:
        policy = ActionOffsetPolicy(defender, offset)
        scores = []
        for episode in range(int(args.adaptation_episodes)):
            seed = int(scenario.seed) + 10_000 + episode
            record = _evaluate_scenario_at(
                args,
                policy,
                scenario,
                seed=seed,
                horizon=int(args.adaptation_horizon),
            )
            scores.append(float(record["final_defense_score"]))
        candidate_records.append(
            {
                "offset": [float(v) for v in offset],
                "mean_defense_score": float(np.mean(scores)) if scores else float("-inf"),
                "scores": scores,
            }
        )
    best = max(candidate_records, key=lambda item: item["mean_defense_score"])
    zero = candidate_records[0]
    best_offset = np.asarray(best["offset"], dtype=np.float32)
    adapted_policy = ActionOffsetPolicy(defender, best_offset)
    adapted_evaluation = _evaluate_scenario(args, adapted_policy, scenario)
    selection = {
        "mode": str(args.few_shot_selection),
        "accepted": True,
        "selected": "adapted",
        "margin": float(args.selection_margin),
        "base_score": float(zero["mean_defense_score"]),
        "adapted_score": float(best["mean_defense_score"]),
        "score_gain": float(best["mean_defense_score"]) - float(zero["mean_defense_score"]),
    }
    evaluation = adapted_evaluation
    if str(args.few_shot_selection) == "guarded":
        selection = _guarded_selection_decision(
            base_score=float(zero["mean_defense_score"]),
            adapted_score=float(best["mean_defense_score"]),
            margin=float(args.selection_margin),
        )
        selection.update(
            {
                "mode": "guarded",
                "margin": float(args.selection_margin),
                "validation_horizon": int(args.adaptation_horizon),
                "validation_episodes": int(args.adaptation_episodes),
            }
        )
        if not selection["accepted"]:
            evaluation = _evaluate_scenario(args, defender, scenario)
    end_action = _action_diagnostics(adapted_policy, probe_obs, bsmg_cfg)
    return {
        "method": "action_offset",
        "episodes": int(args.adaptation_episodes),
        "horizon": int(args.adaptation_horizon),
        "updates_per_episode": 0,
        "noise": 0.0,
        "lr_scale": 0.0,
        "offset_step": float(args.offset_step),
        "candidate_scores": candidate_records,
        "selected_offset": [float(v) for v in best_offset],
        "probe_observation": {
            "scenario": "clean",
            "seed": int(args.seed),
            "description": "shared clean initial observation used only for action-space diagnostics",
        },
        "transition": {
            "from": start_action,
            "to": end_action,
            "delta": _action_delta(start_action, end_action),
        },
        "trace": [
            {"shot": 0, "updates": 0, "action": start_action},
            {"shot": int(args.adaptation_episodes), "updates": 0, "action": end_action},
        ],
        "num_transitions": int(args.adaptation_horizon) * int(args.adaptation_episodes) * len(candidates),
        "num_updates": 0,
        "last_update_loss": {},
        "selection": selection,
        "adapted_evaluation": adapted_evaluation,
        "evaluation": evaluation,
    }


def _offset_candidates(*, act_dim: int, step: float) -> list[np.ndarray]:
    zero = np.zeros(int(act_dim), dtype=np.float32)
    candidates = [zero]
    for idx in range(int(act_dim)):
        pos = np.zeros(int(act_dim), dtype=np.float32)
        neg = np.zeros(int(act_dim), dtype=np.float32)
        pos[idx] = float(step)
        neg[idx] = -float(step)
        candidates.extend([pos, neg])
    return candidates


def _make_common_probe_obs(args, scenario: Scenario) -> np.ndarray:
    env = _make_env(args, scenario, seed=int(args.seed), horizon=int(args.adaptation_horizon))
    return env.reset(seed=int(args.seed))


def _make_env(args, scenario: Scenario, *, seed: int, horizon: int) -> BSMGEnv:
    coordinator = FLSandboxCoordinatorAdapter(
        _sandbox_config(args, scenario.attack_name, seed=seed, patch=scenario.patch)
    )
    attack_strategy = None if scenario.attack_name == "clean" else SandboxAttackMarker(scenario.attack_type)
    return BSMGEnv(
        coordinator=coordinator,
        attack_type=scenario.attack_type,
        attack_strategy=attack_strategy,
        defense_strategy=PaperDefenseStrategy(),
        config=_bsmg_config(args, horizon=horizon),
        evaluator=getattr(coordinator, "evaluate_weights", None),
    )


def _action_diagnostics(defender: TD3Agent, obs: np.ndarray, config: BSMGConfig) -> dict:
    raw = defender.get_action(obs, noise=0.0)
    decision = DefenseDecision.from_raw(
        raw,
        alpha_min=config.alpha_min,
        alpha_max=config.alpha_max,
        beta_min=config.beta_min,
        beta_max=config.beta_max,
        eps_min=config.eps_min,
        eps_max=config.eps_max,
        use_neuroclip=config.use_neuroclip,
        third_action=config.third_action,
        server_lr_min=config.server_lr_min,
        server_lr_max=config.server_lr_max,
    )
    return {
        "raw": [float(v) for v in np.asarray(raw, dtype=np.float32)],
        "alpha": float(decision.norm_bound_alpha),
        "beta": float(decision.trimmed_mean_beta),
        "neuroclip": float(decision.neuroclip_epsilon)
        if decision.neuroclip_epsilon is not None
        else float("nan"),
        "server_lr": float(decision.server_lr) if decision.server_lr is not None else float("nan"),
    }


def _action_delta(start: dict, end: dict) -> dict:
    keys = ("alpha", "beta", "neuroclip", "server_lr")
    return {
        key: float(end[key]) - float(start[key])
        for key in keys
        if key in start and key in end and np.isfinite(float(start[key])) and np.isfinite(float(end[key]))
    }


def _sandbox_config(args, attack_name: str, *, seed: int, patch: dict) -> object:
    values = {
        "dataset": args.dataset,
        "attack_type": attack_name,
        "defense_type": "paper_norm_trimmed_mean",
        "rounds": int(args.H),
        "num_clients": int(args.num_clients),
        "num_attackers": int(args.num_attackers),
        "subsample_rate": float(args.subsample_rate),
        "seed": int(seed),
        "device": str(_resolve_device(args.device)),
        "parallel_clients": 1,
        "num_workers": 0,
        "local_epochs": 1,
        "lr": 0.05,
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "max_client_samples_per_client": int(args.client_samples),
        "max_eval_samples": int(args.eval_samples),
        "fltrust_root_size": 0,
        "krum_attackers": int(args.num_attackers),
        "trimmed_mean_ratio": 0.2,
        "clipped_median_norm": 2.0,
    }
    values.update(patch)
    return SandboxConfig(**values)


def _bsmg_config(args, *, horizon: int | None = None) -> BSMGConfig:
    return BSMGConfig(
        horizon=int(args.H if horizon is None else horizon),
        eval_every=1,
        history_len=0,
        lambda_bd=float(args.lambda_bd),
        reward_mode="accuracy",
        third_action=args.defender_third_action,
        server_lr_min=float(args.server_lr_min),
        server_lr_max=float(args.server_lr_max),
        server_lr_penalty_weight=float(args.server_lr_penalty_weight),
        attack_context_names=_attack_context_names(args),
    )


def _rl_patch(args) -> dict:
    return {
        "rl_distribution_dir": str(args.rl_distribution_dir),
        "rl_attack_start_round": 6,
        "rl_policy_train_end_round": int(args.H),
        "rl_distribution_steps": 5,
        "rl_inversion_steps": 20,
        "rl_reconstruction_batch_size": 8,
        "rl_simulator_horizon": 5,
        "rl_policy_train_episodes_per_round": 1,
        "rl_policy_warmup_steps": 0,
        "rl_policy_warmup_random_steps": 0,
        "rl_policy_checkpoint_path": str(args.rl_policy_checkpoint),
    }


def _backdoor_patch(args) -> dict:
    return {
        "bfl_poison_frac": 1.0,
        "dba_poison_frac": 0.5,
        "dba_num_sub_triggers": 4,
        "rl_backdoor_default_action": (1.0, 0.0, -1.0, 0.0),
        "rl_backdoor_stealth_norm_cap": True,
        "rl_backdoor_freeze_boost": 5.0,
        "rl_backdoor_warmup_fixed_rollouts": 0,
        "rl_backdoor_simulator_shadow_clients": min(5, int(args.num_clients)),
        "rl_backdoor_simulator_shadow_samples_per_client": 50,
        "rl_backdoor_reward_mode": "paper",
        "rl_backdoor_reward_clean_lambda": 0.375,
        "rl_policy_train_steps_per_round": 1,
        "rl_attack_start_round": max(2, min(6, int(args.H))),
        "rl_policy_train_end_round": max(2, int(args.H)),
    }


def _scenarios(args) -> list[Scenario]:
    clean = [Scenario("clean", _attack_type("clean"), "clean", {"num_attackers": 0}, int(args.seed))]
    poisoning = [
        Scenario("clean", _attack_type("clean"), "clean", {"num_attackers": 0}, int(args.seed)),
        Scenario("ipm", _attack_type("ipm"), "ipm", {"ipm_scaling": 2.0}, int(args.seed)),
        Scenario("lmp", _attack_type("lmp"), "lmp", {"lmp_scale": 2.0}, int(args.seed)),
        Scenario("rl", _attack_type("rl"), "rl", _rl_patch(args), int(args.rl_seed)),
    ]
    backdoor = _backdoor_patch(args)
    backdoor_scenarios = [
        Scenario("clean", _attack_type("clean"), "clean", {"num_attackers": 0}, int(args.seed)),
        Scenario("bfl", _attack_type("bfl"), "bfl", dict(backdoor), int(args.seed)),
        Scenario("dba", _attack_type("dba"), "dba", dict(backdoor), int(args.seed)),
        Scenario("rl_backdoor", _attack_type("rl_backdoor"), "rl_backdoor", dict(backdoor), int(args.rl_seed)),
    ]
    if args.scenario_set == "backdoor":
        return backdoor_scenarios
    if args.scenario_set == "mixed":
        return [
            *clean,
            *poisoning[1:],
            *backdoor_scenarios[1:],
        ]
    return poisoning


def _attack_context_names(args) -> tuple[str, ...]:
    if not bool(getattr(args, "attack_context", False)):
        return ()
    if args.scenario_set == "backdoor":
        return ("bfl", "dba", "rl_backdoor")
    if args.scenario_set == "mixed":
        return ("ipm", "lmp", "rl", "bfl", "dba", "rl_backdoor")
    return ("ipm", "lmp", "rl")


def _defender_action_dim(args) -> int:
    return 4 if str(args.defender_third_action) == "both" else 3


def _attack_type(name: str) -> AttackType:
    return AttackType(
        name=name,
        objective="targeted" if name in {"bfl", "dba", "rl_backdoor", "brl"} else "untargeted",
        adaptive=(name in {"rl", "brl"}),
    )


def _checkpoint_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "defender_meta.pt"
    return str(candidate)


def _resolve_device(device: str) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


if __name__ == "__main__":
    main()
