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
from meta_sg.learning.residual_adapter import load_residual_adapter, predict_residual_offset
from meta_sg.learning.replay_buffer import ReplayBuffer
from meta_sg.learning.sac import SACAgent
from meta_sg.learning.td3 import TD3Agent
from meta_sg.simulation.fl_sandbox_adapter import FLSandboxCoordinatorAdapter, SandboxConfig
from meta_sg.strategies.defenses.paper import PaperDefenseStrategy
from meta_sg.strategies.types import AttackType, DefenseDecision
from src.defenses import apply_post_defense


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


class ScheduledActionOffsetPolicy(ActionOffsetPolicy):
    """Action offset policy that only applies inside a half-open round window."""

    def __init__(
        self,
        base_policy,
        offset: np.ndarray,
        *,
        start_round: int = 0,
        end_round: int | None = None,
    ) -> None:
        super().__init__(base_policy, offset)
        self.start_round = max(0, int(start_round))
        self.end_round = None if end_round is None else max(0, int(end_round))
        self._round = 0

    def reset(self) -> None:
        self._round = 0

    def get_action(self, obs, noise: float = 0.0) -> np.ndarray:
        base_action = np.asarray(self.base_policy.get_action(obs, noise=noise), dtype=np.float32)
        apply_offset = self._round >= self.start_round and (
            self.end_round is None or self._round < self.end_round
        )
        self._round += 1
        if not apply_offset:
            return np.clip(base_action, -1.0, 1.0).astype(np.float32)
        return np.clip(base_action + self.offset, -1.0, 1.0).astype(np.float32)


class PhysicalTargetPolicy:
    """Policy adapter that pins decoded alpha/beta to physical target values."""

    def __init__(
        self,
        base_policy,
        config: BSMGConfig,
        *,
        target_alpha: float | None = None,
        target_beta: float | None = None,
    ) -> None:
        self.base_policy = base_policy
        self.config = config
        self.target_alpha = None if target_alpha is None else float(target_alpha)
        self.target_beta = None if target_beta is None else float(target_beta)
        self.obs_dim = int(getattr(base_policy, "obs_dim"))
        self.act_dim = int(getattr(base_policy, "act_dim"))

    def get_action(self, obs, noise: float = 0.0) -> np.ndarray:
        raw = np.asarray(self.base_policy.get_action(obs, noise=noise), dtype=np.float32).copy()
        if self.target_alpha is not None:
            raw[0] = _raw_for_physical_target(
                self.target_alpha,
                min_value=float(self.config.alpha_min),
                max_value=float(self.config.alpha_max),
            )
        if self.target_beta is not None and raw.shape[0] >= 2:
            raw[1] = _raw_for_physical_target(
                self.target_beta,
                min_value=float(self.config.beta_min),
                max_value=float(self.config.beta_max),
            )
        return np.clip(raw, -1.0, 1.0).astype(np.float32)


class ScheduledPhysicalTargetPolicy(PhysicalTargetPolicy):
    """Physical target policy that only pins alpha/beta inside a round window."""

    def __init__(
        self,
        base_policy,
        config: BSMGConfig,
        *,
        target_alpha: float | None = None,
        target_beta: float | None = None,
        start_round: int = 0,
        end_round: int | None = None,
    ) -> None:
        super().__init__(
            base_policy,
            config,
            target_alpha=target_alpha,
            target_beta=target_beta,
        )
        self.start_round = max(0, int(start_round))
        self.end_round = None if end_round is None else max(0, int(end_round))
        self._round = 0

    def reset(self) -> None:
        self._round = 0

    def get_action(self, obs, noise: float = 0.0) -> np.ndarray:
        apply_target = self._round >= self.start_round and (
            self.end_round is None or self._round < self.end_round
        )
        self._round += 1
        if apply_target:
            return super().get_action(obs, noise=noise)
        return np.clip(self.base_policy.get_action(obs, noise=noise), -1.0, 1.0).astype(np.float32)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to defender_meta.pt or checkpoint directory.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    parser.add_argument(
        "--scenario-set",
        choices=["model_poisoning", "backdoor", "clean_mixed_backdoor", "mixed", "clean_global_backdoor_mixed"],
        default="model_poisoning",
    )
    parser.add_argument(
        "--scenario-filter",
        default=None,
        help="Comma-separated scenario names to evaluate after building the selected scenario set.",
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
    parser.add_argument(
        "--post-defense-mode",
        choices=["weight_copy", "model_aware_neuroclip", "model_aware_pruning"],
        default="weight_copy",
        help=(
            "Post-training evaluation path. weight_copy preserves the legacy "
            "PaperDefenseStrategy weight-value clipping; model_aware_neuroclip "
            "wraps the fl_sandbox model with activation-clamping NeuroClip; "
            "model_aware_pruning wraps it with activation-based pruning."
        ),
    )
    parser.add_argument(
        "--fixed-neuroclip-epsilon",
        type=float,
        default=None,
        help=(
            "Optional fixed NeuroClip epsilon for post-defense evaluation. "
            "Useful for overlaying model-aware NeuroClip on a server_lr checkpoint."
        ),
    )
    parser.add_argument("--neuroclip-eps-min", type=float, default=1.0)
    parser.add_argument("--neuroclip-eps-max", type=float, default=10.0)
    parser.add_argument("--fixed-pruning-mask-rate", type=float, default=None)
    parser.add_argument(
        "--neuroclip-log-scale",
        action="store_true",
        help="Decode learned NeuroClip epsilon actions logarithmically.",
    )
    parser.add_argument("--server-lr-min", type=float, default=0.0)
    parser.add_argument("--server-lr-max", type=float, default=1.0)
    parser.add_argument("--server-lr-penalty-weight", type=float, default=0.0)
    parser.add_argument("--few-shot", action="store_true", help="Run per-scenario adaptation from the loaded checkpoint.")
    parser.add_argument(
        "--few-shot-method",
        choices=[
            "td3",
            "sac",
            "action_offset",
            "cem_offset",
            "beta_offset",
            "axis_offset",
            "axis_rule_offset",
            "axis_rule_v2_offset",
            "physical_rule_target",
            "physical_target_selector",
            "paper_online_td3",
            "paper_online_backdoor_td3",
            "paper_online_proxy_td3",
            "residual_adapter",
            "trained_residual_adapter",
        ],
        default="td3",
    )
    parser.add_argument("--adaptation-horizon", type=int, default=5)
    parser.add_argument("--adaptation-episodes", type=int, default=2)
    parser.add_argument("--adaptation-updates", type=int, default=10)
    parser.add_argument("--adaptation-batch-size", type=int, default=None)
    parser.add_argument("--adaptation-warmup-steps", type=int, default=0)
    parser.add_argument("--adaptation-noise", type=float, default=0.05)
    parser.add_argument("--adaptation-lr-scale", type=float, default=0.25)
    parser.add_argument(
        "--attacker-source",
        choices=["native", "zero"],
        default="native",
        help=(
            "Attacker action source for direct evaluation. native lets the sandbox/client attack "
            "module provide attack behavior; zero explicitly passes a zero action for ablation."
        ),
    )
    parser.add_argument(
        "--adaptation-attacker-source",
        choices=["native", "zero"],
        default=None,
        help="Optional attacker action source override for adaptation rollouts. Defaults to --attacker-source.",
    )
    parser.add_argument("--paper-online-windows", type=int, default=10)
    parser.add_argument("--paper-online-window-horizon", type=int, default=20)
    parser.add_argument("--paper-online-updates-per-window", type=int, default=10)
    parser.add_argument(
        "--backdoor-reward-mode",
        choices=["environment", "clean_minus_asr", "clean_gated_asr"],
        default="clean_gated_asr",
        help="Replay-buffer reward used by paper_online_backdoor_td3.",
    )
    parser.add_argument("--backdoor-clean-floor", type=float, default=0.90)
    parser.add_argument("--backdoor-reward-lambda", type=float, default=1.0)
    parser.add_argument("--backdoor-clean-penalty", type=float, default=2.0)
    parser.add_argument(
        "--proxy-reward-mode",
        choices=["clean_update_anomaly", "clean_synthetic_trigger", "clean_proxy_backdoor"],
        default="clean_update_anomaly",
        help=(
            "Replay-buffer reward used by paper_online_proxy_td3. "
            "This path does not consume true backdoor_acc/ASR during adaptation."
        ),
    )
    parser.add_argument("--proxy-clean-weight", type=float, default=1.0)
    parser.add_argument("--proxy-update-anomaly-weight", type=float, default=0.2)
    parser.add_argument("--proxy-server-lr-weight", type=float, default=0.1)
    parser.add_argument("--proxy-synthetic-trigger-weight", type=float, default=1.0)
    parser.add_argument(
        "--proxy-synthetic-trigger-patterns",
        default="corner_square,opposite_corner,center_square",
        help="Comma-separated synthetic trigger patterns for proxy backdoor scoring.",
    )
    parser.add_argument(
        "--proxy-synthetic-trigger-targets",
        default="all",
        help="'all' or comma-separated target labels for synthetic trigger proxy scoring.",
    )
    parser.add_argument("--proxy-synthetic-trigger-max-batches", type=int, default=1)
    parser.add_argument("--proxy-synthetic-trigger-size", type=int, default=3)
    parser.add_argument("--proxy-synthetic-trigger-value", type=float, default=1.0)
    parser.add_argument("--proxy-synthetic-trigger-eval-interval", type=int, default=20)
    parser.add_argument(
        "--proxy-server-lr-offset-step",
        type=float,
        default=0.0,
        help=(
            "Raw-action step used by paper_online_proxy_td3 to deploy an additional "
            "negative server_lr residual during adaptation. Disabled when zero."
        ),
    )
    parser.add_argument(
        "--proxy-server-lr-offset-max-steps",
        type=int,
        default=0,
        help=(
            "Number of raw-action steps to subtract from the server_lr dimension in "
            "paper_online_proxy_td3. For a saturated +1 action, step=0.5 and max_steps=4 "
            "moves the deployed action to -1 after clipping."
        ),
    )
    parser.add_argument(
        "--continuous-window-gate",
        action="store_true",
        help="After each online window, roll back the adapted TD3 network if clean/ASR validation worsens.",
    )
    parser.add_argument("--continuous-window-clean-drop-tolerance", type=float, default=0.02)
    parser.add_argument("--continuous-window-asr-improvement-margin", type=float, default=0.0)
    parser.add_argument("--sac-alpha", type=float, default=0.2)
    parser.add_argument("--sac-conditioned-sigma", action="store_true")
    parser.add_argument("--offset-step", type=float, default=0.1)
    parser.add_argument("--beta-offset-step", type=float, default=0.25)
    parser.add_argument("--beta-offset-max-steps", type=int, default=2)
    parser.add_argument("--beta-offset-asr-reduction-margin", type=float, default=0.005)
    parser.add_argument("--beta-offset-clean-floor", type=float, default=None)
    parser.add_argument("--beta-offset-clean-drop-tolerance", type=float, default=None)
    parser.add_argument("--beta-offset-deployment-beta-ceiling", type=float, default=None)
    parser.add_argument("--beta-offset-start-round", type=int, default=0)
    parser.add_argument("--beta-offset-end-round", type=int, default=None)
    parser.add_argument("--axis-offset-step", type=float, default=0.25)
    parser.add_argument("--axis-offset-max-steps", type=int, default=1)
    parser.add_argument("--axis-offset-asr-reduction-margin", type=float, default=0.005)
    parser.add_argument("--axis-offset-pessimistic-asr-margin", type=float, default=0.0)
    parser.add_argument("--axis-offset-clean-floor", type=float, default=None)
    parser.add_argument("--axis-offset-clean-drop-tolerance", type=float, default=None)
    parser.add_argument("--axis-rule-beta-threshold", type=float, default=0.35)
    parser.add_argument("--axis-rule-low-beta-threshold", type=float, default=0.34)
    parser.add_argument("--physical-rule-low-beta-threshold", type=float, default=0.34)
    parser.add_argument("--physical-rule-beta-threshold", type=float, default=0.35)
    parser.add_argument("--physical-rule-low-alpha-target", type=float, default=0.10)
    parser.add_argument("--physical-rule-near-alpha-target", type=float, default=0.14)
    parser.add_argument("--physical-rule-beta-target", type=float, default=0.38)
    parser.add_argument("--physical-target-alpha-candidates", default="0.10,0.12,0.14")
    parser.add_argument("--physical-target-beta-target", type=float, default=0.38)
    parser.add_argument("--physical-target-asr-reduction-margin", type=float, default=0.005)
    parser.add_argument("--physical-target-clean-floor", type=float, default=0.92)
    parser.add_argument("--physical-target-clean-drop-tolerance", type=float, default=0.04)
    parser.add_argument("--physical-target-score-slack", type=float, default=0.0)
    parser.add_argument(
        "--physical-target-selection-mode",
        choices=["score", "deployment_clean_recovery"],
        default="score",
        help=(
            "Selection objective for physical_target_selector. "
            "'deployment_clean_recovery' is an offline diagnostic mode that validates candidates "
            "on the deployment seed before selecting."
        ),
    )
    parser.add_argument("--physical-target-deployment-clean-floor", type=float, default=None)
    parser.add_argument("--physical-target-deployment-clean-drop-tolerance", type=float, default=None)
    parser.add_argument("--physical-target-deployment-asr-ceiling", type=float, default=None)
    parser.add_argument("--physical-target-start-round", type=int, default=0)
    parser.add_argument("--physical-target-end-round", type=int, default=None)
    parser.add_argument(
        "--physical-target-end-round-candidates",
        default=None,
        help=(
            "Comma-separated end-round candidates for physical_target_selector, "
            "for example 'full,35,40'. Overrides --physical-target-end-round per candidate."
        ),
    )
    parser.add_argument("--cem-iterations", type=int, default=2)
    parser.add_argument("--cem-population", type=int, default=8)
    parser.add_argument("--cem-elites", type=int, default=3)
    parser.add_argument("--cem-init-sigma", type=float, default=0.15)
    parser.add_argument("--cem-min-sigma", type=float, default=0.03)
    parser.add_argument("--cem-offset-bound", type=float, default=0.5)
    parser.add_argument("--residual-bound", type=float, default=0.2)
    parser.add_argument("--residual-candidates", type=int, default=5)
    parser.add_argument("--residual-query-episodes", type=int, default=2)
    parser.add_argument("--residual-degradation-margin", type=float, default=0.001)
    parser.add_argument("--residual-objective-gate", choices=["all", "targeted", "untargeted"], default="all")
    parser.add_argument(
        "--residual-selection-metric",
        choices=["mean", "worst", "lcb"],
        default="mean",
        help="Query statistic used to select residual candidates.",
    )
    parser.add_argument(
        "--residual-lcb-std-weight",
        type=float,
        default=1.0,
        help="Standard-deviation penalty for --residual-selection-metric lcb.",
    )
    parser.add_argument("--residual-adapter-checkpoint", default=None)
    parser.add_argument("--residual-dump-supervision-jsonl", default=None)
    parser.add_argument(
        "--few-shot-selection",
        choices=["always", "guarded", "backdoor_guarded"],
        default="always",
        help=(
            "Whether to always deploy the adapted policy or accept it only after a validation rollout. "
            "backdoor_guarded requires clean accuracy to stay within tolerance and ASR to drop."
        ),
    )
    parser.add_argument("--selection-margin", type=float, default=0.0)
    parser.add_argument(
        "--selection-clean-drop-tolerance",
        type=float,
        default=0.0,
        help="Maximum validation clean-accuracy drop allowed by --few-shot-selection backdoor_guarded.",
    )
    parser.add_argument("--selection-horizon", type=int, default=None)
    parser.add_argument("--selection-seed-offset", type=int, default=20_000)
    parser.add_argument(
        "--selection-repeats",
        type=int,
        default=1,
        help="Number of validation seeds averaged by guarded few-shot selection.",
    )
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
        evaluator=None
        if str(args.post_defense_mode) == "model_aware_neuroclip"
        else getattr(probe, "evaluate_weights", None),
        post_training_evaluator=_post_training_evaluator_for_args(args, probe),
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


def _adaptation_attacker_source(args) -> str:
    return str(args.adaptation_attacker_source or args.attacker_source)


def _attacker_action_for_source(args, *, source: str | None = None):
    del args
    selected = str(source or "native")
    if selected == "native":
        return None
    if selected == "zero":
        return np.zeros(3, dtype=np.float32)
    raise ValueError(f"Unsupported attacker source: {selected}")


def _evaluate_scenario_at(args, defender: TD3Agent, scenario: Scenario, *, seed: int, horizon: int) -> dict:
    env = _make_env(args, scenario, seed=seed, horizon=int(horizon))
    obs = env.reset(seed=seed)
    if hasattr(defender, "reset"):
        defender.reset()
    initial_action = _action_diagnostics(defender, obs, _bsmg_config(args, horizon=int(args.H)))
    if hasattr(defender, "reset"):
        defender.reset()
    rewards_d: list[float] = []
    rewards_a: list[float] = []
    last_info = {}
    attacker_source = str(args.attacker_source)
    for _ in range(int(args.H)):
        defender_action = defender.get_action(obs, noise=0.0)
        attacker_action = _attacker_action_for_source(args, source=attacker_source)
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
        "attacker_source": attacker_source,
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
        "post_defense_mode": str(args.post_defense_mode),
        "fixed_neuroclip_epsilon": float(args.fixed_neuroclip_epsilon)
        if args.fixed_neuroclip_epsilon is not None
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
    if str(args.few_shot_method) == "cem_offset":
        return _cem_offset_adapt_and_evaluate(args, defender, scenario, probe_obs=probe_obs)
    if str(args.few_shot_method) == "beta_offset":
        return _beta_offset_adapt_and_evaluate(args, defender, scenario, probe_obs=probe_obs)
    if str(args.few_shot_method) == "axis_offset":
        return _axis_offset_adapt_and_evaluate(args, defender, scenario, probe_obs=probe_obs)
    if str(args.few_shot_method) == "axis_rule_offset":
        return _axis_rule_offset_adapt_and_evaluate(args, defender, scenario, probe_obs=probe_obs)
    if str(args.few_shot_method) == "axis_rule_v2_offset":
        return _axis_rule_v2_offset_adapt_and_evaluate(args, defender, scenario, probe_obs=probe_obs)
    if str(args.few_shot_method) == "physical_rule_target":
        return _physical_rule_target_adapt_and_evaluate(args, defender, scenario, probe_obs=probe_obs)
    if str(args.few_shot_method) == "physical_target_selector":
        return _physical_target_selector_adapt_and_evaluate(args, defender, scenario, probe_obs=probe_obs)
    if str(args.few_shot_method) == "paper_online_td3":
        return _paper_online_td3_adapt_and_evaluate(args, defender, scenario, probe_obs=probe_obs)
    if str(args.few_shot_method) == "paper_online_backdoor_td3":
        return _paper_online_td3_adapt_and_evaluate(
            args,
            defender,
            scenario,
            probe_obs=probe_obs,
            backdoor_aware=True,
        )
    if str(args.few_shot_method) == "paper_online_proxy_td3":
        return _paper_online_td3_adapt_and_evaluate(
            args,
            defender,
            scenario,
            probe_obs=probe_obs,
            proxy_reward=True,
        )
    if str(args.few_shot_method) == "residual_adapter":
        return _residual_adapter_adapt_and_evaluate(args, defender, scenario, probe_obs=probe_obs)
    if str(args.few_shot_method) == "trained_residual_adapter":
        return _trained_residual_adapter_adapt_and_evaluate(args, defender, scenario, probe_obs=probe_obs)

    if str(args.few_shot_method) == "sac":
        adapted = SACAgent.from_td3(
            defender,
            alpha=float(args.sac_alpha),
            conditioned_sigma=bool(args.sac_conditioned_sigma),
        )
    else:
        adapted = defender.clone()
    if float(args.adaptation_lr_scale) != 1.0:
        adapted.set_learning_rates(
            policy_lr=float(adapted.cfg.policy_lr) * float(args.adaptation_lr_scale),
            critic_lr=float(adapted.cfg.critic_lr) * float(args.adaptation_lr_scale),
        )
    if args.adaptation_batch_size is not None and hasattr(adapted, "cfg"):
        adapted.cfg.batch_size = max(1, int(args.adaptation_batch_size))

    bsmg_cfg = _bsmg_config(args, horizon=int(args.adaptation_horizon))
    start_action = _action_diagnostics(defender, probe_obs, bsmg_cfg)
    trace = [{"shot": 0, "updates": 0, "action": start_action}]

    buffer = ReplayBuffer(
        capacity=max(
            32,
            int(args.adaptation_horizon) * int(args.adaptation_episodes)
            + int(args.adaptation_warmup_steps),
        ),
        obs_dim=defender.obs_dim,
        act_dim=defender.act_dim,
    )
    update_losses: list[dict] = []
    warmup_steps = max(0, int(args.adaptation_warmup_steps))
    attacker_source = _adaptation_attacker_source(args)
    if warmup_steps:
        warmup_seed = int(scenario.seed) + 9_000
        env = _make_env(args, scenario, seed=warmup_seed, horizon=max(1, warmup_steps))
        obs = env.reset(seed=warmup_seed)
        for _ in range(warmup_steps):
            action = np.random.uniform(-1.0, 1.0, size=defender.act_dim).astype(np.float32)
            attacker_action = _attacker_action_for_source(args, source=attacker_source)
            next_obs, reward_d, _reward_a, done, _info = env.step(action, attacker_action)
            buffer.add(obs, action, reward_d, next_obs, done)
            obs = next_obs
            if done:
                obs = env.reset(seed=warmup_seed)
    for episode in range(int(args.adaptation_episodes)):
        seed = int(scenario.seed) + 10_000 + episode
        env = _make_env(args, scenario, seed=seed, horizon=int(args.adaptation_horizon))
        obs = env.reset(seed=seed)
        for _ in range(int(args.adaptation_horizon)):
            action = adapted.get_action(obs, noise=float(args.adaptation_noise))
            attacker_action = _attacker_action_for_source(args, source=attacker_source)
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
        validation_repeats = max(1, int(args.selection_repeats))
        validation_seeds = [
            int(scenario.seed) + int(args.selection_seed_offset) + repeat
            for repeat in range(validation_repeats)
        ]
        base_validation_records = [
            _evaluate_scenario_at(args, defender, scenario, seed=seed, horizon=validation_horizon)
            for seed in validation_seeds
        ]
        adapted_validation_records = [
            _evaluate_scenario_at(args, adapted, scenario, seed=seed, horizon=validation_horizon)
            for seed in validation_seeds
        ]
        base_validation = _validation_summary(base_validation_records)
        adapted_validation = _validation_summary(adapted_validation_records)
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
                "validation_seed": validation_seeds[0],
                "validation_seeds": validation_seeds,
                "validation_repeats": validation_repeats,
                "base_validation": base_validation,
                "adapted_validation": adapted_validation,
                "base_validation_records": base_validation_records,
                "adapted_validation_records": adapted_validation_records,
            }
        )
        if not selection["accepted"]:
            evaluation = _evaluate_scenario(args, defender, scenario)
    return {
        "method": str(args.few_shot_method),
        "episodes": int(args.adaptation_episodes),
        "horizon": int(args.adaptation_horizon),
        "updates_per_episode": int(args.adaptation_updates),
        "batch_size": int(adapted.cfg.batch_size) if hasattr(adapted, "cfg") else None,
        "attacker_source": attacker_source,
        "warmup_steps": warmup_steps,
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


def _backdoor_guarded_selection_decision(
    *,
    base_clean_acc: float,
    adapted_clean_acc: float,
    base_backdoor_acc: float,
    adapted_backdoor_acc: float,
    asr_reduction_margin: float,
    clean_drop_tolerance: float,
) -> dict:
    clean_drop = float(base_clean_acc) - float(adapted_clean_acc)
    asr_reduction = float(base_backdoor_acc) - float(adapted_backdoor_acc)
    clean_safe = bool(clean_drop <= float(clean_drop_tolerance))
    asr_safe = bool(asr_reduction > float(asr_reduction_margin))
    accepted = bool(clean_safe and asr_safe)
    return {
        "accepted": accepted,
        "selected": "adapted" if accepted else "base",
        "base_clean_acc": float(base_clean_acc),
        "adapted_clean_acc": float(adapted_clean_acc),
        "clean_drop": clean_drop,
        "clean_drop_tolerance": float(clean_drop_tolerance),
        "clean_safe": clean_safe,
        "base_backdoor_acc": float(base_backdoor_acc),
        "adapted_backdoor_acc": float(adapted_backdoor_acc),
        "asr_reduction": round(asr_reduction, 12),
        "asr_reduction_margin": float(asr_reduction_margin),
        "asr_safe": asr_safe,
    }


def _metric_from_info(info: dict, key: str, default: float = 0.0) -> float:
    value = info.get(key, default)
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return float(default)
    return metric if np.isfinite(metric) else float(default)


def _mean_record_value(records: list[dict], key: str) -> float:
    values = []
    for record in records:
        try:
            value = float(record.get(key, float("nan")))
        except (TypeError, ValueError):
            value = float("nan")
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def _validation_summary(records: list[dict]) -> dict:
    scenario = str(records[0].get("scenario", "")) if records else ""
    attack_type = str(records[0].get("attack_type", "")) if records else ""
    seeds = [int(record["seed"]) for record in records if "seed" in record]
    horizons = [int(record["horizon"]) for record in records if "horizon" in record]
    return {
        "scenario": scenario,
        "attack_type": attack_type,
        "seed": seeds[0] if seeds else None,
        "seeds": seeds,
        "horizon": horizons[0] if horizons else None,
        "num_records": len(records),
        "final_clean_acc": _mean_record_value(records, "final_clean_acc"),
        "final_backdoor_acc": _mean_record_value(records, "final_backdoor_acc"),
        "final_defense_score": _mean_record_value(records, "final_defense_score"),
        "final_defender_reward": _mean_record_value(records, "final_defender_reward"),
        "mean_defender_reward": _mean_record_value(records, "mean_defender_reward"),
        "mean_attacker_reward": _mean_record_value(records, "mean_attacker_reward"),
    }


def _backdoor_aware_replay_reward(
    info: dict,
    *,
    raw_reward: float,
    mode: str,
    clean_floor: float,
    lambda_bd: float,
    clean_penalty: float,
) -> float:
    if str(mode) == "environment":
        return float(raw_reward)
    clean_acc = _metric_from_info(info, "clean_acc")
    backdoor_acc = _metric_from_info(info, "backdoor_acc")
    if str(mode) == "clean_minus_asr":
        return float(clean_acc - float(lambda_bd) * backdoor_acc)
    clean_shortfall = max(0.0, float(clean_floor) - clean_acc)
    return float(-float(lambda_bd) * backdoor_acc - float(clean_penalty) * clean_shortfall)


def _proxy_update_anomaly_score(info: dict) -> float:
    norms = []
    for key in ("benign_update_norms", "malicious_update_norms"):
        for value in info.get(key, []) or []:
            try:
                metric = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(metric):
                norms.append(metric)
    if len(norms) < 2:
        return 0.0
    values = np.asarray(norms, dtype=np.float64)
    median = float(np.median(values))
    scale = max(abs(median), 1e-8)
    mad = float(np.median(np.abs(values - median))) / scale
    upper_tail = max(0.0, float(np.max(values)) / scale - 1.0)
    return float(mad + upper_tail)


def _proxy_replay_reward(
    info: dict,
    *,
    args,
    proxy_backdoor_acc: float | None = None,
) -> dict:
    clean_acc = _metric_from_info(info, "clean_acc")
    update_anomaly = _proxy_update_anomaly_score(info)
    defense_decision = info.get("defense_decision")
    server_lr = (
        float(defense_decision.server_lr)
        if defense_decision is not None and defense_decision.server_lr is not None
        else 0.0
    )
    server_lr_span = max(1e-8, float(args.server_lr_max) - float(args.server_lr_min))
    server_lr_fraction = max(0.0, (server_lr - float(args.server_lr_min)) / server_lr_span)
    proxy_bd = None if proxy_backdoor_acc is None else float(proxy_backdoor_acc)

    reward = float(args.proxy_clean_weight) * clean_acc
    reward -= float(args.proxy_update_anomaly_weight) * update_anomaly
    reward -= float(args.proxy_server_lr_weight) * server_lr_fraction * server_lr_fraction
    if str(args.proxy_reward_mode) in {"clean_synthetic_trigger", "clean_proxy_backdoor"} and proxy_bd is not None:
        reward -= float(args.proxy_synthetic_trigger_weight) * proxy_bd

    diagnostics = {
        "clean_acc": float(clean_acc),
        "update_anomaly_score": float(update_anomaly),
        "server_lr_fraction": float(server_lr_fraction),
        "proxy_reward": float(reward),
    }
    if proxy_bd is not None:
        diagnostics["proxy_backdoor_acc"] = float(proxy_bd)
    return {"reward": float(reward), "diagnostics": diagnostics}


def _server_lr_action_index(args, *, act_dim: int) -> int | None:
    third_action = str(args.defender_third_action)
    if third_action == "server_lr" and int(act_dim) >= 3:
        return 2
    if third_action == "both" and int(act_dim) >= 4:
        return 3
    return None


def _proxy_server_lr_offset(args, *, act_dim: int) -> np.ndarray:
    offset = np.zeros(int(act_dim), dtype=np.float32)
    index = _server_lr_action_index(args, act_dim=act_dim)
    if index is None:
        return offset
    magnitude = max(0.0, float(args.proxy_server_lr_offset_step)) * max(
        0, int(args.proxy_server_lr_offset_max_steps)
    )
    if magnitude <= 0.0:
        return offset
    offset[index] = -float(magnitude)
    return offset


def _parse_csv_items(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_proxy_target_classes(value: str, *, num_classes: int | None = None) -> list[int] | None:
    text = str(value).strip().lower()
    if text == "all":
        if num_classes is None:
            return None
        return list(range(int(num_classes)))
    return [int(item) for item in _parse_csv_items(value)]


def _apply_synthetic_trigger_batch(
    images: torch.Tensor,
    *,
    pattern: str,
    trigger_size: int,
    trigger_value: float,
) -> torch.Tensor:
    triggered = images.clone()
    if triggered.ndim != 4:
        raise ValueError("Synthetic trigger proxy expects images shaped [B, C, H, W].")
    size = max(1, min(int(trigger_size), int(triggered.shape[-1]), int(triggered.shape[-2])))
    pattern_name = str(pattern)
    if pattern_name == "corner_square":
        rows = slice(-size, None)
        cols = slice(-size, None)
    elif pattern_name == "opposite_corner":
        rows = slice(0, size)
        cols = slice(0, size)
    elif pattern_name == "center_square":
        row0 = max(0, (int(triggered.shape[-2]) - size) // 2)
        col0 = max(0, (int(triggered.shape[-1]) - size) // 2)
        rows = slice(row0, row0 + size)
        cols = slice(col0, col0 + size)
    else:
        raise ValueError(f"Unsupported synthetic trigger pattern: {pattern}")
    triggered[:, :, rows, cols] = float(trigger_value)
    return triggered


def _synthetic_trigger_proxy_backdoor_acc(
    model: torch.nn.Module,
    loader,
    *,
    device: torch.device,
    target_classes: list[int] | None,
    patterns: list[str],
    trigger_size: int,
    trigger_value: float,
    max_batches: int,
) -> dict:
    model.eval()
    best = {
        "proxy_backdoor_acc": 0.0,
        "target_class": None,
        "pattern": None,
        "num_examples": 0,
    }
    cached_batches = []
    with torch.no_grad():
        for batch_idx, (images, _labels) in enumerate(loader):
            if batch_idx >= max(1, int(max_batches)):
                break
            cached_batches.append(images.to(device))
    if not cached_batches:
        return best
    resolved_targets = target_classes
    if resolved_targets is None:
        with torch.no_grad():
            logits = model(cached_batches[0])
        resolved_targets = list(range(int(logits.shape[1])))
    with torch.no_grad():
        for pattern in patterns:
            for target in resolved_targets:
                correct = 0
                total = 0
                for images in cached_batches:
                    triggered = _apply_synthetic_trigger_batch(
                        images,
                        pattern=pattern,
                        trigger_size=int(trigger_size),
                        trigger_value=float(trigger_value),
                    )
                    preds = model(triggered).argmax(dim=1)
                    correct += int((preds == int(target)).sum().item())
                    total += int(preds.numel())
                acc = float(correct / total) if total else 0.0
                if acc > float(best["proxy_backdoor_acc"]):
                    best = {
                        "proxy_backdoor_acc": acc,
                        "target_class": int(target),
                        "pattern": str(pattern),
                        "num_examples": int(total),
                    }
    return best


def _proxy_synthetic_trigger_for_env(args, env: BSMGEnv, info: dict) -> dict:
    coordinator = getattr(env, "coordinator", None)
    runner = getattr(coordinator, "runner", None)
    if runner is None or not hasattr(runner, "test_loader"):
        return {"proxy_backdoor_acc": None}
    model = getattr(runner, "model", None)
    if model is None:
        return {"proxy_backdoor_acc": None}
    decision = info.get("defense_decision")
    eval_model = model
    if str(args.post_defense_mode) == "model_aware_neuroclip" and decision is not None:
        epsilon = _neuroclip_epsilon_for_post_eval(args, decision)
        if epsilon is not None:
            eval_model = apply_post_defense(model, "neuroclip", float(epsilon))
    elif str(args.post_defense_mode) == "model_aware_pruning" and decision is not None:
        mask_rate = getattr(args, "fixed_pruning_mask_rate", None)
        if mask_rate is not None:
            eval_model = apply_post_defense(
                model,
                "pruning",
                float(mask_rate),
                eval_loader=getattr(runner, "test_loader", None),
                device=getattr(runner, "device", None),
            )
    patterns = _parse_csv_items(args.proxy_synthetic_trigger_patterns)
    targets = _parse_proxy_target_classes(args.proxy_synthetic_trigger_targets)
    return _synthetic_trigger_proxy_backdoor_acc(
        eval_model,
        runner.test_loader,
        device=getattr(runner, "device", _resolve_device(args.device)),
        target_classes=targets,
        patterns=patterns,
        trigger_size=int(args.proxy_synthetic_trigger_size),
        trigger_value=float(args.proxy_synthetic_trigger_value),
        max_batches=int(args.proxy_synthetic_trigger_max_batches),
    )


def _should_refresh_proxy_trigger(args, round_idx: int, *, force: bool = False) -> bool:
    if str(args.proxy_reward_mode) not in {"clean_synthetic_trigger", "clean_proxy_backdoor"}:
        return False
    if force:
        return True
    interval = max(1, int(args.proxy_synthetic_trigger_eval_interval))
    return int(round_idx) == 1 or int(round_idx) % interval == 0


def _coerce_optional_float(value) -> float | None:
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return None
    return metric if np.isfinite(metric) else None


def _mean_optional_metric(records: list[dict], key: str) -> float:
    values = []
    for record in records:
        value = _coerce_optional_float(record.get(key))
        if value is not None:
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def _continuous_backdoor_window_gate_decision(
    *,
    previous_clean_acc: float,
    current_clean_acc: float,
    previous_backdoor_acc: float,
    current_backdoor_acc: float,
    clean_drop_tolerance: float,
    asr_improvement_margin: float,
) -> dict:
    clean_drop = float(previous_clean_acc) - float(current_clean_acc)
    asr_reduction = float(previous_backdoor_acc) - float(current_backdoor_acc)
    clean_safe = bool(clean_drop <= float(clean_drop_tolerance))
    asr_safe = bool(asr_reduction > float(asr_improvement_margin))
    accepted = bool(clean_safe and asr_safe)
    return {
        "accepted": accepted,
        "selected": "current" if accepted else "previous",
        "previous_clean_acc": float(previous_clean_acc),
        "current_clean_acc": float(current_clean_acc),
        "clean_drop": clean_drop,
        "clean_drop_tolerance": float(clean_drop_tolerance),
        "clean_safe": clean_safe,
        "previous_backdoor_acc": float(previous_backdoor_acc),
        "current_backdoor_acc": float(current_backdoor_acc),
        "asr_reduction": round(asr_reduction, 12),
        "asr_improvement_margin": float(asr_improvement_margin),
        "asr_safe": asr_safe,
    }


def _paper_online_td3_adapt_and_evaluate(
    args,
    defender: TD3Agent,
    scenario: Scenario,
    *,
    probe_obs: np.ndarray,
    backdoor_aware: bool = False,
    proxy_reward: bool = False,
) -> dict:
    adapted = defender.clone()
    if float(args.adaptation_lr_scale) != 1.0:
        adapted.set_learning_rates(
            policy_lr=float(adapted.cfg.policy_lr) * float(args.adaptation_lr_scale),
            critic_lr=float(adapted.cfg.critic_lr) * float(args.adaptation_lr_scale),
        )
    if args.adaptation_batch_size is not None:
        adapted.cfg.batch_size = max(1, int(args.adaptation_batch_size))

    windows = max(1, int(args.paper_online_windows))
    window_horizon = max(1, int(args.paper_online_window_horizon))
    updates_per_window = max(0, int(args.paper_online_updates_per_window))
    total_horizon = windows * window_horizon
    warmup_steps = max(0, int(args.adaptation_warmup_steps))
    buffer = ReplayBuffer(
        capacity=max(32, total_horizon + warmup_steps),
        obs_dim=defender.obs_dim,
        act_dim=defender.act_dim,
    )
    bsmg_cfg = _bsmg_config(args, horizon=total_horizon)
    start_action = _action_diagnostics(defender, probe_obs, bsmg_cfg)
    trace = [{"window": 0, "round": 0, "updates": 0, "action": start_action}]
    update_losses: list[dict] = []
    window_records: list[dict] = []
    accepted_policy = adapted.clone()
    accepted_clean_acc = None
    accepted_backdoor_acc = None
    attacker_source = _adaptation_attacker_source(args)
    latest_proxy_backdoor_acc: float | None = None
    latest_proxy_diagnostics: dict = {}
    proxy_server_lr_offset = (
        _proxy_server_lr_offset(args, act_dim=defender.act_dim)
        if proxy_reward
        else np.zeros(int(defender.act_dim), dtype=np.float32)
    )
    uses_proxy_server_lr_offset = bool(proxy_reward and not np.allclose(proxy_server_lr_offset, 0.0))

    def deployed_policy(policy):
        return ActionOffsetPolicy(policy, proxy_server_lr_offset) if uses_proxy_server_lr_offset else policy

    if warmup_steps:
        warmup_seed = int(scenario.seed) + 29_000
        env = _make_env(args, scenario, seed=warmup_seed, horizon=max(1, warmup_steps))
        obs = env.reset(seed=warmup_seed)
        for warmup_step in range(warmup_steps):
            action = np.random.uniform(-1.0, 1.0, size=defender.act_dim).astype(np.float32)
            attacker_action = _attacker_action_for_source(args, source=attacker_source)
            next_obs, reward_d, _reward_a, done, info = env.step(action, attacker_action)
            if proxy_reward:
                if _should_refresh_proxy_trigger(args, warmup_step + 1, force=done):
                    latest_proxy_diagnostics = _proxy_synthetic_trigger_for_env(args, env, dict(info))
                    latest_proxy_backdoor_acc = _coerce_optional_float(
                        latest_proxy_diagnostics.get("proxy_backdoor_acc")
                    )
                proxy_result = _proxy_replay_reward(
                    dict(info),
                    args=args,
                    proxy_backdoor_acc=latest_proxy_backdoor_acc,
                )
                replay_reward = float(proxy_result["reward"])
            elif backdoor_aware:
                replay_reward = _backdoor_aware_replay_reward(
                    dict(info),
                    raw_reward=float(reward_d),
                    mode=str(args.backdoor_reward_mode),
                    clean_floor=float(args.backdoor_clean_floor),
                    lambda_bd=float(args.backdoor_reward_lambda),
                    clean_penalty=float(args.backdoor_clean_penalty),
                )
            else:
                replay_reward = float(reward_d)
            buffer.add(obs, action, replay_reward, next_obs, done)
            obs = next_obs
            if done:
                break

    seed = int(scenario.seed) + 30_000
    env = _make_env(args, scenario, seed=seed, horizon=total_horizon)
    obs = env.reset(seed=seed)
    if hasattr(adapted, "reset"):
        adapted.reset()
    global_round = 0
    done = False
    for window in range(windows):
        rewards: list[float] = []
        env_rewards: list[float] = []
        proxy_records: list[dict] = []
        last_info: dict = {}
        for step_in_window in range(window_horizon):
            action = deployed_policy(adapted).get_action(obs, noise=float(args.adaptation_noise))
            attacker_action = _attacker_action_for_source(args, source=attacker_source)
            next_obs, reward_d, _reward_a, done, info = env.step(action, attacker_action)
            last_info = dict(info)
            if proxy_reward:
                round_for_proxy = global_round + 1
                should_refresh = _should_refresh_proxy_trigger(
                    args,
                    round_for_proxy,
                    force=done or step_in_window == window_horizon - 1,
                )
                if should_refresh:
                    latest_proxy_diagnostics = _proxy_synthetic_trigger_for_env(args, env, last_info)
                    latest_proxy_backdoor_acc = _coerce_optional_float(
                        latest_proxy_diagnostics.get("proxy_backdoor_acc")
                    )
                proxy_result = _proxy_replay_reward(
                    last_info,
                    args=args,
                    proxy_backdoor_acc=latest_proxy_backdoor_acc,
                )
                proxy_diagnostics = dict(latest_proxy_diagnostics)
                proxy_diagnostics.update(proxy_result["diagnostics"])
                replay_reward = float(proxy_result["reward"])
                proxy_records.append(proxy_diagnostics)
            elif backdoor_aware:
                replay_reward = _backdoor_aware_replay_reward(
                    last_info,
                    raw_reward=float(reward_d),
                    mode=str(args.backdoor_reward_mode),
                    clean_floor=float(args.backdoor_clean_floor),
                    lambda_bd=float(args.backdoor_reward_lambda),
                    clean_penalty=float(args.backdoor_clean_penalty),
                )
            else:
                replay_reward = float(reward_d)
            buffer.add(obs, action, replay_reward, next_obs, done)
            rewards.append(float(replay_reward))
            env_rewards.append(float(reward_d))
            obs = next_obs
            global_round += 1
            if done:
                break
        for _ in range(updates_per_window):
            losses = adapted.update(buffer)
            if losses:
                update_losses.append({k: float(v) for k, v in losses.items()})
        current_clean_acc = _metric_from_info(last_info, "clean_acc", default=float("nan"))
        current_backdoor_acc = _metric_from_info(last_info, "backdoor_acc", default=float("nan"))
        window_gate = {
            "enabled": bool(backdoor_aware and args.continuous_window_gate),
            "accepted": True,
            "selected": "current",
        }
        if backdoor_aware and bool(args.continuous_window_gate) and np.isfinite(current_clean_acc):
            if accepted_clean_acc is None or not np.isfinite(float(accepted_clean_acc)):
                accepted_clean_acc = current_clean_acc
                accepted_backdoor_acc = current_backdoor_acc
                accepted_policy = adapted.clone()
            else:
                window_gate = _continuous_backdoor_window_gate_decision(
                    previous_clean_acc=float(accepted_clean_acc),
                    current_clean_acc=float(current_clean_acc),
                    previous_backdoor_acc=float(accepted_backdoor_acc),
                    current_backdoor_acc=float(current_backdoor_acc),
                    clean_drop_tolerance=float(args.continuous_window_clean_drop_tolerance),
                    asr_improvement_margin=float(args.continuous_window_asr_improvement_margin),
                )
                window_gate["enabled"] = True
                if bool(window_gate["accepted"]):
                    accepted_clean_acc = current_clean_acc
                    accepted_backdoor_acc = current_backdoor_acc
                    accepted_policy = adapted.clone()
                else:
                    adapted = accepted_policy.clone()
        trace.append(
            {
                "window": window + 1,
                "round": global_round,
                "updates": len(update_losses),
                "action": _action_diagnostics(deployed_policy(adapted), probe_obs, bsmg_cfg),
                "actor_action": _action_diagnostics(adapted, probe_obs, bsmg_cfg)
                if uses_proxy_server_lr_offset
                else None,
            }
        )
        defense_decision = last_info.get("defense_decision")
        window_records.append(
            {
                "window": window + 1,
                "round": global_round,
                "mean_reward": float(np.mean(rewards)) if rewards else float("nan"),
                "mean_environment_reward": float(np.mean(env_rewards)) if env_rewards else float("nan"),
                "clean_acc": float(last_info.get("clean_acc", float("nan"))),
                "backdoor_acc": float(last_info.get("backdoor_acc", float("nan"))),
                "window_gate": window_gate,
                "alpha": float(defense_decision.norm_bound_alpha)
                if defense_decision is not None
                else float("nan"),
                "beta": float(defense_decision.trimmed_mean_beta)
                if defense_decision is not None
                else float("nan"),
                "server_lr": float(defense_decision.server_lr)
                if defense_decision is not None and defense_decision.server_lr is not None
                else float("nan"),
                "proxy_reward": float(np.mean([item["proxy_reward"] for item in proxy_records]))
                if proxy_records
                else float("nan"),
                "proxy_update_anomaly_score": float(
                    np.mean([item["update_anomaly_score"] for item in proxy_records])
                )
                if proxy_records
                else float("nan"),
                "proxy_backdoor_acc": _mean_optional_metric(proxy_records, "proxy_backdoor_acc"),
            }
        )
        if done:
            break

    adapted_policy = deployed_policy(adapted)
    adapted_evaluation = _evaluate_scenario(args, adapted_policy, scenario)
    selection = {
        "mode": str(args.few_shot_selection),
        "accepted": True,
        "selected": "adapted",
        "margin": float(args.selection_margin),
    }
    evaluation = adapted_evaluation
    if str(args.few_shot_selection) in {"guarded", "backdoor_guarded"}:
        validation_horizon = int(args.selection_horizon or args.H)
        validation_repeats = max(1, int(args.selection_repeats))
        validation_seeds = [
            int(scenario.seed) + int(args.selection_seed_offset) + repeat
            for repeat in range(validation_repeats)
        ]
        base_validation_records = [
            _evaluate_scenario_at(args, defender, scenario, seed=seed, horizon=validation_horizon)
            for seed in validation_seeds
        ]
        adapted_validation_records = [
            _evaluate_scenario_at(args, adapted_policy, scenario, seed=seed, horizon=validation_horizon)
            for seed in validation_seeds
        ]
        base_validation = _validation_summary(base_validation_records)
        adapted_validation = _validation_summary(adapted_validation_records)
        if str(args.few_shot_selection) == "backdoor_guarded":
            selection = _backdoor_guarded_selection_decision(
                base_clean_acc=float(base_validation["final_clean_acc"]),
                adapted_clean_acc=float(adapted_validation["final_clean_acc"]),
                base_backdoor_acc=float(base_validation["final_backdoor_acc"]),
                adapted_backdoor_acc=float(adapted_validation["final_backdoor_acc"]),
                asr_reduction_margin=float(args.selection_margin),
                clean_drop_tolerance=float(args.selection_clean_drop_tolerance),
            )
        else:
            selection = _guarded_selection_decision(
                base_score=float(base_validation["final_defense_score"]),
                adapted_score=float(adapted_validation["final_defense_score"]),
                margin=float(args.selection_margin),
            )
        selection.update(
            {
                "mode": str(args.few_shot_selection),
                "margin": float(args.selection_margin),
                "validation_horizon": validation_horizon,
                "validation_seed": validation_seeds[0],
                "validation_seeds": validation_seeds,
                "validation_repeats": validation_repeats,
                "base_validation": base_validation,
                "adapted_validation": adapted_validation,
                "base_validation_records": base_validation_records,
                "adapted_validation_records": adapted_validation_records,
            }
        )
        if not selection["accepted"]:
            evaluation = _evaluate_scenario(args, defender, scenario)

    return {
        "method": (
            "paper_online_backdoor_td3"
            if backdoor_aware
            else "paper_online_proxy_td3"
            if proxy_reward
            else "paper_online_td3"
        ),
        "paper_style": {
            "reward_source": "backdoor_aware_clean_asr_reward"
            if backdoor_aware
            else "paper_like_proxy_reward"
            if proxy_reward
            else "sandbox_oracle_metric_reward",
            "backdoor_reward_mode": str(args.backdoor_reward_mode) if backdoor_aware else "environment",
            "backdoor_clean_floor": float(args.backdoor_clean_floor) if backdoor_aware else None,
            "backdoor_reward_lambda": float(args.backdoor_reward_lambda) if backdoor_aware else None,
            "backdoor_clean_penalty": float(args.backdoor_clean_penalty) if backdoor_aware else None,
            "proxy_reward_mode": str(args.proxy_reward_mode) if proxy_reward else None,
            "proxy_uses_true_asr": False if proxy_reward else None,
            "proxy_synthetic_trigger_patterns": _parse_csv_items(args.proxy_synthetic_trigger_patterns)
            if proxy_reward
            else None,
            "proxy_synthetic_trigger_targets": str(args.proxy_synthetic_trigger_targets)
            if proxy_reward
            else None,
            "proxy_server_lr_offset_active": uses_proxy_server_lr_offset if proxy_reward else None,
            "proxy_server_lr_offset": [float(v) for v in proxy_server_lr_offset]
            if proxy_reward
            else None,
            "proxy_server_lr_offset_step": float(args.proxy_server_lr_offset_step)
            if proxy_reward
            else None,
            "proxy_server_lr_offset_max_steps": int(args.proxy_server_lr_offset_max_steps)
            if proxy_reward
            else None,
            "attacker_source": attacker_source,
            "online_windows": windows,
            "window_horizon": window_horizon,
            "updates_per_window": updates_per_window,
            "total_online_horizon": total_horizon,
            "continuous_trajectory": True,
            "continuous_window_gate": bool(backdoor_aware and args.continuous_window_gate),
            "action_space": str(args.defender_third_action),
            "server_lr_available": bool(str(args.defender_third_action) in {"server_lr", "both"}),
            "action_space_warning": None
            if str(args.defender_third_action) in {"server_lr", "both"}
            else (
                "server_lr is unavailable for this checkpoint action space; "
                "use a server_lr or 4D both checkpoint to adapt learning-rate control."
            ),
        },
        "episodes": windows,
        "horizon": window_horizon,
        "updates_per_episode": updates_per_window,
        "attacker_source": attacker_source,
        "batch_size": int(adapted.cfg.batch_size),
        "warmup_steps": warmup_steps,
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
        "window_records": window_records,
        "num_transitions": len(buffer),
        "num_updates": len(update_losses),
        "last_update_loss": update_losses[-1] if update_losses else {},
        "selection": selection,
        "adapted_evaluation": adapted_evaluation,
        "evaluation": evaluation,
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


def _beta_offset_adapt_and_evaluate(
    args,
    defender,
    scenario: Scenario,
    *,
    probe_obs: np.ndarray,
) -> dict:
    bsmg_cfg = _bsmg_config(args, horizon=int(args.selection_horizon or args.H))
    start_action = _action_diagnostics(defender, probe_obs, bsmg_cfg)
    zero_offset = np.zeros(int(defender.act_dim), dtype=np.float32)
    if str(scenario.attack_type.objective) != "targeted":
        evaluation = _evaluate_scenario(args, defender, scenario)
        return {
            "method": "beta_offset",
            "episodes": 0,
            "horizon": int(args.selection_horizon or args.H),
            "updates_per_episode": 0,
            "noise": 0.0,
            "lr_scale": 0.0,
            "beta_offset_step": float(args.beta_offset_step),
            "beta_offset_max_steps": int(args.beta_offset_max_steps),
            "candidate_scores": [],
            "selected_offset": [float(v) for v in zero_offset],
            "probe_observation": {
                "scenario": "clean",
                "seed": int(args.seed),
                "description": "shared clean initial observation used only for action-space diagnostics",
            },
            "transition": {
                "from": start_action,
                "to": start_action,
                "delta": _action_delta(start_action, start_action),
            },
            "trace": [{"shot": 0, "updates": 0, "action": start_action}],
            "num_transitions": 0,
            "num_updates": 0,
            "last_update_loss": {},
            "selection": {
                "mode": "objective_gated_beta_offset",
                "accepted": False,
                "selected": "base",
                "skip_reason": "scenario objective is outside beta offset gate",
                "scenario_objective": str(scenario.attack_type.objective),
            },
            "adapted_evaluation": evaluation,
            "evaluation": evaluation,
        }

    deployment_base_action = _scenario_initial_action_diagnostics(
        args,
        defender,
        scenario,
        horizon=int(args.H),
    )
    if _beta_offset_deployment_beta_blocked(
        deployment_base_action,
        beta_ceiling=args.beta_offset_deployment_beta_ceiling,
    ):
        evaluation = _evaluate_scenario(args, defender, scenario)
        return {
            "method": "beta_offset",
            "episodes": 0,
            "horizon": int(args.selection_horizon or args.H),
            "updates_per_episode": 0,
            "noise": 0.0,
            "lr_scale": 0.0,
            "beta_offset_step": float(args.beta_offset_step),
            "beta_offset_max_steps": int(args.beta_offset_max_steps),
            "candidate_scores": [],
            "selected_offset": [float(v) for v in zero_offset],
            "deployed_offset": [float(v) for v in zero_offset],
            "deployment_base_action": deployment_base_action,
            "probe_observation": {
                "scenario": "clean",
                "seed": int(args.seed),
                "description": "shared clean initial observation used only for action-space diagnostics",
            },
            "transition": {
                "from": start_action,
                "to": start_action,
                "delta": _action_delta(start_action, start_action),
            },
            "trace": [{"shot": 0, "updates": 0, "action": start_action}],
            "num_transitions": 0,
            "num_updates": 0,
            "last_update_loss": {},
            "selection": {
                "mode": "deployment_beta_ceiling_beta_offset",
                "accepted": False,
                "selected": "base",
                "skip_reason": "deployment initial beta is already at or above beta offset ceiling",
                "deployment_beta": float(deployment_base_action.get("beta", float("nan"))),
                "deployment_beta_ceiling": float(args.beta_offset_deployment_beta_ceiling),
            },
            "adapted_evaluation": evaluation,
            "evaluation": evaluation,
        }

    candidates = _beta_offset_candidates(
        act_dim=int(defender.act_dim),
        step=float(args.beta_offset_step),
        max_steps=int(args.beta_offset_max_steps),
    )
    query_horizon = int(args.selection_horizon or args.H)
    query_episodes = max(1, int(args.adaptation_episodes))
    candidate_records = []
    for idx, (label, offset) in enumerate(candidates):
        policy = _scheduled_offset_policy_for_candidate(args, defender, offset)
        query_records = []
        for episode in range(query_episodes):
            seed = int(scenario.seed) + int(args.selection_seed_offset) + episode
            query_records.append(
                _evaluate_scenario_at(args, policy, scenario, seed=seed, horizon=query_horizon)
            )
        candidate_records.append(
            {
                "index": int(idx),
                "offset_label": str(label),
                "offset": [float(v) for v in offset],
                "query_records": query_records,
                **_beta_offset_query_summary(query_records),
            }
        )

    selected_record, selection = _select_beta_offset_from_query_records(
        candidate_records,
        asr_reduction_margin=float(args.beta_offset_asr_reduction_margin),
        clean_floor=args.beta_offset_clean_floor,
        clean_drop_tolerance=args.beta_offset_clean_drop_tolerance,
    )
    selected_offset = np.asarray(selected_record["offset"], dtype=np.float32)
    selected_policy = _scheduled_offset_policy_for_candidate(args, defender, selected_offset)
    evaluation = _evaluate_scenario(args, selected_policy if selection["accepted"] else defender, scenario)
    end_action = _action_diagnostics(selected_policy, probe_obs, bsmg_cfg)
    selection.update(
        {
            "mode": "query_guarded_beta_offset",
            "query_horizon": query_horizon,
            "query_episodes": query_episodes,
            "asr_reduction_margin": float(args.beta_offset_asr_reduction_margin),
            "clean_floor": args.beta_offset_clean_floor,
            "clean_drop_tolerance": args.beta_offset_clean_drop_tolerance,
            "selected_candidate": selected_record,
        }
    )
    return {
        "method": "beta_offset",
        "episodes": query_episodes,
        "horizon": query_horizon,
        "updates_per_episode": 0,
        "noise": 0.0,
        "lr_scale": 0.0,
        "beta_offset_step": float(args.beta_offset_step),
        "beta_offset_max_steps": int(args.beta_offset_max_steps),
        "beta_offset_start_round": int(args.beta_offset_start_round),
        "beta_offset_end_round": args.beta_offset_end_round,
        "candidate_scores": candidate_records,
        "selected_offset": [float(v) for v in selected_offset],
        "deployed_offset": [float(v) for v in selected_offset] if selection["accepted"] else [float(v) for v in zero_offset],
        "deployment_base_action": deployment_base_action,
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
            {"shot": query_episodes, "updates": 0, "action": end_action},
        ],
        "num_transitions": query_horizon * query_episodes * len(candidates),
        "num_updates": 0,
        "last_update_loss": {},
        "selection": selection,
        "adapted_evaluation": evaluation,
        "evaluation": evaluation,
    }


def _axis_offset_adapt_and_evaluate(
    args,
    defender,
    scenario: Scenario,
    *,
    probe_obs: np.ndarray,
) -> dict:
    bsmg_cfg = _bsmg_config(args, horizon=int(args.selection_horizon or args.H))
    start_action = _action_diagnostics(defender, probe_obs, bsmg_cfg)
    zero_offset = np.zeros(int(defender.act_dim), dtype=np.float32)
    if str(scenario.attack_type.objective) != "targeted":
        evaluation = _evaluate_scenario(args, defender, scenario)
        return {
            "method": "axis_offset",
            "episodes": 0,
            "horizon": int(args.selection_horizon or args.H),
            "updates_per_episode": 0,
            "noise": 0.0,
            "lr_scale": 0.0,
            "axis_offset_step": float(args.axis_offset_step),
            "axis_offset_max_steps": int(args.axis_offset_max_steps),
            "candidate_scores": [],
            "selected_offset": [float(v) for v in zero_offset],
            "deployed_offset": [float(v) for v in zero_offset],
            "probe_observation": {
                "scenario": "clean",
                "seed": int(args.seed),
                "description": "shared clean initial observation used only for action-space diagnostics",
            },
            "transition": {
                "from": start_action,
                "to": start_action,
                "delta": _action_delta(start_action, start_action),
            },
            "trace": [{"shot": 0, "updates": 0, "action": start_action}],
            "num_transitions": 0,
            "num_updates": 0,
            "last_update_loss": {},
            "selection": {
                "mode": "objective_gated_axis_offset",
                "accepted": False,
                "selected": "base",
                "skip_reason": "scenario objective is outside axis offset gate",
                "scenario_objective": str(scenario.attack_type.objective),
            },
            "adapted_evaluation": evaluation,
            "evaluation": evaluation,
        }

    candidates = _axis_offset_candidates(
        act_dim=int(defender.act_dim),
        step=float(args.axis_offset_step),
        max_steps=int(args.axis_offset_max_steps),
    )
    query_horizon = int(args.selection_horizon or args.H)
    query_episodes = max(1, int(args.adaptation_episodes))
    candidate_records = []
    base_query_records: list[dict] = []
    evaluated_query_records = 0
    for idx, (label, offset) in enumerate(candidates):
        policy = defender if np.allclose(offset, 0.0) else ActionOffsetPolicy(defender, offset)
        query_records = []
        pruned = False
        prune_reason = None
        for episode in range(query_episodes):
            seed = int(scenario.seed) + int(args.selection_seed_offset) + episode
            query_records.append(
                _evaluate_scenario_at(args, policy, scenario, seed=seed, horizon=query_horizon)
            )
            evaluated_query_records += 1
            if not np.allclose(offset, 0.0):
                prune_reason = _axis_offset_prune_reason(
                    base_query_records,
                    query_records,
                    pessimistic_asr_margin=float(args.axis_offset_pessimistic_asr_margin),
                )
                if prune_reason is not None:
                    pruned = True
                    break
        if np.allclose(offset, 0.0):
            base_query_records = list(query_records)
        candidate_records.append(
            {
                "index": int(idx),
                "offset_label": str(label),
                "offset": [float(v) for v in offset],
                "query_records": query_records,
                "query_episodes_completed": len(query_records),
                "query_episodes_requested": query_episodes,
                "pruned": bool(pruned),
                "prune_reason": prune_reason,
                **_beta_offset_query_summary(query_records),
            }
        )

    selected_record, selection = _select_axis_offset_from_query_records(
        candidate_records,
        asr_reduction_margin=float(args.axis_offset_asr_reduction_margin),
        pessimistic_asr_margin=float(args.axis_offset_pessimistic_asr_margin),
        clean_floor=args.axis_offset_clean_floor,
        clean_drop_tolerance=args.axis_offset_clean_drop_tolerance,
    )
    selected_offset = np.asarray(selected_record["offset"], dtype=np.float32)
    selected_policy = defender if np.allclose(selected_offset, 0.0) else ActionOffsetPolicy(defender, selected_offset)
    evaluation = _evaluate_scenario(args, selected_policy if selection["accepted"] else defender, scenario)
    end_action = _action_diagnostics(selected_policy, probe_obs, bsmg_cfg)
    selection.update(
        {
            "mode": "query_guarded_axis_offset",
            "query_horizon": query_horizon,
            "query_episodes": query_episodes,
            "asr_reduction_margin": float(args.axis_offset_asr_reduction_margin),
            "pessimistic_asr_margin": float(args.axis_offset_pessimistic_asr_margin),
            "clean_floor": args.axis_offset_clean_floor,
            "clean_drop_tolerance": args.axis_offset_clean_drop_tolerance,
            "selected_candidate": selected_record,
        }
    )
    return {
        "method": "axis_offset",
        "episodes": query_episodes,
        "horizon": query_horizon,
        "updates_per_episode": 0,
        "noise": 0.0,
        "lr_scale": 0.0,
        "axis_offset_step": float(args.axis_offset_step),
        "axis_offset_max_steps": int(args.axis_offset_max_steps),
        "candidate_scores": candidate_records,
        "selected_offset": [float(v) for v in selected_offset],
        "deployed_offset": [float(v) for v in selected_offset] if selection["accepted"] else [float(v) for v in zero_offset],
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
            {"shot": query_episodes, "updates": 0, "action": end_action},
        ],
        "num_transitions": query_horizon * evaluated_query_records,
        "num_updates": 0,
        "last_update_loss": {},
        "selection": selection,
        "adapted_evaluation": evaluation,
        "evaluation": evaluation,
    }


def _axis_rule_offset_adapt_and_evaluate(
    args,
    defender,
    scenario: Scenario,
    *,
    probe_obs: np.ndarray,
) -> dict:
    bsmg_cfg = _bsmg_config(args, horizon=int(args.H))
    start_action = _action_diagnostics(defender, probe_obs, bsmg_cfg)
    zero_offset = np.zeros(int(defender.act_dim), dtype=np.float32)
    if str(scenario.attack_type.objective) != "targeted":
        evaluation = _evaluate_scenario(args, defender, scenario)
        return {
            "method": "axis_rule_offset",
            "episodes": 0,
            "horizon": int(args.H),
            "updates_per_episode": 0,
            "noise": 0.0,
            "lr_scale": 0.0,
            "axis_offset_step": float(args.axis_offset_step),
            "axis_rule_beta_threshold": float(args.axis_rule_beta_threshold),
            "candidate_scores": [],
            "selected_offset": [float(v) for v in zero_offset],
            "deployed_offset": [float(v) for v in zero_offset],
            "probe_observation": {
                "scenario": "clean",
                "seed": int(args.seed),
                "description": "shared clean initial observation used only for action-space diagnostics",
            },
            "transition": {
                "from": start_action,
                "to": start_action,
                "delta": _action_delta(start_action, start_action),
            },
            "trace": [{"shot": 0, "updates": 0, "action": start_action}],
            "num_transitions": 0,
            "num_updates": 0,
            "last_update_loss": {},
            "selection": {
                "mode": "objective_gated_axis_rule_offset",
                "accepted": False,
                "selected": "base",
                "skip_reason": "scenario objective is outside axis rule offset gate",
                "scenario_objective": str(scenario.attack_type.objective),
            },
            "adapted_evaluation": evaluation,
            "evaluation": evaluation,
        }

    deployment_base_action = _scenario_initial_action_diagnostics(
        args,
        defender,
        scenario,
        horizon=int(args.H),
    )
    selected_label, selected_offset = _axis_rule_offset_for_action(
        deployment_base_action,
        act_dim=int(defender.act_dim),
        step=float(args.axis_offset_step),
        beta_threshold=float(args.axis_rule_beta_threshold),
        attack_name=str(scenario.attack_name),
    )
    selected_policy = defender if np.allclose(selected_offset, 0.0) else ActionOffsetPolicy(defender, selected_offset)
    evaluation = _evaluate_scenario(args, selected_policy, scenario)
    end_action = _action_diagnostics(selected_policy, probe_obs, bsmg_cfg)
    accepted = not np.allclose(selected_offset, 0.0)
    return {
        "method": "axis_rule_offset",
        "episodes": 0,
        "horizon": int(args.H),
        "updates_per_episode": 0,
        "noise": 0.0,
        "lr_scale": 0.0,
        "axis_offset_step": float(args.axis_offset_step),
        "axis_rule_beta_threshold": float(args.axis_rule_beta_threshold),
        "candidate_scores": [],
        "selected_offset": [float(v) for v in selected_offset],
        "deployed_offset": [float(v) for v in selected_offset],
        "deployment_base_action": deployment_base_action,
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
            {"shot": 0, "updates": 0, "action": end_action},
        ],
        "num_transitions": 0,
        "num_updates": 0,
        "last_update_loss": {},
        "selection": {
            "mode": "deployment_beta_rule_axis_offset",
            "accepted": bool(accepted),
            "selected": "adapted" if accepted else "base",
            "selected_offset_label": selected_label,
            "deployment_beta": float(deployment_base_action.get("beta", float("nan"))),
            "axis_rule_beta_threshold": float(args.axis_rule_beta_threshold),
        },
        "adapted_evaluation": evaluation,
        "evaluation": evaluation,
    }


def _axis_rule_v2_offset_adapt_and_evaluate(
    args,
    defender,
    scenario: Scenario,
    *,
    probe_obs: np.ndarray,
) -> dict:
    bsmg_cfg = _bsmg_config(args, horizon=int(args.H))
    start_action = _action_diagnostics(defender, probe_obs, bsmg_cfg)
    zero_offset = np.zeros(int(defender.act_dim), dtype=np.float32)
    if str(scenario.attack_type.objective) != "targeted":
        evaluation = _evaluate_scenario(args, defender, scenario)
        return {
            "method": "axis_rule_v2_offset",
            "episodes": 0,
            "horizon": int(args.H),
            "updates_per_episode": 0,
            "noise": 0.0,
            "lr_scale": 0.0,
            "axis_offset_step": float(args.axis_offset_step),
            "axis_rule_beta_threshold": float(args.axis_rule_beta_threshold),
            "axis_rule_low_beta_threshold": float(args.axis_rule_low_beta_threshold),
            "candidate_scores": [],
            "selected_offset": [float(v) for v in zero_offset],
            "deployed_offset": [float(v) for v in zero_offset],
            "probe_observation": {
                "scenario": "clean",
                "seed": int(args.seed),
                "description": "shared clean initial observation used only for action-space diagnostics",
            },
            "transition": {
                "from": start_action,
                "to": start_action,
                "delta": _action_delta(start_action, start_action),
            },
            "trace": [{"shot": 0, "updates": 0, "action": start_action}],
            "num_transitions": 0,
            "num_updates": 0,
            "last_update_loss": {},
            "selection": {
                "mode": "objective_gated_axis_rule_v2_offset",
                "accepted": False,
                "selected": "base",
                "skip_reason": "scenario objective is outside axis rule v2 offset gate",
                "scenario_objective": str(scenario.attack_type.objective),
            },
            "adapted_evaluation": evaluation,
            "evaluation": evaluation,
        }

    deployment_base_action = _scenario_initial_action_diagnostics(
        args,
        defender,
        scenario,
        horizon=int(args.H),
    )
    selected_label, selected_offset = _axis_rule_v2_offset_for_action(
        deployment_base_action,
        act_dim=int(defender.act_dim),
        step=float(args.axis_offset_step),
        beta_threshold=float(args.axis_rule_beta_threshold),
        low_beta_threshold=float(args.axis_rule_low_beta_threshold),
        attack_name=str(scenario.attack_name),
    )
    selected_policy = defender if np.allclose(selected_offset, 0.0) else ActionOffsetPolicy(defender, selected_offset)
    evaluation = _evaluate_scenario(args, selected_policy, scenario)
    end_action = _action_diagnostics(selected_policy, probe_obs, bsmg_cfg)
    accepted = not np.allclose(selected_offset, 0.0)
    return {
        "method": "axis_rule_v2_offset",
        "episodes": 0,
        "horizon": int(args.H),
        "updates_per_episode": 0,
        "noise": 0.0,
        "lr_scale": 0.0,
        "axis_offset_step": float(args.axis_offset_step),
        "axis_rule_beta_threshold": float(args.axis_rule_beta_threshold),
        "axis_rule_low_beta_threshold": float(args.axis_rule_low_beta_threshold),
        "candidate_scores": [],
        "selected_offset": [float(v) for v in selected_offset],
        "deployed_offset": [float(v) for v in selected_offset],
        "deployment_base_action": deployment_base_action,
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
            {"shot": 0, "updates": 0, "action": end_action},
        ],
        "num_transitions": 0,
        "num_updates": 0,
        "last_update_loss": {},
        "selection": {
            "mode": "deployment_beta_rule_v2_axis_offset",
            "accepted": bool(accepted),
            "selected": "adapted" if accepted else "base",
            "selected_offset_label": selected_label,
            "deployment_beta": float(deployment_base_action.get("beta", float("nan"))),
            "axis_rule_beta_threshold": float(args.axis_rule_beta_threshold),
            "axis_rule_low_beta_threshold": float(args.axis_rule_low_beta_threshold),
        },
        "adapted_evaluation": evaluation,
        "evaluation": evaluation,
    }


def _physical_rule_target_adapt_and_evaluate(
    args,
    defender,
    scenario: Scenario,
    *,
    probe_obs: np.ndarray,
) -> dict:
    bsmg_cfg = _bsmg_config(args, horizon=int(args.H))
    start_action = _action_diagnostics(defender, probe_obs, bsmg_cfg)
    zero_offset = np.zeros(int(defender.act_dim), dtype=np.float32)
    if str(scenario.attack_type.objective) != "targeted":
        evaluation = _evaluate_scenario(args, defender, scenario)
        return {
            "method": "physical_rule_target",
            "episodes": 0,
            "horizon": int(args.H),
            "updates_per_episode": 0,
            "noise": 0.0,
            "lr_scale": 0.0,
            "candidate_scores": [],
            "selected_offset": [float(v) for v in zero_offset],
            "deployed_offset": [float(v) for v in zero_offset],
            "target_alpha": None,
            "target_beta": None,
            "probe_observation": {
                "scenario": "clean",
                "seed": int(args.seed),
                "description": "shared clean initial observation used only for action-space diagnostics",
            },
            "transition": {
                "from": start_action,
                "to": start_action,
                "delta": _action_delta(start_action, start_action),
            },
            "trace": [{"shot": 0, "updates": 0, "action": start_action}],
            "num_transitions": 0,
            "num_updates": 0,
            "last_update_loss": {},
            "selection": {
                "mode": "objective_gated_physical_rule_target",
                "accepted": False,
                "selected": "base",
                "skip_reason": "scenario objective is outside physical rule target gate",
                "scenario_objective": str(scenario.attack_type.objective),
            },
            "adapted_evaluation": evaluation,
            "evaluation": evaluation,
        }

    deployment_base_action = _scenario_initial_action_diagnostics(
        args,
        defender,
        scenario,
        horizon=int(args.H),
    )
    rule = _physical_rule_target_for_action(
        deployment_base_action,
        act_dim=int(defender.act_dim),
        step=float(args.axis_offset_step),
        low_beta_threshold=float(args.physical_rule_low_beta_threshold),
        beta_threshold=float(args.physical_rule_beta_threshold),
        low_alpha_target=float(args.physical_rule_low_alpha_target),
        near_alpha_target=float(args.physical_rule_near_alpha_target),
        beta_target=float(args.physical_rule_beta_target),
        attack_name=str(scenario.attack_name),
    )
    selected_offset = np.asarray(rule["offset"], dtype=np.float32)
    target_alpha = rule["target_alpha"]
    target_beta = rule["target_beta"]
    if target_alpha is not None or target_beta is not None:
        selected_policy = PhysicalTargetPolicy(
            defender,
            bsmg_cfg,
            target_alpha=target_alpha,
            target_beta=target_beta,
        )
    elif np.allclose(selected_offset, 0.0):
        selected_policy = defender
    else:
        selected_policy = ActionOffsetPolicy(defender, selected_offset)
    evaluation = _evaluate_scenario(args, selected_policy, scenario)
    end_action = _action_diagnostics(selected_policy, probe_obs, bsmg_cfg)
    accepted = not (
        np.allclose(selected_offset, 0.0) and target_alpha is None and target_beta is None
    )
    return {
        "method": "physical_rule_target",
        "episodes": 0,
        "horizon": int(args.H),
        "updates_per_episode": 0,
        "noise": 0.0,
        "lr_scale": 0.0,
        "candidate_scores": [],
        "selected_offset": [float(v) for v in selected_offset],
        "deployed_offset": [float(v) for v in selected_offset],
        "target_alpha": None if target_alpha is None else float(target_alpha),
        "target_beta": None if target_beta is None else float(target_beta),
        "deployment_base_action": deployment_base_action,
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
            {"shot": 0, "updates": 0, "action": end_action},
        ],
        "num_transitions": 0,
        "num_updates": 0,
        "last_update_loss": {},
        "selection": {
            "mode": "deployment_beta_physical_rule_target",
            "accepted": bool(accepted),
            "selected": "adapted" if accepted else "base",
            "selected_offset_label": str(rule["selected_offset_label"]),
            "deployment_beta": float(deployment_base_action.get("beta", float("nan"))),
            "physical_rule_low_beta_threshold": float(args.physical_rule_low_beta_threshold),
            "physical_rule_beta_threshold": float(args.physical_rule_beta_threshold),
            "target_alpha": None if target_alpha is None else float(target_alpha),
            "target_beta": None if target_beta is None else float(target_beta),
        },
        "adapted_evaluation": evaluation,
        "evaluation": evaluation,
    }


def _physical_target_selector_adapt_and_evaluate(
    args,
    defender,
    scenario: Scenario,
    *,
    probe_obs: np.ndarray,
) -> dict:
    query_horizon = int(args.selection_horizon or args.H)
    query_episodes = max(1, int(args.adaptation_episodes))
    bsmg_cfg = _bsmg_config(args, horizon=query_horizon)
    start_action = _action_diagnostics(defender, probe_obs, bsmg_cfg)
    zero_offset = np.zeros(int(defender.act_dim), dtype=np.float32)
    if str(scenario.attack_type.objective) != "targeted":
        evaluation = _evaluate_scenario(args, defender, scenario)
        return {
            "method": "physical_target_selector",
            "episodes": 0,
            "horizon": query_horizon,
            "updates_per_episode": 0,
            "noise": 0.0,
            "lr_scale": 0.0,
            "candidate_scores": [],
            "selected_offset": [float(v) for v in zero_offset],
            "deployed_offset": [float(v) for v in zero_offset],
            "target_alpha": None,
            "target_beta": None,
            "probe_observation": {
                "scenario": "clean",
                "seed": int(args.seed),
                "description": "shared clean initial observation used only for action-space diagnostics",
            },
            "transition": {
                "from": start_action,
                "to": start_action,
                "delta": _action_delta(start_action, start_action),
            },
            "trace": [{"shot": 0, "updates": 0, "action": start_action}],
            "num_transitions": 0,
            "num_updates": 0,
            "last_update_loss": {},
            "selection": {
                "mode": "objective_gated_physical_target_selector",
                "accepted": False,
                "selected": "base",
                "skip_reason": "scenario objective is outside physical target selector gate",
                "scenario_objective": str(scenario.attack_type.objective),
            },
            "adapted_evaluation": evaluation,
            "evaluation": evaluation,
        }

    candidates = _physical_target_candidates(
        act_dim=int(defender.act_dim),
        alpha_candidates=str(args.physical_target_alpha_candidates),
        beta_target=float(args.physical_target_beta_target),
        start_round=int(getattr(args, "physical_target_start_round", 0)),
        end_round_candidates=_parse_optional_int_candidates(
            getattr(args, "physical_target_end_round_candidates", None)
        ),
    )
    candidate_records = []
    for idx, candidate in enumerate(candidates):
        policy = _physical_target_policy_for_candidate(
            args,
            defender,
            bsmg_cfg,
            candidate,
        )
        query_records = []
        for episode in range(query_episodes):
            seed = int(scenario.seed) + int(args.selection_seed_offset) + episode
            query_records.append(
                _evaluate_scenario_at(args, policy, scenario, seed=seed, horizon=query_horizon)
            )
        candidate_records.append(
            {
                "index": int(idx),
                "offset_label": str(candidate["offset_label"]),
                "offset": [float(v) for v in candidate["offset"]],
                "target_alpha": candidate["target_alpha"],
                "target_beta": candidate["target_beta"],
                **(
                    {
                        "target_start_round": candidate.get("target_start_round"),
                        "target_end_round": candidate.get("target_end_round"),
                    }
                    if "target_end_round" in candidate or "target_start_round" in candidate
                    else {}
                ),
                "query_records": query_records,
                **_beta_offset_query_summary(query_records),
            }
        )

    deployment_validations = 0
    selection_mode = str(getattr(args, "physical_target_selection_mode", "score"))
    if selection_mode == "deployment_clean_recovery":
        for candidate_record in candidate_records:
            policy = _physical_target_policy_for_candidate(
                args,
                defender,
                bsmg_cfg,
                candidate_record,
            )
            deployment_record = _evaluate_scenario(args, policy, scenario)
            candidate_record["deployment_record"] = deployment_record
            candidate_record.update(_deployment_record_summary(deployment_record))
            deployment_validations += 1
        selected_record, selection = _select_physical_target_with_deployment_records(
            candidate_records,
            asr_reduction_margin=float(args.physical_target_asr_reduction_margin),
            clean_floor=args.physical_target_clean_floor,
            clean_drop_tolerance=args.physical_target_clean_drop_tolerance,
            score_slack=float(getattr(args, "physical_target_score_slack", 0.0)),
            deployment_clean_floor=getattr(args, "physical_target_deployment_clean_floor", None),
            deployment_clean_drop_tolerance=getattr(
                args,
                "physical_target_deployment_clean_drop_tolerance",
                None,
            ),
            deployment_asr_ceiling=getattr(args, "physical_target_deployment_asr_ceiling", None),
        )
    else:
        selected_record, selection = _select_physical_target_from_query_records(
            candidate_records,
            asr_reduction_margin=float(args.physical_target_asr_reduction_margin),
            clean_floor=args.physical_target_clean_floor,
            clean_drop_tolerance=args.physical_target_clean_drop_tolerance,
            score_slack=float(getattr(args, "physical_target_score_slack", 0.0)),
        )
    selected_policy = _physical_target_policy_for_candidate(args, defender, bsmg_cfg, selected_record)
    deployed_policy = selected_policy if selection["accepted"] else defender
    evaluation = selected_record.get("deployment_record")
    if evaluation is None:
        evaluation = _evaluate_scenario(args, deployed_policy, scenario)
    end_action = _action_diagnostics(deployed_policy, probe_obs, bsmg_cfg)
    target_alpha = selected_record["target_alpha"] if selection["accepted"] else None
    target_beta = selected_record["target_beta"] if selection["accepted"] else None
    deployed_start_round = _candidate_target_start_round(args, selected_record) if selection["accepted"] else int(
        getattr(args, "physical_target_start_round", 0)
    )
    deployed_end_round = _candidate_target_end_round(args, selected_record) if selection["accepted"] else getattr(
        args,
        "physical_target_end_round",
        None,
    )
    selection.update(
        {
            "mode": "deployment_clean_recovery_physical_target_selector"
            if selection_mode == "deployment_clean_recovery"
            else "query_guarded_physical_target_selector",
            "query_horizon": query_horizon,
            "query_episodes": query_episodes,
            "asr_reduction_margin": float(args.physical_target_asr_reduction_margin),
            "clean_floor": args.physical_target_clean_floor,
            "clean_drop_tolerance": args.physical_target_clean_drop_tolerance,
            "score_slack": float(getattr(args, "physical_target_score_slack", 0.0)),
            "selection_mode": selection_mode,
            "start_round": deployed_start_round,
            "end_round": deployed_end_round,
            "selected_candidate": selected_record,
        }
    )
    return {
        "method": "physical_target_selector",
        "episodes": query_episodes,
        "horizon": query_horizon,
        "updates_per_episode": 0,
        "noise": 0.0,
        "lr_scale": 0.0,
        "physical_target_alpha_candidates": str(args.physical_target_alpha_candidates),
        "physical_target_beta_target": float(args.physical_target_beta_target),
        "physical_target_start_round": deployed_start_round,
        "physical_target_end_round": deployed_end_round,
        "physical_target_end_round_candidates": getattr(args, "physical_target_end_round_candidates", None),
        "physical_target_selection_mode": selection_mode,
        "candidate_scores": candidate_records,
        "selected_offset": [float(v) for v in np.asarray(selected_record["offset"], dtype=np.float32)],
        "deployed_offset": [float(v) for v in np.asarray(selected_record["offset"], dtype=np.float32)]
        if selection["accepted"]
        else [float(v) for v in zero_offset],
        "target_alpha": None if target_alpha is None else float(target_alpha),
        "target_beta": None if target_beta is None else float(target_beta),
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
            {"shot": query_episodes, "updates": 0, "action": end_action},
        ],
        "num_transitions": query_horizon * query_episodes * len(candidates)
        + int(args.H) * deployment_validations,
        "num_deployment_validations": deployment_validations,
        "num_updates": 0,
        "last_update_loss": {},
        "selection": selection,
        "adapted_evaluation": evaluation,
        "evaluation": evaluation,
    }


def _cem_offset_adapt_and_evaluate(
    args,
    defender,
    scenario: Scenario,
    *,
    probe_obs: np.ndarray,
) -> dict:
    bsmg_cfg = _bsmg_config(args, horizon=int(args.adaptation_horizon))
    start_action = _action_diagnostics(defender, probe_obs, bsmg_cfg)
    act_dim = int(defender.act_dim)
    bound = abs(float(args.cem_offset_bound))
    mean = np.zeros(act_dim, dtype=np.float32)
    sigma = np.full(act_dim, float(args.cem_init_sigma), dtype=np.float32)
    rng = np.random.default_rng(int(scenario.seed) + 40_000)
    iteration_records = []

    best_record = {
        "offset": [0.0 for _ in range(act_dim)],
        "mean_defense_score": float("-inf"),
        "scores": [],
        "iteration": -1,
    }
    for iteration in range(max(1, int(args.cem_iterations))):
        candidates = rng.normal(
            loc=mean,
            scale=sigma,
            size=(max(1, int(args.cem_population)), act_dim),
        ).astype(np.float32)
        candidates = np.clip(candidates, -bound, bound)
        candidates[0] = np.zeros(act_dim, dtype=np.float32)

        candidate_records = []
        for idx, offset in enumerate(candidates):
            policy = ActionOffsetPolicy(defender, offset)
            scores = []
            for episode in range(int(args.adaptation_episodes)):
                seed = int(scenario.seed) + 10_000 + iteration * 1_000 + idx * 100 + episode
                record = _evaluate_scenario_at(
                    args,
                    policy,
                    scenario,
                    seed=seed,
                    horizon=int(args.adaptation_horizon),
                )
                scores.append(float(record["final_defense_score"]))
            candidate_record = {
                "offset": [float(v) for v in offset],
                "mean_defense_score": float(np.mean(scores)) if scores else float("-inf"),
                "scores": scores,
                "iteration": int(iteration),
            }
            candidate_records.append(candidate_record)
            if candidate_record["mean_defense_score"] > best_record["mean_defense_score"]:
                best_record = dict(candidate_record)

        mean, sigma, elite_indices = _cem_update_distribution(
            candidates,
            [float(item["mean_defense_score"]) for item in candidate_records],
            elite_count=int(args.cem_elites),
            min_sigma=float(args.cem_min_sigma),
        )
        mean = np.clip(mean, -bound, bound).astype(np.float32)
        sigma = np.asarray(sigma, dtype=np.float32)
        iteration_records.append(
            {
                "iteration": int(iteration),
                "mean": [float(v) for v in mean],
                "sigma": [float(v) for v in sigma],
                "elite_indices": elite_indices,
                "candidate_scores": candidate_records,
            }
        )

    zero_record = _cem_zero_record(iteration_records)
    best_offset = np.asarray(best_record["offset"], dtype=np.float32)
    adapted_policy = ActionOffsetPolicy(defender, best_offset)
    adapted_evaluation = _evaluate_scenario(args, adapted_policy, scenario)
    selection = {
        "mode": str(args.few_shot_selection),
        "accepted": True,
        "selected": "adapted",
        "margin": float(args.selection_margin),
        "base_score": float(zero_record["mean_defense_score"]),
        "adapted_score": float(best_record["mean_defense_score"]),
        "score_gain": float(best_record["mean_defense_score"]) - float(zero_record["mean_defense_score"]),
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
            adapted_policy,
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
                "base_support": zero_record,
                "adapted_support": best_record,
            }
        )
        if not selection["accepted"]:
            evaluation = _evaluate_scenario(args, defender, scenario)
    end_action = _action_diagnostics(adapted_policy, probe_obs, bsmg_cfg)
    return {
        "method": "cem_offset",
        "episodes": int(args.adaptation_episodes),
        "horizon": int(args.adaptation_horizon),
        "updates_per_episode": 0,
        "noise": 0.0,
        "lr_scale": 0.0,
        "cem_iterations": int(args.cem_iterations),
        "cem_population": int(args.cem_population),
        "cem_elites": int(args.cem_elites),
        "cem_init_sigma": float(args.cem_init_sigma),
        "cem_min_sigma": float(args.cem_min_sigma),
        "cem_offset_bound": float(args.cem_offset_bound),
        "iterations": iteration_records,
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
        "num_transitions": (
            int(args.adaptation_horizon)
            * int(args.adaptation_episodes)
            * int(args.cem_population)
            * int(args.cem_iterations)
        ),
        "num_updates": 0,
        "last_update_loss": {},
        "selection": selection,
        "adapted_evaluation": adapted_evaluation,
        "evaluation": evaluation,
    }


def _cem_update_distribution(
    candidates: np.ndarray,
    scores: list[float],
    *,
    elite_count: int,
    min_sigma: float,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if len(candidates) == 0:
        raise ValueError("CEM requires at least one candidate")
    elite_count = max(1, min(int(elite_count), len(candidates)))
    elite_indices = sorted(range(len(scores)), key=lambda idx: float(scores[idx]), reverse=True)[:elite_count]
    elites = np.asarray([candidates[idx] for idx in elite_indices], dtype=np.float32)
    mean = np.mean(elites, axis=0).astype(np.float32)
    sigma = np.maximum(np.std(elites, axis=0), float(min_sigma)).astype(np.float32)
    return mean, sigma, elite_indices


def _cem_zero_record(iteration_records: list[dict]) -> dict:
    for iteration in iteration_records:
        candidates = iteration.get("candidate_scores", [])
        if candidates:
            return candidates[0]
    return {
        "offset": [],
        "mean_defense_score": float("-inf"),
        "scores": [],
        "iteration": -1,
    }


def _residual_adapter_adapt_and_evaluate(
    args,
    defender,
    scenario: Scenario,
    *,
    probe_obs: np.ndarray,
) -> dict:
    bsmg_cfg = _bsmg_config(args, horizon=int(args.adaptation_horizon))
    start_action = _action_diagnostics(defender, probe_obs, bsmg_cfg)
    if not _residual_objective_allowed(str(args.residual_objective_gate), scenario):
        evaluation = _evaluate_scenario(args, defender, scenario)
        zero_offset = [0.0 for _ in range(int(defender.act_dim))]
        selection = {
            "mode": "objective_gated_residual",
            "accepted": False,
            "selected": "base",
            "base_score": float(evaluation["final_defense_score"]),
            "adapted_score": float(evaluation["final_defense_score"]),
            "score_gain": 0.0,
            "margin": float(args.residual_degradation_margin),
            "objective_gate": str(args.residual_objective_gate),
            "scenario_objective": str(scenario.attack_type.objective),
            "skip_reason": "scenario objective is outside residual adapter gate",
        }
        return {
            "method": "residual_adapter",
            "episodes": int(args.adaptation_episodes),
            "horizon": int(args.adaptation_horizon),
            "updates_per_episode": 0,
            "noise": 0.0,
            "lr_scale": 0.0,
            "residual_bound": float(args.residual_bound),
            "residual_candidates": int(args.residual_candidates),
            "residual_query_episodes": int(args.residual_query_episodes),
            "residual_degradation_margin": float(args.residual_degradation_margin),
            "residual_objective_gate": str(args.residual_objective_gate),
            "residual_selection_metric": str(args.residual_selection_metric),
            "residual_lcb_std_weight": float(args.residual_lcb_std_weight),
            "support_records": [],
            "support_summary": {},
            "candidate_scores": [],
            "best_offset": zero_offset,
            "selected_offset": zero_offset,
            "probe_observation": {
                "scenario": "clean",
                "seed": int(args.seed),
                "description": "shared clean initial observation used only for action-space diagnostics",
            },
            "transition": {
                "from": start_action,
                "to": start_action,
                "delta": {},
            },
            "trace": [
                {"shot": 0, "updates": 0, "action": start_action},
                {"shot": int(args.adaptation_episodes), "updates": 0, "action": start_action},
            ],
            "num_transitions": 0,
            "num_updates": 0,
            "last_update_loss": {},
            "selection": selection,
            "adapted_evaluation": evaluation,
            "evaluation": evaluation,
        }
    support_records = []
    for episode in range(max(1, int(args.adaptation_episodes))):
        seed = int(scenario.seed) + 10_000 + episode
        support_records.append(
            _evaluate_scenario_at(
                args,
                defender,
                scenario,
                seed=seed,
                horizon=int(args.adaptation_horizon),
            )
        )

    act_dim = int(defender.act_dim)
    support_summary = _residual_support_summary(support_records, act_dim=act_dim)
    offsets = _residual_candidate_offsets(
        np.asarray(support_summary["direction"], dtype=np.float32),
        act_dim=act_dim,
        bound=float(args.residual_bound),
        candidate_count=int(args.residual_candidates),
    )
    query_horizon = int(args.selection_horizon or args.adaptation_horizon)
    query_episodes = max(1, int(args.residual_query_episodes))
    candidate_records = []
    for idx, offset in enumerate(offsets):
        policy = ActionOffsetPolicy(defender, offset)
        query_records = []
        scores = []
        for episode in range(query_episodes):
            seed = int(scenario.seed) + int(args.selection_seed_offset) + episode
            record = _evaluate_scenario_at(
                args,
                policy,
                scenario,
                seed=seed,
                horizon=query_horizon,
            )
            query_records.append(record)
            scores.append(float(record["final_defense_score"]))
        score_summary = _residual_query_score_summary(
            scores,
            metric=str(args.residual_selection_metric),
            lcb_std_weight=float(args.residual_lcb_std_weight),
        )
        candidate_records.append(
            {
                "index": int(idx),
                "offset": [float(v) for v in np.asarray(offset, dtype=np.float32)],
                **score_summary,
                "query_scores": scores,
                "query_records": query_records,
            }
        )

    selected_record, selection = _select_residual_from_query_scores(
        candidate_records,
        margin=float(args.residual_degradation_margin),
        metric=str(args.residual_selection_metric),
        lcb_std_weight=float(args.residual_lcb_std_weight),
    )
    best_record = max(
        candidate_records,
        key=lambda item: _residual_selection_score(
            item,
            metric=str(args.residual_selection_metric),
            lcb_std_weight=float(args.residual_lcb_std_weight),
        ),
    )
    best_offset = np.asarray(best_record["offset"], dtype=np.float32)
    selected_offset = np.asarray(selected_record["offset"], dtype=np.float32)
    best_policy = ActionOffsetPolicy(defender, best_offset)
    selected_policy = ActionOffsetPolicy(defender, selected_offset)
    adapted_evaluation = _evaluate_scenario(args, best_policy, scenario)
    evaluation = _evaluate_scenario(args, selected_policy if selection["accepted"] else defender, scenario)
    selection.update(
        {
            "mode": "query_guarded_residual",
            "margin": float(args.residual_degradation_margin),
            "query_horizon": query_horizon,
            "query_episodes": query_episodes,
            "best_candidate": best_record,
            "selected_candidate": selected_record,
        }
    )
    _dump_residual_supervision_sample(
        args,
        scenario=scenario,
        support_summary=support_summary,
        candidate_records=candidate_records,
        selected_offset=selected_offset,
        selection=selection,
    )
    end_action = _action_diagnostics(selected_policy if selection["accepted"] else defender, probe_obs, bsmg_cfg)
    return {
        "method": "residual_adapter",
        "episodes": int(args.adaptation_episodes),
        "horizon": int(args.adaptation_horizon),
        "updates_per_episode": 0,
        "noise": 0.0,
        "lr_scale": 0.0,
        "residual_bound": float(args.residual_bound),
        "residual_candidates": int(args.residual_candidates),
        "residual_query_episodes": int(args.residual_query_episodes),
        "residual_degradation_margin": float(args.residual_degradation_margin),
        "residual_objective_gate": str(args.residual_objective_gate),
        "residual_selection_metric": str(args.residual_selection_metric),
        "residual_lcb_std_weight": float(args.residual_lcb_std_weight),
        "support_records": support_records,
        "support_summary": support_summary,
        "candidate_scores": candidate_records,
        "best_offset": [float(v) for v in best_offset],
        "selected_offset": [float(v) for v in selected_offset],
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
        "num_transitions": (
            int(args.adaptation_horizon) * int(args.adaptation_episodes)
            + query_horizon * query_episodes * len(offsets)
        ),
        "num_updates": 0,
        "last_update_loss": {},
        "selection": selection,
        "adapted_evaluation": adapted_evaluation,
        "evaluation": evaluation,
    }


def _trained_residual_adapter_adapt_and_evaluate(
    args,
    defender,
    scenario: Scenario,
    *,
    probe_obs: np.ndarray,
) -> dict:
    if not args.residual_adapter_checkpoint:
        raise ValueError("--residual-adapter-checkpoint is required for trained_residual_adapter")
    bsmg_cfg = _bsmg_config(args, horizon=int(args.adaptation_horizon))
    start_action = _action_diagnostics(defender, probe_obs, bsmg_cfg)
    if not _residual_objective_allowed(str(args.residual_objective_gate), scenario):
        evaluation = _evaluate_scenario(args, defender, scenario)
        zero_offset = [0.0 for _ in range(int(defender.act_dim))]
        return {
            "method": "trained_residual_adapter",
            "episodes": int(args.adaptation_episodes),
            "horizon": int(args.adaptation_horizon),
            "updates_per_episode": 0,
            "noise": 0.0,
            "lr_scale": 0.0,
            "residual_adapter_checkpoint": str(args.residual_adapter_checkpoint),
            "residual_objective_gate": str(args.residual_objective_gate),
            "residual_selection_metric": str(args.residual_selection_metric),
            "residual_lcb_std_weight": float(args.residual_lcb_std_weight),
            "support_records": [],
            "support_summary": {},
            "predicted_offset": zero_offset,
            "selected_offset": zero_offset,
            "transition": {"from": start_action, "to": start_action, "delta": {}},
            "trace": [
                {"shot": 0, "updates": 0, "action": start_action},
                {"shot": int(args.adaptation_episodes), "updates": 0, "action": start_action},
            ],
            "num_transitions": 0,
            "num_updates": 0,
            "last_update_loss": {},
            "selection": {
                "mode": "objective_gated_trained_residual",
                "accepted": False,
                "selected": "base",
                "base_score": float(evaluation["final_defense_score"]),
                "adapted_score": float(evaluation["final_defense_score"]),
                "score_gain": 0.0,
                "margin": float(args.residual_degradation_margin),
                "objective_gate": str(args.residual_objective_gate),
                "scenario_objective": str(scenario.attack_type.objective),
                "skip_reason": "scenario objective is outside residual adapter gate",
            },
            "adapted_evaluation": evaluation,
            "evaluation": evaluation,
        }

    support_records = []
    for episode in range(max(1, int(args.adaptation_episodes))):
        seed = int(scenario.seed) + 10_000 + episode
        support_records.append(
            _evaluate_scenario_at(
                args,
                defender,
                scenario,
                seed=seed,
                horizon=int(args.adaptation_horizon),
            )
        )
    support_summary = _residual_support_summary(support_records, act_dim=int(defender.act_dim))
    adapter, adapter_metrics = load_residual_adapter(args.residual_adapter_checkpoint)
    sample = {
        "support_summary": support_summary,
        "attack_name": scenario.attack_name,
        "attack_objective": scenario.attack_type.objective,
    }
    predicted_offset = predict_residual_offset(adapter, sample, bound=float(args.residual_bound))
    zero_offset = np.zeros(int(defender.act_dim), dtype=np.float32)
    query_horizon = int(args.selection_horizon or args.adaptation_horizon)
    query_episodes = max(1, int(args.residual_query_episodes))
    candidate_records = []
    for idx, offset in enumerate([zero_offset, predicted_offset]):
        policy = ActionOffsetPolicy(defender, offset)
        query_records = []
        scores = []
        for episode in range(query_episodes):
            seed = int(scenario.seed) + int(args.selection_seed_offset) + episode
            record = _evaluate_scenario_at(args, policy, scenario, seed=seed, horizon=query_horizon)
            query_records.append(record)
            scores.append(float(record["final_defense_score"]))
        score_summary = _residual_query_score_summary(
            scores,
            metric=str(args.residual_selection_metric),
            lcb_std_weight=float(args.residual_lcb_std_weight),
        )
        candidate_records.append(
            {
                "index": int(idx),
                "offset": [float(v) for v in np.asarray(offset, dtype=np.float32)],
                **score_summary,
                "query_scores": scores,
                "query_records": query_records,
            }
        )
    selected_record, selection = _select_residual_from_query_scores(
        candidate_records,
        margin=float(args.residual_degradation_margin),
        metric=str(args.residual_selection_metric),
        lcb_std_weight=float(args.residual_lcb_std_weight),
    )
    selected_offset = np.asarray(selected_record["offset"], dtype=np.float32)
    predicted_policy = ActionOffsetPolicy(defender, predicted_offset)
    selected_policy = ActionOffsetPolicy(defender, selected_offset)
    adapted_evaluation = _evaluate_scenario(args, predicted_policy, scenario)
    evaluation = _evaluate_scenario(args, selected_policy if selection["accepted"] else defender, scenario)
    selection.update(
        {
            "mode": "query_guarded_trained_residual",
            "margin": float(args.residual_degradation_margin),
            "query_horizon": query_horizon,
            "query_episodes": query_episodes,
            "adapter_metrics": adapter_metrics,
            "selected_candidate": selected_record,
        }
    )
    end_action = _action_diagnostics(selected_policy if selection["accepted"] else defender, probe_obs, bsmg_cfg)
    return {
        "method": "trained_residual_adapter",
        "episodes": int(args.adaptation_episodes),
        "horizon": int(args.adaptation_horizon),
        "updates_per_episode": 0,
        "noise": 0.0,
        "lr_scale": 0.0,
        "residual_bound": float(args.residual_bound),
        "residual_query_episodes": int(args.residual_query_episodes),
        "residual_degradation_margin": float(args.residual_degradation_margin),
        "residual_objective_gate": str(args.residual_objective_gate),
        "residual_selection_metric": str(args.residual_selection_metric),
        "residual_lcb_std_weight": float(args.residual_lcb_std_weight),
        "residual_adapter_checkpoint": str(args.residual_adapter_checkpoint),
        "support_records": support_records,
        "support_summary": support_summary,
        "candidate_scores": candidate_records,
        "predicted_offset": [float(v) for v in predicted_offset],
        "selected_offset": [float(v) for v in selected_offset],
        "transition": {
            "from": start_action,
            "to": end_action,
            "delta": _action_delta(start_action, end_action),
        },
        "trace": [
            {"shot": 0, "updates": 0, "action": start_action},
            {"shot": int(args.adaptation_episodes), "updates": 0, "action": end_action},
        ],
        "num_transitions": (
            int(args.adaptation_horizon) * int(args.adaptation_episodes)
            + query_horizon * query_episodes * len(candidate_records)
        ),
        "num_updates": 0,
        "last_update_loss": {},
        "selection": selection,
        "adapted_evaluation": adapted_evaluation,
        "evaluation": evaluation,
    }


def _dump_residual_supervision_sample(
    args,
    *,
    scenario: Scenario,
    support_summary: dict,
    candidate_records: list[dict],
    selected_offset: np.ndarray,
    selection: dict,
) -> None:
    path_value = getattr(args, "residual_dump_supervision_jsonl", None)
    if not path_value:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    sample = {
        "scenario": scenario.name,
        "attack_name": scenario.attack_name,
        "attack_objective": scenario.attack_type.objective,
        "seed": int(scenario.seed),
        "support_summary": support_summary,
        "candidate_scores": candidate_records,
        "selected_offset": [float(v) for v in np.asarray(selected_offset, dtype=np.float32)],
        "accepted": bool(selection.get("accepted", False)),
        "selection": selection,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sample, sort_keys=True) + "\n")


def _residual_support_summary(records: list[dict], *, act_dim: int) -> dict:
    scores = _metric_series(records, "final_defense_score")
    clean = _metric_series(records, "final_clean_acc")
    backdoor = _metric_series(records, "final_backdoor_acc")
    reward = _metric_series(records, "mean_defender_reward")
    score_slope = _series_slope(scores)
    clean_slope = _series_slope(clean)
    backdoor_slope = _series_slope(backdoor)
    reward_slope = _series_slope(reward)
    source = np.asarray([score_slope, -backdoor_slope, clean_slope, reward_slope], dtype=np.float32)
    direction = np.zeros(int(act_dim), dtype=np.float32)
    usable = min(len(source), int(act_dim))
    direction[:usable] = source[:usable]
    direction_norm = float(np.linalg.norm(direction))
    normalized = direction / direction_norm if direction_norm > 0.0 else direction
    return {
        "support_score_mean": float(np.mean(scores)) if scores else float("nan"),
        "support_clean_mean": float(np.mean(clean)) if clean else float("nan"),
        "support_backdoor_mean": float(np.mean(backdoor)) if backdoor else float("nan"),
        "support_reward_mean": float(np.mean(reward)) if reward else float("nan"),
        "support_score_slope": float(score_slope),
        "support_clean_slope": float(clean_slope),
        "support_backdoor_slope": float(backdoor_slope),
        "support_reward_slope": float(reward_slope),
        "direction": [float(v) for v in direction],
        "normalized_direction": [float(v) for v in normalized],
        "direction_norm": direction_norm,
    }


def _metric_series(records: list[dict], key: str) -> list[float]:
    values = []
    for record in records:
        value = float(record.get(key, float("nan")))
        if np.isfinite(value):
            values.append(value)
    return values


def _series_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(values[-1] - values[0])


def _residual_candidate_offsets(
    direction: np.ndarray,
    *,
    act_dim: int,
    bound: float,
    candidate_count: int,
) -> list[np.ndarray]:
    limit = max(1, int(candidate_count))
    bound = abs(float(bound))
    offsets: list[np.ndarray] = [np.zeros(int(act_dim), dtype=np.float32)]
    direction = np.asarray(direction, dtype=np.float32).reshape(-1)
    padded = np.zeros(int(act_dim), dtype=np.float32)
    usable = min(len(direction), int(act_dim))
    padded[:usable] = direction[:usable]
    norm = float(np.linalg.norm(padded))
    if norm > 0.0:
        unit = padded / norm
        for scale in (1.0, 0.5):
            offsets.append(np.clip(bound * scale * unit, -bound, bound).astype(np.float32))
            offsets.append(np.clip(-bound * scale * unit, -bound, bound).astype(np.float32))
            if len(offsets) >= limit:
                return _dedupe_offsets(offsets)[:limit]
    for idx in range(int(act_dim)):
        pos = np.zeros(int(act_dim), dtype=np.float32)
        neg = np.zeros(int(act_dim), dtype=np.float32)
        pos[idx] = bound
        neg[idx] = -bound
        offsets.extend([pos, neg])
        if len(offsets) >= limit:
            break
    return _dedupe_offsets(offsets)[:limit]


def _dedupe_offsets(offsets: list[np.ndarray]) -> list[np.ndarray]:
    seen = set()
    unique = []
    for offset in offsets:
        arr = np.asarray(offset, dtype=np.float32)
        key = tuple(np.round(arr, 8).tolist())
        if key in seen:
            continue
        seen.add(key)
        unique.append(arr)
    return unique


def _select_residual_from_query_scores(
    candidates: list[dict],
    *,
    margin: float,
    metric: str = "mean",
    lcb_std_weight: float = 1.0,
) -> tuple[dict, dict]:
    if not candidates:
        raise ValueError("residual adapter requires at least one candidate")
    metric = str(metric)
    base = next(
        (
            item
            for item in candidates
            if np.allclose(np.asarray(item["offset"], dtype=np.float32), 0.0)
        ),
        candidates[0],
    )
    best = max(
        candidates,
        key=lambda item: _residual_selection_score(
            item,
            metric=metric,
            lcb_std_weight=float(lcb_std_weight),
        ),
    )
    base_score = _residual_selection_score(
        base,
        metric=metric,
        lcb_std_weight=float(lcb_std_weight),
    )
    adapted_score = _residual_selection_score(
        best,
        metric=metric,
        lcb_std_weight=float(lcb_std_weight),
    )
    score_gain = float(adapted_score) - float(base_score)
    accepted = bool(best is not base and score_gain >= float(margin))
    selected = best if accepted else base
    return selected, {
        "accepted": accepted,
        "selected": "adapted" if accepted else "base",
        "selection_metric": metric,
        "base_score": float(base_score),
        "adapted_score": float(adapted_score),
        "score_gain": score_gain,
        "base_score_mean": _residual_selection_score(base, metric="mean", lcb_std_weight=float(lcb_std_weight)),
        "adapted_score_mean": _residual_selection_score(best, metric="mean", lcb_std_weight=float(lcb_std_weight)),
    }


def _residual_query_score_summary(
    scores: list[float],
    *,
    metric: str,
    lcb_std_weight: float,
) -> dict:
    finite_scores = [float(score) for score in scores if np.isfinite(float(score))]
    if not finite_scores:
        mean = float("-inf")
        worst = float("-inf")
        std = float("inf")
        lcb = float("-inf")
    else:
        arr = np.asarray(finite_scores, dtype=np.float32)
        mean = float(np.mean(arr))
        worst = float(np.min(arr))
        std = float(np.std(arr))
        lcb = float(mean - float(lcb_std_weight) * std)
    summary = {
        "query_score_mean": mean,
        "query_score_min": worst,
        "query_score_std": std,
        "query_score_lcb": lcb,
    }
    summary["query_score_selected"] = _residual_selection_score(
        summary,
        metric=str(metric),
        lcb_std_weight=float(lcb_std_weight),
    )
    summary["query_score_metric"] = str(metric)
    return summary


def _residual_selection_score(
    candidate: dict,
    *,
    metric: str,
    lcb_std_weight: float,
) -> float:
    metric = str(metric)
    if metric == "mean":
        return float(candidate.get("query_score_mean", _score_mean(candidate.get("query_scores", []))))
    if metric == "worst":
        return float(candidate.get("query_score_min", _score_min(candidate.get("query_scores", []))))
    if metric == "lcb":
        if "query_score_lcb" in candidate:
            return float(candidate["query_score_lcb"])
        mean = float(candidate.get("query_score_mean", _score_mean(candidate.get("query_scores", []))))
        std = float(candidate.get("query_score_std", _score_std(candidate.get("query_scores", []))))
        return float(mean - float(lcb_std_weight) * std)
    raise ValueError(f"Unknown residual selection metric: {metric}")


def _score_mean(scores: list[float]) -> float:
    finite_scores = [float(score) for score in scores if np.isfinite(float(score))]
    return float(np.mean(finite_scores)) if finite_scores else float("-inf")


def _score_min(scores: list[float]) -> float:
    finite_scores = [float(score) for score in scores if np.isfinite(float(score))]
    return float(np.min(finite_scores)) if finite_scores else float("-inf")


def _score_std(scores: list[float]) -> float:
    finite_scores = [float(score) for score in scores if np.isfinite(float(score))]
    return float(np.std(np.asarray(finite_scores, dtype=np.float32))) if finite_scores else float("inf")


def _residual_objective_allowed(gate: str, scenario: Scenario) -> bool:
    if str(gate) == "all":
        return True
    return str(scenario.attack_type.objective) == str(gate)


def _beta_offset_candidates(*, act_dim: int, step: float, max_steps: int) -> list[tuple[str, np.ndarray]]:
    act_dim = int(act_dim)
    if act_dim < 2:
        raise ValueError("beta_offset requires defender action dimension >= 2")
    step = abs(float(step))
    max_steps = max(1, int(max_steps))
    candidates: list[tuple[str, np.ndarray]] = [("zero", np.zeros(act_dim, dtype=np.float32))]
    for scale in range(1, max_steps + 1):
        offset = np.zeros(act_dim, dtype=np.float32)
        offset[1] = float(step * scale)
        candidates.append((f"beta_plus_{scale}", np.clip(offset, -1.0, 1.0).astype(np.float32)))
    offset = np.zeros(act_dim, dtype=np.float32)
    offset[1] = -float(step)
    candidates.append(("beta_minus_1", np.clip(offset, -1.0, 1.0).astype(np.float32)))
    return _dedupe_labeled_offsets(candidates)


def _axis_offset_candidates(*, act_dim: int, step: float, max_steps: int) -> list[tuple[str, np.ndarray]]:
    act_dim = int(act_dim)
    if act_dim < 2:
        raise ValueError("axis_offset requires defender action dimension >= 2")
    step = abs(float(step))
    max_steps = max(1, int(max_steps))
    candidates: list[tuple[str, np.ndarray]] = [("zero", np.zeros(act_dim, dtype=np.float32))]
    for scale in range(1, max_steps + 1):
        alpha_minus = np.zeros(act_dim, dtype=np.float32)
        alpha_plus = np.zeros(act_dim, dtype=np.float32)
        beta_plus = np.zeros(act_dim, dtype=np.float32)
        beta_minus = np.zeros(act_dim, dtype=np.float32)
        beta_plus_alpha_minus = np.zeros(act_dim, dtype=np.float32)
        alpha_minus[0] = -float(step * scale)
        alpha_plus[0] = float(step * scale)
        beta_plus[1] = float(step * scale)
        beta_minus[1] = -float(step * scale)
        beta_plus_alpha_minus[0] = -float(step * scale)
        beta_plus_alpha_minus[1] = float(step * scale)
        candidates.extend(
            [
                (f"alpha_minus_{scale}", np.clip(alpha_minus, -1.0, 1.0).astype(np.float32)),
                (f"alpha_plus_{scale}", np.clip(alpha_plus, -1.0, 1.0).astype(np.float32)),
                (f"beta_plus_{scale}", np.clip(beta_plus, -1.0, 1.0).astype(np.float32)),
                (f"beta_minus_{scale}", np.clip(beta_minus, -1.0, 1.0).astype(np.float32)),
                (
                    f"beta_plus_{scale}_alpha_minus_{scale}",
                    np.clip(beta_plus_alpha_minus, -1.0, 1.0).astype(np.float32),
                ),
            ]
        )
    return _dedupe_labeled_offsets(candidates)


def _axis_rule_offset_for_action(
    action: dict,
    *,
    act_dim: int,
    step: float,
    beta_threshold: float,
    attack_name: str | None = None,
) -> tuple[str, np.ndarray]:
    act_dim = int(act_dim)
    if act_dim < 2:
        raise ValueError("axis_rule_offset requires defender action dimension >= 2")
    beta = float(action.get("beta", float("nan")))
    offset = np.zeros(act_dim, dtype=np.float32)
    if str(attack_name or "") == "dba" and np.isfinite(beta) and beta >= float(beta_threshold):
        return "zero", offset
    if np.isfinite(beta) and beta >= float(beta_threshold):
        offset[0] = -abs(float(step))
        return "alpha_minus_1", np.clip(offset, -1.0, 1.0).astype(np.float32)
    offset[1] = abs(float(step))
    return "beta_plus_1", np.clip(offset, -1.0, 1.0).astype(np.float32)


def _axis_rule_v2_offset_for_action(
    action: dict,
    *,
    act_dim: int,
    step: float,
    beta_threshold: float,
    low_beta_threshold: float,
    attack_name: str | None = None,
) -> tuple[str, np.ndarray]:
    act_dim = int(act_dim)
    if act_dim < 2:
        raise ValueError("axis_rule_v2_offset requires defender action dimension >= 2")
    beta = float(action.get("beta", float("nan")))
    offset = np.zeros(act_dim, dtype=np.float32)
    step = abs(float(step))
    low_beta_threshold = min(float(low_beta_threshold), float(beta_threshold))
    if np.isfinite(beta) and beta >= low_beta_threshold:
        if str(attack_name or "") == "dba":
            return "zero", offset
        offset[0] = -step
        return "alpha_minus_1", np.clip(offset, -1.0, 1.0).astype(np.float32)
    offset[0] = -step
    offset[1] = step
    return "alpha_minus_1_beta_plus_1", np.clip(offset, -1.0, 1.0).astype(np.float32)


def _physical_rule_target_for_action(
    action: dict,
    *,
    act_dim: int,
    step: float,
    low_beta_threshold: float,
    beta_threshold: float,
    low_alpha_target: float,
    near_alpha_target: float,
    beta_target: float,
    attack_name: str | None = None,
) -> dict:
    act_dim = int(act_dim)
    if act_dim < 2:
        raise ValueError("physical_rule_target requires defender action dimension >= 2")
    beta = float(action.get("beta", float("nan")))
    offset = np.zeros(act_dim, dtype=np.float32)
    low_beta_threshold = min(float(low_beta_threshold), float(beta_threshold))
    if np.isfinite(beta) and beta < low_beta_threshold:
        return {
            "selected_offset_label": f"physical_a{float(low_alpha_target):.2f}_b{float(beta_target):.2f}",
            "offset": offset,
            "target_alpha": float(low_alpha_target),
            "target_beta": float(beta_target),
        }
    if np.isfinite(beta) and beta < float(beta_threshold):
        if str(attack_name or "") == "dba":
            return {
                "selected_offset_label": "zero",
                "offset": offset,
                "target_alpha": None,
                "target_beta": None,
            }
        return {
            "selected_offset_label": f"physical_a{float(near_alpha_target):.2f}_b{float(beta_target):.2f}",
            "offset": offset,
            "target_alpha": float(near_alpha_target),
            "target_beta": float(beta_target),
        }
    if str(attack_name or "") == "dba":
        return {
            "selected_offset_label": "zero",
            "offset": offset,
            "target_alpha": None,
            "target_beta": None,
        }
    offset[0] = -abs(float(step))
    return {
        "selected_offset_label": "alpha_minus_1",
        "offset": np.clip(offset, -1.0, 1.0).astype(np.float32),
        "target_alpha": None,
        "target_beta": None,
    }


def _physical_target_candidates(
    *,
    act_dim: int,
    alpha_candidates: str,
    beta_target: float,
    start_round: int = 0,
    end_round_candidates: list[int | None] | None = None,
) -> list[dict]:
    act_dim = int(act_dim)
    if act_dim < 2:
        raise ValueError("physical_target_selector requires defender action dimension >= 2")
    zero_offset = [0.0 for _ in range(act_dim)]
    candidates = [
        {
            "offset_label": "zero",
            "offset": list(zero_offset),
            "target_alpha": None,
            "target_beta": None,
        }
    ]
    window_candidates = list(end_round_candidates or [])
    expand_windows = bool(window_candidates)
    start_round = max(0, int(start_round))
    if expand_windows:
        candidates[0]["target_start_round"] = None
        candidates[0]["target_end_round"] = None
    for alpha in _parse_float_candidates(alpha_candidates):
        base_label = f"physical_a{float(alpha):.2f}_b{float(beta_target):.2f}"
        if not expand_windows:
            candidates.append(
                {
                    "offset_label": base_label,
                    "offset": list(zero_offset),
                    "target_alpha": float(alpha),
                    "target_beta": float(beta_target),
                }
            )
            continue
        for end_round in window_candidates:
            label = f"{base_label}_full" if end_round is None else f"{base_label}_w{start_round}_{int(end_round)}"
            candidates.append(
                {
                    "offset_label": label,
                    "offset": list(zero_offset),
                    "target_alpha": float(alpha),
                    "target_beta": float(beta_target),
                    "target_start_round": start_round,
                    "target_end_round": None if end_round is None else int(end_round),
                }
            )
    return candidates


def _parse_float_candidates(value: str | list[float] | tuple[float, ...]) -> list[float]:
    if isinstance(value, (list, tuple)):
        raw_values = list(value)
    else:
        raw_values = [part.strip() for part in str(value).split(",")]
    parsed: list[float] = []
    seen = set()
    for raw in raw_values:
        if raw == "":
            continue
        candidate = float(raw)
        if not np.isfinite(candidate):
            continue
        key = round(candidate, 8)
        if key in seen:
            continue
        seen.add(key)
        parsed.append(float(candidate))
    if not parsed:
        raise ValueError("physical target selector requires at least one alpha candidate")
    return parsed


def _parse_optional_int_candidates(value: str | list[int | None] | tuple[int | None, ...] | None) -> list[int | None]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw_values = list(value)
    else:
        raw_values = [part.strip() for part in str(value).split(",")]
    parsed: list[int | None] = []
    seen = set()
    for raw in raw_values:
        if raw is None:
            candidate = None
        else:
            text = str(raw).strip().lower()
            if text == "":
                continue
            if text in {"full", "none", "null"}:
                candidate = None
            else:
                candidate = int(text)
                if candidate < 0:
                    raise ValueError("round candidates must be non-negative")
        key = "full" if candidate is None else int(candidate)
        if key in seen:
            continue
        seen.add(key)
        parsed.append(candidate)
    return parsed


def _candidate_target_start_round(args, candidate: dict) -> int:
    if "target_start_round" in candidate and candidate.get("target_start_round") is not None:
        return max(0, int(candidate["target_start_round"]))
    return max(0, int(getattr(args, "physical_target_start_round", 0)))


def _candidate_target_end_round(args, candidate: dict) -> int | None:
    if "target_end_round" in candidate:
        end_round = candidate.get("target_end_round")
        return None if end_round is None else max(0, int(end_round))
    end_round = getattr(args, "physical_target_end_round", None)
    return None if end_round is None else max(0, int(end_round))


def _physical_target_policy_for_candidate(args, defender, config: BSMGConfig, candidate: dict):
    target_alpha = candidate.get("target_alpha")
    target_beta = candidate.get("target_beta")
    if target_alpha is None and target_beta is None:
        return defender
    end_round = _candidate_target_end_round(args, candidate)
    if end_round is None:
        return PhysicalTargetPolicy(
            defender,
            config,
            target_alpha=None if target_alpha is None else float(target_alpha),
            target_beta=None if target_beta is None else float(target_beta),
        )
    return ScheduledPhysicalTargetPolicy(
        defender,
        config,
        target_alpha=None if target_alpha is None else float(target_alpha),
        target_beta=None if target_beta is None else float(target_beta),
        start_round=_candidate_target_start_round(args, candidate),
        end_round=end_round,
    )


def _raw_for_physical_target(value: float, *, min_value: float, max_value: float) -> float:
    span = float(max_value) - float(min_value)
    if span <= 0.0:
        return 0.0
    return float(2.0 * (float(value) - float(min_value)) / span - 1.0)


def _dedupe_labeled_offsets(candidates: list[tuple[str, np.ndarray]]) -> list[tuple[str, np.ndarray]]:
    seen = set()
    result: list[tuple[str, np.ndarray]] = []
    for label, offset in candidates:
        arr = np.asarray(offset, dtype=np.float32)
        key = tuple(np.round(arr, 8).tolist())
        if key in seen:
            continue
        seen.add(key)
        result.append((label, arr))
    return result


def _select_beta_offset_from_query_records(
    candidates: list[dict],
    *,
    asr_reduction_margin: float,
    clean_floor: float | None,
    clean_drop_tolerance: float | None,
) -> tuple[dict, dict]:
    if not candidates:
        raise ValueError("beta offset selection requires at least one candidate")
    enriched = [{**candidate, **_beta_offset_query_summary(candidate.get("query_records", []))} for candidate in candidates]
    base = next(
        (
            item
            for item in enriched
            if np.allclose(np.asarray(item["offset"], dtype=np.float32), 0.0)
        ),
        enriched[0],
    )
    base_backdoor = float(base["query_backdoor_mean"])
    base_clean = float(base["query_clean_mean"])
    eligible = [
        item
        for item in enriched
        if item is not base
        and not bool(item.get("pruned", False))
        and _beta_offset_candidate_clean_accept(
            base_clean=base_clean,
            adapted_clean=float(item["query_clean_mean"]),
            clean_floor=clean_floor,
            clean_drop_tolerance=clean_drop_tolerance,
        )
        and base_backdoor - float(item["query_backdoor_mean"]) >= float(asr_reduction_margin) - 1e-12
    ]
    selected = max(eligible, key=lambda item: _beta_offset_rank_key(item, base)) if eligible else base
    reduction = base_backdoor - float(selected["query_backdoor_mean"])
    score_gain = float(selected["query_score_mean"]) - float(base["query_score_mean"])
    accepted = selected is not base
    return selected, {
        "accepted": bool(accepted),
        "selected": "adapted" if accepted else "base",
        "base_offset_label": str(base.get("offset_label", "base")),
        "selected_offset_label": str(selected.get("offset_label", "base")),
        "base_score_mean": float(base["query_score_mean"]),
        "adapted_score_mean": float(selected["query_score_mean"]),
        "score_gain": score_gain,
        "base_clean_mean": base_clean,
        "adapted_clean_mean": float(selected["query_clean_mean"]),
        "base_backdoor_mean": base_backdoor,
        "adapted_backdoor_mean": float(selected["query_backdoor_mean"]),
        "backdoor_reduction": reduction,
    }


def _select_axis_offset_from_query_records(
    candidates: list[dict],
    *,
    asr_reduction_margin: float,
    pessimistic_asr_margin: float,
    clean_floor: float | None,
    clean_drop_tolerance: float | None,
) -> tuple[dict, dict]:
    if not candidates:
        raise ValueError("axis offset selection requires at least one candidate")
    enriched = [{**candidate, **_beta_offset_query_summary(candidate.get("query_records", []))} for candidate in candidates]
    base = next(
        (
            item
            for item in enriched
            if np.allclose(np.asarray(item["offset"], dtype=np.float32), 0.0)
        ),
        enriched[0],
    )
    base_clean = float(base["query_clean_mean"])
    for item in enriched:
        reductions = _axis_offset_backdoor_reductions(
            base.get("query_records", []),
            item.get("query_records", []),
        )
        if reductions:
            item["query_backdoor_reduction_mean"] = float(np.mean(reductions))
            item["query_backdoor_reduction_min"] = float(np.min(reductions))
        else:
            item["query_backdoor_reduction_mean"] = float("nan")
            item["query_backdoor_reduction_min"] = float("nan")

    eligible = [
        item
        for item in enriched
        if item is not base
        and _beta_offset_candidate_clean_accept(
            base_clean=base_clean,
            adapted_clean=float(item["query_clean_mean"]),
            clean_floor=clean_floor,
            clean_drop_tolerance=clean_drop_tolerance,
        )
        and float(item["query_backdoor_reduction_mean"]) >= float(asr_reduction_margin) - 1e-12
        and float(item["query_backdoor_reduction_min"]) >= float(pessimistic_asr_margin) - 1e-12
    ]
    selected = max(eligible, key=lambda item: _axis_offset_rank_key(item, base)) if eligible else base
    accepted = selected is not base
    score_gain = float(selected["query_score_mean"]) - float(base["query_score_mean"])
    return selected, {
        "accepted": bool(accepted),
        "selected": "adapted" if accepted else "base",
        "base_offset_label": str(base.get("offset_label", "base")),
        "selected_offset_label": str(selected.get("offset_label", "base")),
        "base_score_mean": float(base["query_score_mean"]),
        "adapted_score_mean": float(selected["query_score_mean"]),
        "score_gain": score_gain,
        "base_clean_mean": base_clean,
        "adapted_clean_mean": float(selected["query_clean_mean"]),
        "base_backdoor_mean": float(base["query_backdoor_mean"]),
        "adapted_backdoor_mean": float(selected["query_backdoor_mean"]),
        "backdoor_reduction_mean": float(selected["query_backdoor_reduction_mean"]),
        "backdoor_reduction_min": float(selected["query_backdoor_reduction_min"]),
    }


def _select_physical_target_from_query_records(
    candidates: list[dict],
    *,
    asr_reduction_margin: float,
    clean_floor: float | None,
    clean_drop_tolerance: float | None,
    score_slack: float = 0.0,
) -> tuple[dict, dict]:
    if not candidates:
        raise ValueError("physical target selection requires at least one candidate")
    enriched = [{**candidate, **_beta_offset_query_summary(candidate.get("query_records", []))} for candidate in candidates]
    base = next(
        (
            item
            for item in enriched
            if str(item.get("offset_label", "")) == "zero"
            or (item.get("target_alpha") is None and item.get("target_beta") is None)
        ),
        enriched[0],
    )
    base_backdoor = float(base["query_backdoor_mean"])
    base_clean = float(base["query_clean_mean"])
    eligible = [
        item
        for item in enriched
        if item is not base
        and _beta_offset_candidate_clean_accept(
            base_clean=base_clean,
            adapted_clean=float(item["query_clean_mean"]),
            clean_floor=clean_floor,
            clean_drop_tolerance=clean_drop_tolerance,
        )
        and base_backdoor - float(item["query_backdoor_mean"]) >= float(asr_reduction_margin) - 1e-12
    ]
    selected = _select_clean_aware_physical_target(eligible, base, score_slack=float(score_slack)) if eligible else base
    reduction = base_backdoor - float(selected["query_backdoor_mean"])
    score_gain = float(selected["query_score_mean"]) - float(base["query_score_mean"])
    accepted = selected is not base
    return selected, {
        "accepted": bool(accepted),
        "selected": "adapted" if accepted else "base",
        "base_offset_label": str(base.get("offset_label", "base")),
        "selected_offset_label": str(selected.get("offset_label", "base")),
        "target_alpha": None if selected.get("target_alpha") is None else float(selected["target_alpha"]),
        "target_beta": None if selected.get("target_beta") is None else float(selected["target_beta"]),
        "target_start_round": None
        if selected.get("target_start_round") is None
        else int(selected["target_start_round"]),
        "target_end_round": None
        if selected.get("target_end_round") is None
        else int(selected["target_end_round"]),
        "base_score_mean": float(base["query_score_mean"]),
        "adapted_score_mean": float(selected["query_score_mean"]),
        "score_gain": score_gain,
        "base_clean_mean": base_clean,
        "adapted_clean_mean": float(selected["query_clean_mean"]),
        "base_backdoor_mean": base_backdoor,
        "adapted_backdoor_mean": float(selected["query_backdoor_mean"]),
        "backdoor_reduction": reduction,
        "score_slack": float(score_slack),
    }


def _select_physical_target_with_deployment_records(
    candidates: list[dict],
    *,
    asr_reduction_margin: float,
    clean_floor: float | None,
    clean_drop_tolerance: float | None,
    score_slack: float = 0.0,
    deployment_clean_floor: float | None,
    deployment_clean_drop_tolerance: float | None,
    deployment_asr_ceiling: float | None,
) -> tuple[dict, dict]:
    if not candidates:
        raise ValueError("physical target deployment selection requires at least one candidate")
    enriched = [
        {
            **candidate,
            **_beta_offset_query_summary(candidate.get("query_records", [])),
            **_deployment_record_summary(candidate.get("deployment_record", {})),
        }
        for candidate in candidates
    ]
    base = next(
        (
            item
            for item in enriched
            if str(item.get("offset_label", "")) == "zero"
            or (item.get("target_alpha") is None and item.get("target_beta") is None)
        ),
        enriched[0],
    )
    base_backdoor = float(base["query_backdoor_mean"])
    base_clean = float(base["query_clean_mean"])
    query_eligible = [
        item
        for item in enriched
        if item is not base
        and _beta_offset_candidate_clean_accept(
            base_clean=base_clean,
            adapted_clean=float(item["query_clean_mean"]),
            clean_floor=clean_floor,
            clean_drop_tolerance=clean_drop_tolerance,
        )
        and base_backdoor - float(item["query_backdoor_mean"]) >= float(asr_reduction_margin) - 1e-12
    ]
    query_selected = (
        _select_clean_aware_physical_target(query_eligible, base, score_slack=float(score_slack))
        if query_eligible
        else base
    )
    base_deployment_clean = float(base["deployment_clean_acc"])
    deployment_eligible = [
        item
        for item in query_eligible
        if _deployment_clean_recovery_accept(
            item,
            base_clean=base_deployment_clean,
            deployment_clean_floor=deployment_clean_floor,
            deployment_clean_drop_tolerance=deployment_clean_drop_tolerance,
            deployment_asr_ceiling=deployment_asr_ceiling,
        )
    ]
    has_deployment_constraint = (
        deployment_clean_floor is not None
        or deployment_clean_drop_tolerance is not None
        or deployment_asr_ceiling is not None
    )
    selected = max(deployment_eligible, key=_deployment_clean_recovery_rank_key) if deployment_eligible else (
        base if has_deployment_constraint else query_selected
    )
    accepted = selected is not base
    reduction = base_backdoor - float(selected["query_backdoor_mean"])
    score_gain = float(selected["query_score_mean"]) - float(base["query_score_mean"])
    deployment_clean_drop = (
        base_deployment_clean - float(selected["deployment_clean_acc"])
        if np.isfinite(base_deployment_clean) and np.isfinite(float(selected["deployment_clean_acc"]))
        else float("nan")
    )
    return selected, {
        "accepted": bool(accepted),
        "selected": "adapted" if accepted else "base",
        "base_offset_label": str(base.get("offset_label", "base")),
        "selected_offset_label": str(selected.get("offset_label", "base")),
        "query_selected_offset_label": str(query_selected.get("offset_label", "base")),
        "selection_stage": "deployment_clean_recovery"
        if deployment_eligible
        else ("deployment_rejected" if has_deployment_constraint else "query_fallback"),
        "target_alpha": None if selected.get("target_alpha") is None else float(selected["target_alpha"]),
        "target_beta": None if selected.get("target_beta") is None else float(selected["target_beta"]),
        "target_start_round": None
        if selected.get("target_start_round") is None
        else int(selected["target_start_round"]),
        "target_end_round": None
        if selected.get("target_end_round") is None
        else int(selected["target_end_round"]),
        "base_score_mean": float(base["query_score_mean"]),
        "adapted_score_mean": float(selected["query_score_mean"]),
        "score_gain": score_gain,
        "base_clean_mean": base_clean,
        "adapted_clean_mean": float(selected["query_clean_mean"]),
        "base_backdoor_mean": base_backdoor,
        "adapted_backdoor_mean": float(selected["query_backdoor_mean"]),
        "backdoor_reduction": reduction,
        "score_slack": float(score_slack),
        "deployment_clean_floor": None
        if deployment_clean_floor is None
        else float(deployment_clean_floor),
        "deployment_clean_drop_tolerance": None
        if deployment_clean_drop_tolerance is None
        else float(deployment_clean_drop_tolerance),
        "deployment_asr_ceiling": None
        if deployment_asr_ceiling is None
        else float(deployment_asr_ceiling),
        "base_deployment_clean_acc": base_deployment_clean,
        "deployment_clean_acc": float(selected["deployment_clean_acc"]),
        "deployment_clean_drop": float(deployment_clean_drop),
        "deployment_backdoor_acc": float(selected["deployment_backdoor_acc"]),
        "deployment_defense_score": float(selected["deployment_defense_score"]),
        "query_eligible_count": int(len(query_eligible)),
        "deployment_eligible_count": int(len(deployment_eligible)),
    }


def _beta_offset_candidate_clean_accept(
    *,
    base_clean: float,
    adapted_clean: float,
    clean_floor: float | None,
    clean_drop_tolerance: float | None,
) -> bool:
    if not np.isfinite(adapted_clean):
        return False
    if clean_floor is not None and adapted_clean < float(clean_floor):
        return False
    if clean_drop_tolerance is not None:
        if not np.isfinite(base_clean):
            return False
        if base_clean - adapted_clean > float(clean_drop_tolerance) + 1e-12:
            return False
    return True


def _deployment_clean_recovery_accept(
    candidate: dict,
    *,
    base_clean: float,
    deployment_clean_floor: float | None,
    deployment_clean_drop_tolerance: float | None,
    deployment_asr_ceiling: float | None,
) -> bool:
    clean = float(candidate.get("deployment_clean_acc", float("nan")))
    backdoor = float(candidate.get("deployment_backdoor_acc", float("nan")))
    if not np.isfinite(clean) or not np.isfinite(backdoor):
        return False
    if deployment_clean_floor is not None and clean < float(deployment_clean_floor) - 1e-12:
        return False
    if deployment_clean_drop_tolerance is not None:
        if not np.isfinite(float(base_clean)):
            return False
        if float(base_clean) - clean > float(deployment_clean_drop_tolerance) + 1e-12:
            return False
    if deployment_asr_ceiling is not None and backdoor > float(deployment_asr_ceiling) + 1e-12:
        return False
    return True


def _deployment_clean_recovery_rank_key(candidate: dict) -> tuple[float, float, float, float]:
    return (
        float(candidate["deployment_clean_acc"]),
        -float(candidate["deployment_backdoor_acc"]),
        float(candidate["deployment_defense_score"]),
        float(candidate["query_score_mean"]),
    )


def _deployment_record_summary(record: dict) -> dict:
    return {
        "deployment_clean_acc": _finite_record_value(record, "final_clean_acc"),
        "deployment_backdoor_acc": _finite_record_value(record, "final_backdoor_acc"),
        "deployment_defense_score": _finite_record_value(record, "final_defense_score"),
    }


def _finite_record_value(record: dict, key: str) -> float:
    value = float(record.get(key, float("nan")))
    return value if np.isfinite(value) else float("nan")


def _beta_offset_rank_key(candidate: dict, base: dict) -> tuple[float, float, float, float]:
    reduction = float(base["query_backdoor_mean"]) - float(candidate["query_backdoor_mean"])
    score_gain = float(candidate["query_score_mean"]) - float(base["query_score_mean"])
    return (
        reduction,
        score_gain,
        float(candidate["query_clean_mean"]),
        -float(candidate["query_backdoor_mean"]),
    )


def _physical_target_rank_key(candidate: dict, base: dict) -> tuple[float, float, float, float]:
    reduction = float(base["query_backdoor_mean"]) - float(candidate["query_backdoor_mean"])
    score_gain = float(candidate["query_score_mean"]) - float(base["query_score_mean"])
    return (
        score_gain,
        reduction,
        float(candidate["query_clean_mean"]),
        -float(candidate["query_backdoor_mean"]),
    )


def _select_clean_aware_physical_target(candidates: list[dict], base: dict, *, score_slack: float) -> dict:
    best_by_score = max(candidates, key=lambda item: _physical_target_rank_key(item, base))
    best_score = float(best_by_score["query_score_mean"])
    score_slack = max(0.0, float(score_slack))
    near_best = [
        item
        for item in candidates
        if best_score - float(item["query_score_mean"]) <= score_slack + 1e-12
    ]
    return max(
        near_best,
        key=lambda item: (
            float(item["query_clean_mean"]),
            -float(item["query_backdoor_mean"]),
            float(item["query_score_mean"]),
            float(base["query_backdoor_mean"]) - float(item["query_backdoor_mean"]),
        ),
    )


def _axis_offset_rank_key(candidate: dict, base: dict) -> tuple[float, float, float, float, float]:
    score_gain = float(candidate["query_score_mean"]) - float(base["query_score_mean"])
    return (
        float(candidate["query_backdoor_reduction_min"]),
        float(candidate["query_backdoor_reduction_mean"]),
        score_gain,
        float(candidate["query_clean_mean"]),
        -float(candidate["query_backdoor_mean"]),
    )


def _axis_offset_prune_reason(
    base_records: list[dict],
    candidate_records: list[dict],
    *,
    pessimistic_asr_margin: float,
) -> str | None:
    reductions = _axis_offset_backdoor_reductions(base_records, candidate_records)
    if reductions and float(np.min(reductions)) < float(pessimistic_asr_margin) - 1e-12:
        return "pessimistic_asr"
    return None


def _axis_offset_backdoor_reductions(base_records: list[dict], candidate_records: list[dict]) -> list[float]:
    reductions: list[float] = []
    for base, candidate in zip(base_records, candidate_records):
        if "final_backdoor_acc" not in base or "final_backdoor_acc" not in candidate:
            continue
        base_backdoor = float(base["final_backdoor_acc"])
        candidate_backdoor = float(candidate["final_backdoor_acc"])
        if np.isfinite(base_backdoor) and np.isfinite(candidate_backdoor):
            reductions.append(base_backdoor - candidate_backdoor)
    return reductions


def _beta_offset_query_summary(records: list[dict]) -> dict:
    return {
        "query_score_mean": _record_mean(records, "final_defense_score"),
        "query_clean_mean": _record_mean(records, "final_clean_acc"),
        "query_backdoor_mean": _record_mean(records, "final_backdoor_acc"),
    }


def _record_mean(records: list[dict], key: str) -> float:
    values = [
        float(record[key])
        for record in records
        if key in record and np.isfinite(float(record[key]))
    ]
    return float(np.mean(values)) if values else float("nan")


def _beta_offset_deployment_beta_blocked(action: dict, *, beta_ceiling: float | None) -> bool:
    if beta_ceiling is None:
        return False
    beta = float(action.get("beta", float("nan")))
    return bool(np.isfinite(beta) and beta >= float(beta_ceiling))


def _scheduled_offset_policy_for_candidate(args, defender, offset: np.ndarray):
    if np.allclose(offset, 0.0):
        return defender
    return ScheduledActionOffsetPolicy(
        defender,
        offset,
        start_round=int(args.beta_offset_start_round),
        end_round=args.beta_offset_end_round,
    )


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


def _scenario_initial_action_diagnostics(args, defender, scenario: Scenario, *, horizon: int) -> dict:
    env = _make_env(args, scenario, seed=int(scenario.seed), horizon=int(horizon))
    obs = env.reset(seed=int(scenario.seed))
    return _action_diagnostics(defender, obs, _bsmg_config(args, horizon=int(horizon)))


def _make_env(args, scenario: Scenario, *, seed: int, horizon: int) -> BSMGEnv:
    coordinator = FLSandboxCoordinatorAdapter(
        _sandbox_config(args, scenario.attack_name, seed=seed, patch=scenario.patch)
    )
    attack_strategy = None if scenario.attack_name == "clean" else SandboxAttackMarker(scenario.attack_type)
    post_training_evaluator = _post_training_evaluator_for_args(args, coordinator)
    return BSMGEnv(
        coordinator=coordinator,
        attack_type=scenario.attack_type,
        attack_strategy=attack_strategy,
        defense_strategy=PaperDefenseStrategy(),
        config=_bsmg_config(args, horizon=horizon),
        evaluator=None
        if post_training_evaluator is not None
        else getattr(coordinator, "evaluate_weights", None),
        post_training_evaluator=post_training_evaluator,
    )


def _post_training_evaluator_for_args(args, coordinator):
    mode = str(getattr(args, "post_defense_mode", "weight_copy"))
    if mode not in {"model_aware_neuroclip", "model_aware_pruning"}:
        return None
    runner = getattr(coordinator, "runner", None)
    evaluate_model = getattr(coordinator, "evaluate_model", None)
    if runner is None or evaluate_model is None:
        raise ValueError(f"{mode} requires coordinator.runner.model and evaluate_model().")

    def evaluator(weights, decision: DefenseDecision) -> dict[str, float]:
        if mode == "model_aware_pruning":
            mask_rate = getattr(args, "fixed_pruning_mask_rate", None)
            if mask_rate is None:
                mask_rate = decision.prun_mask_rate
            if mask_rate is None:
                evaluate_weights = getattr(coordinator, "evaluate_weights", None)
                if evaluate_weights is None:
                    raise ValueError("No pruning mask rate was provided and coordinator cannot evaluate weights.")
                return evaluate_weights(weights)
            model = getattr(runner, "model", None)
            if model is None:
                raise ValueError("model_aware_pruning requires coordinator.runner.model at evaluation time.")
            defended_model = apply_post_defense(
                model,
                "pruning",
                float(mask_rate),
                eval_loader=getattr(runner, "test_loader", None),
                device=getattr(runner, "device", None),
            )
            return evaluate_model(defended_model, weights)
        epsilon = _neuroclip_epsilon_for_post_eval(args, decision)
        if epsilon is None:
            evaluate_weights = getattr(coordinator, "evaluate_weights", None)
            if evaluate_weights is None:
                raise ValueError("No NeuroClip epsilon was provided and coordinator cannot evaluate weights.")
            return evaluate_weights(weights)
        model = getattr(runner, "model", None)
        if model is None:
            raise ValueError("model_aware_neuroclip requires coordinator.runner.model at evaluation time.")
        defended_model = apply_post_defense(model, "neuroclip", float(epsilon))
        return evaluate_model(defended_model, weights)

    return evaluator


def _neuroclip_epsilon_for_post_eval(args, decision: DefenseDecision) -> float | None:
    fixed = getattr(args, "fixed_neuroclip_epsilon", None)
    if fixed is not None:
        return float(fixed)
    if decision.neuroclip_epsilon is not None:
        return float(decision.neuroclip_epsilon)
    return None


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
        eps_log_scale=config.eps_log_scale,
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
        eps_min=float(getattr(args, "neuroclip_eps_min", 1.0)),
        eps_max=float(getattr(args, "neuroclip_eps_max", 10.0)),
        eps_log_scale=bool(getattr(args, "neuroclip_log_scale", False)),
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
    clean_mixed_backdoor_scenarios = [
        Scenario("clean", _attack_type("clean"), "clean", {"num_attackers": 0}, int(args.seed)),
        Scenario(
            "mixed_backdoor",
            _attack_type("mixed_backdoor"),
            "mixed_backdoor",
            dict(backdoor, num_attackers=int(args.num_attackers)),
            int(args.rl_seed),
        ),
    ]
    clean_global_backdoor_mixed_scenarios = [
        *clean,
        *poisoning[1:],
        *backdoor_scenarios[1:],
        clean_mixed_backdoor_scenarios[1],
    ]
    if args.scenario_set == "backdoor":
        return _filter_scenarios(backdoor_scenarios, getattr(args, "scenario_filter", None))
    if args.scenario_set == "clean_mixed_backdoor":
        return _filter_scenarios(clean_mixed_backdoor_scenarios, getattr(args, "scenario_filter", None))
    if args.scenario_set == "clean_global_backdoor_mixed":
        return _filter_scenarios(clean_global_backdoor_mixed_scenarios, getattr(args, "scenario_filter", None))
    if args.scenario_set == "mixed":
        return _filter_scenarios(
            [
                *clean,
                *poisoning[1:],
                *backdoor_scenarios[1:],
            ],
            getattr(args, "scenario_filter", None),
        )
    return _filter_scenarios(poisoning, getattr(args, "scenario_filter", None))


def _filter_scenarios(scenarios: list[Scenario], scenario_filter: str | None) -> list[Scenario]:
    if scenario_filter is None or not str(scenario_filter).strip():
        return scenarios
    requested = [name.strip() for name in str(scenario_filter).split(",") if name.strip()]
    scenario_by_name = {scenario.name: scenario for scenario in scenarios}
    unknown = [name for name in requested if name not in scenario_by_name]
    if unknown:
        valid = ", ".join(scenario_by_name)
        raise ValueError(f"Unknown scenario(s) in --scenario-filter: {', '.join(unknown)}. Valid: {valid}")
    return [scenario_by_name[name] for name in requested]


def _attack_context_names(args) -> tuple[str, ...]:
    if not bool(getattr(args, "attack_context", False)):
        return ()
    if args.scenario_set == "backdoor":
        return ("bfl", "dba", "rl_backdoor")
    if args.scenario_set == "clean_mixed_backdoor":
        return ("mixed_backdoor",)
    if args.scenario_set == "clean_global_backdoor_mixed":
        return ("clean", "ipm", "lmp", "rl", "bfl", "dba", "rl_backdoor", "mixed_backdoor")
    if args.scenario_set == "mixed":
        return ("ipm", "lmp", "rl", "bfl", "dba", "rl_backdoor")
    return ("ipm", "lmp", "rl")


def _defender_action_dim(args) -> int:
    return 4 if str(args.defender_third_action) == "both" else 3


def _attack_type(name: str) -> AttackType:
    return AttackType(
        name=name,
        objective="targeted" if name in {"bfl", "dba", "rl_backdoor", "mixed_backdoor", "brl"} else "untargeted",
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
