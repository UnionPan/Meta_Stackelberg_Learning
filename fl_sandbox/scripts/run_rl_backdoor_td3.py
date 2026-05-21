"""TD3 paper-style RL backdoor attacker benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from fl_sandbox.attacks import create_attack
from fl_sandbox.attacks.rl_backdoor.action import decode_backdoor_action
from fl_sandbox.attacks.rl_backdoor.config import BackdoorRLConfig
from fl_sandbox.attacks.rl_backdoor.env import BackdoorPolicyGymEnv
from fl_sandbox.attacks.rl_backdoor.observation import BackdoorObservationBuilder
from fl_sandbox.attacks.rl_backdoor.attack import RLBackdoorAttack
from fl_sandbox.attacks.rl_backdoor.policy import TD3BackdoorPolicy
from fl_sandbox.attacks.rl_backdoor.reward import BackdoorRewardFn, BackdoorRewardInputs
from fl_sandbox.config import RunConfig
from fl_sandbox.core.experiment_service import rl_training_tensorboard_scalars
from fl_sandbox.federation.runner import MinimalFLRunner

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - tests can use the local box fallback.
    gym = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mnist", choices=("mnist", "fmnist", "cifar10"))
    parser.add_argument("--defense-type", default="fedavg")
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--train-horizon", type=int, default=20)
    parser.add_argument("--num-clients", type=int, default=100)
    parser.add_argument("--num-attackers", type=int, default=10)
    parser.add_argument("--subsample-rate", type=float, default=0.1)
    parser.add_argument("--client-samples", type=int, default=0, help="<=0 uses full client splits")
    parser.add_argument("--eval-samples", type=int, default=0, help="<=0 uses full target-class eval split")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--base-class", type=int, default=1)
    parser.add_argument("--target-class", type=int, default=7)
    parser.add_argument("--pattern-type", default="square")
    parser.add_argument("--bfl-poison-frac", type=float, default=0.5)
    parser.add_argument("--dba-poison-frac", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden-sizes", default="256,256")
    parser.add_argument("--policy-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--exploration-noise", type=float, default=0.15)
    parser.add_argument("--replay-capacity", type=int, default=50000)
    parser.add_argument("--rl-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=Path("fl_sandbox/outputs/rl_backdoor_td3"))
    parser.add_argument("--tb-dir", type=Path, default=None)
    parser.add_argument("--policy-checkpoint", type=Path, default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--eval-every", type=int, default=1, help="Kept for compatibility; this benchmark records every round.")
    parser.add_argument("--log-every", type=int, default=10, help="Print and flush live progress every N rounds.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rl_config = BackdoorRLConfig(
        seed=args.seed,
        train_horizon=args.train_horizon,
        train_steps=args.train_steps,
        hidden_sizes=tuple(int(part) for part in args.hidden_sizes.split(",") if part),
        policy_lr=args.policy_lr,
        critic_lr=args.critic_lr,
        exploration_noise=args.exploration_noise,
        replay_capacity=args.replay_capacity,
        batch_size=args.rl_batch_size,
    )
    env = BackdoorPolicyGymEnv(
        runner_factory=lambda seed_offset=0: MinimalFLRunner(
            _config(args, "rl_backdoor", rounds=args.train_horizon, seed=args.seed + int(seed_offset))
        ),
        config=rl_config,
    )
    policy = TD3BackdoorPolicy(rl_config)
    policy_path = args.output_dir / "rl_backdoor_td3_policy.pt"
    checkpoint_obs_dim = _checkpoint_observation_dim(args.policy_checkpoint) if args.policy_checkpoint else None
    policy_observation_space = env.observation_space
    legacy_obs_dim = None
    if checkpoint_obs_dim is not None and int(checkpoint_obs_dim) != int(env.observation_space.shape[0]):
        if int(checkpoint_obs_dim) != 1298:
            raise ValueError(
                f"Policy checkpoint observation dim {checkpoint_obs_dim} does not match current "
                f"tail-layer state dim {env.observation_space.shape[0]}. Retrain the TD3 policy "
                "for the new state; only legacy 1298-dim checkpoints are auto-compatible."
            )
        legacy_obs_dim = int(checkpoint_obs_dim)
        policy_observation_space = _box_space(low=-np.inf, high=np.inf, shape=(legacy_obs_dim,), dtype=np.float32)
    if args.policy_checkpoint:
        print(json.dumps({"phase": "load_policy", "path": str(args.policy_checkpoint), "obs_dim": int(policy_observation_space.shape[0])}), flush=True)
        policy.load(args.policy_checkpoint, policy_observation_space, env.action_space)
        train_stats = None
        policy_path = args.policy_checkpoint
    elif args.skip_train:
        print(json.dumps({"phase": "init_policy", "obs_dim": int(env.observation_space.shape[0])}), flush=True)
        policy.ensure_initialized(env.observation_space, env.action_space)
        train_stats = None
    else:
        print(json.dumps({"phase": "train_policy_start", "steps": args.train_steps, "horizon": args.train_horizon, "obs_dim": int(env.observation_space.shape[0])}), flush=True)
        train_stats = policy.train(env)
        policy.save(policy_path)
        print(
            json.dumps(
                {
                    "phase": "train_policy_done",
                    "policy": str(policy_path),
                    "collect_steps": train_stats.collect.steps,
                    "reward_mean": train_stats.collect.reward_mean,
                    "update_loss": train_stats.update.loss,
                }
            ),
            flush=True,
        )

    eval_rows = {
        "bfl": _run_fixed("bfl", args, action=None, seed=args.seed + 1000),
        "dba": _run_fixed("dba", args, action=None, seed=args.seed + 2000),
        "rl_backdoor_td3": _run_td3_policy(args, policy, rl_config, seed=args.seed + 3000, legacy_obs_dim=legacy_obs_dim),
    }
    payload = {
        "policy_path": str(policy_path),
        "train": {
            "collect_steps": train_stats.collect.steps if train_stats is not None else 0,
            "reward_mean": train_stats.collect.reward_mean if train_stats is not None else 0.0,
            "update_loss": train_stats.update.loss if train_stats is not None else 0.0,
        },
        "eval": eval_rows,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_round_csvs(args, eval_rows)
    _write_tensorboard(args, eval_rows)
    print("SUMMARY", summary_path)
    print("POLICY", policy_path)
    for name, row in eval_rows.items():
        print("RESULT", name, "clean=", round(row["final_clean_acc"], 4), "asr=", round(row["final_backdoor_acc"], 4), "mean_asr=", round(row["mean_backdoor_acc"], 4))


def _config(args: argparse.Namespace, attack_type: str, *, rounds: int, seed: int) -> RunConfig:
    return RunConfig.from_flat_dict(
        {
            "dataset": args.dataset,
            "attack_type": attack_type,
            "defense_type": args.defense_type,
            "rounds": rounds,
            "device": args.device,
            "num_clients": args.num_clients,
            "num_attackers": args.num_attackers,
            "subsample_rate": args.subsample_rate,
            "local_epochs": 1,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "max_client_samples_per_client": args.client_samples if args.client_samples > 0 else None,
            "max_eval_samples": args.eval_samples if args.eval_samples > 0 else None,
            "seed": seed,
            "bfl_poison_frac": args.bfl_poison_frac,
            "dba_poison_frac": args.dba_poison_frac,
            "dba_num_sub_triggers": 4,
            "base_class": args.base_class,
            "target_class": args.target_class,
            "pattern_type": args.pattern_type,
        }
    )


def _run_fixed(attack_type: str, args: argparse.Namespace, *, action, seed: int) -> dict[str, Any]:
    runner = MinimalFLRunner(_config(args, attack_type, rounds=args.rounds, seed=seed))
    attack = create_attack(runner.config.attacker)
    return _rollout(runner, attack, args.rounds, action_fn=lambda _summary: action, eval_every=args.eval_every, args=args, attack_name=attack_type)


def _run_td3_policy(
    args: argparse.Namespace,
    policy: TD3BackdoorPolicy,
    rl_config: BackdoorRLConfig,
    *,
    seed: int,
    legacy_obs_dim: int | None = None,
) -> dict[str, Any]:
    runner = MinimalFLRunner(_config(args, "rl_backdoor", rounds=args.rounds, seed=seed))
    attack = RLBackdoorAttack()
    observation_builder = BackdoorObservationBuilder(rl_config)
    reward_fn = BackdoorRewardFn(
        clean_weight=rl_config.reward_clean_weight,
        norm_weight=rl_config.reward_norm_weight,
    )
    previous_weights = [layer.copy() for layer in runner.current_weights]
    last_action = np.zeros(rl_config.action_dim, dtype=np.float32)
    last_clean = 0.0
    last_asr = 0.0

    def action_fn(summary):
        nonlocal previous_weights, last_action, last_clean, last_asr
        round_idx = int(getattr(summary, "round_idx", 0) or 0)
        clean = _finite(getattr(summary, "clean_acc", last_clean), last_clean) if summary is not None else last_clean
        asr = _finite(getattr(summary, "backdoor_acc", last_asr), last_asr) if summary is not None else last_asr
        sampled_attackers, sampled_clients = _attacker_sampling_counts(runner, max(1, round_idx + 1))
        attacker_frac = float(sampled_attackers) / max(1.0, float(len(getattr(runner, "attacker_ids", []) or [])))
        if legacy_obs_dim is not None:
            obs = _legacy_eval_observe(
                weights=runner.current_weights,
                last_action=last_action,
                round_idx=round_idx,
                total_rounds=args.rounds,
                clean_acc=clean,
                asr=asr,
                obs_dim=legacy_obs_dim,
                action_dim=rl_config.action_dim,
            )
        else:
            obs = observation_builder.build(
                weights=runner.current_weights,
                previous_weights=previous_weights,
                last_action=last_action,
                round_idx=round_idx,
                total_rounds=args.rounds,
                sampled_attacker_count=sampled_attackers,
                num_attackers=len(getattr(runner, "attacker_ids", []) or []),
                sampled_client_count=sampled_clients,
                clean_acc=clean,
                asr=asr,
            )
        previous_weights = [layer.copy() for layer in runner.current_weights]
        last_action = policy.act(obs, deterministic=True)
        metadata = _rl_action_metadata(last_action)
        metadata.update(_rl_state_metadata(obs))
        metadata["rl_attacker_fraction"] = attacker_frac
        if summary is not None:
            norm_ratio = _norm_ratio(summary)
            metadata["rl_real_reward"] = reward_fn(
                BackdoorRewardInputs(
                    asr_before=last_asr,
                    asr_after=asr,
                    clean_before=last_clean,
                    clean_after=clean,
                    norm_ratio=norm_ratio,
                )
            )
            metadata["rl_norm_ratio"] = norm_ratio
            last_clean = clean
            last_asr = asr
        else:
            metadata["rl_real_reward"] = 0.0
            metadata["rl_norm_ratio"] = 1.0
        return last_action, metadata

    return _rollout(runner, attack, args.rounds, action_fn=action_fn, eval_every=args.eval_every, args=args, attack_name="rl_backdoor_td3")


def _rollout(
    runner: MinimalFLRunner,
    attack,
    rounds: int,
    *,
    action_fn,
    eval_every: int,
    args: argparse.Namespace | None = None,
    attack_name: str = "attack",
) -> dict[str, Any]:
    summaries = []
    series: list[dict[str, float]] = []
    previous = None
    del eval_every
    live = _LiveRolloutLogger(args=args, attack_name=attack_name, rounds=rounds) if args is not None else None
    try:
        if live is not None:
            live.start()
        for round_idx in range(1, rounds + 1):
            action_payload = action_fn(previous)
            action, action_metadata = _normalize_action_payload(action_payload)
            previous = runner.run_round(round_idx, attack=attack, evaluate=True, attacker_action=action)
            summaries.append(previous)
            row = _round_metrics_row(previous)
            row.update(action_metadata)
            series.append(row)
            if live is not None:
                live.record(row)
    finally:
        if live is not None:
            live.close()
    clean_values = [_finite(s.clean_acc, float("nan")) for s in summaries]
    asr_values = [_finite(s.backdoor_acc, float("nan")) for s in summaries]
    clean_finite = [v for v in clean_values if math.isfinite(v)]
    asr_finite = [v for v in asr_values if math.isfinite(v)]
    return {
        "final_clean_acc": clean_finite[-1] if clean_finite else float("nan"),
        "final_backdoor_acc": asr_finite[-1] if asr_finite else float("nan"),
        "mean_clean_acc": float(np.mean(clean_finite)) if clean_finite else float("nan"),
        "mean_backdoor_acc": float(np.mean(asr_finite)) if asr_finite else float("nan"),
        "last100_backdoor_acc": float(np.mean(asr_finite[-100:])) if asr_finite else float("nan"),
        "series": series,
    }


class _LiveRolloutLogger:
    def __init__(self, *, args: argparse.Namespace, attack_name: str, rounds: int) -> None:
        self.args = args
        self.attack_name = attack_name
        self.rounds = int(rounds)
        self.log_every = max(1, int(getattr(args, "log_every", 10)))
        self.jsonl = None
        self.writer = None

    def start(self) -> None:
        out_dir = self.args.output_dir / self.attack_name
        out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = (out_dir / "round_metrics.live.jsonl").open("w", encoding="utf-8")
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_root = self.args.tb_dir or (self.args.output_dir / "tensorboard")
            self.writer = SummaryWriter(str(tb_root / self.attack_name))
            self.writer.add_text("config/json", json.dumps(vars(self.args), indent=2, default=str), global_step=0)
        except Exception:
            self.writer = None
        print(json.dumps({"phase": "rollout_start", "attack": self.attack_name, "rounds": self.rounds}), flush=True)

    def record(self, row: dict[str, float]) -> None:
        round_idx = int(row["round"])
        if self.jsonl is not None:
            self.jsonl.write(json.dumps(row, sort_keys=True) + "\n")
            self.jsonl.flush()
        if self.writer is not None:
            _write_standard_round_scalars(self.writer, row, round_idx, attack_type=self.attack_name)
            for tag, value in rl_training_tensorboard_scalars(row):
                self.writer.add_scalar(tag, value, round_idx)
            self.writer.flush()
        if round_idx <= 3 or round_idx == self.rounds or round_idx % self.log_every == 0:
            print(
                json.dumps(
                    {
                        "phase": "rollout_round",
                        "attack": self.attack_name,
                        "round": round_idx,
                        "clean_acc": round(row.get("clean_acc", float("nan")), 4),
                        "asr": round(row.get("asr", float("nan")), 4),
                        "selected_attackers": int(row.get("num_selected_attackers", 0.0)),
                        "round_seconds": round(row.get("round_seconds", 0.0), 3),
                    }
                ),
                flush=True,
            )

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        if self.jsonl is not None:
            self.jsonl.close()
        print(json.dumps({"phase": "rollout_done", "attack": self.attack_name}), flush=True)


def _write_round_csvs(args: argparse.Namespace, eval_rows: dict[str, dict[str, Any]]) -> None:
    for name, row in eval_rows.items():
        series = row.get("series", [])
        if not series:
            continue
        path = args.output_dir / name / "round_metrics.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for point in series for key in point.keys()})
        if "round" in fieldnames:
            fieldnames.remove("round")
            fieldnames.insert(0, "round")
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(series)


def _write_tensorboard(args: argparse.Namespace, eval_rows: dict[str, dict[str, Any]]) -> None:
    tb_dir = args.tb_dir or (args.output_dir / "tensorboard")
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        return
    for _step, (name, row) in enumerate(eval_rows.items()):
        writer = SummaryWriter(str(tb_dir / name))
        writer.add_text("config/json", json.dumps(vars(args), indent=2, default=str), global_step=0)
        for key, value in row.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                writer.add_scalar(f"final/{key}", float(value), int(args.rounds))
        for point in row.get("series", []):
            round_idx = int(point["round"])
            _write_standard_round_scalars(writer, point, round_idx, attack_type=name)
            for tag, value in rl_training_tensorboard_scalars(point):
                writer.add_scalar(tag, value, round_idx)
        writer.flush()
        writer.close()


def _write_standard_round_scalars(writer, point: dict[str, float], round_idx: int, *, attack_type: str) -> None:
    clean_loss = point.get("clean_loss")
    clean_acc = point.get("clean_acc")
    backdoor_acc = point.get("backdoor_acc")
    if _is_finite_number(clean_loss):
        writer.add_scalar("metrics/loss", float(clean_loss), round_idx)
    if _is_finite_number(clean_acc):
        writer.add_scalar("metrics/accuracy", float(clean_acc), round_idx)
        writer.add_scalar("paper/clean_acc", float(clean_acc), round_idx)
    if _is_finite_number(backdoor_acc):
        writer.add_scalar("metrics/backdoor_accuracy", float(backdoor_acc), round_idx)
        writer.add_scalar("metrics/asr", float(backdoor_acc), round_idx)
        writer.add_scalar("paper/backdoor_acc", float(backdoor_acc), round_idx)
        writer.add_scalar("paper/asr", float(backdoor_acc), round_idx)
    round_seconds = point.get("round_seconds")
    if _is_finite_number(round_seconds):
        writer.add_scalar("metrics/round_duration_seconds", float(round_seconds), round_idx)
    for key in ("num_sampled_clients", "num_selected_attackers", "mean_benign_norm"):
        value = point.get(key)
        if _is_finite_number(value):
            writer.add_scalar(f"metrics/{key}", float(value), round_idx)
    if attack_type != "clean":
        for key in ("mean_malicious_norm", "mean_malicious_cosine"):
            value = point.get(key)
            if _is_finite_number(value):
                writer.add_scalar(f"attack_only/{key}", float(value), round_idx)
        for key, value in point.items():
            if key.startswith("attack_") and _is_finite_number(value):
                writer.add_scalar(f"attack_only/{key.removeprefix('attack_')}", float(value), round_idx)


def _normalize_action_payload(payload) -> tuple[np.ndarray | None, dict[str, float]]:
    if isinstance(payload, tuple) and len(payload) == 2:
        action, metadata = payload
        return action, _numeric_metadata(metadata)
    return payload, {}


def _numeric_metadata(metadata) -> dict[str, float]:
    if not isinstance(metadata, dict):
        return {}
    return {
        str(key): float(value)
        for key, value in metadata.items()
        if isinstance(value, (int, float, np.number)) and math.isfinite(float(value))
    }


def _round_metrics_row(summary) -> dict[str, float]:
    row = {
        "round": float(getattr(summary, "round_idx", 0)),
        "clean_loss": _finite(getattr(summary, "clean_loss", float("nan")), float("nan")),
        "clean_acc": _finite(getattr(summary, "clean_acc", float("nan")), float("nan")),
        "backdoor_acc": _finite(getattr(summary, "backdoor_acc", float("nan")), float("nan")),
        "asr": _finite(getattr(summary, "backdoor_acc", float("nan")), float("nan")),
        "round_seconds": _finite(getattr(summary, "round_seconds", float("nan")), float("nan")),
        "num_sampled_clients": float(len(getattr(summary, "sampled_clients", []) or [])),
        "num_selected_attackers": float(len(getattr(summary, "selected_attackers", []) or [])),
        "mean_benign_norm": _mean_or_nan(getattr(summary, "benign_update_norms", []) or []),
        "mean_malicious_norm": _mean_or_nan(getattr(summary, "malicious_update_norms", []) or []),
        "mean_malicious_cosine": _mean_or_nan(getattr(summary, "malicious_cosines_to_benign", []) or []),
    }
    for key, value in (getattr(summary, "attack_metrics", {}) or {}).items():
        if isinstance(value, (int, float, np.number)) and math.isfinite(float(value)):
            row[f"attack_{key}"] = float(value)
    return row


def _rl_action_metadata(action: np.ndarray) -> dict[str, float]:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    decoded = decode_backdoor_action(action)
    payload = {
        "rl_action_poison_frac": float(decoded.poison_frac),
        "rl_action_local_lr": float(decoded.local_lr),
        "rl_action_local_epochs": float(decoded.local_epochs),
        "rl_action_boost": float(decoded.boost),
    }
    for idx, value in enumerate(action[:4]):
        payload[f"rl_action_raw_{idx}"] = float(value)
    return payload


def _rl_state_metadata(obs: np.ndarray) -> dict[str, float]:
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    if obs.size == 0:
        return {
            "rl_observation_dim": 0.0,
            "rl_observation_norm": 0.0,
            "rl_observation_mean": 0.0,
            "rl_observation_std": 0.0,
            "rl_observation_min": 0.0,
            "rl_observation_max": 0.0,
            "rl_observation_absmax": 0.0,
        }
    return {
        "rl_observation_dim": float(obs.size),
        "rl_observation_norm": float(np.linalg.norm(obs)),
        "rl_observation_mean": float(np.mean(obs)),
        "rl_observation_std": float(np.std(obs)),
        "rl_observation_min": float(np.min(obs)),
        "rl_observation_max": float(np.max(obs)),
        "rl_observation_absmax": float(np.max(np.abs(obs))),
    }


def _is_finite_number(value) -> bool:
    return isinstance(value, (int, float, np.number)) and math.isfinite(float(value))


def _mean_or_nan(values) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(values)) if values else float("nan")


def _norm_ratio(summary) -> float:
    benign = getattr(summary, "benign_update_norms", []) or []
    malicious = getattr(summary, "malicious_update_norms", []) or []
    benign_mean = _mean_or_nan(benign)
    malicious_mean = _mean_or_nan(malicious)
    if not math.isfinite(benign_mean) or benign_mean <= 1e-12 or not math.isfinite(malicious_mean):
        return 1.0
    return malicious_mean / benign_mean


def _legacy_eval_observe(
    *,
    weights,
    last_action,
    round_idx: int,
    total_rounds: int,
    clean_acc: float,
    asr: float,
    obs_dim: int,
    action_dim: int,
) -> np.ndarray:
    feature_dim = max(1, int(obs_dim) - int(action_dim) - 4)
    arrays = [np.asarray(layer, dtype=np.float32).reshape(-1) for layer in (weights or [])[-2:]]
    vec = np.concatenate(arrays).astype(np.float32) if arrays else np.zeros(feature_dim, dtype=np.float32)
    vec = (vec - float(np.mean(vec))) / (float(np.std(vec)) + 1e-6)
    vec = np.clip(vec, -5.0, 5.0).astype(np.float32)
    if vec.size < feature_dim:
        vec = np.pad(vec, (0, feature_dim - vec.size))
    else:
        vec = vec[:feature_dim]
    action = np.asarray(last_action, dtype=np.float32).reshape(-1)
    if action.size < action_dim:
        action = np.pad(action, (0, action_dim - action.size))
    feedback = np.asarray(
        [
            float(round_idx) / max(1.0, float(total_rounds)),
            0.0,
            float(np.nan_to_num(clean_acc, nan=0.0)),
            float(np.nan_to_num(asr, nan=0.0)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([vec, action[:action_dim], feedback], axis=0).astype(np.float32)


def _checkpoint_observation_dim(path: Path | None) -> int | None:
    if not path:
        return None
    try:
        import torch

        payload = torch.load(path, map_location="cpu")
    except Exception:
        return None
    algorithm = payload.get("algorithm", payload) if isinstance(payload, dict) else {}
    if not isinstance(algorithm, dict):
        return None
    for key, value in algorithm.items():
        if key.endswith("model.0.weight") and hasattr(value, "shape") and len(value.shape) == 2:
            return int(value.shape[1])
    return None


def _box_space(*, low, high, shape, dtype=np.float32):
    if gym is not None:
        return gym.spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
    low_array = np.full(shape, low, dtype=dtype)
    high_array = np.full(shape, high, dtype=dtype)
    return type("Box", (), {"low": low_array, "high": high_array, "shape": tuple(shape), "dtype": dtype})()


def _finite(value, fallback: float) -> float:
    try:
        value = float(value)
    except Exception:
        return float(fallback)
    return value if math.isfinite(value) else float(fallback)


def _attacker_fraction(runner, round_idx: int) -> float:
    sampled_attackers, _sampled_clients = _attacker_sampling_counts(runner, round_idx)
    attacker_ids = set(getattr(runner, "attacker_ids", []) or [])
    if not attacker_ids:
        return 0.0
    return float(sampled_attackers) / max(1.0, float(len(attacker_ids)))


def _attacker_sampling_counts(runner, round_idx: int) -> tuple[int, int]:
    attacker_ids = set(getattr(runner, "attacker_ids", []) or [])
    if not attacker_ids:
        return 0, 0
    sampled = set(runner._sample_clients(round_idx))
    return len(attacker_ids.intersection(sampled)), len(sampled)


if __name__ == "__main__":
    main(sys.argv[1:])
