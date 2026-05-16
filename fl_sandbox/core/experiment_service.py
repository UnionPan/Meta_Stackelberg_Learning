"""Run orchestration for sandbox experiments."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .experiment_builders import build_attack, build_config, default_output_dir, default_tb_dir
from .postprocess import build_postprocess_hint_lines
from .postprocess.tensorboard_utils import build_summary_writer
from .runtime import ExperimentTimer, client_metrics_to_rows, summaries_to_dict
from fl_sandbox.config import RunConfig


CLIENT_METRICS_FIELDNAMES = [
    "round_idx",
    "client_id",
    "selected",
    "is_attacker",
    "train_loss",
    "train_acc",
    "update_norm",
]

ROUND_METRICS_FIELDNAMES = [
    "round_idx",
    "clean_loss",
    "clean_acc",
    "backdoor_acc",
    "asr",
    "round_seconds",
    "num_sampled_clients",
    "num_selected_attackers",
]


RL_TRAINING_TENSORBOARD_TAGS = {
    "rl_trainer_actor_loss": "rl_training/actor_loss",
    "rl_trainer_critic1_loss": "rl_training/critic1_loss",
    "rl_trainer_critic2_loss": "rl_training/critic2_loss",
    "rl_trainer_loss": "rl_training/total_loss",
    "rl_trainer_last_update_loss": "rl_training/last_update_loss",
    "rl_trainer_reward_mean": "rl_training/reward_mean",
    "rl_real_reward": "rl_training/real_reward",
    "rl_simulated_reward": "rl_training/simulated_reward",
    "rl_sim2real_gap": "rl_training/sim2real_gap",
    "rl_trainer_replay_size": "rl_training/replay_size",
    "rl_trainer_update_steps": "rl_training/update_steps",
    "rl_trainer_collect_steps": "rl_training/collect_steps",
    "rl_trainer_train_time": "rl_training/train_time_seconds",
    "rl_action_gamma_scale": "rl_action/gamma_scale",
    "rl_action_local_steps": "rl_action/local_steps",
    "rl_action_lambda_stealth": "rl_action/lambda_stealth",
    "rl_action_template_index": "rl_action/template_index",
    "rl_action_raw_0": "rl_action/raw_0",
    "rl_action_raw_1": "rl_action/raw_1",
    "rl_action_raw_2": "rl_action/raw_2",
    "rl_observation_dim": "rl_observation/dim",
    "rl_observation_norm": "rl_observation/norm",
    "rl_observation_mean": "rl_observation/mean",
    "rl_observation_std": "rl_observation/std",
    "rl_observation_min": "rl_observation/min",
    "rl_observation_max": "rl_observation/max",
    "rl_observation_absmax": "rl_observation/absmax",
    "rl_gap_loss_mean": "rl_reward/loss_delta_mean",
    "rl_gap_acc_mean": "rl_reward/acc_delta_mean",
    "rl_gap_bypass_mean": "rl_reward/bypass_mean",
    "rl_gap_smoothness_mean": "rl_reward/smoothness_mean",
    "rl_gap_oob_mean": "rl_reward/action_saturation_mean",
    "rl_bypass_score": "rl_krum/bypass_score",
    "rl_krum_projection_applied": "rl_krum/projection_applied",
    "rl_krum_raw_selected": "rl_krum/raw_selected",
    "rl_krum_raw_rank": "rl_krum/raw_rank",
    "rl_krum_raw_score_ratio": "rl_krum/raw_score_ratio",
    "rl_krum_projected_selected": "rl_krum/projected_selected",
    "rl_krum_projected_rank": "rl_krum/projected_rank",
    "rl_krum_projected_score_ratio": "rl_krum/projected_score_ratio",
    "rl_krum_projection_alpha": "rl_krum/projection_alpha",
    "rl_krum_projection_max_alpha": "rl_krum/projection_max_alpha",
    "rl_krum_raw_delta_norm": "rl_krum/raw_delta_norm",
    "rl_krum_projected_delta_norm": "rl_krum/projected_delta_norm",
    "rl_krum_mean_benign_norm": "rl_krum/mean_benign_norm",
    "rl_krum_selected_attackers": "rl_krum/selected_attackers",
    "rl_krum_num_byzantine": "rl_krum/num_byzantine",
    "rl_krum_feasible_byzantine": "rl_krum/feasible_byzantine",
    "rl_krum_neighbor_count": "rl_krum/neighbor_count",
    "rl_krum_actual_selected": "rl_krum/actual_selected",
    "rl_krum_actual_best_rank": "rl_krum/actual_best_rank",
    "rl_krum_actual_score_ratio": "rl_krum/actual_score_ratio",
    "rl_krum_actual_feasible_byzantine": "rl_krum/actual_feasible_byzantine",
    "rl_krum_actual_neighbor_count": "rl_krum/actual_neighbor_count",
}


def rl_training_tensorboard_scalars(metrics: dict[str, object]) -> list[tuple[str, float]]:
    scalars: list[tuple[str, float]] = []
    for metric_key, tag in RL_TRAINING_TENSORBOARD_TAGS.items():
        value = metrics.get(metric_key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            scalars.append((tag, float(value)))

    critic1 = metrics.get("rl_trainer_critic1_loss")
    critic2 = metrics.get("rl_trainer_critic2_loss")
    if (
        isinstance(critic1, (int, float))
        and isinstance(critic2, (int, float))
        and math.isfinite(float(critic1))
        and math.isfinite(float(critic2))
    ):
        scalars.append(("rl_training/critic_loss", float(critic1) + float(critic2)))
    return scalars


class LiveMetricsLogger:
    """Append per-round metrics to disk and TensorBoard during execution."""

    def __init__(
        self,
        *,
        output_dir: Path,
        tb_dir: Path,
        attack_type: str,
        config_payload: dict[str, object],
        target_rounds: int,
        payload_factory,
    ) -> None:
        self.output_dir = output_dir
        self.tb_dir = tb_dir
        self.attack_type = attack_type
        self.config_payload = config_payload
        self.target_rounds = int(target_rounds)
        self._payload_factory = payload_factory
        self.csv_path = output_dir / "live_metrics.csv"
        self._csv_file = self.csv_path.open("w", encoding="utf-8", newline="")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=[
                "round_idx",
                "clean_loss",
                "clean_acc",
                "backdoor_acc",
                "asr",
                "round_seconds",
                "num_sampled_clients",
                "num_selected_attackers",
                "mean_benign_norm",
                "mean_malicious_norm",
                "mean_malicious_cosine",
            ],
        )
        self._csv_writer.writeheader()
        self._round_metrics_path = output_dir / "round_metrics.csv"
        self._round_metrics_file = self._round_metrics_path.open("w", encoding="utf-8", newline="")
        self._round_metrics_writer = csv.DictWriter(
            self._round_metrics_file,
            fieldnames=ROUND_METRICS_FIELDNAMES,
        )
        self._round_metrics_writer.writeheader()
        self._client_metrics_path = output_dir / "client_metrics.csv"
        self._client_metrics_file = self._client_metrics_path.open("w", encoding="utf-8", newline="")
        self._client_metrics_writer = csv.DictWriter(
            self._client_metrics_file,
            fieldnames=CLIENT_METRICS_FIELDNAMES,
        )
        self._client_metrics_writer.writeheader()
        self._writer = build_summary_writer(tb_dir)
        self._writer.add_text("config/json", json.dumps(config_payload, indent=2), global_step=0)
        self._writer.add_text("run/status", "in_progress", global_step=0)
        self._elapsed_seconds = 0.0
        self._summaries = []
        self._last_total_seconds = 0.0
        self._run_status = "in_progress"
        self._started_at = time.time()

    def log_round(self, summary) -> None:
        self._summaries.append(summary)
        mean_benign_norm = (
            float(sum(summary.benign_update_norms) / len(summary.benign_update_norms))
            if summary.benign_update_norms
            else 0.0
        )
        mean_malicious_norm = (
            float(sum(summary.malicious_update_norms) / len(summary.malicious_update_norms))
            if summary.malicious_update_norms
            else 0.0
        )
        mean_malicious_cosine = (
            float(sum(summary.malicious_cosines_to_benign) / len(summary.malicious_cosines_to_benign))
            if summary.malicious_cosines_to_benign
            else 0.0
        )
        row = {
            "round_idx": summary.round_idx,
            "clean_loss": summary.clean_loss,
            "clean_acc": summary.clean_acc,
            "backdoor_acc": summary.backdoor_acc,
            "asr": summary.backdoor_acc,
            "round_seconds": summary.round_seconds,
            "num_sampled_clients": len(summary.sampled_clients),
            "num_selected_attackers": len(summary.selected_attackers),
            "mean_benign_norm": mean_benign_norm,
            "mean_malicious_norm": mean_malicious_norm,
            "mean_malicious_cosine": mean_malicious_cosine,
        }
        self._csv_writer.writerow(row)
        self._csv_file.flush()
        self._round_metrics_writer.writerow(
            {
                "round_idx": summary.round_idx,
                "clean_loss": summary.clean_loss,
                "clean_acc": summary.clean_acc,
                "backdoor_acc": summary.backdoor_acc,
                "asr": summary.backdoor_acc,
                "round_seconds": summary.round_seconds,
                "num_sampled_clients": len(summary.sampled_clients),
                "num_selected_attackers": len(summary.selected_attackers),
            }
        )
        self._round_metrics_file.flush()
        self._client_metrics_writer.writerows(client_metrics_to_rows([summary]))
        self._client_metrics_file.flush()

        step = int(summary.round_idx)
        if not math.isnan(summary.clean_loss):
            self._writer.add_scalar("metrics/loss", summary.clean_loss, step)
        if not math.isnan(summary.clean_acc):
            self._writer.add_scalar("metrics/accuracy", summary.clean_acc, step)
            self._writer.add_scalar("paper/clean_acc", summary.clean_acc, step)
        if not math.isnan(summary.backdoor_acc):
            self._writer.add_scalar("metrics/backdoor_accuracy", summary.backdoor_acc, step)
            self._writer.add_scalar("metrics/asr", summary.backdoor_acc, step)
            self._writer.add_scalar("paper/backdoor_acc", summary.backdoor_acc, step)
            self._writer.add_scalar("paper/asr", summary.backdoor_acc, step)
        self._writer.add_scalar("metrics/round_duration_seconds", summary.round_seconds, step)
        self._elapsed_seconds += float(summary.round_seconds)
        self._writer.add_scalar("metrics/elapsed_seconds", self._elapsed_seconds, step)
        self._writer.add_scalar("metrics/num_sampled_clients", len(summary.sampled_clients), step)
        self._writer.add_scalar("metrics/num_selected_attackers", len(summary.selected_attackers), step)
        self._writer.add_scalar("metrics/mean_benign_norm", mean_benign_norm, step)
        if self.attack_type != "clean":
            self._writer.add_scalar("attack_only/mean_malicious_norm", mean_malicious_norm, step)
            self._writer.add_scalar("attack_only/mean_malicious_cosine", mean_malicious_cosine, step)
            for key, value in summary.attack_metrics.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    self._writer.add_scalar(f"attack_only/{key}", float(value), step)
            for tag, value in rl_training_tensorboard_scalars(summary.attack_metrics):
                self._writer.add_scalar(tag, value, step)
        self._writer.flush()
        self._write_partial_summary()

    def mark_completed(self, total_seconds: float) -> None:
        self._last_total_seconds = float(total_seconds)
        self._run_status = "completed"
        self._write_partial_summary()
        self._writer.add_text("run/status", "completed", global_step=max(len(self._summaries), 1))
        self._writer.flush()

    def mark_interrupted(self, total_seconds: float) -> None:
        self._last_total_seconds = float(total_seconds)
        self._run_status = "interrupted"
        self._write_partial_summary()
        self._writer.add_text("run/status", "interrupted", global_step=max(len(self._summaries), 1))
        self._writer.flush()

    def _write_partial_summary(self) -> None:
        payload = self._payload_factory(self._summaries, self._last_total_seconds or self._elapsed_seconds)
        payload["run_status"] = self._run_status
        payload["completed_rounds"] = len(self._summaries)
        payload["target_rounds"] = self.target_rounds
        payload["wall_clock_started_at"] = self._started_at
        payload["wall_clock_updated_at"] = time.time()
        write_summary_json(self.output_dir, payload)

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()
        self._csv_file.close()
        self._round_metrics_file.close()
        self._client_metrics_file.close()


class ExperimentCheckpointManager:
    """Persist lightweight per-round checkpoints for RL attacker runs."""

    def __init__(
        self,
        *,
        output_dir: Path,
        run_config: RunConfig,
        config_payload: dict[str, object],
        attack: Any,
        model: Any,
    ) -> None:
        self.output_dir = output_dir
        self.run_config = run_config
        self.config_payload = dict(config_payload)
        self.attack = attack
        self.model = model
        self.checkpoint_dir = output_dir / "checkpoints"

    def maybe_save(self, *, round_idx: int) -> list[Path]:
        if self.run_config.attacker.type != "rl":
            return []
        if not self._should_save(round_idx):
            return []

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        policy_paths = self._save_rl_policy(round_idx)
        paths.extend(policy_paths)
        model_paths = self._save_global_model(round_idx)
        paths.extend(model_paths)
        return paths

    def _should_save(self, round_idx: int) -> bool:
        interval = max(0, int(self.run_config.attacker.rl_checkpoint_interval or 0))
        is_interval_round = interval > 0 and int(round_idx) % interval == 0
        is_final_round = int(round_idx) == int(self.run_config.runtime.rounds)
        return is_interval_round or (bool(self.run_config.attacker.rl_save_final_checkpoint) and is_final_round)

    def _save_rl_policy(self, round_idx: int) -> list[Path]:
        trainer = getattr(self.attack, "trainer", None)
        if trainer is None or not hasattr(trainer, "save"):
            return []

        latest_path = self.checkpoint_dir / "rl_policy_latest.pt"
        round_path = self.checkpoint_dir / f"rl_policy_round_{int(round_idx):06d}.pt"
        paths = [latest_path, round_path]
        for path in paths:
            trainer.save(str(path))
            self._annotate_rl_policy_checkpoint(path, round_idx)
        return paths

    def _annotate_rl_policy_checkpoint(self, path: Path, round_idx: int) -> None:
        try:
            import torch

            payload = torch.load(path, map_location="cpu")
            if not isinstance(payload, dict):
                payload = {"payload": payload}
            payload.update(
                {
                    "kind": "rl_policy",
                    "round_idx": int(round_idx),
                    "config": self.config_payload,
                }
            )
            torch.save(payload, path)
        except Exception:
            # The trainer checkpoint itself has already been written; metadata is
            # helpful but should not make a long FL run fail.
            return

    def _save_global_model(self, round_idx: int) -> list[Path]:
        if self.model is None or not hasattr(self.model, "state_dict"):
            return []
        import torch

        payload = {
            "kind": "global_model",
            "round_idx": int(round_idx),
            "config": self.config_payload,
            "state_dict": self.model.state_dict(),
        }
        latest_path = self.checkpoint_dir / "global_model_latest.pt"
        round_path = self.checkpoint_dir / f"global_model_round_{int(round_idx):06d}.pt"
        for path in (latest_path, round_path):
            torch.save(payload, path)
        return [latest_path, round_path]


@dataclass
class ExperimentRunResult:
    args: Any
    run_config: RunConfig
    config: Any
    attack: Any
    output_dir: Path
    tb_dir: Path
    summaries: list[Any]
    series: dict[str, list[float]]
    payload: dict[str, object]
    total_seconds: float
    runtime_device: Any
    client_metric_rows: list[dict[str, object]]


def write_tensorboard_logs(
    tb_dir: Path,
    attack_type: str,
    summaries,
    series: dict[str, list[float]],
    total_seconds: float,
    config_payload: dict[str, object],
) -> None:
    writer = build_summary_writer(tb_dir)
    writer.add_text("config/json", json.dumps(config_payload, indent=2), global_step=0)

    for summary in summaries:
        if not math.isnan(summary.clean_loss):
            writer.add_scalar("metrics/loss", summary.clean_loss, summary.round_idx)
        if not math.isnan(summary.clean_acc):
            writer.add_scalar("metrics/accuracy", summary.clean_acc, summary.round_idx)
            writer.add_scalar("paper/clean_acc", summary.clean_acc, summary.round_idx)
        if not math.isnan(summary.backdoor_acc):
            writer.add_scalar("metrics/backdoor_accuracy", summary.backdoor_acc, summary.round_idx)
            writer.add_scalar("metrics/asr", summary.backdoor_acc, summary.round_idx)
            writer.add_scalar("paper/backdoor_acc", summary.backdoor_acc, summary.round_idx)
            writer.add_scalar("paper/asr", summary.backdoor_acc, summary.round_idx)
        writer.add_scalar("metrics/round_duration_seconds", summary.round_seconds, summary.round_idx)
        writer.add_scalar("metrics/num_sampled_clients", len(summary.sampled_clients), summary.round_idx)
        writer.add_scalar("metrics/num_selected_attackers", len(summary.selected_attackers), summary.round_idx)

    for round_idx, value in enumerate(series["mean_benign_norm"], start=1):
        writer.add_scalar("metrics/mean_benign_norm", value, round_idx)
    if attack_type != "clean":
        for round_idx, value in enumerate(series["mean_malicious_norm"], start=1):
            writer.add_scalar("attack_only/mean_malicious_norm", value, round_idx)
        for round_idx, value in enumerate(series["mean_malicious_cosine"], start=1):
            writer.add_scalar("attack_only/mean_malicious_cosine", value, round_idx)
        for summary in summaries:
            for key, value in summary.attack_metrics.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    writer.add_scalar(f"attack_only/{key}", float(value), summary.round_idx)
            for tag, value in rl_training_tensorboard_scalars(summary.attack_metrics):
                writer.add_scalar(tag, value, summary.round_idx)

    writer.add_scalar("metrics/total_seconds", total_seconds, 0)
    writer.flush()
    writer.close()


def build_payload(
    args,
    config,
    run_config: RunConfig,
    series: dict[str, list[float]],
    summaries,
    total_seconds: float,
) -> dict[str, object]:
    final_clean_acc = float("nan")
    final_backdoor_acc = float("nan")
    final_mean_benign_norm = float("nan")
    final_mean_malicious_norm = float("nan")
    final_mean_malicious_cosine = float("nan")
    for summary in reversed(summaries):
        if (
            math.isfinite(summary.clean_loss)
            and math.isfinite(summary.clean_acc)
            and math.isfinite(summary.backdoor_acc)
        ):
            final_clean_acc = summary.clean_acc
            final_backdoor_acc = summary.backdoor_acc
            idx = summary.round_idx - 1
            if 0 <= idx < len(series["mean_benign_norm"]):
                final_mean_benign_norm = series["mean_benign_norm"][idx]
                final_mean_malicious_norm = series["mean_malicious_norm"][idx]
                final_mean_malicious_cosine = series["mean_malicious_cosine"][idx]
            break
    config_payload = {
        "dataset": config.dataset,
        "data_dir": config.data_dir,
        "device": args.device,
        "seed": config.seed,
        "init_mode": config.init_mode,
        "init_checkpoint_path": config.init_checkpoint_path,
        "num_clients": config.num_clients,
        "num_attackers": config.num_attackers,
        "subsample_rate": config.subsample_rate,
        "local_epochs": config.local_epochs,
        "lr": config.lr,
        "batch_size": config.batch_size,
        "eval_batch_size": config.eval_batch_size,
        "num_workers": config.num_workers,
        "parallel_clients": config.parallel_clients,
        "eval_every": run_config.runtime.eval_every,
        "base_class": config.base_class,
        "target_class": config.target_class,
        "pattern_type": config.pattern_type,
        "attack_type": args.attack_type,
        "ipm_scaling": args.ipm_scaling,
        "lmp_scale": args.lmp_scale,
        "alie_tau": args.alie_tau,
        "gaussian_sigma": args.gaussian_sigma,
        "bfl_poison_frac": args.bfl_poison_frac,
        "dba_poison_frac": args.dba_poison_frac,
        "dba_num_sub_triggers": args.dba_num_sub_triggers,
        "attacker_action": list(args.attacker_action),
        "rl_algorithm": args.rl_algorithm,
        "rl_attacker_semantics": args.rl_attacker_semantics,
        "rl_policy_lr": args.rl_policy_lr,
        "rl_critic_lr": args.rl_critic_lr,
        "rl_gamma": args.rl_gamma,
        "rl_replay_capacity": args.rl_replay_capacity,
        "rl_batch_size": args.rl_batch_size,
        "rl_hidden_sizes": list(args.rl_hidden_sizes),
        "rl_exploration_noise": args.rl_exploration_noise,
        "rl_train_freq_steps": args.rl_train_freq_steps,
        "rl_policy_train_steps_per_round": args.rl_policy_train_steps_per_round,
        "rl_policy_checkpoint_path": args.rl_policy_checkpoint_path,
        "rl_policy_checkpoint_dir": args.rl_policy_checkpoint_dir,
        "rl_freeze_policy": args.rl_freeze_policy,
        "rl_strict_reproduction_initial_samples": args.rl_strict_reproduction_initial_samples,
        "rl_strict_reproduction_samples_per_epoch": args.rl_strict_reproduction_samples_per_epoch,
        "defense_type": config.defense_type,
        "krum_attackers": config.krum_attackers,
        "multi_krum_selected": config.multi_krum_selected,
        "clipped_median_norm": config.clipped_median_norm,
        "trimmed_mean_ratio": config.trimmed_mean_ratio,
        "geometric_median_iters": config.geometric_median_iters,
        "fltrust_root_size": config.fltrust_root_size,
        "split_mode": config.split_mode,
        "noniid_q": config.noniid_q,
        "rl_distribution_steps": config.rl_distribution_steps,
        "rl_attack_start_round": config.rl_attack_start_round,
        "rl_policy_train_end_round": config.rl_policy_train_end_round,
        "rl_inversion_steps": config.rl_inversion_steps,
        "rl_reconstruction_batch_size": config.rl_reconstruction_batch_size,
        "rl_policy_train_episodes_per_round": config.rl_policy_train_episodes_per_round,
        "rl_simulator_horizon": config.rl_simulator_horizon,
        "rl_ppo_real_rollout_steps": config.rl_ppo_real_rollout_steps,
        "rl_checkpoint_interval": args.rl_checkpoint_interval,
        "rl_save_final_checkpoint": args.rl_save_final_checkpoint,
        "rounds": args.rounds,
    }
    benchmark_protocol = run_config.benchmark_protocol_payload()
    if benchmark_protocol is not None:
        config_payload["benchmark_protocol"] = benchmark_protocol
    return {
        "config": config_payload,
        "attack_type": args.attack_type,
        "total_seconds": total_seconds,
        "series": series,
        "rounds": [
            {
                "round_idx": summary.round_idx,
                "defense_name": summary.defense_name,
                "sampled_clients": summary.sampled_clients,
                "selected_attackers": summary.selected_attackers,
                "clean_loss": summary.clean_loss,
                "clean_acc": summary.clean_acc,
                "backdoor_acc": summary.backdoor_acc,
                "asr": summary.backdoor_acc,
                "round_seconds": summary.round_seconds,
                "mean_benign_norm": series["mean_benign_norm"][summary.round_idx - 1],
                "mean_malicious_norm": series["mean_malicious_norm"][summary.round_idx - 1],
                "mean_malicious_cosine": series["mean_malicious_cosine"][summary.round_idx - 1],
                "attack_metrics": summary.attack_metrics,
                "evaluated": not math.isnan(summary.clean_acc),
            }
            for summary in summaries
        ],
        "final": {
            "clean_acc": final_clean_acc,
            "backdoor_acc": final_backdoor_acc,
            "asr": final_backdoor_acc,
            "mean_benign_norm": final_mean_benign_norm,
            "mean_malicious_norm": final_mean_malicious_norm,
            "mean_malicious_cosine": final_mean_malicious_cosine,
        },
    }


def _json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def write_summary_json(output_dir: Path, payload: dict[str, object]) -> Path:
    path = output_dir / "summary.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(payload), fh, indent=2, allow_nan=False)
    return path


def write_client_metrics_csv(output_dir: Path, rows: list[dict[str, object]]) -> Path:
    path = output_dir / "client_metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CLIENT_METRICS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_round_metrics_csv(output_dir: Path, summaries) -> Path:
    path = output_dir / "round_metrics.csv"
    rows = []
    for summary in summaries:
        rows.append(
            {
                "round_idx": summary.round_idx,
                "clean_loss": summary.clean_loss,
                "clean_acc": summary.clean_acc,
                "backdoor_acc": summary.backdoor_acc,
                "asr": summary.backdoor_acc,
                "round_seconds": summary.round_seconds,
                "num_sampled_clients": len(summary.sampled_clients),
                "num_selected_attackers": len(summary.selected_attackers),
            }
        )
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=ROUND_METRICS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return path


def execute_experiment(
    args,
    *,
    progress_desc: str | None = None,
    run_config: RunConfig | None = None,
) -> ExperimentRunResult:
    from fl_sandbox.core.fl_runner import MinimalFLRunner

    run_config = (run_config or RunConfig.from_flat_dict(vars(args))).normalize()
    config = build_config(run_config)
    attack = build_attack(run_config.attacker)
    output_dir = Path(args.output_dir or default_output_dir(run_config.attacker, run_config.defender, run_config.data))
    tb_dir = Path(args.tb_dir or default_tb_dir(run_config.attacker, run_config.defender, run_config.data))
    output_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    live_config_payload = {
        "dataset": config.dataset,
        "device": args.device,
        "attack_type": args.attack_type,
        "defense_type": config.defense_type,
        "rounds": args.rounds,
        "num_clients": config.num_clients,
        "num_attackers": config.num_attackers,
        "subsample_rate": config.subsample_rate,
        "local_epochs": config.local_epochs,
        "lr": config.lr,
        "batch_size": config.batch_size,
        "eval_batch_size": config.eval_batch_size,
        "split_mode": config.split_mode,
        "noniid_q": config.noniid_q,
        "rl_distribution_steps": config.rl_distribution_steps,
        "rl_attack_start_round": config.rl_attack_start_round,
        "rl_policy_train_end_round": config.rl_policy_train_end_round,
        "rl_policy_train_episodes_per_round": config.rl_policy_train_episodes_per_round,
        "rl_simulator_horizon": config.rl_simulator_horizon,
        "rl_ppo_real_rollout_steps": config.rl_ppo_real_rollout_steps,
        "rl_attacker_semantics": config.rl_attacker_semantics,
        "rl_policy_checkpoint_path": args.rl_policy_checkpoint_path,
        "rl_policy_checkpoint_dir": args.rl_policy_checkpoint_dir,
        "rl_freeze_policy": args.rl_freeze_policy,
        "rl_checkpoint_interval": args.rl_checkpoint_interval,
        "rl_save_final_checkpoint": args.rl_save_final_checkpoint,
    }
    live_logger = LiveMetricsLogger(
        output_dir=output_dir,
        tb_dir=tb_dir,
        attack_type=args.attack_type,
        config_payload=live_config_payload,
        target_rounds=run_config.runtime.rounds,
        payload_factory=lambda live_summaries, total_seconds: build_payload(
            args,
            config,
            run_config,
            summaries_to_dict(live_summaries),
            live_summaries,
            total_seconds,
        ),
    )

    runner = MinimalFLRunner(config)
    checkpoint_manager = ExperimentCheckpointManager(
        output_dir=output_dir,
        run_config=run_config,
        config_payload=live_config_payload,
        attack=attack,
        model=runner.model,
    )

    def _after_round(summary) -> None:
        live_logger.log_round(summary)
        checkpoint_paths = checkpoint_manager.maybe_save(round_idx=int(summary.round_idx))
        if checkpoint_paths:
            print(
                f"Saved checkpoint(s) for round {summary.round_idx}: "
                f"{', '.join(str(path) for path in checkpoint_paths)}",
                flush=True,
            )

    timer = ExperimentTimer.start()
    from fl_sandbox.attacks import RLAttack
    # For RL attacks let the internal policy pick the action (pass None).  For all other
    # parameterised attacks (e.g. BRL) pass the configured default action so it is used
    # as a fallback when no per-round override is provided.
    attacker_action_arg = None
    if attack is not None and not isinstance(attack, RLAttack):
        attacker_action_arg = tuple(run_config.attacker.attacker_action)
    completed = False
    try:
        summaries = runner.run_many_rounds(
            run_config.runtime.rounds,
            attack=attack,
            show_progress=True,
            progress_desc=progress_desc or f"{run_config.attacker.type} ({config.dataset})",
            eval_every=run_config.runtime.eval_every,
            attacker_action=attacker_action_arg,
            per_round_callback=_after_round,
        )
        completed = True
        live_logger.mark_completed(timer.elapsed_seconds())
    finally:
        if not completed:
            live_logger.mark_interrupted(timer.elapsed_seconds())
        live_logger.close()
    total_seconds = timer.elapsed_seconds()
    series = summaries_to_dict(summaries)
    payload = build_payload(args, config, run_config, series, summaries, total_seconds)
    return ExperimentRunResult(
        args=args,
        run_config=run_config,
        config=config,
        attack=attack,
        output_dir=output_dir,
        tb_dir=tb_dir,
        summaries=summaries,
        series=series,
        payload=payload,
        total_seconds=total_seconds,
        runtime_device=runner.device,
        client_metric_rows=client_metrics_to_rows(summaries),
    )


def persist_experiment_artifacts(
    result: ExperimentRunResult,
    *,
    payload_override: dict[str, object] | None = None,
    write_client_metrics: bool = True,
    write_tensorboard: bool = True,
    write_round_metrics: bool = False,
) -> None:
    payload = payload_override or result.payload
    write_summary_json(result.output_dir, payload)
    if write_client_metrics:
        write_client_metrics_csv(result.output_dir, result.client_metric_rows)
    if write_round_metrics:
        write_round_metrics_csv(result.output_dir, result.summaries)
    if write_tensorboard:
        write_tensorboard_logs(
            result.tb_dir,
            result.args.attack_type,
            result.summaries,
            result.series,
            result.total_seconds,
            payload["config"],
        )


def completion_lines(result: ExperimentRunResult) -> list[str]:
    attack_type = result.run_config.attacker.type
    lines = [
        "Sandbox run finished.",
        f"Mode: {attack_type}",
        f"Defense: {result.config.defense_type}",
        f"Output directory: {result.output_dir}",
        f"TensorBoard dir: {result.tb_dir}",
        f"Summary file: {result.output_dir / 'summary.json'}",
        f"Client metrics table: {result.output_dir / 'client_metrics.csv'}",
    ]
    lines.extend(
        build_postprocess_hint_lines(
            attack_type=attack_type,
            defense_type=result.config.defense_type,
            split_mode=result.config.split_mode,
            noniid_q=result.config.noniid_q,
            output_dir=result.output_dir,
            tb_dir=result.tb_dir,
        )
    )
    lines.extend(
        [
            f"Runtime device: {result.runtime_device}",
            f"Total seconds: {result.total_seconds:.3f}",
            f"Final clean acc: {result.payload['final']['clean_acc']:.4f}",
            f"Final backdoor acc: {result.payload['final']['backdoor_acc']:.4f}",
        ]
    )
    if attack_type != "clean":
        lines.extend(
            [
                f"Final mean malicious norm: {result.payload['final']['mean_malicious_norm']:.4f}",
                f"Final mean malicious cosine: {result.payload['final']['mean_malicious_cosine']:.4f}",
            ]
        )
    return lines
