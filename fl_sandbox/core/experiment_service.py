"""Run orchestration for sandbox experiments."""

from __future__ import annotations

import csv
import json
import math
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

    metric_prefix = "clean" if attack_type == "clean" else "attack"
    for summary in summaries:
        if not math.isnan(summary.clean_loss):
            writer.add_scalar(f"{metric_prefix}/loss", summary.clean_loss, summary.round_idx)
        if not math.isnan(summary.clean_acc):
            writer.add_scalar(f"{metric_prefix}/accuracy", summary.clean_acc, summary.round_idx)
        if not math.isnan(summary.backdoor_acc):
            writer.add_scalar(f"{metric_prefix}/backdoor_accuracy", summary.backdoor_acc, summary.round_idx)
            writer.add_scalar(f"{metric_prefix}/asr", summary.backdoor_acc, summary.round_idx)
        writer.add_scalar(f"{metric_prefix}/round_seconds", summary.round_seconds, summary.round_idx)
        writer.add_scalar(f"{metric_prefix}/num_sampled_clients", len(summary.sampled_clients), summary.round_idx)

    if attack_type != "clean":
        for round_idx, value in enumerate(series["mean_benign_norm"], start=1):
            writer.add_scalar("attack/mean_benign_norm", value, round_idx)
        for round_idx, value in enumerate(series["mean_malicious_norm"], start=1):
            writer.add_scalar("attack/mean_malicious_norm", value, round_idx)
        for round_idx, value in enumerate(series["mean_malicious_cosine"], start=1):
            writer.add_scalar("attack/mean_malicious_cosine", value, round_idx)

    writer.add_scalar(f"{metric_prefix}/total_seconds", total_seconds, 0)
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
    final_clean_acc = next((acc for acc in reversed(series["clean_acc"]) if not math.isnan(acc)), float("nan"))
    final_backdoor_acc = next((acc for acc in reversed(series["backdoor_acc"]) if not math.isnan(acc)), float("nan"))
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
        "eval_every": args.eval_every,
        "base_class": config.base_class,
        "target_class": config.target_class,
        "pattern_type": config.pattern_type,
        "attack_type": args.attack_type,
        "ipm_scaling": args.ipm_scaling,
        "lmp_scale": args.lmp_scale,
        "bfl_poison_frac": args.bfl_poison_frac,
        "dba_poison_frac": args.dba_poison_frac,
        "dba_num_sub_triggers": args.dba_num_sub_triggers,
        "attacker_action": list(args.attacker_action),
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
                "evaluated": not math.isnan(summary.clean_acc),
            }
            for summary in summaries
        ],
        "final": {
            "clean_acc": final_clean_acc,
            "backdoor_acc": final_backdoor_acc,
            "asr": final_backdoor_acc,
            "mean_benign_norm": series["mean_benign_norm"][-1] if series["mean_benign_norm"] else float("nan"),
            "mean_malicious_norm": series["mean_malicious_norm"][-1]
            if series["mean_malicious_norm"] else float("nan"),
            "mean_malicious_cosine": series["mean_malicious_cosine"][-1]
            if series["mean_malicious_cosine"] else float("nan"),
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

    runner = MinimalFLRunner(config)
    timer = ExperimentTimer.start()
    summaries = runner.run_many_rounds(
        run_config.runtime.rounds,
        attack=attack,
        show_progress=True,
        progress_desc=progress_desc or f"{run_config.attacker.type} ({config.dataset})",
        eval_every=run_config.runtime.eval_every,
        attacker_action=None if attack is None else tuple(run_config.attacker.attacker_action),
    )
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
