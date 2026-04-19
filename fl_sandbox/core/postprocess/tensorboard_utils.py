"""Shared TensorBoard export helpers for sandbox postprocess flows."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


def _coerce_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def _is_finite_scalar(value: object) -> bool:
    return bool(np.isfinite(_coerce_float(value)))


def _last_finite(values: Iterable[float]) -> float:
    resolved = list(values)
    for value in reversed(resolved):
        if _is_finite_scalar(value):
            return float(value)
    return float('nan')


def coerce_series(raw_series: Mapping[str, Iterable[float]]) -> dict[str, list[float]]:
    series: dict[str, list[float]] = {}
    for key, values in raw_series.items():
        try:
            series[key] = [_coerce_float(value) for value in values]
        except TypeError:
            continue
    if 'asr' not in series and 'backdoor_acc' in series:
        series['asr'] = list(series['backdoor_acc'])
    return series


def payload_to_series(payload: dict) -> dict[str, list[float]]:
    if 'series' in payload:
        return coerce_series(payload['series'])

    rounds = payload['rounds']
    backdoor_acc = [_coerce_float(item.get('backdoor_acc')) for item in rounds]
    return {
        'clean_loss': [_coerce_float(item.get('clean_loss')) for item in rounds],
        'clean_acc': [_coerce_float(item.get('clean_acc')) for item in rounds],
        'backdoor_acc': backdoor_acc,
        'asr': backdoor_acc,
        'round_seconds': [float(item['round_seconds']) for item in rounds],
        'mean_benign_norm': [_coerce_float(item.get('mean_benign_norm', 0.0)) for item in rounds],
        'mean_malicious_norm': [_coerce_float(item.get('mean_malicious_norm', 0.0)) for item in rounds],
        'mean_malicious_cosine': [_coerce_float(item.get('mean_malicious_cosine', 0.0)) for item in rounds],
    }


def build_summary_writer(log_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter

        return TorchSummaryWriter(log_dir=str(log_dir))
    except ModuleNotFoundError:
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        class BasicSummaryWriter:
            def __init__(self, path: Path) -> None:
                self._writer = EventFileWriter(str(path))

            def add_text(self, tag: str, text_string: str, global_step: int = 0) -> None:
                return None

            def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
                summary = Summary(value=[Summary.Value(tag=tag, simple_value=float(scalar_value))])
                event = Event(wall_time=time.time(), step=int(global_step), summary=summary)
                self._writer.add_event(event)

            def flush(self) -> None:
                self._writer.flush()

            def close(self) -> None:
                self._writer.close()

        return BasicSummaryWriter(log_dir)


def _write_scalar_series(writer, tag: str, values: Iterable[float]) -> None:
    for round_idx, value in enumerate(values, start=1):
        if _is_finite_scalar(value):
            writer.add_scalar(tag, float(value), round_idx)


def write_common_scalars(writer, series: Mapping[str, list[float]]) -> None:
    _write_scalar_series(writer, 'metrics/loss', series['clean_loss'])
    _write_scalar_series(writer, 'metrics/accuracy', series['clean_acc'])
    _write_scalar_series(writer, 'metrics/backdoor_accuracy', series.get('backdoor_acc', []))
    _write_scalar_series(writer, 'metrics/asr', series.get('asr', series.get('backdoor_acc', [])))
    _write_scalar_series(writer, 'metrics/round_duration_seconds', series['round_seconds'])
    _write_scalar_series(writer, 'metrics/elapsed_seconds', np.cumsum(series['round_seconds']).tolist())
    _write_scalar_series(writer, 'metrics/mean_benign_norm', series['mean_benign_norm'])


def write_attack_scalars(writer, series: Mapping[str, list[float]]) -> None:
    _write_scalar_series(writer, 'attack_only/mean_malicious_norm', series['mean_malicious_norm'])
    _write_scalar_series(writer, 'attack_only/mean_malicious_cosine', series['mean_malicious_cosine'])


def build_run_summary_text(
    run_name: str,
    config: Mapping[str, object],
    series: Mapping[str, list[float]],
    *,
    include_attack: bool,
    total_seconds: float | None = None,
) -> str:
    resolved_total_seconds = (
        float(total_seconds) if total_seconds is not None else float(np.sum(series['round_seconds']))
    )
    lines = [
        f'run: {run_name}',
        f"dataset: {config.get('dataset', 'unknown')}",
        f"rounds: {len(series['clean_acc'])}",
        f"num_clients: {config.get('num_clients', 'unknown')}",
        f"subsample_rate: {config.get('subsample_rate', 'unknown')}",
        f"local_epochs: {config.get('local_epochs', 'unknown')}",
        f"lr: {config.get('lr', 'unknown')}",
        f"batch_size: {config.get('batch_size', 'unknown')}",
        f"final_accuracy: {_last_finite(series['clean_acc']):.4f}",
        f"final_backdoor_accuracy: {_last_finite(series.get('backdoor_acc', [0.0])):.4f}",
        f"final_loss: {_last_finite(series['clean_loss']):.4f}",
        f"mean_round_duration_seconds: {resolved_total_seconds / len(series['round_seconds']):.4f}",
        f'total_elapsed_seconds: {resolved_total_seconds:.4f}',
    ]
    if include_attack:
        lines.append(f"final_mean_malicious_norm: {_last_finite(series['mean_malicious_norm']):.4f}")
        lines.append(f"final_mean_malicious_cosine: {_last_finite(series['mean_malicious_cosine']):.4f}")
    return '\n'.join(lines)


def write_run_text(
    writer,
    run_name: str,
    config: Mapping[str, object],
    series: Mapping[str, list[float]],
    *,
    include_attack: bool,
    total_seconds: float | None = None,
) -> None:
    writer.add_text('config/json', json.dumps(dict(config), indent=2), global_step=0)
    writer.add_text(
        'run/summary',
        build_run_summary_text(
            run_name,
            config,
            series,
            include_attack=include_attack,
            total_seconds=total_seconds,
        ),
        global_step=0,
    )


def write_standard_tensorboard_run(
    writer,
    run_name: str,
    config: Mapping[str, object],
    series: Mapping[str, list[float]],
    *,
    include_attack: bool,
    total_seconds: float | None = None,
) -> None:
    write_common_scalars(writer, series)
    if include_attack:
        write_attack_scalars(writer, series)
    write_run_text(
        writer,
        run_name,
        config,
        series,
        include_attack=include_attack,
        total_seconds=total_seconds,
    )


def write_client_stat_scalars(writer, client_rows: list[dict[str, str]], round_count: int) -> None:
    per_round_client_loss = []
    per_round_client_acc = []
    per_round_update_norm = []
    selected_client_counts = []

    for round_idx in range(1, round_count + 1):
        round_rows = [row for row in client_rows if int(row['round_idx']) == round_idx]
        selected_rows = [row for row in round_rows if row['selected'] == 'True']
        selected_client_counts.append(len(selected_rows))
        per_round_client_loss.append(
            [float(row['train_loss']) for row in selected_rows if row['train_loss'] not in ('', None)]
        )
        per_round_client_acc.append(
            [float(row['train_acc']) for row in selected_rows if row['train_acc'] not in ('', None)]
        )
        per_round_update_norm.append(
            [float(row['update_norm']) for row in selected_rows if row['update_norm'] not in ('', None)]
        )

    mean_client_loss = [float(np.mean(values)) if values else 0.0 for values in per_round_client_loss]
    std_client_loss = [float(np.std(values)) if values else 0.0 for values in per_round_client_loss]
    mean_client_acc = [float(np.mean(values)) if values else 0.0 for values in per_round_client_acc]
    std_client_acc = [float(np.std(values)) if values else 0.0 for values in per_round_client_acc]
    mean_update_norm = [float(np.mean(values)) if values else 0.0 for values in per_round_update_norm]
    std_update_norm = [float(np.std(values)) if values else 0.0 for values in per_round_update_norm]

    _write_scalar_series(writer, 'client_stats/selected_clients', selected_client_counts)
    _write_scalar_series(writer, 'client_stats/train_loss_mean', mean_client_loss)
    _write_scalar_series(writer, 'client_stats/train_loss_std', std_client_loss)
    _write_scalar_series(writer, 'client_stats/train_acc_mean', mean_client_acc)
    _write_scalar_series(writer, 'client_stats/train_acc_std', std_client_acc)
    _write_scalar_series(writer, 'client_stats/update_norm_mean', mean_update_norm)
    _write_scalar_series(writer, 'client_stats/update_norm_std', std_update_norm)
