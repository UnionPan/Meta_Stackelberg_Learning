#!/usr/bin/env python3
"""Monitor the Algo2 gate, run the fresh formal job, then evaluate it."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO = Path('/home/antik/rl/Meta_Stackelberg_Learning')
sys.path.insert(0, str(REPO))

from meta_stackelberg.experiments.scaled_training_checkpoint import (  # noqa: E402
    load_scaled_training_checkpoint,
)


INTERVAL_SECONDS = 600
GATE_DIR = REPO / (
    'runs/meta_stackelberg/'
    'algo2_global_iid_standard_t10_k5_h100_l10_w20_s4_a4_'
    'alpha075_meta025_seed41_20260717'
)
FORMAL_DIR = REPO / (
    'runs/meta_stackelberg/'
    'algo2_global_formal_iid_standard_normmedian_t100_k5_h100_l10_w20_s4_a4_'
    'alpha075_meta025_seed41_20260718'
)
ATTACK_DOMAIN = REPO / (
    'runs/meta_stackelberg/paper_mnist_t100_k5_h200_seed41_20260715/'
    'attack-domain.pt'
)
GATE_CHECKPOINT = GATE_DIR / 'training.pt'
FORMAL_CHECKPOINT = FORMAL_DIR / 'training.pt'
FROZEN_DIR = FORMAL_DIR / 'evaluation_frozen_h100'
ONLINE_DIR = FORMAL_DIR / 'evaluation_online_h100'
STATUS_PATH = FORMAL_DIR / 'pipeline_status.json'
HISTORY_PATH = FORMAL_DIR / 'monitor_history.jsonl'
EVENTS_PATH = FORMAL_DIR / 'pipeline_events.jsonl'
LOCK_PATH = FORMAL_DIR / '.pipeline.lock'


def _now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def _append_json(path: Path, payload: Any) -> None:
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _event(kind: str, **fields: Any) -> None:
    _append_json(EVENTS_PATH, {'timestamp': _now(), 'kind': kind, **fields})


def _running_pids(marker: str) -> list[int]:
    result = []
    own_pid = os.getpid()
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            command = (entry / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if marker in command:
            result.append(int(entry.name))
    return sorted(result)


def _gpu_metrics() -> list[dict[str, Any]]:
    command = [
        'nvidia-smi',
        '--query-compute-apps=pid,used_memory',
        '--format=csv,noheader,nounits',
    ]
    try:
        output = subprocess.check_output(command, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return []
    result = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(',')]
        if len(fields) == 2 and all(field.isdigit() for field in fields):
            result.append({'pid': int(fields[0]), 'used_memory_mib': int(fields[1])})
    return result


def _checkpoint_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {'exists': False}
    try:
        checkpoint = load_scaled_training_checkpoint(path)
    except Exception as error:  # preserve evidence and retry next interval
        return {
            'exists': True,
            'read_error': f'{type(error).__name__}: {error}',
            'bytes': path.stat().st_size,
        }
    config = checkpoint.config_signature
    iterations = checkpoint.algorithm2_iterations
    latest_tasks = []
    if iterations:
        for task in iterations[-1].tasks:
            critic = [float(item.critic_loss) for item in task.update_stats]
            actor = [
                float(item.actor_loss)
                for item in task.update_stats
                if item.actor_loss is not None
            ]
            latest_tasks.append({
                'task': task.task,
                'critic_loss_min': min(critic),
                'critic_loss_max': max(critic),
                'actor_loss_last': actor[-1] if actor else None,
                'all_update_metrics_finite': all(
                    math.isfinite(value) for value in critic + actor
                ),
            })
    defender_finite = all(
        bool(parameter.isfinite().all())
        for parameter in checkpoint.algorithm2_defender.actor.values()
    )
    completed = int(checkpoint.algorithm2_completed)
    return {
        'exists': True,
        'phase': checkpoint.phase,
        'algorithm1_completed': int(checkpoint.algorithm1_completed),
        'algorithm2_completed': completed,
        'target_T': int(config['T']),
        'estimated_fl_rounds': (
            completed * int(config['K']) * int(config['l']) * int(config['H'])
        ),
        'support_seed_count': len(checkpoint.support_seeds),
        'replay_serial': int(checkpoint.replay_serial),
        'latest_iteration_tasks': latest_tasks,
        'defender_actor_parameters_finite': defender_finite,
        'bytes': path.stat().st_size,
        'mtime': path.stat().st_mtime,
    }


def _common_training_args(*, target_t: int, checkpoint: Path) -> list[str]:
    return [
        str(REPO / '.venv/bin/python'), '-u', '-m',
        'meta_stackelberg.experiments.run_paper_meta_rl',
        '--data-root', 'data',
        '--checkpoint', str(checkpoint.relative_to(REPO)),
        '--T', str(target_t), '--K', '5', '--H', '100', '--l', '10',
        '--seed', '41', '--partition-seed', '17', '--model-seed', '99',
        '--support-seed', '9900000',
        '--workers', '20', '--untargeted-attackers', '4', '--sample-size', '4',
        '--parallel-tasks', '5', '--parallel-clients', '1', '--cpu-threads', '1',
        '--deterministic-torch', '--checkpoint-interval', '1',
        '--device', 'cuda:0', '--materialize-mnist',
        '--mnist-normalization', 'standard', '--data-split', 'iid',
        '--alpha-floor-ratio', '0.75', '--post-defense-mode', 'identity',
        '--defender-norm-reference', 'median',
        '--task-domain', 'global',
        '--attack-domain', str(ATTACK_DOMAIN.relative_to(REPO)),
        '--task-sampler', 'balanced', '--meta-update-step', '0.25',
    ]


def _evaluation_args(*, output: Path, online: bool) -> list[str]:
    command = [
        str(REPO / '.venv/bin/python'), '-u', '-m',
        'meta_stackelberg.experiments.run_model_poisoning_evaluation',
        '--checkpoint', str(FORMAL_CHECKPOINT.relative_to(REPO)),
        '--data-root', 'data', '--output', str(output.relative_to(REPO)),
        '--T', '100', '--K', '5', '--H', '100', '--training-H', '100',
        '--seed', '101', '--partition-seed', '17', '--model-seed', '99',
        '--workers', '20', '--untargeted-attackers', '4', '--sample-size', '4',
        '--parallel-clients', '1', '--cpu-threads', '1', '--deterministic-torch',
        '--device', 'cuda:0', '--materialize-mnist',
        '--mnist-normalization', 'standard', '--data-split', 'iid',
        '--alpha-floor-ratio', '0.75', '--post-defense-mode', 'identity',
        '--defender-norm-reference', 'median',
        '--attack-domain', str(ATTACK_DOMAIN.relative_to(REPO)),
        '--method', 'meta-rl',
    ]
    if online:
        command.extend([
            '--online-T', '10', '--online-H', '100', '--online-l', '10',
            '--online-steps', '100', '--online-batch-size', '256',
            '--online-learning-starts', '100',
            '--online-adaptation-step', '0.001',
            '--online-actor-logit-l2', '1.0',
            '--online-actor-logit-l2-mask', '1', '0', '0',
            '--online-selection', 'reward-guarded',
            '--online-selection-repeats', '2',
        ])
    else:
        command.append('--skip-online-adaptation')
    return command


def _spawn(command: list[str], log_path: Path, marker: str) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('ab', buffering=0) as stream:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    _event('process_started', marker=marker, pid=process.pid, command=command)
    return process.pid


def _ensure_training(
    *, target_t: int, checkpoint: Path, log_path: Path,
) -> tuple[list[int], bool]:
    metrics = _checkpoint_metrics(checkpoint)
    if metrics.get('algorithm2_completed') == target_t and metrics.get('phase') == 'complete':
        return [], True
    marker = str(checkpoint.relative_to(REPO))
    pids = _running_pids(marker)
    if not pids:
        command = _common_training_args(target_t=target_t, checkpoint=checkpoint)
        if checkpoint.is_file():
            command.append('--resume')
        pids = [_spawn(command, log_path, marker)]
    return pids, False


def _ensure_evaluation(output: Path, *, online: bool) -> tuple[list[int], bool]:
    summary = output / 'summary.json'
    if summary.is_file():
        return [], True
    marker = str(output.relative_to(REPO))
    pids = _running_pids(marker)
    if not pids:
        pids = [_spawn(
            _evaluation_args(output=output, online=online),
            output.parent / f'{output.name}.log',
            marker,
        )]
    return pids, False


def _export_training_metrics() -> None:
    destination = FORMAL_DIR / 'training_metrics.json'
    checkpoint = load_scaled_training_checkpoint(FORMAL_CHECKPOINT)
    iterations = []
    for iteration in checkpoint.algorithm2_iterations:
        tasks = []
        for task in iteration.tasks:
            critic = [float(item.critic_loss) for item in task.update_stats]
            actor = [
                float(item.actor_loss) for item in task.update_stats
                if item.actor_loss is not None
            ]
            tasks.append({
                'task': task.task,
                'critic_loss_min': min(critic),
                'critic_loss_max': max(critic),
                'critic_loss_last': critic[-1],
                'actor_loss_last': actor[-1] if actor else None,
                'all_update_metrics_finite': all(
                    math.isfinite(value) for value in critic + actor
                ),
                'adapted_defender_fingerprint': task.adapted_defender_fingerprint,
            })
        iterations.append({
            'iteration': int(iteration.meta_iteration) + 1,
            'tasks': tasks,
            'meta_defender_before': iteration.meta_defender_before,
            'meta_defender_after': iteration.meta_defender_after,
        })
    _atomic_json(destination, {
        'protocol': 'algo2-formal-training-metrics-v1',
        'generated_at': _now(),
        'checkpoint': str(FORMAL_CHECKPOINT),
        'checkpoint_sha256': _sha256(FORMAL_CHECKPOINT),
        'config_signature': checkpoint.config_signature,
        'completed_iterations': checkpoint.algorithm2_completed,
        'estimated_fl_rounds': (
            checkpoint.algorithm2_completed
            * checkpoint.config_signature['K']
            * checkpoint.config_signature['l']
            * checkpoint.config_signature['H']
        ),
        'iterations': iterations,
    })


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _compile_final_summary() -> None:
    frozen = json.loads((FROZEN_DIR / 'summary.json').read_text())
    online = json.loads((ONLINE_DIR / 'summary.json').read_text())
    _atomic_json(FORMAL_DIR / 'final_results.json', {
        'protocol': 'algo2-formal-pipeline-results-v1',
        'completed_at': _now(),
        'training_checkpoint': str(FORMAL_CHECKPOINT),
        'training_checkpoint_sha256': _sha256(FORMAL_CHECKPOINT),
        'frozen': {
            name: values
            for name, values in frozen['scenarios'].items()
        },
        'online_adapted': {
            name: values
            for name, values in online['scenarios'].items()
        },
    })


def _write_protocol() -> None:
    path = FORMAL_DIR / 'protocol.json'
    if path.exists():
        return
    _atomic_json(path, {
        'protocol': 'algo2-formal-pipeline-v1',
        'created_at': _now(),
        'method': 'Algorithm 2 Meta-RL',
        'fresh_formal_training': True,
        'scale': {'T': 100, 'K': 5, 'H': 100, 'l': 10},
        'topology': {'workers': 20, 'attackers': 4, 'sample_size': 4},
        'tasks': ['na', 'ipm', 'lmp', 'rl-krum', 'rl-clipmed'],
        'data_split': 'iid',
        'mnist_normalization': 'standard',
        'local_training': 'one-minibatch-step-per-local-iteration',
        'post_defense_mode': 'identity',
        'alpha_floor_ratio': 0.75,
        'defender_norm_reference': 'median',
        'meta_update_step': 0.25,
        'checkpoint_interval_outer_iterations': 1,
        'monitor_interval_seconds': INTERVAL_SECONDS,
        'evaluation': {
            'frozen_H': 100,
            'online_T': 10,
            'online_H': 100,
            'online_l': 10,
            'online_steps': 100,
            'online_adaptation_step': 0.01,
        },
    })


def _advance() -> tuple[str, list[int], bool]:
    formal_pids, formal_complete = _ensure_training(
        target_t=100,
        checkpoint=FORMAL_CHECKPOINT,
        log_path=FORMAL_DIR / 'training.log',
    )
    if not formal_complete:
        return 'formal_training', formal_pids, False
    if not (FORMAL_DIR / 'training_metrics.json').is_file():
        _export_training_metrics()

    frozen_pids, frozen_complete = _ensure_evaluation(FROZEN_DIR, online=False)
    if not frozen_complete:
        return 'frozen_evaluation', frozen_pids, False

    online_pids, online_complete = _ensure_evaluation(ONLINE_DIR, online=True)
    if not online_complete:
        return 'online_evaluation', online_pids, False

    if not (FORMAL_DIR / 'final_results.json').is_file():
        _compile_final_summary()
    return 'complete', [], True


def main() -> int:
    FORMAL_DIR.mkdir(parents=True, exist_ok=True)
    lock_stream = LOCK_PATH.open('w')
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('formal pipeline monitor is already running')
    _write_protocol()
    _event('pipeline_started', pid=os.getpid(), interval_seconds=INTERVAL_SECONDS)
    while True:
        try:
            stage, pids, complete = _advance()
            status = {
                'protocol': 'algo2-formal-pipeline-status-v1',
                'timestamp': _now(),
                'stage': stage,
                'complete': complete,
                'active_pids': pids,
                'abandoned_max_norm_gate_checkpoint': (
                    _checkpoint_metrics(GATE_CHECKPOINT)
                ),
                'stability_gate': json.loads(
                    (FORMAL_DIR / 'stability_gate.json').read_text()
                ),
                'formal_checkpoint': _checkpoint_metrics(FORMAL_CHECKPOINT),
                'gpu_processes': _gpu_metrics(),
                'next_check_seconds': None if complete else INTERVAL_SECONDS,
            }
        except Exception as error:
            status = {
                'protocol': 'algo2-formal-pipeline-status-v1',
                'timestamp': _now(),
                'stage': 'monitor_error',
                'complete': False,
                'error': f'{type(error).__name__}: {error}',
                'next_check_seconds': INTERVAL_SECONDS,
            }
            _event('monitor_error', error=status['error'])
        _atomic_json(STATUS_PATH, status)
        _append_json(HISTORY_PATH, status)
        if status['complete']:
            _event('pipeline_complete', results=str(FORMAL_DIR / 'final_results.json'))
            return 0
        time.sleep(INTERVAL_SECONDS)


if __name__ == '__main__':
    raise SystemExit(main())
