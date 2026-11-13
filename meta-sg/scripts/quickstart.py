"""Design: demonstrate the public FL, Histogram40 and Meta-SG APIs on MNIST.
Data flow: short root-based calibration -> frozen histogram -> one meta iteration
-> frozen defender under IPM -> measured metrics and a portable summary.
This small single-seed run verifies integration, not research performance.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from meta_sg.experiments.artifacts import write_json
from meta_sg.experiments.config import load_config
from meta_sg.experiments.meta import run_meta
from meta_sg.experiments.run import run_experiment
from meta_sg.rl.distributions import DistributionTransform


def run_demo(config, output, *, smoke=False):
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f'Quickstart requires an empty output directory: {output}')
    config = deepcopy(config)
    if smoke:
        config['partition']['root_samples_per_class'] = 1
        config['training'].update(rounds=2, batch_size=16)
        config['meta'].update(horizon=2, inner_steps=1, response_steps=1, post_critic_steps=1)
    started = time.monotonic()
    calibration = deepcopy(config)
    calibration['rl']['mode'] = 'fixed'
    calibration['output']['directory'] = str(output / 'calibration')
    print('1/4 Collect training calibration contexts', flush=True)
    samples = list(run_experiment(calibration))
    features = [np.asarray(row['aggregation']['client_features'], dtype=np.float32) for row in samples if row['round'] > 0]
    # A few calibration rounds cannot estimate reliable per-bin standard deviations.
    transform = DistributionTransform.fit(features, 'histogram40', standardized=False)
    transform.save(output / 'histogram40.json')
    print('2/4 Freeze Histogram40 fitted on training contexts only', flush=True)
    config['rl']['observation'] = {'name': 'histogram40', 'parameters': {'transform_path': str(output / 'histogram40.json')}}
    config['output']['directory'] = str(output / 'meta')
    print('3/4 Train one Meta-SG outer iteration with independent RL/BRL followers', flush=True)
    meta_results = list(run_meta(config))
    evaluation = deepcopy(config)
    evaluation['rl'].update(mode='frozen', checkpoint=str(output / 'meta/policy.zip'))
    evaluation['attack'].update(name='ipm', malicious_clients=[0], start_round=2, parameters={'scaling_factor': 5.})
    evaluation['output']['directory'] = str(output / 'evaluation')
    print('4/4 Evaluate the frozen defender under IPM-5', flush=True)
    records = list(run_experiment(evaluation))
    meta_summary = meta_results[-1]
    evaluation_rounds = [row['round'] for row in records]
    assert evaluation_rounds == list(range(config['training']['rounds'] + 1))
    assert all(np.isfinite(row['evaluation']['accuracy']) and np.isfinite(row['evaluation']['loss']) for row in records)
    assert meta_summary['actor_delta_norm'] > 0
    metrics = [json.loads(line) for line in (output / 'meta/metrics.jsonl').read_text().splitlines()]
    phases = {row['phase'] for row in metrics}
    assert {'inner_1', 'response_1', 'post_response', 'outer'} <= phases
    for folder in ['calibration', 'meta', 'evaluation']:
        assert (output / folder / 'latest.pt').is_file()
        assert json.loads((output / folder / 'status.json').read_text())['phase'] == 'complete'
    source = Path(__file__).resolve().parents[1]
    code_hashes = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted((source / 'src').rglob('*.py'))}
    code_hashes['scripts/quickstart.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(output / 'code_hashes.json', code_hashes)
    summary = {
        'purpose': 'single-seed integration demo, not a performance benchmark',
        'smoke': smoke, 'seed': config['training']['seed'], 'device': config['training']['device'],
        'python': platform.python_version(), 'torch': str(torch.__version__),
        'gpu': torch.cuda.get_device_name() if torch.device(config['training']['device']).type == 'cuda' else None,
        'wall_seconds': time.monotonic() - started,
        'calibration_fl_rounds': len(features), 'meta': meta_summary,
        'evaluation_fl_rounds': len(records) - 1, 'evaluation_attack': evaluation['attack'],
        'initial_test_accuracy_percent': records[0]['evaluation']['accuracy'] * 100,
        'final_test_accuracy_percent': records[-1]['evaluation']['accuracy'] * 100,
        'final_test_loss': records[-1]['evaluation']['loss'],
        'observation_dimensions': 40, 'follower_observation_dimensions': 8,
        'histogram_standardized': False,
        'defender_hidden_units': config['rl']['agent']['hidden_units'],
        'checks': {'all_stages_complete': True, 'checkpoints_present': True,
                   'outer_actor_changed': True, 'finite_evaluation_metrics': True},
    }
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description='Run the small MNIST Meta-SG integration demo')
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[1] / 'configs/quickstart.json'))
    parser.add_argument('--data-root')
    parser.add_argument('--device')
    parser.add_argument('--output')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = load_config(args.config)
    if args.data_root:
        config['dataset']['data_root'] = args.data_root
    if args.device:
        config['training']['device'] = args.device
    run_demo(config, args.output or config['output']['directory'], smoke=args.smoke)


if __name__ == '__main__':
    main()
