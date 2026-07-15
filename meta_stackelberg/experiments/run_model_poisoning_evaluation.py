"""Evaluate a completed canonical Meta-SG checkpoint on clean/IPM/LMP/RL."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.agents.td3.config import PaperMetaSGConfig
from meta_stackelberg.agents.td3.replay import flatten_observation
from meta_stackelberg.experiments.model_poisoning_evaluation import (
    canonical_model_poisoning_scenarios,
    evaluate_model_poisoning_scenario,
    summarize_model_poisoning_evaluation,
)
from meta_stackelberg.experiments.paper_meta_sg import (
    ATTACKER_OBSERVATION_KEYS,
    DEFENDER_OBSERVATION_KEYS,
)
from meta_stackelberg.experiments.paper_mnist_env import (
    PaperMNISTEnvironmentFactory,
    load_paper_mnist_datasets,
)
from meta_stackelberg.experiments.scaled_training_checkpoint import (
    load_scaled_training_checkpoint,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--partition-seed', type=int, default=17)
    parser.add_argument('--model-seed', type=int, default=99)
    parser.add_argument('--T', type=int, default=100)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--H', type=int, default=200)
    parser.add_argument('--ipm-scale', type=float, default=2.0)
    parser.add_argument('--lmp-scale', type=float, default=2.0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument(
        '--method', choices=('meta-sg', 'meta-rl'), default='meta-sg',
        help='select the independently trained policy to evaluate',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = load_scaled_training_checkpoint(args.checkpoint)
    _validate_checkpoint(checkpoint, args)
    paper = PaperMetaSGConfig()
    datasets = load_paper_mnist_datasets(
        args.data_root,
        seed=args.partition_seed,
        root_samples=paper.root_samples_mnist,
        download=False,
    )
    factory = PaperMNISTEnvironmentFactory(
        train_dataset=datasets.client_train,
        root_dataset=datasets.root,
        partition_seed=args.partition_seed,
        model_seed=args.model_seed,
        workers=paper.workers,
        untargeted_attackers=paper.untargeted_attackers,
        sample_size=int(paper.workers * paper.subsampling_rate),
        non_iid_q=paper.non_iid_q,
        fl_batch_size=paper.fl_batch_size,
        local_iterations=paper.local_iterations,
        client_learning_rate=paper.client_learning_rate,
        local_search_batch_size=128,
        local_search_gradient_norm_cap=1.0,
        device=args.device,
    )
    probe = factory.make(seed=args.seed, horizon=args.H, task_id='eval-probe')
    defender_dim = len(flatten_observation(
        probe.defender_observation(), DEFENDER_OBSERVATION_KEYS,
    ))
    attacker_probe = factory.make(
        seed=args.seed, horizon=args.H, task_id='eval-attacker-probe',
    )
    attacker_dim = len(flatten_observation(
        attacker_probe.begin_round(
            np.zeros(3, dtype=np.float32),
        ).attacker_observation,
        ATTACKER_OBSERVATION_KEYS,
    ))
    defender = _agent(paper, defender_dim, 'defender', 1, args.device)
    defender.restore(
        checkpoint.algorithm1_defender
        if args.method == 'meta-sg'
        else checkpoint.algorithm2_defender
    )
    attackers = {}
    for index, (label, snapshot) in enumerate(
        sorted(checkpoint.algorithm1_attackers.items()),
    ):
        policy = _agent(
            paper, attacker_dim, 'attacker', 10 + index, args.device,
        )
        policy.restore(snapshot)
        attackers[str(label)] = policy
    scenarios = canonical_model_poisoning_scenarios(
        tuple(attackers),
        ipm_scale=args.ipm_scale,
        lmp_scale=args.lmp_scale,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    for scenario in scenarios:
        path = output / f'{scenario.name}.json'
        if path.exists():
            record = json.loads(path.read_text(encoding='utf-8'))
            if _record_matches(
                record,
                scenario,
                args.seed,
                args.H,
                defender.fingerprint(),
                attackers[scenario.attacker_label].fingerprint(),
            ):
                records[scenario.name] = record
                continue
        record = evaluate_model_poisoning_scenario(
            scenario=scenario,
            factory=factory,
            test_dataset=datasets.test,
            defender=defender,
            attacker=attackers[scenario.attacker_label],
            seed=args.seed,
            horizon=args.H,
            device=args.device,
        )
        _atomic_json(path, record)
        records[scenario.name] = record
        print(json.dumps({
            'scenario': scenario.name,
            'final_delivered_clean_accuracy': record[
                'final_delivered_clean_accuracy'
            ],
        }, sort_keys=True), flush=True)
    summary = summarize_model_poisoning_evaluation(
        records,
        checkpoint=str(Path(args.checkpoint).resolve()),
        seed=args.seed,
        horizon=args.H,
    )
    summary['training_configuration'] = dict(checkpoint.config_signature)
    summary['method'] = args.method
    summary['data_provenance'] = {
        key: getattr(datasets.provenance, key)
        for key in datasets.provenance.__dataclass_fields__
    }
    _atomic_json(output / 'summary.json', summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


def _validate_checkpoint(checkpoint, args) -> None:
    if checkpoint.phase != 'complete':
        raise ValueError('final evaluation requires a complete training checkpoint')
    expected = {'T': args.T, 'K': args.K, 'H': args.H}
    observed = dict(checkpoint.config_signature)
    if any(observed.get(key) != value for key, value in expected.items()):
        raise ValueError('training checkpoint T/K/H does not match evaluation')
    recorded_method = observed.get('training_method')
    if recorded_method not in {None, args.method, 'both'}:
        raise ValueError('training checkpoint method does not match evaluation')
    if args.method == 'meta-sg':
        if checkpoint.algorithm1_completed != observed.get('N_D'):
            raise ValueError('Meta-SG checkpoint Algorithm 1 is incomplete')
    elif checkpoint.algorithm2_completed != args.T:
        raise ValueError('Meta-RL checkpoint Algorithm 2 is incomplete')


def _agent(paper, obs_dim, role, seed, device):
    return TD3Agent(
        obs_dim=obs_dim,
        action_dim=3,
        role=role,
        seed=seed,
        hidden_sizes=paper.hidden_sizes,
        learning_rate=paper.policy_learning_rate,
        gamma=paper.gamma,
        tau=paper.tau,
        policy_delay=paper.policy_delay,
        target_policy_noise=paper.target_policy_noise,
        noise_clip=paper.noise_clip,
        device=device,
    )


def _record_matches(
    record, scenario, seed, horizon, defender_fingerprint, attacker_fingerprint,
):
    return (
        record.get('protocol') == 'canonical-model-poisoning-final-evaluation-v1'
        and record.get('scenario') == scenario.name
        and record.get('attack_family') == scenario.attack_family
        and record.get('fixed_attack_scale') == scenario.scale
        and record.get('seed') == seed
        and record.get('horizon') == horizon
        and record.get('defender_fingerprint') == defender_fingerprint
        and record.get('attacker_fingerprint') == attacker_fingerprint
        and len(record.get('round_metrics', ())) == horizon
    )


def _atomic_json(path: Path, payload) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


if __name__ == '__main__':
    raise SystemExit(main())
