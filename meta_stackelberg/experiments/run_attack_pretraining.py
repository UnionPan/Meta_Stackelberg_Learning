"""Build paper Meta-SG RL attack types pre-trained against fixed defenses."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import tempfile

from meta_stackelberg.agents.td3.config import PaperMetaSGConfig
from meta_stackelberg.experiments.attack_domain import save_attack_type_domain
from meta_stackelberg.experiments.attack_pretraining import (
    AttackPolicyPretrainingConfig,
    AttackPretrainingTaskSpec,
    pretrain_attack_type_domain,
)
from meta_stackelberg.experiments.paper_cifar_env import (
    PaperCIFAREnvironmentFactory,
    load_paper_cifar_datasets,
)
from meta_stackelberg.experiments.paper_mnist_env import (
    PaperMNISTEnvironmentFactory,
    load_paper_mnist_datasets,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=('mnist', 'cifar'))
    parser.add_argument('--profile', default='micro', choices=('micro', 'paper'))
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--clip-radius', type=float, required=True)
    parser.add_argument('--krum-byzantine-count', type=int)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--allow-paper-scale', action='store_true')
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--partition-seed', type=int, default=17)
    parser.add_argument('--model-seed', type=int, default=99)
    parser.add_argument('--local-search-batch-size', type=int, default=128)
    return parser


def make_pretraining_config(
    args,
) -> tuple[AttackPolicyPretrainingConfig, tuple[int, ...]]:
    paper = PaperMetaSGConfig()
    if args.profile == 'paper':
        if not args.allow_paper_scale:
            raise ValueError('paper profile requires --allow-paper-scale')
        return AttackPolicyPretrainingConfig.from_paper(paper), paper.hidden_sizes
    return AttackPolicyPretrainingConfig(
        fl_rounds=4,
        batch_size=2,
        learning_starts=2,
        train_freq=1,
        gradient_steps=1,
        replay_capacity=256,
    ), (8,)


def make_task_specs(
    args,
    *,
    sample_size: int,
    attacker_fraction: float,
) -> tuple[AttackPretrainingTaskSpec, ...]:
    if not math.isfinite(attacker_fraction) or not 0 < attacker_fraction < 1:
        raise ValueError('attacker_fraction must be within (0, 1)')
    byzantine_count = args.krum_byzantine_count
    if byzantine_count is None:
        byzantine_count = max(1, int(round(sample_size * attacker_fraction)))
    if sample_size <= 2 * byzantine_count + 2:
        raise ValueError('Krum requires sample_size > 2f + 2')
    return (
        AttackPretrainingTaskSpec(
            'rl-krum', 'krum', byzantine_count=byzantine_count,
        ),
        AttackPretrainingTaskSpec(
            'rl-clipmed', 'clipmed', clip_radius=args.clip_radius,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paper = PaperMetaSGConfig()
    config, hidden_sizes = make_pretraining_config(args)
    sample_size = int(paper.workers * paper.subsampling_rate)
    tasks = make_task_specs(
        args,
        sample_size=sample_size,
        attacker_fraction=paper.untargeted_attackers / paper.workers,
    )
    if args.dataset == 'mnist':
        datasets = load_paper_mnist_datasets(
            args.data_root,
            seed=args.partition_seed,
            root_samples=paper.root_samples_mnist,
            download=args.download,
        )
        factory = PaperMNISTEnvironmentFactory(
            train_dataset=datasets.client_train,
            root_dataset=datasets.root,
            partition_seed=args.partition_seed,
            model_seed=args.model_seed,
            workers=paper.workers,
            untargeted_attackers=paper.untargeted_attackers,
            sample_size=sample_size,
            non_iid_q=paper.non_iid_q,
            fl_batch_size=paper.fl_batch_size,
            local_iterations=paper.local_iterations,
            client_learning_rate=paper.client_learning_rate,
            local_search_batch_size=args.local_search_batch_size,
        )
    else:
        datasets = load_paper_cifar_datasets(
            args.data_root,
            seed=args.partition_seed,
            root_samples=paper.root_samples_cifar,
            download=args.download,
        )
        factory = PaperCIFAREnvironmentFactory(
            train_dataset=datasets.client_train,
            root_dataset=datasets.root,
            partition_seed=args.partition_seed,
            model_seed=args.model_seed,
            workers=paper.workers,
            untargeted_attackers=paper.untargeted_attackers,
            sample_size=sample_size,
            non_iid_q=paper.non_iid_q,
            fl_batch_size=paper.fl_batch_size,
            local_iterations=paper.local_iterations,
            client_learning_rate=paper.client_learning_rate,
            local_search_batch_size=args.local_search_batch_size,
        )
    result = pretrain_attack_type_domain(
        config=config,
        paper=paper,
        env_factory=lambda seed, horizon, task_id: factory.make(
            seed=seed, horizon=horizon, task_id=task_id,
        ),
        tasks=tasks,
        hidden_sizes=hidden_sizes,
        seed=args.seed,
    )
    output = Path(args.output)
    save_attack_type_domain(output, result.domain)
    manifest_path = output.with_suffix(output.suffix + '.manifest.json')
    manifest = {
        'schema_version': 1,
        'protocol': result.protocol,
        'dataset': args.dataset,
        'profile': args.profile,
        'output': str(output),
        'pretraining_config': asdict(config),
        'tasks': [asdict(task) for task in tasks],
        'task_origins': dict(result.domain.origins),
        'total_fl_round_count': result.total_fl_round_count,
        'td3_update_counts': {
            task.label: task.td3_update_count for task in result.tasks
        },
        'parameter_sources': {
            'fl_rounds': 'paper.rl_training_rounds',
            'K': 'not-used-in-attack-pretraining',
            'N_A': 'not-used-in-attack-pretraining',
            'fixed_defender_raw_action': (
                'implementation-contract-aggregator-overrides-alpha-beta-'
                'and-post-defense-disabled'
            ),
            'clip_radius': 'explicit-cli-experiment-parameter',
            'krum_byzantine_count': (
                'explicit-cli-experiment-parameter'
                if args.krum_byzantine_count is not None
                else 'derived-from-sampled-attacker-fraction'
            ),
        },
        'data_provenance': {
            key: getattr(datasets.provenance, key)
            for key in datasets.provenance.__dataclass_fields__
        },
    }
    _atomic_json(manifest_path, manifest)
    print(json.dumps({
        'attack_domain': str(output),
        'manifest': str(manifest_path),
        'total_fl_round_count': result.total_fl_round_count,
        'labels': list(result.domain.snapshots),
    }, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


if __name__ == '__main__':
    raise SystemExit(main())
