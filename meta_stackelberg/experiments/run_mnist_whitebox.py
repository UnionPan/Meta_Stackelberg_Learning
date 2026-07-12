"""CLI for MNIST white-box BRL pre-training and Meta-SG Algorithm 1."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile

from meta_stackelberg.agents.td3.config import PaperMetaSGConfig
from meta_stackelberg.experiments.attack_domain import (
    load_attack_type_domain,
    save_attack_type_domain,
)
from meta_stackelberg.experiments.attack_pretraining import AttackPolicyPretrainingConfig
from meta_stackelberg.experiments.backdoor_attack_pretraining import (
    BackdoorAttackPretrainingTask,
    pretrain_backdoor_attack_type_domain,
)
from meta_stackelberg.experiments.paper_mnist_backdoor_env import (
    PaperMNISTBackdoorEnvironmentFactory,
    load_whitebox_mnist_datasets,
)
from meta_stackelberg.experiments.paper_mnist_backdoor_meta_sg import (
    MNISTWhiteBoxMetaSGConfig,
    run_mnist_whitebox_backdoor_meta_sg,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)

    pretrain = subparsers.add_parser('pretrain')
    _common(pretrain)
    pretrain.add_argument('--output', required=True)
    pretrain.add_argument('--norm-clip-radius', type=float, default=1.0)
    pretrain.add_argument('--neuroclip-epsilon', type=float, default=7.0)
    pretrain.add_argument('--checkpoint-dir')
    pretrain.add_argument('--checkpoint-interval', type=int, default=25)
    pretrain.add_argument('--resume', action='store_true')

    meta = subparsers.add_parser('meta-sg')
    _common(meta)
    meta.add_argument('--attack-domain', required=True)
    meta.add_argument('--output', required=True)
    meta.add_argument('--support-seed', type=int, default=1_000_000)
    meta.add_argument('--query-seed', type=int, default=101)
    meta.set_defaults(execution_only=True)
    return parser


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--profile', choices=('micro', 'paper'), default='micro')
    parser.add_argument('--allow-paper-scale', action='store_true')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--partition-seed', type=int, default=17)
    parser.add_argument('--model-seed', type=int, default=99)
    parser.add_argument('--reward-samples', type=int, default=200)
    parser.add_argument('--workers', type=int, default=100)
    parser.add_argument('--backdoor-attackers', type=int, default=5)
    parser.add_argument('--sample-size', type=int, default=10)


def make_pretraining_config(
    args,
) -> tuple[AttackPolicyPretrainingConfig, tuple[int, ...]]:
    paper = PaperMetaSGConfig()
    if args.profile == 'paper':
        if not args.allow_paper_scale:
            raise ValueError('paper profile requires --allow-paper-scale')
        return AttackPolicyPretrainingConfig.from_paper(paper), paper.hidden_sizes
    return AttackPolicyPretrainingConfig(
        fl_rounds=2,
        batch_size=1,
        learning_starts=1,
        train_freq=1,
        gradient_steps=1,
        replay_capacity=256,
    ), (8,)


def make_meta_config(args) -> MNISTWhiteBoxMetaSGConfig:
    if args.profile == 'paper':
        if not args.allow_paper_scale:
            raise ValueError('paper profile requires --allow-paper-scale')
        args.execution_only = False
        return MNISTWhiteBoxMetaSGConfig()
    args.execution_only = True
    return MNISTWhiteBoxMetaSGConfig(
        N_D=1,
        K=1,
        N_A=1,
        H=1,
        td3_batch_size=1,
        learning_starts=1,
        replay_capacity=256,
        hidden_sizes=(8,),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paper = PaperMetaSGConfig()
    if args.profile == 'paper' and (
        args.workers != paper.workers
        or args.backdoor_attackers != paper.backdoor_attackers
        or args.sample_size != int(paper.workers * paper.subsampling_rate)
    ):
        raise ValueError('paper profile requires 100 workers, 5 attackers, sample size 10')
    datasets = load_whitebox_mnist_datasets(
        args.data_root,
        seed=args.partition_seed,
        reward_samples=args.reward_samples,
        download=args.download,
    )
    factory = PaperMNISTBackdoorEnvironmentFactory(
        datasets=datasets,
        partition_seed=args.partition_seed,
        model_seed=args.model_seed,
        workers=args.workers,
        backdoor_attackers=args.backdoor_attackers,
        sample_size=args.sample_size,
        fl_batch_size=paper.fl_batch_size,
        local_iterations=paper.local_iterations,
        client_learning_rate=paper.client_learning_rate,
        malicious_batch_size=paper.fl_batch_size,
        defender_lambda=paper.default_backdoor_reward_lambda,
        attacker_lambda=paper.default_backdoor_reward_lambda,
    )
    if args.command == 'pretrain':
        return _pretrain(args, paper, factory)
    return _meta_sg(args, factory)


def _pretrain(args, paper, factory) -> int:
    config, hidden_sizes = make_pretraining_config(args)
    output = Path(args.output)
    checkpoint_directory = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else output.with_suffix(output.suffix + '.checkpoints')
    )
    result = pretrain_backdoor_attack_type_domain(
        config=config,
        paper=paper,
        environment_factory=factory,
        tasks=(
            BackdoorAttackPretrainingTask(
                'brl-norm', 'norm-bounding',
                clip_radius=args.norm_clip_radius,
            ),
            BackdoorAttackPretrainingTask(
                'brl-neuroclip', 'neuroclip',
                epsilon=args.neuroclip_epsilon,
            ),
        ),
        hidden_sizes=hidden_sizes,
        seed=args.seed,
        checkpoint_directory=checkpoint_directory,
        checkpoint_interval=args.checkpoint_interval,
        resume_checkpoints=args.resume,
    )
    save_attack_type_domain(output, result.domain)
    manifest = {
        'schema_version': 1,
        'protocol': result.protocol,
        'dataset': 'MNIST',
        'knowledge': 'white-box',
        'profile': args.profile,
        'execution_only': args.profile == 'micro',
        'client_training_samples': len(factory.datasets.client_train),
        'reward_view_samples': len(factory.datasets.reward),
        'query_data_used_for_training': False,
        'pretraining_config': asdict(config),
        'attack_origins': dict(result.domain.origins),
        'norm_clip_radius': args.norm_clip_radius,
        'neuroclip_epsilon': args.neuroclip_epsilon,
        'total_fl_round_count': result.total_fl_round_count,
        'checkpoint_directory': str(checkpoint_directory),
    }
    manifest_path = output.with_suffix(output.suffix + '.manifest.json')
    _atomic_json(manifest_path, manifest)
    print(json.dumps({
        'attack_domain': str(output),
        'manifest': str(manifest_path),
        'labels': list(result.domain.snapshots),
        'total_fl_round_count': result.total_fl_round_count,
    }, indent=2, sort_keys=True))
    return 0


def _meta_sg(args, factory) -> int:
    config = make_meta_config(args)
    result = run_mnist_whitebox_backdoor_meta_sg(
        environment_factory=factory,
        attack_domain=load_attack_type_domain(args.attack_domain),
        output_dir=args.output,
        seed=args.seed,
        support_seed=args.support_seed,
        config=config,
        query_seeds=(args.query_seed, args.query_seed + 1),
    )
    print(json.dumps({
        'run': str(Path(args.output)),
        'profile': args.profile,
        'execution_only': args.execution_only,
        'algorithm': 'meta-sg-algorithm1-reptile',
        'trajectory_count': result.trajectory_count,
        'defender_fingerprint': result.defender.fingerprint(),
    }, indent=2, sort_keys=True))
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
