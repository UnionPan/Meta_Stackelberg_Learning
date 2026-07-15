"""CLI for reproducible MNIST/CIFAR Meta-SG evidence runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from meta_stackelberg.agents.td3.config import (
    PaperMetaSGConfig,
    ScaledMetaSGConfig,
)
from meta_stackelberg.experiments.paper_cifar_env import load_paper_cifar_datasets
from meta_stackelberg.experiments.paper_cifar_evidence import (
    run_paper_cifar_scaled_evidence,
)
from meta_stackelberg.experiments.paper_mnist_env import load_paper_mnist_datasets
from meta_stackelberg.experiments.paper_mnist_evidence import (
    run_paper_mnist_scaled_evidence,
)
from meta_stackelberg.experiments.scaled_artifact import (
    save_scaled_evidence_artifact,
)
from meta_stackelberg.experiments.scientific_gate import ScientificGateThresholds
from meta_stackelberg.experiments.attack_domain import load_attack_type_domain


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=('mnist', 'cifar'))
    parser.add_argument(
        '--profile', default='actor-active',
        choices=('micro', 'actor-active', 'declared-scaled', 'paper'),
    )
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--allow-paper-scale', action='store_true')
    parser.add_argument(
        '--task-batch-size', '--K', dest='task_batch_size', type=int,
        help=(
            'override the number of attack tasks sampled per outer iteration; '
            'defaults to the selected profile value'
        ),
    )
    parser.add_argument('--attack-domain')
    parser.add_argument('--allow-random-attacker-init', action='store_true')
    parser.add_argument('--require-gate-pass', action='store_true')
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--partition-seed', type=int, default=17)
    parser.add_argument('--model-seed', type=int, default=99)
    parser.add_argument('--query-seed', type=int, default=101)
    parser.add_argument('--attacker-improvement', type=float, default=0.001)
    parser.add_argument('--response-difference', type=float, default=0.001)
    parser.add_argument('--adaptation-improvement', type=float, default=0.001)
    parser.add_argument('--meta-advantage', type=float, default=0.001)
    parser.add_argument('--oracle-regret', type=float, default=0.1)
    parser.add_argument('--action-difference', type=float, default=0.001)
    parser.add_argument('--attacker-plateau-gap', type=float, default=0.001)
    parser.add_argument('--local-search-batch-size', type=int, default=128)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--training-checkpoint')
    parser.add_argument('--training-checkpoint-interval', type=int, default=1)
    parser.add_argument('--resume-training', action='store_true')
    return parser


def make_profile_config(args) -> ScaledMetaSGConfig:
    paper = PaperMetaSGConfig()
    if args.task_batch_size is not None and args.task_batch_size <= 0:
        raise ValueError('--task-batch-size must be positive')
    common = dict(
        workers=paper.workers,
        untargeted_attackers=paper.untargeted_attackers,
        sample_size=int(paper.workers * paper.subsampling_rate),
    )
    if args.profile == 'micro':
        values = dict(
            T=1, K=1, H=1, l=1, N_A=1, N_D=1,
            td3_batch_size=1, learning_starts=1,
            hidden_sizes=(8,), replay_capacity=256,
        )
    elif args.profile == 'actor-active':
        values = dict(
            T=1, K=1, H=1, l=2, N_A=2, N_D=2,
            td3_batch_size=2, learning_starts=2,
            hidden_sizes=(8,), replay_capacity=512,
        )
    elif args.profile == 'declared-scaled':
        values = dict(
            T=2, K=2, H=8, l=2, N_A=2, N_D=2,
            td3_batch_size=16, learning_starts=16,
            hidden_sizes=(64, 64), replay_capacity=4096,
        )
    else:
        if not args.allow_paper_scale:
            raise ValueError('paper profile requires --allow-paper-scale')
        values = dict(
            T=paper.T,
            K=paper.K,
            H=paper.H_mnist if args.dataset == 'mnist' else paper.H_cifar,
            l=paper.l,
            N_A=paper.N_A,
            N_D=paper.N_D,
            td3_batch_size=paper.td3_batch_size,
            learning_starts=paper.learning_starts,
            hidden_sizes=paper.hidden_sizes,
            replay_capacity=paper.replay_capacity,
        )
    if args.task_batch_size is not None:
        values['K'] = args.task_batch_size
    return paper.scaled(**values, **common)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.resume_training and not args.training_checkpoint:
        raise ValueError('--resume-training requires --training-checkpoint')
    if args.training_checkpoint_interval <= 0:
        raise ValueError('--training-checkpoint-interval must be positive')
    config = make_profile_config(args)
    validate_attack_initialization(args)
    attack_domain = (
        load_attack_type_domain(args.attack_domain)
        if args.attack_domain else None
    )
    paper = config.paper_reference
    thresholds = ScientificGateThresholds(
        args.attacker_improvement,
        args.response_difference,
        args.adaptation_improvement,
        args.meta_advantage,
        args.oracle_regret,
        args.action_difference,
        attacker_plateau_gap=args.attacker_plateau_gap,
    )
    query_seeds = (args.query_seed, args.query_seed + 1)
    scientific_support_seeds = tuple(range(2_000_000, 2_010_000))
    if args.dataset == 'mnist':
        datasets = load_paper_mnist_datasets(
            args.data_root,
            seed=args.partition_seed,
            root_samples=paper.root_samples_mnist,
            download=args.download,
        )
        result = run_paper_mnist_scaled_evidence(
            config=config,
            datasets=datasets,
            thresholds=thresholds,
            query_seeds=query_seeds,
            training_support_seed=1_000_000,
            scientific_support_seeds=scientific_support_seeds,
            seed=args.seed,
            partition_seed=args.partition_seed,
            model_seed=args.model_seed,
            local_search_batch_size=args.local_search_batch_size,
            attack_domain=attack_domain,
            device=args.device,
            training_checkpoint_path=args.training_checkpoint,
            resume_training=args.resume_training,
            training_checkpoint_interval=args.training_checkpoint_interval,
        )
    else:
        datasets = load_paper_cifar_datasets(
            args.data_root,
            seed=args.partition_seed,
            root_samples=paper.root_samples_cifar,
            download=args.download,
        )
        result = run_paper_cifar_scaled_evidence(
            config=config,
            datasets=datasets,
            thresholds=thresholds,
            query_seeds=query_seeds,
            training_support_seed=1_000_000,
            scientific_support_seeds=scientific_support_seeds,
            seed=args.seed,
            partition_seed=args.partition_seed,
            model_seed=args.model_seed,
            local_search_batch_size=args.local_search_batch_size,
            attack_domain=attack_domain,
            device=args.device,
            training_checkpoint_path=args.training_checkpoint,
            resume_training=args.resume_training,
            training_checkpoint_interval=args.training_checkpoint_interval,
        )
    artifact = save_scaled_evidence_artifact(Path(args.output), result)
    summary = {
        'artifact': str(artifact.directory),
        'dataset': args.dataset,
        'profile': args.profile,
        'gate_passed': result.scientific.gate.passed,
        'checks': [as_check_dict(check) for check in result.scientific.gate.checks],
        'training_trajectories': result.training.trajectory_count,
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    if args.require_gate_pass and not result.scientific.gate.passed:
        return 2
    return 0


def validate_attack_initialization(args) -> None:
    if (
        args.profile == 'paper'
        and not args.attack_domain
        and not args.allow_random_attacker_init
    ):
        raise ValueError(
            'paper profile requires --attack-domain or explicit '
            '--allow-random-attacker-init deviation',
        )


def as_check_dict(check) -> dict:
    return {
        'name': check.name,
        'passed': check.passed,
        'observed': check.observed,
        'threshold': check.threshold,
        'alternative_observed': check.alternative_observed,
        'alternative_threshold': check.alternative_threshold,
    }


if __name__ == '__main__':
    raise SystemExit(main())
