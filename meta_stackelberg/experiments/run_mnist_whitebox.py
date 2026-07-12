"""CLI for MNIST white-box BRL pre-training and Meta-SG Algorithm 1."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile

import torch

from meta_stackelberg.agents.td3.agent import TD3Agent
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
    load_mnist_whitebox_policy_artifact,
    run_mnist_whitebox_backdoor_meta_sg,
)
from meta_stackelberg.experiments.paper_mnist_backdoor_scientific import (
    run_mnist_whitebox_scientific_evidence,
)
from meta_stackelberg.experiments.scientific_gate import (
    QueryEvidencePlan,
    ScientificGateThresholds,
)
from meta_stackelberg.experiments.whitebox_backdoor_evidence import (
    WhiteBoxSafetyThresholds,
)
from meta_stackelberg.security.data.mnist_global_trigger import mnist_global_trigger


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

    scientific = subparsers.add_parser('scientific')
    _common(scientific)
    scientific.add_argument('--attack-domain', required=True)
    scientific.add_argument('--policy-artifact', required=True)
    scientific.add_argument('--output', required=True)
    scientific.add_argument('--task', default='brl-norm')
    scientific.add_argument('--support-seed', type=int, default=2_000_000)
    scientific.add_argument('--support-seed-count', type=int, default=10_000)
    scientific.add_argument('--query-seed', type=int, default=101)
    scientific.add_argument('--query-seed-count', type=int, default=2)
    scientific.add_argument('--adaptation-steps', type=int)
    scientific.add_argument('--attacker-improvement', type=float, default=0.001)
    scientific.add_argument('--response-difference', type=float, default=0.001)
    scientific.add_argument('--adaptation-improvement', type=float, default=0.001)
    scientific.add_argument('--meta-advantage', type=float, default=0.001)
    scientific.add_argument('--oracle-regret', type=float, default=0.1)
    scientific.add_argument('--action-difference', type=float, default=0.001)
    scientific.add_argument('--attacker-plateau-gap', type=float, default=0.001)
    scientific.add_argument('--clean-accuracy-floor', type=float, default=0.8)
    scientific.add_argument('--asr-ceiling', type=float, default=0.2)
    scientific.add_argument('--asr-reduction', type=float, default=0.2)
    scientific.add_argument('--require-gate-pass', action='store_true')
    return parser


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--profile', choices=('micro', 'actor-active', 'paper'), default='micro',
    )
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
    if args.profile == 'actor-active':
        return AttackPolicyPretrainingConfig(
            fl_rounds=8,
            batch_size=2,
            learning_starts=2,
            train_freq=1,
            gradient_steps=1,
            replay_capacity=512,
        ), (8,)
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
    if args.profile == 'actor-active':
        args.execution_only = True
        return MNISTWhiteBoxMetaSGConfig(
            N_D=2,
            K=2,
            N_A=2,
            H=2,
            td3_batch_size=2,
            learning_starts=2,
            replay_capacity=512,
            hidden_sizes=(8,),
        )
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
    if args.command == 'meta-sg':
        return _meta_sg(args, factory)
    return _scientific(args, paper, factory)


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
    fixture = mnist_global_trigger()
    manifest = {
        'schema_version': 1,
        'protocol': result.protocol,
        'dataset': 'MNIST',
        'knowledge': 'white-box',
        'profile': args.profile,
        'execution_only': args.profile != 'paper',
        'client_training_samples': len(factory.datasets.client_train),
        'reward_view_samples': len(factory.datasets.reward),
        'query_data_used_for_training': False,
        'query_samples': len(factory.datasets.query),
        'reward_indices': list(factory.datasets.reward_indices),
        'client_partition_sha256': factory.client_partition_sha256,
        'trigger_id': fixture.identifier,
        'trigger_sha256': fixture.sha256,
        'source_class': fixture.source_class,
        'target_class': fixture.target_class,
        'workers': factory.workers,
        'backdoor_attackers': factory.backdoor_attackers,
        'sample_size': factory.sample_size,
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


def _scientific(args, paper, factory) -> int:
    artifact = load_mnist_whitebox_policy_artifact(args.policy_artifact)
    domain = load_attack_type_domain(args.attack_domain)
    if dict(artifact.attack_origins) != dict(domain.origins):
        raise ValueError('policy artifact and attack domain origins differ')
    if args.task not in domain.snapshots:
        raise ValueError(f'attack domain does not contain task {args.task!r}')
    learned_config = MNISTWhiteBoxMetaSGConfig(**dict(artifact.config))
    learned = _policy_agent(
        factory.defender_observation_dim,
        'defender',
        args.seed,
        learned_config,
    )
    learned.restore(artifact.defender)
    random_defender = _policy_agent(
        factory.defender_observation_dim,
        'defender',
        args.seed + 1,
        learned_config,
    )
    initial_attacker = _policy_agent(
        factory.attacker_observation_dim,
        'attacker',
        args.seed + 2,
        learned_config,
    )
    initial_attacker.restore(domain.snapshots[args.task])
    adaptation_steps = args.adaptation_steps
    if adaptation_steps is None:
        adaptation_steps = paper.l if args.profile == 'paper' else 1
    scientific_config = paper.scaled(
        T=1,
        K=learned_config.K,
        H=learned_config.H,
        l=adaptation_steps,
        N_A=learned_config.N_A,
        N_D=learned_config.N_D,
        workers=factory.workers,
        untargeted_attackers=factory.backdoor_attackers,
        sample_size=factory.sample_size,
        td3_batch_size=learned_config.td3_batch_size,
        learning_starts=learned_config.learning_starts,
        hidden_sizes=learned_config.hidden_sizes,
        replay_capacity=learned_config.replay_capacity,
    )
    support_seeds = tuple(range(
        args.support_seed,
        args.support_seed + args.support_seed_count,
    ))
    query_seeds = tuple(range(
        args.query_seed,
        args.query_seed + args.query_seed_count,
    ))
    result = run_mnist_whitebox_scientific_evidence(
        config=scientific_config,
        environment_factory=factory,
        evidence_plan=QueryEvidencePlan(support_seeds, query_seeds),
        meta_thresholds=ScientificGateThresholds(
            args.attacker_improvement,
            args.response_difference,
            args.adaptation_improvement,
            args.meta_advantage,
            args.oracle_regret,
            args.action_difference,
            attacker_plateau_gap=args.attacker_plateau_gap,
        ),
        safety_thresholds=WhiteBoxSafetyThresholds(
            args.clean_accuracy_floor,
            args.asr_ceiling,
            args.asr_reduction,
        ),
        learned_defender=learned,
        random_defender=random_defender,
        initial_attacker=initial_attacker,
        specialized_defenders={
            'tight': _constant_policy(learned, (-0.8, 0.0, -0.8)),
            'balanced': _constant_policy(learned, (0.0, 0.0, 0.0)),
            'loose': _constant_policy(learned, (0.8, 0.0, 0.8)),
        },
        attacker_oracle_policies={
            'low-poison': _constant_policy(
                initial_attacker, (-0.8, 0.0, -0.9),
            ),
            'mid-poison': _constant_policy(
                initial_attacker, (0.0, 0.0, -0.9),
            ),
            'high-poison': _constant_policy(
                initial_attacker, (0.8, 0.0, -0.9),
            ),
        },
        task=args.task,
        query_batch_size=paper.fl_batch_size,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        'schema_version': 1,
        'protocol': result.protocol,
        'profile': args.profile,
        'execution_only': args.profile != 'paper',
        'passed': result.passed,
        'meta_gate': {
            'passed': result.meta_sg.gate.passed,
            'thresholds': asdict(result.meta_sg.gate.thresholds),
            'checks': [asdict(check) for check in result.meta_sg.gate.checks],
        },
        'safety_gate': {
            'passed': result.safety_gate.passed,
            'thresholds': asdict(result.safety_gate.thresholds),
            'checks': [asdict(check) for check in result.safety_gate.checks],
        },
        'metrics': {
            label: {
                'mean_clean_accuracy': item.mean_clean_accuracy,
                'mean_clean_loss': item.mean_clean_loss,
                'mean_attack_success_rate': item.mean_attack_success_rate,
                'mean_safe_loss': item.mean_safe_loss,
                'mean_target_loss': item.mean_target_loss,
                'query_seeds': list(item.query_seeds),
                'per_seed': [asdict(metric) for metric in item.per_seed],
            }
            for label, item in result.metrics.items()
        },
        'budgets': {
            label: asdict(budget)
            for label, budget in result.meta_sg.budgets.items()
        },
        'used_support_seeds': list(result.meta_sg.used_support_seeds),
        'query_seeds': list(result.meta_sg.query_seeds),
        'specialized_oracle_label': result.meta_sg.specialized_oracle_label,
        'attacker_oracle_label': result.meta_sg.attacker_oracle_label,
        'query_data_used_for_training': False,
        'client_partition_sha256': factory.client_partition_sha256,
        'trigger_sha256': mnist_global_trigger().sha256,
    }
    evidence_path = output / 'scientific.json'
    _atomic_json(evidence_path, payload)
    print(json.dumps({
        'scientific_evidence': str(evidence_path),
        'passed': result.passed,
        'meta_gate_passed': result.meta_sg.gate.passed,
        'safety_gate_passed': result.safety_gate.passed,
    }, indent=2, sort_keys=True))
    if args.require_gate_pass and not result.passed:
        return 2
    return 0


def _policy_agent(
    obs_dim: int,
    role: str,
    seed: int,
    config: MNISTWhiteBoxMetaSGConfig,
) -> TD3Agent:
    return TD3Agent(
        obs_dim=obs_dim,
        action_dim=3,
        role=role,
        seed=seed,
        hidden_sizes=tuple(config.hidden_sizes),
        learning_rate=config.policy_learning_rate,
        gamma=config.gamma,
        tau=config.tau,
        policy_delay=config.policy_delay,
        target_policy_noise=config.target_policy_noise,
        noise_clip=config.noise_clip,
    )


def _constant_policy(
    source: TD3Agent,
    raw_action: tuple[float, float, float],
) -> TD3Agent:
    result = source.clone()
    for parameter in result.actor.parameters():
        parameter.data.zero_()
    final = tuple(result.actor.modules())[-1]
    final.bias.data.copy_(torch.atanh(torch.tensor(raw_action) * 0.999))
    return result


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
