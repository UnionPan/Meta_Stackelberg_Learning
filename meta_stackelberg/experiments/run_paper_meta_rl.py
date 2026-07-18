"""Train the independent Algorithm 2 Meta-RL baseline on fixed attacks."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.agents.td3.config import PaperMetaSGConfig
from meta_stackelberg.agents.td3.replay import flatten_observation
from meta_stackelberg.experiments.attack_domain import (
    BalancedAttackTypeSampler,
    UniformAttackTypeSampler,
    load_attack_type_domain,
)
from meta_stackelberg.experiments.paper_meta_sg import (
    ATTACKER_OBSERVATION_KEYS,
    DEFENDER_OBSERVATION_KEYS,
    ScaledPaperMetaSGTrainingRunner,
)
from meta_stackelberg.experiments.paper_mnist_backdoor_env import (
    PaperMNISTBackdoorEnvironmentFactory,
    load_whitebox_mnist_datasets,
)
from meta_stackelberg.experiments.paper_mnist_env import (
    PaperMNISTEnvironmentFactory,
    load_paper_mnist_datasets,
)
from meta_stackelberg.security.attacks.ipm import IPMAttack
from meta_stackelberg.security.attacks.lmp import LMPAttack


TABLE4_TASKS = ('na', 'ipm', 'lmp', 'bfl', 'dba')
GLOBAL_TASKS = ('na', 'ipm', 'lmp', 'rl-krum', 'rl-clipmed')
# Backward-compatible public name for the paper Table 4 baseline.
TASKS = TABLE4_TASKS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--T', type=int, default=100)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--H', type=int, default=200)
    parser.add_argument('--l', type=int, default=10)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--partition-seed', type=int, default=17)
    parser.add_argument('--model-seed', type=int, default=99)
    parser.add_argument('--support-seed', type=int, default=4_000_000)
    parser.add_argument('--ipm-scale', type=float, default=2.0)
    parser.add_argument('--lmp-scale', type=float, default=2.0)
    parser.add_argument('--backdoor-poison-fraction', type=float, default=1.0)
    parser.add_argument('--local-search-batch-size', type=int, default=128)
    parser.add_argument('--td3-batch-size', type=int, default=256)
    parser.add_argument('--learning-starts', type=int, default=100)
    parser.add_argument('--workers', type=int, default=100)
    parser.add_argument('--untargeted-attackers', type=int, default=20)
    parser.add_argument('--sample-size', type=int, default=10)
    parser.add_argument('--parallel-tasks', type=int, default=1)
    parser.add_argument('--parallel-clients', type=int, default=1)
    parser.add_argument(
        '--cpu-threads', type=int, default=0,
        help='PyTorch intra-op CPU threads; zero keeps the process default',
    )
    parser.add_argument('--deterministic-torch', action='store_true')
    parser.add_argument(
        '--meta-update-step', type=float,
        help='optional Reptile outer-step stabilization override (paper: 1.0)',
    )
    parser.add_argument('--checkpoint-interval', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument(
        '--materialize-mnist', action='store_true',
        help='materialize torchvision MNIST as tensors once at startup',
    )
    parser.add_argument(
        '--mnist-normalization', choices=('none', 'standard'), default='none',
    )
    parser.add_argument(
        '--data-split', choices=('iid', 'paper-q'), default='paper-q',
    )
    parser.add_argument(
        '--alpha-floor-ratio', type=float, default=0.0,
        help='minimum alpha as a fraction of the observed update norm',
    )
    parser.add_argument(
        '--defender-norm-reference',
        choices=('max', 'median'),
        default='max',
        help='update-norm statistic used to decode the alpha action',
    )
    parser.add_argument(
        '--post-defense-mode',
        choices=('neuroclip', 'identity'),
        default='neuroclip',
        help=(
            'post-aggregation model defense used to define the root-loss '
            'reward; identity disables NeuroClip for global-only studies'
        ),
    )
    parser.add_argument(
        '--task-domain', choices=('table4', 'global'), default='table4',
        help='Table 4 fixed tasks or the global model-poisoning comparison tasks',
    )
    parser.add_argument(
        '--attack-domain',
        help='pretrained RL attacker artifact required by --task-domain global',
    )
    parser.add_argument(
        '--task-sampler', choices=('uniform', 'balanced'), default='uniform',
        help='balanced guarantees task coverage for small-scale validation',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.parallel_tasks <= 0 or args.parallel_clients <= 0:
        raise ValueError('parallel task/client counts must be positive')
    if args.cpu_threads < 0:
        raise ValueError('--cpu-threads must be non-negative')
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(1)
    if args.deterministic_torch:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if args.resume and not Path(args.checkpoint).is_file():
        raise ValueError('--resume requires an existing checkpoint')
    if args.task_domain == 'global' and not args.attack_domain:
        raise ValueError('--task-domain global requires --attack-domain')
    tasks = TABLE4_TASKS if args.task_domain == 'table4' else GLOBAL_TASKS
    paper = PaperMetaSGConfig()
    if args.meta_update_step is not None:
        paper = replace(paper, meta_update_step=args.meta_update_step)
    config = paper.scaled(
        T=args.T,
        K=args.K,
        H=args.H,
        l=args.l,
        N_A=paper.N_A,
        N_D=paper.N_D,
        workers=args.workers,
        untargeted_attackers=args.untargeted_attackers,
        sample_size=args.sample_size,
        td3_batch_size=args.td3_batch_size,
        learning_starts=args.learning_starts,
        hidden_sizes=paper.hidden_sizes,
        replay_capacity=paper.replay_capacity,
    )
    datasets = load_paper_mnist_datasets(
        args.data_root,
        seed=args.partition_seed,
        root_samples=paper.root_samples_mnist,
        download=False,
        materialize=args.materialize_mnist,
        normalization=args.mnist_normalization,
    )
    factory = PaperMNISTEnvironmentFactory(
        train_dataset=datasets.client_train,
        root_dataset=datasets.root,
        partition_seed=args.partition_seed,
        model_seed=args.model_seed,
        workers=args.workers,
        untargeted_attackers=args.untargeted_attackers,
        sample_size=args.sample_size,
        non_iid_q=paper.non_iid_q,
        partition_mode=args.data_split,
        fl_batch_size=paper.fl_batch_size,
        local_iterations=paper.local_iterations,
        client_learning_rate=paper.client_learning_rate,
        local_search_batch_size=args.local_search_batch_size,
        local_search_gradient_norm_cap=1.0,
        device=args.device,
        post_defense_mode=args.post_defense_mode,
        parallel_clients=args.parallel_clients,
        defender_alpha_floor_ratio=args.alpha_floor_ratio,
        defender_norm_reference=args.defender_norm_reference,
    )
    backdoor_factory = None
    if args.task_domain == 'table4':
        backdoor_datasets = load_whitebox_mnist_datasets(
            args.data_root,
            seed=args.partition_seed,
            reward_samples=200,
            download=False,
        )
        backdoor_factory = PaperMNISTBackdoorEnvironmentFactory(
            datasets=backdoor_datasets,
            partition_seed=args.partition_seed,
            model_seed=args.model_seed,
            workers=config.workers,
            backdoor_attackers=paper.backdoor_attackers,
            sample_size=config.sample_size,
            fl_batch_size=paper.fl_batch_size,
            local_iterations=paper.local_iterations,
            client_learning_rate=paper.client_learning_rate,
            malicious_batch_size=paper.fl_batch_size,
            fixed_attack_poison_fraction=args.backdoor_poison_fraction,
            device=args.device,
        )
    probe = factory.make(seed=args.seed, horizon=args.H, task_id='meta-rl-probe')
    defender_dim = len(flatten_observation(
        probe.defender_observation(), DEFENDER_OBSERVATION_KEYS,
    ))
    attacker_probe = factory.make(
        seed=args.seed, horizon=args.H, task_id='meta-rl-attacker-probe',
    )
    attacker_dim = len(flatten_observation(
        attacker_probe.begin_round(
            np.zeros(3, dtype=np.float32),
        ).attacker_observation,
        ATTACKER_OBSERVATION_KEYS,
    ))
    if backdoor_factory is not None and (
        backdoor_factory.defender_observation_dim != defender_dim
        or backdoor_factory.attacker_observation_dim != attacker_dim
    ):
        raise RuntimeError(
            'untargeted and backdoor Meta-RL observation spaces differ',
        )
    initial_defender = _agent(
        paper, defender_dim, 'defender', args.seed + 2, args.device,
    )
    fixed_responses = {
        task: _agent(
            paper, attacker_dim, 'attacker', args.seed + 10 + index,
            args.device,
        )
        for index, task in enumerate(tasks)
    }
    attack_origins = {}
    if args.task_domain == 'global':
        source = load_attack_type_domain(args.attack_domain)
        expected_rl = set(GLOBAL_TASKS) - {'na', 'ipm', 'lmp'}
        if set(source.snapshots) != expected_rl:
            raise ValueError(
                'global attack domain must contain rl-krum and rl-clipmed',
            )
        for task, snapshot in source.snapshots.items():
            fixed_responses[task].restore(snapshot)
        attack_origins = dict(source.origins)

    def env_factory(task, rollout_seed, horizon):
        if task not in tasks:
            raise ValueError(f'unknown Meta-RL fixed task {task!r}')
        if task in {'bfl', 'dba'}:
            if backdoor_factory is None:
                raise RuntimeError('backdoor factory is unavailable')
            return backdoor_factory.make(
                seed=rollout_seed,
                horizon=horizon,
                task_id=f'meta-rl-{task}',
                fixed_attack=task,
            )
        malicious_ids = () if task == 'na' else factory.malicious_ids
        if task == 'ipm':
            attack_factory = lambda action: IPMAttack(
                scale=args.ipm_scale,
                num_examples_by_client=factory.attacker_num_examples,
            )
        elif task == 'lmp':
            attack_factory = lambda action: LMPAttack(
                scale=args.lmp_scale,
                num_examples_by_client=factory.attacker_num_examples,
            )
        else:
            attack_factory = None
        return factory.make(
            seed=rollout_seed,
            horizon=horizon,
            task_id=f'meta-rl-{task}',
            malicious_ids=malicious_ids,
            attack_generator_factory=attack_factory,
        )

    sampler_type = (
        UniformAttackTypeSampler
        if args.task_sampler == 'uniform'
        else BalancedAttackTypeSampler
    )
    sampler = sampler_type(tasks, seed=args.seed + 1_000)
    result = ScaledPaperMetaSGTrainingRunner(
        config=config,
        env_factory=env_factory,
        defender_obs_dim=defender_dim,
        attacker_obs_dim=attacker_dim,
        support_seed=args.support_seed,
        protocol_signature={
            'dataset': 'MNIST',
            'method': 'meta-rl',
            'algorithm': 'Algorithm 2',
            'fixed_attack_domain': tasks,
            'scope': (
                'table4-mixed-fixed-attacks'
                if args.task_domain == 'table4'
                else 'global-model-poisoning-tasks'
            ),
            'task_sampler': args.task_sampler,
            'meta_update_step': paper.meta_update_step,
            'meta_update_step_source': (
                'paper-explicit'
                if args.meta_update_step is None
                else 'declared-stabilization-override'
            ),
            'rl_attack_origins': attack_origins,
            'ipm_scale': args.ipm_scale,
            'lmp_scale': args.lmp_scale,
            'backdoor_poison_fraction': args.backdoor_poison_fraction,
            'backdoor_attackers': paper.backdoor_attackers,
            'dba_subtrigger_rule': 'per-sampled-attacker-uniform-random',
            'table4_interpretation': (
                'Table 4 five-task domain; Appendix C text differs'
            ),
            'partition_seed': args.partition_seed,
            'model_seed': args.model_seed,
            'device': str(args.device),
            'local_search_gradient_norm_cap': 1.0,
            'local_training_budget': (
                'one-minibatch-step-per-local-iteration'
            ),
            'post_defense_mode': args.post_defense_mode,
            'root_loss_evaluation': (
                'raw-model-cached-sequential-before-after'
                if args.post_defense_mode == 'identity'
                else 'neuroclip-before-after-every-round'
            ),
            'mnist_input_pipeline': (
                'materialized-tensor-dataset'
                if args.materialize_mnist
                else 'torchvision-transform-per-sample'
            ),
            'mnist_normalization': args.mnist_normalization,
            'mnist_partition': args.data_split,
            'non_iid_q': paper.non_iid_q if args.data_split == 'paper-q' else None,
            'defender_alpha_floor_ratio': args.alpha_floor_ratio,
            'defender_norm_reference': args.defender_norm_reference,
            'local_model_workspace': 'reused-and-reset-per-client-update',
            'parallel_tasks': min(args.parallel_tasks, args.K),
            'parallel_clients': min(args.parallel_clients, args.sample_size),
            'cpu_threads': args.cpu_threads,
            'deterministic_torch': args.deterministic_torch,
            'attack_client_sampling': (
                'uniform-conditioned-on-at-least-one-benign-reference'
            ),
        },
        parallel_tasks=args.parallel_tasks,
    ).run(
        initial_defender=initial_defender,
        initial_attackers=fixed_responses,
        sample_tasks=sampler,
        training_method='meta-rl',
        checkpoint_path=args.checkpoint,
        resume=args.resume,
        checkpoint_interval=args.checkpoint_interval,
    )
    print(json.dumps({
        'protocol': 'paper-meta-rl-fixed-attack-training-v1',
        'method': 'meta-rl',
        'algorithm': 'Algorithm 2',
        'scope': (
            'table4-mixed-fixed-attacks'
            if args.task_domain == 'table4'
            else 'global-model-poisoning-tasks'
        ),
        'fixed_attack_domain': tasks,
        'task_sampler': args.task_sampler,
        'meta_update_step': paper.meta_update_step,
        'T': args.T,
        'K': args.K,
        'H': args.H,
        'l': args.l,
        'workers': args.workers,
        'untargeted_attackers': args.untargeted_attackers,
        'sample_size': args.sample_size,
        'parallel_tasks': min(args.parallel_tasks, args.K),
        'parallel_clients': min(args.parallel_clients, args.sample_size),
        'cpu_threads': args.cpu_threads,
        'deterministic_torch': args.deterministic_torch,
        'local_training_budget': 'one-minibatch-step-per-local-iteration',
        'post_defense_mode': args.post_defense_mode,
        'mnist_input_pipeline': (
            'materialized-tensor-dataset'
            if args.materialize_mnist
            else 'torchvision-transform-per-sample'
        ),
        'mnist_normalization': args.mnist_normalization,
        'mnist_partition': args.data_split,
        'defender_alpha_floor_ratio': args.alpha_floor_ratio,
        'defender_norm_reference': args.defender_norm_reference,
        'algorithm1_completed': len(result.algorithm1.iterations),
        'algorithm2_completed': len(result.algorithm2.iterations),
        'trajectory_count': result.trajectory_count,
        'checkpoint': str(Path(args.checkpoint).resolve()),
    }, sort_keys=True), flush=True)
    return 0


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


if __name__ == '__main__':
    raise SystemExit(main())
