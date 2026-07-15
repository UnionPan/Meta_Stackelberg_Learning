"""Train the independent Algorithm 2 Meta-RL baseline on fixed attacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.agents.td3.config import PaperMetaSGConfig
from meta_stackelberg.agents.td3.replay import flatten_observation
from meta_stackelberg.experiments.attack_domain import UniformAttackTypeSampler
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


TASKS = ('na', 'ipm', 'lmp', 'bfl', 'dba')


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
    parser.add_argument('--checkpoint-interval', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.resume and not Path(args.checkpoint).is_file():
        raise ValueError('--resume requires an existing checkpoint')
    paper = PaperMetaSGConfig()
    config = paper.scaled(
        T=args.T,
        K=args.K,
        H=args.H,
        l=args.l,
        N_A=paper.N_A,
        N_D=paper.N_D,
        workers=paper.workers,
        untargeted_attackers=paper.untargeted_attackers,
        sample_size=int(paper.workers * paper.subsampling_rate),
        td3_batch_size=paper.td3_batch_size,
        learning_starts=paper.learning_starts,
        hidden_sizes=paper.hidden_sizes,
        replay_capacity=paper.replay_capacity,
    )
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
        workers=config.workers,
        untargeted_attackers=config.untargeted_attackers,
        sample_size=config.sample_size,
        non_iid_q=paper.non_iid_q,
        fl_batch_size=paper.fl_batch_size,
        local_iterations=paper.local_iterations,
        client_learning_rate=paper.client_learning_rate,
        local_search_batch_size=args.local_search_batch_size,
        local_search_gradient_norm_cap=1.0,
        device=args.device,
    )
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
    if (
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
        for index, task in enumerate(TASKS)
    }

    def env_factory(task, rollout_seed, horizon):
        if task not in TASKS:
            raise ValueError(f'unknown Meta-RL fixed task {task!r}')
        if task in {'bfl', 'dba'}:
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

    sampler = UniformAttackTypeSampler(TASKS, seed=args.seed + 1_000)
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
            'fixed_attack_domain': TASKS,
            'scope': 'table4-mixed-fixed-attacks',
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
        },
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
        'scope': 'table4-mixed-fixed-attacks',
        'fixed_attack_domain': TASKS,
        'T': args.T,
        'K': args.K,
        'H': args.H,
        'l': args.l,
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
