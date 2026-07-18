"""Evaluate a completed canonical Meta-SG checkpoint on clean/IPM/LMP/RL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.agents.td3.config import PaperMetaSGConfig
from meta_stackelberg.agents.td3.replay import flatten_observation
from meta_stackelberg.experiments.model_poisoning_evaluation import (
    canonical_model_poisoning_scenarios,
    evaluate_model_poisoning_scenario,
    model_poisoning_attack_factory,
    summarize_model_poisoning_evaluation,
)
from meta_stackelberg.experiments.attack_domain import load_attack_type_domain
from meta_stackelberg.experiments.paper_meta_sg import (
    ATTACKER_OBSERVATION_KEYS,
    DEFENDER_OBSERVATION_KEYS,
    PaperOnlineAdaptationTrainingRunner,
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
    parser.add_argument(
        '--training-H', type=int,
        help='checkpoint training horizon when evaluation uses a different H',
    )
    parser.add_argument('--ipm-scale', type=float, default=2.0)
    parser.add_argument('--lmp-scale', type=float, default=2.0)
    parser.add_argument('--workers', type=int, default=100)
    parser.add_argument('--untargeted-attackers', type=int, default=20)
    parser.add_argument('--sample-size', type=int, default=10)
    parser.add_argument('--parallel-clients', type=int, default=1)
    parser.add_argument('--cpu-threads', type=int, default=0)
    parser.add_argument('--deterministic-torch', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--materialize-mnist', action='store_true')
    parser.add_argument(
        '--mnist-normalization', choices=('none', 'standard'), default='none',
    )
    parser.add_argument(
        '--data-split', choices=('iid', 'paper-q'), default='paper-q',
    )
    parser.add_argument('--alpha-floor-ratio', type=float, default=0.0)
    parser.add_argument(
        '--defender-norm-reference',
        choices=('max', 'median'),
        default='max',
    )
    parser.add_argument(
        '--post-defense-mode',
        choices=('neuroclip', 'identity'),
        default='neuroclip',
    )
    parser.add_argument('--online-support-seed', type=int, default=3_000_000)
    parser.add_argument('--online-T', type=int, default=10)
    parser.add_argument('--online-H', type=int, default=100)
    parser.add_argument('--online-l', type=int, default=10)
    parser.add_argument('--online-steps', type=int, default=100)
    parser.add_argument('--online-batch-size', type=int, default=256)
    parser.add_argument('--online-learning-starts', type=int, default=100)
    parser.add_argument(
        '--online-adaptation-step', type=float,
        help='online TD3 optimizer step; defaults to the paper-declared 0.01',
    )
    parser.add_argument(
        '--online-actor-logit-l2', type=float, default=0.0,
        help=(
            'optional online-only pre-tanh L2 penalty used to recover from '
            'a saturated Meta actor; zero preserves ordinary TD3'
        ),
    )
    parser.add_argument(
        '--online-actor-logit-l2-mask', nargs=3, type=float,
        metavar=('ALPHA', 'BETA', 'EPSILON'),
        help=(
            'optional per-action weights for --online-actor-logit-l2; '
            'defaults to all three dimensions'
        ),
    )
    parser.add_argument(
        '--skip-online-adaptation', action='store_true',
        help='evaluate the frozen Meta policy before any online TD3 updates',
    )
    parser.add_argument(
        '--online-selection',
        choices=('always', 'reward-guarded'),
        default='reward-guarded',
        help=(
            'deploy the adapted policy unconditionally or require an '
            'improvement in held-out defender reward'
        ),
    )
    parser.add_argument('--online-selection-seed', type=int, default=6_000_000)
    parser.add_argument('--online-selection-repeats', type=int, default=2)
    parser.add_argument('--online-selection-horizon', type=int)
    parser.add_argument('--online-selection-margin', type=float, default=0.0)
    parser.add_argument(
        '--attack-domain',
        help='pretrained RL attack domain required for Meta-RL evaluation',
    )
    parser.add_argument(
        '--method', choices=('meta-sg', 'meta-rl'), default='meta-sg',
        help='select the independently trained policy to evaluate',
    )
    parser.add_argument(
        '--scenarios', nargs='+',
        help=(
            'optional scenario names to evaluate; defaults to the complete '
            'clean/IPM/LMP/RL domain'
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.parallel_clients <= 0 or args.cpu_threads < 0:
        raise ValueError(
            'parallel clients must be positive and CPU threads non-negative'
        )
    if (
        args.online_selection_repeats <= 0
        or (
            args.online_selection_horizon is not None
            and args.online_selection_horizon <= 0
        )
        or not np.isfinite(args.online_selection_margin)
        or args.online_selection_margin < 0
    ):
        raise ValueError('online reward-guard settings are invalid')
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(1)
    if args.deterministic_torch:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    checkpoint = load_scaled_training_checkpoint(args.checkpoint)
    _validate_checkpoint(checkpoint, args)
    paper = PaperMetaSGConfig()
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
        local_search_batch_size=128,
        local_search_gradient_norm_cap=1.0,
        device=args.device,
        post_defense_mode=args.post_defense_mode,
        parallel_clients=args.parallel_clients,
        defender_alpha_floor_ratio=args.alpha_floor_ratio,
        defender_norm_reference=args.defender_norm_reference,
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
    meta_defender = _agent(paper, defender_dim, 'defender', 1, args.device)
    meta_defender.restore(
        checkpoint.algorithm1_defender
        if args.method == 'meta-sg'
        else checkpoint.algorithm2_defender
    )
    if args.method == 'meta-rl' and not args.attack_domain:
        raise ValueError('Meta-RL evaluation requires --attack-domain')
    attacker_snapshots = (
        load_attack_type_domain(args.attack_domain).snapshots
        if args.method == 'meta-rl'
        else checkpoint.algorithm1_attackers
    )
    attackers = {}
    for index, (label, snapshot) in enumerate(sorted(attacker_snapshots.items())):
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
    if args.scenarios:
        requested = tuple(dict.fromkeys(args.scenarios))
        available = {scenario.name: scenario for scenario in scenarios}
        unknown = tuple(name for name in requested if name not in available)
        if unknown:
            raise ValueError(
                f'unknown evaluation scenarios {unknown!r}; '
                f'available={tuple(available)!r}',
            )
        scenarios = tuple(available[name] for name in requested)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    online_directory = output / 'online_adaptation'
    if not args.skip_online_adaptation:
        online_directory.mkdir(parents=True, exist_ok=True)
    training_checkpoint_sha256 = _sha256(Path(args.checkpoint))
    online_config = paper.scaled_online(
        online_T=args.online_T,
        online_H=args.online_H,
        online_l=args.online_l,
        online_steps=args.online_steps,
        td3_batch_size=args.online_batch_size,
        learning_starts=args.online_learning_starts,
        replay_capacity=paper.replay_capacity,
        adaptation_step=args.online_adaptation_step,
    )
    records = {}
    online_records = {}
    for scenario_index, scenario in enumerate(scenarios):
        attacker = attackers[scenario.attacker_label]
        if args.skip_online_adaptation:
            defender = meta_defender
            online_record = None
        else:
            trajectories_per_update = max(
                1,
                int(np.ceil(
                    online_config.learning_starts / online_config.online_H
                )),
            )
            support_count = (
                trajectories_per_update + online_config.online_T - 1
            )
            support_start = args.online_support_seed + scenario_index * 10_000
            support_seeds = tuple(
                range(support_start, support_start + support_count),
            )

            def online_env_factory(task, rollout_seed, horizon, scenario=scenario):
                del task
                return factory.make(
                    seed=rollout_seed,
                    horizon=horizon,
                    task_id=f'online-adaptation-{scenario.name}',
                    malicious_ids=(
                        () if scenario.attack_family == 'clean'
                        else factory.malicious_ids
                    ),
                    attack_generator_factory=model_poisoning_attack_factory(
                        scenario, factory,
                    ),
                )

            online_checkpoint = online_directory / f'{scenario.name}.pt'
            online = PaperOnlineAdaptationTrainingRunner(
                config=online_config,
                env_factory=online_env_factory,
                defender_obs_dim=defender_dim,
                attacker_obs_dim=attacker_dim,
                support_seeds=support_seeds,
                protocol_signature={
                    'method': args.method,
                    'scenario': scenario.name,
                    'attack_family': scenario.attack_family,
                    'fixed_attack_scale': scenario.scale,
                    'training_checkpoint_sha256': training_checkpoint_sha256,
                    'partition_seed': args.partition_seed,
                    'model_seed': args.model_seed,
                    'device': str(args.device),
                },
                actor_logit_l2=args.online_actor_logit_l2,
                actor_logit_l2_mask=(
                    tuple(args.online_actor_logit_l2_mask)
                    if args.online_actor_logit_l2_mask is not None
                    else None
                ),
            ).run(
                task=scenario.name,
                meta_defender=meta_defender,
                attacker=attacker,
                checkpoint_path=online_checkpoint,
                resume=online_checkpoint.is_file(),
                checkpoint_interval=1,
            )
            candidate_defender = online.adaptation.adapted_defender
            selection = {
                'mode': args.online_selection,
                'accepted': True,
                'selected': 'adapted',
            }
            if args.online_selection == 'reward-guarded':
                selection_horizon = (
                    args.online_selection_horizon or args.online_H
                )
                selection_seeds = tuple(
                    args.online_selection_seed + scenario_index * 10_000 + i
                    for i in range(args.online_selection_repeats)
                )
                base_scores = tuple(
                    _mean_defender_reward(
                        policy=meta_defender,
                        attacker=attacker,
                        env=online_env_factory(
                            scenario.name, seed, selection_horizon,
                        ),
                    )
                    for seed in selection_seeds
                )
                adapted_scores = tuple(
                    _mean_defender_reward(
                        policy=candidate_defender,
                        attacker=attacker,
                        env=online_env_factory(
                            scenario.name, seed, selection_horizon,
                        ),
                    )
                    for seed in selection_seeds
                )
                selection = _reward_guard_decision(
                    base_score=float(np.mean(base_scores)),
                    adapted_score=float(np.mean(adapted_scores)),
                    margin=args.online_selection_margin,
                )
                selection.update({
                    'mode': 'reward-guarded',
                    'metric': 'mean_defender_reward',
                    'uses_test_dataset': False,
                    'validation_seeds': selection_seeds,
                    'validation_horizon': selection_horizon,
                    'base_scores': base_scores,
                    'adapted_scores': adapted_scores,
                })
            defender = (
                candidate_defender
                if selection['accepted'] else meta_defender
            )
            online_record = {
                'protocol': 'paper-online-adaptation-scenario-v1',
                'method': args.method,
                'scenario': scenario.name,
                'attack_family': scenario.attack_family,
                'fixed_attack_scale': scenario.scale,
                'checkpoint': str(online_checkpoint.resolve()),
                'training_checkpoint_sha256': training_checkpoint_sha256,
                'online_T': online_config.online_T,
                'online_H': online_config.online_H,
                'online_l': online_config.online_l,
                'online_steps': online_config.online_steps,
                'adaptation_step': online_config.adaptation_step,
                'actor_logit_l2': args.online_actor_logit_l2,
                'actor_logit_l2_mask': args.online_actor_logit_l2_mask,
                'completed_iterations': len(online.adaptation.iterations),
                'trajectory_count': online.trajectory_count,
                'fl_round_count': online.fl_round_count,
                'trajectories_per_update': online.trajectories_per_update,
                'trajectory_collection': (
                    'one-fresh-trajectory-per-outer-iteration;'
                    'sb3-uniform-random-learning-starts-replacement-v3'
                ),
                'warmup_trajectories': online.trajectories_per_update,
                'fresh_trajectories_per_later_outer_iteration': 1,
                'support_seed_start': support_seeds[0],
                'support_seed_stop_exclusive': support_seeds[-1] + 1,
                'support_seed_count': len(support_seeds),
                'meta_defender_fingerprint': (
                    online.adaptation.initial_defender_fingerprint
                ),
                'adapted_defender_fingerprint': (
                    online.adaptation.adapted_defender_fingerprint
                ),
                'selected_defender_fingerprint': defender.fingerprint(),
                'selection': selection,
                'attacker_fingerprint': online.adaptation.attacker_fingerprint,
            }
            _atomic_json(
                online_directory / f'{scenario.name}.json', online_record,
            )
            online_records[scenario.name] = online_record
        path = output / f'{scenario.name}.json'
        if path.exists():
            record = json.loads(path.read_text(encoding='utf-8'))
            if _record_matches(
                record,
                scenario,
                args.seed,
                args.H,
                defender.fingerprint(),
                attacker.fingerprint(),
                args.method,
                online_record,
            ):
                records[scenario.name] = record
                continue
        record = evaluate_model_poisoning_scenario(
            scenario=scenario,
            factory=factory,
            test_dataset=datasets.test,
            defender=defender,
            attacker=attacker,
            seed=args.seed,
            horizon=args.H,
            device=args.device,
        )
        record['method'] = args.method
        record['pretraining_domain_member'] = _pretraining_domain_member(
            scenario,
            method=args.method,
            training_domain=tuple(
                checkpoint.config_signature.get(
                    'protocol_signature', {}
                ).get('fixed_attack_domain', ()),
            ),
        )
        record['online_adaptation'] = online_record
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
        allow_partial=bool(args.scenarios),
    )
    summary['training_configuration'] = dict(checkpoint.config_signature)
    summary['method'] = args.method
    summary['policy_stage'] = (
        'pre-online-adaptation'
        if args.skip_online_adaptation else 'post-online-adaptation'
    )
    summary['training_checkpoint_sha256'] = training_checkpoint_sha256
    summary['online_adaptation'] = online_records
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
    expected = {
        'T': args.T,
        'K': args.K,
        'H': getattr(args, 'training_H', None) or args.H,
        'workers': args.workers,
        'untargeted_attackers': args.untargeted_attackers,
        'sample_size': args.sample_size,
    }
    observed = dict(checkpoint.config_signature)
    if any(observed.get(key) != value for key, value in expected.items()):
        raise ValueError(
            'training checkpoint scale/topology does not match evaluation'
        )
    protocol_signature = dict(observed.get('protocol_signature', {}))
    expected_protocol = {
        'local_training_budget': (
            'one-minibatch-step-per-local-iteration'
        ),
        'attack_client_sampling': (
            'uniform-conditioned-on-at-least-one-benign-reference'
        ),
        'post_defense_mode': args.post_defense_mode,
        'mnist_input_pipeline': (
            'materialized-tensor-dataset'
            if args.materialize_mnist
            else 'torchvision-transform-per-sample'
        ),
        'mnist_normalization': getattr(args, 'mnist_normalization', 'none'),
        'mnist_partition': getattr(args, 'data_split', 'paper-q'),
        'non_iid_q': (
            PaperMetaSGConfig().non_iid_q
            if getattr(args, 'data_split', 'paper-q') == 'paper-q'
            else None
        ),
        'defender_alpha_floor_ratio': getattr(
            args, 'alpha_floor_ratio', 0.0,
        ),
        'defender_norm_reference': getattr(
            args, 'defender_norm_reference', 'max',
        ),
        'local_model_workspace': 'reused-and-reset-per-client-update',
        'parallel_clients': min(args.parallel_clients, args.sample_size),
    }
    backward_compatible_optional = {
        'mnist_normalization', 'mnist_partition', 'non_iid_q',
        'defender_alpha_floor_ratio',
        'defender_norm_reference',
    }
    if any(
        protocol_signature.get(key) != value
        for key, value in expected_protocol.items()
        if key not in backward_compatible_optional or key in protocol_signature
    ):
        raise ValueError(
            'training checkpoint local/sampling protocol does not match '
            'evaluation'
        )
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
    method, expected_online_adaptation,
):
    recorded_online_adaptation = record.get('online_adaptation')
    if expected_online_adaptation is None:
        online_adaptation_matches = recorded_online_adaptation is None
    else:
        expected_selected = expected_online_adaptation.get(
            'selected_defender_fingerprint',
            expected_online_adaptation.get('adapted_defender_fingerprint'),
        )
        online_adaptation_matches = (
            isinstance(recorded_online_adaptation, dict)
            and recorded_online_adaptation.get(
                'selected_defender_fingerprint',
                recorded_online_adaptation.get(
                    'adapted_defender_fingerprint'
                ),
            ) == defender_fingerprint == expected_selected
            and (
                'selected_defender_fingerprint'
                not in expected_online_adaptation
                or recorded_online_adaptation.get(
                    'selected_defender_fingerprint'
                ) == expected_selected
            )
        )
    return (
        record.get('protocol') == 'canonical-model-poisoning-final-evaluation-v1'
        and record.get('scenario') == scenario.name
        and record.get('attack_family') == scenario.attack_family
        and record.get('fixed_attack_scale') == scenario.scale
        and record.get('seed') == seed
        and record.get('horizon') == horizon
        and record.get('defender_fingerprint') == defender_fingerprint
        and record.get('attacker_fingerprint') == attacker_fingerprint
        and record.get('method') == method
        and online_adaptation_matches
        and len(record.get('round_metrics', ())) == horizon
    )


def _reward_guard_decision(
    *, base_score: float, adapted_score: float, margin: float,
) -> dict[str, object]:
    values = (base_score, adapted_score, margin)
    if not all(np.isfinite(value) for value in values) or margin < 0:
        raise ValueError('reward guard values must be finite and margin non-negative')
    gain = float(adapted_score) - float(base_score)
    accepted = gain > float(margin)
    return {
        'accepted': accepted,
        'selected': 'adapted' if accepted else 'frozen',
        'base_score': float(base_score),
        'adapted_score': float(adapted_score),
        'score_gain': gain,
        'margin': float(margin),
    }


def _mean_defender_reward(*, policy, attacker, env) -> float:
    policy_guard = policy.freeze_guard()
    attacker_guard = attacker.freeze_guard()
    rewards = []
    while env.state.round_index < env.horizon:
        defender_observation = flatten_observation(
            env.defender_observation(), DEFENDER_OBSERVATION_KEYS,
        )
        defender_action = policy.act(
            defender_observation, deterministic=True,
        )
        pending = env.begin_round(defender_action)
        attacker_observation = flatten_observation(
            pending.attacker_observation, ATTACKER_OBSERVATION_KEYS,
        )
        attacker_action = attacker.act(
            attacker_observation, deterministic=True,
        )
        step = env.finish_round(attacker_action)
        rewards.append(float(step.defender_reward.scalar))
    policy_guard.verify()
    attacker_guard.verify()
    if not rewards or not np.all(np.isfinite(rewards)):
        raise RuntimeError('reward guard produced invalid defender rewards')
    return float(np.mean(rewards))


def _pretraining_domain_member(scenario, *, method, training_domain) -> bool:
    if method == 'meta-sg':
        return scenario.attack_family == 'rl'
    task = {
        'clean': 'na',
        'ipm': 'ipm',
        'lmp': 'lmp',
        'bfl': 'bfl',
        'dba': 'dba',
    }.get(scenario.attack_family, scenario.attacker_label)
    return task in set(training_domain)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


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
