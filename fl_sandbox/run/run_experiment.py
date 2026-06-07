"""Unified single-run experiment entry point for fl_sandbox."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fl_sandbox.config import PROTOCOL_CHOICES, RunConfig, load_run_config, merge_cli_overrides
from fl_sandbox.attacks import ATTACK_CHOICES
from fl_sandbox.defenders import DEFENSE_CHOICES
from fl_sandbox.experiments.builders import build_run_name
from fl_sandbox.experiments.service import completion_lines, execute_experiment, persist_experiment_artifacts


DATASET_CHOICES = ('mnist', 'fmnist', 'cifar10')
SPLIT_MODE_CHOICES = ('iid', 'noniid', 'paper_q')


def _default_kwargs(default, use_defaults: bool) -> dict[str, object]:
    return {'default': default if use_defaults else argparse.SUPPRESS}


def _build_parser(
    *,
    description: str,
    use_defaults: bool,
) -> argparse.ArgumentParser:
    defaults = RunConfig()
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', type=str, default='' if use_defaults else argparse.SUPPRESS)
    parser.add_argument(
        '--attack_type',
        type=str,
        choices=list(ATTACK_CHOICES),
        **_default_kwargs(defaults.attacker.type, use_defaults),
    )
    parser.add_argument(
        '--defense_type',
        type=str,
        choices=list(DEFENSE_CHOICES),
        **_default_kwargs(defaults.defender.type, use_defaults),
    )
    parser.add_argument(
        '--protocol',
        type=str,
        choices=PROTOCOL_CHOICES,
        **_default_kwargs(defaults.protocol.name, use_defaults),
    )
    parser.add_argument(
        '--warmup_rounds',
        type=int,
        **_default_kwargs(defaults.protocol.warmup_rounds, use_defaults),
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=list(DATASET_CHOICES),
        **_default_kwargs(defaults.data.dataset, use_defaults),
    )
    parser.add_argument(
        '--split_mode',
        type=str,
        choices=list(SPLIT_MODE_CHOICES),
        **_default_kwargs(defaults.data.split_mode, use_defaults),
    )
    parser.add_argument('--noniid_q', type=float, **_default_kwargs(defaults.data.noniid_q, use_defaults))
    parser.add_argument('--rounds', type=int, **_default_kwargs(defaults.runtime.rounds, use_defaults))
    parser.add_argument('--start_round_idx', type=int, **_default_kwargs(defaults.runtime.start_round_idx, use_defaults))
    parser.add_argument('--device', type=str, **_default_kwargs(defaults.runtime.device, use_defaults))
    parser.add_argument('--num_clients', type=int, **_default_kwargs(defaults.fl.num_clients, use_defaults))
    parser.add_argument('--num_attackers', type=int, **_default_kwargs(defaults.fl.num_attackers, use_defaults))
    parser.add_argument('--subsample_rate', type=float, **_default_kwargs(defaults.fl.subsample_rate, use_defaults))
    parser.add_argument('--local_epochs', type=int, **_default_kwargs(defaults.fl.local_epochs, use_defaults))
    parser.add_argument('--lr', type=float, **_default_kwargs(defaults.runtime.lr, use_defaults))
    parser.add_argument('--batch_size', type=int, **_default_kwargs(defaults.runtime.batch_size, use_defaults))
    parser.add_argument('--eval_batch_size', type=int, **_default_kwargs(defaults.runtime.eval_batch_size, use_defaults))
    parser.add_argument(
        '--max_client_samples_per_client',
        type=int,
        **_default_kwargs(defaults.runtime.max_client_samples_per_client, use_defaults),
    )
    parser.add_argument(
        '--max_eval_samples',
        type=int,
        **_default_kwargs(defaults.runtime.max_eval_samples, use_defaults),
    )
    parser.add_argument('--num_workers', type=int, **_default_kwargs(defaults.runtime.num_workers, use_defaults))
    parser.add_argument(
        '--parallel_clients',
        type=int,
        **_default_kwargs(defaults.runtime.parallel_clients, use_defaults),
    )
    parser.add_argument('--eval_every', type=int, **_default_kwargs(defaults.runtime.eval_every, use_defaults))
    parser.add_argument('--seed', type=int, **_default_kwargs(defaults.runtime.seed, use_defaults))
    parser.add_argument(
        '--init_mode',
        type=str,
        choices=['seed', 'checkpoint'],
        **_default_kwargs(defaults.init.init_mode, use_defaults),
    )
    parser.add_argument(
        '--init_checkpoint_path',
        type=str,
        **_default_kwargs(defaults.init.init_checkpoint_path, use_defaults),
    )
    parser.add_argument('--ipm_scaling', type=float, **_default_kwargs(defaults.attacker.ipm_scaling, use_defaults))
    parser.add_argument('--lmp_scale', type=float, **_default_kwargs(defaults.attacker.lmp_scale, use_defaults))
    parser.add_argument('--alie_tau', type=float, **_default_kwargs(defaults.attacker.alie_tau, use_defaults))
    parser.add_argument('--gaussian_sigma', type=float, **_default_kwargs(defaults.attacker.gaussian_sigma, use_defaults))
    parser.add_argument('--base_class', type=int, **_default_kwargs(defaults.attacker.base_class, use_defaults))
    parser.add_argument('--target_class', type=int, **_default_kwargs(defaults.attacker.target_class, use_defaults))
    parser.add_argument('--pattern_type', type=str, **_default_kwargs(defaults.attacker.pattern_type, use_defaults))
    parser.add_argument('--bfl_poison_frac', type=float, **_default_kwargs(defaults.attacker.bfl_poison_frac, use_defaults))
    parser.add_argument('--dba_poison_frac', type=float, **_default_kwargs(defaults.attacker.dba_poison_frac, use_defaults))
    parser.add_argument(
        '--dba_num_sub_triggers',
        type=int,
        **_default_kwargs(defaults.attacker.dba_num_sub_triggers, use_defaults),
    )
    parser.add_argument(
        '--attacker_action',
        type=float,
        nargs=3,
        **_default_kwargs(defaults.attacker.attacker_action, use_defaults),
    )
    parser.add_argument(
        '--rl_algorithm',
        choices=('td3', 'ppo'),
        **_default_kwargs(defaults.attacker.rl_algorithm, use_defaults),
    )
    parser.add_argument(
        '--rl_attacker_semantics',
        choices=('paper_clipped_median',),
        **_default_kwargs(defaults.attacker.rl_attacker_semantics, use_defaults),
    )
    parser.add_argument('--rl_policy_lr', type=float, **_default_kwargs(defaults.attacker.rl_policy_lr, use_defaults))
    parser.add_argument('--rl_critic_lr', type=float, **_default_kwargs(defaults.attacker.rl_critic_lr, use_defaults))
    parser.add_argument('--rl_gamma', type=float, **_default_kwargs(defaults.attacker.rl_gamma, use_defaults))
    parser.add_argument(
        '--rl_replay_capacity',
        type=int,
        **_default_kwargs(defaults.attacker.rl_replay_capacity, use_defaults),
    )
    parser.add_argument('--rl_batch_size', type=int, **_default_kwargs(defaults.attacker.rl_batch_size, use_defaults))
    parser.add_argument(
        '--rl_hidden_sizes',
        type=int,
        nargs='+',
        **_default_kwargs(defaults.attacker.rl_hidden_sizes, use_defaults),
    )
    parser.add_argument(
        '--rl_exploration_noise',
        type=float,
        **_default_kwargs(defaults.attacker.rl_exploration_noise, use_defaults),
    )
    parser.add_argument(
        '--rl_train_freq_steps',
        type=int,
        **_default_kwargs(defaults.attacker.rl_train_freq_steps, use_defaults),
    )
    parser.add_argument(
        '--rl_reward_transform',
        choices=('raw', 'tanh_delta'),
        **_default_kwargs(defaults.attacker.rl_reward_transform, use_defaults),
    )
    parser.add_argument(
        '--rl_reward_scale',
        type=float,
        **_default_kwargs(defaults.attacker.rl_reward_scale, use_defaults),
    )
    parser.add_argument(
        '--rl_policy_train_steps_per_round',
        type=int,
        **_default_kwargs(defaults.attacker.rl_policy_train_steps_per_round, use_defaults),
    )
    parser.add_argument(
        '--rl_policy_checkpoint_path',
        type=str,
        **_default_kwargs(defaults.attacker.rl_policy_checkpoint_path, use_defaults),
    )
    parser.add_argument(
        '--rl_policy_checkpoint_dir',
        type=str,
        **_default_kwargs(defaults.attacker.rl_policy_checkpoint_dir, use_defaults),
    )
    parser.add_argument(
        '--rl_freeze_policy',
        action=argparse.BooleanOptionalAction,
        **_default_kwargs(defaults.attacker.rl_freeze_policy, use_defaults),
    )
    parser.add_argument(
        '--distribution_dir',
        '--rl_distribution_dir',
        dest='rl_distribution_dir',
        type=str,
        **_default_kwargs(defaults.attacker.rl_distribution_dir, use_defaults),
    )
    parser.add_argument(
        '--distribution_split',
        '--rl_distribution_split',
        dest='rl_distribution_split',
        type=str,
        **_default_kwargs(defaults.attacker.rl_distribution_split, use_defaults),
    )
    parser.add_argument(
        '--rl_policy_warmup_steps',
        type=int,
        **_default_kwargs(defaults.attacker.rl_policy_warmup_steps, use_defaults),
    )
    parser.add_argument(
        '--rl_policy_warmup_random_steps',
        type=int,
        **_default_kwargs(defaults.attacker.rl_policy_warmup_random_steps, use_defaults),
    )
    parser.add_argument(
        '--rl_policy_warmup_checkpoint_interval',
        type=int,
        **_default_kwargs(defaults.attacker.rl_policy_warmup_checkpoint_interval, use_defaults),
    )
    parser.add_argument(
        '--rl_policy_warmup_checkpoint_dir',
        type=str,
        **_default_kwargs(defaults.attacker.rl_policy_warmup_checkpoint_dir, use_defaults),
    )
    parser.add_argument(
        '--rl_distribution_growth_mode',
        choices=('paper_growth', 'full'),
        **_default_kwargs(defaults.attacker.rl_distribution_growth_mode, use_defaults),
    )
    parser.add_argument(
        '--rl_backdoor_reward_mode',
        choices=('paper', 'henger_li', 'delta', 'asr_delta', 'poi_delta', 'stealth'),
        **_default_kwargs(defaults.attacker.rl_backdoor_reward_mode, use_defaults),
    )
    parser.add_argument(
        '--rl_backdoor_reward_clean_lambda',
        type=float,
        **_default_kwargs(defaults.attacker.rl_backdoor_reward_clean_lambda, use_defaults),
    )
    parser.add_argument(
        '--rl_backdoor_stealth_norm_cap',
        action=argparse.BooleanOptionalAction,
        **_default_kwargs(defaults.attacker.rl_backdoor_stealth_norm_cap, use_defaults),
    )
    parser.add_argument(
        '--rl_backdoor_reward_norm_lambda',
        type=float,
        **_default_kwargs(defaults.attacker.rl_backdoor_reward_norm_lambda, use_defaults),
    )
    parser.add_argument(
        '--rl_backdoor_freeze_boost',
        type=float,
        **_default_kwargs(defaults.attacker.rl_backdoor_freeze_boost, use_defaults),
    )
    parser.add_argument(
        '--rl_backdoor_warmup_fixed_rollouts',
        type=int,
        **_default_kwargs(defaults.attacker.rl_backdoor_warmup_fixed_rollouts, use_defaults),
    )
    parser.add_argument(
        '--rl_backdoor_simulator_shadow_clients',
        type=int,
        **_default_kwargs(defaults.attacker.rl_backdoor_simulator_shadow_clients, use_defaults),
    )
    parser.add_argument(
        '--rl_backdoor_simulator_shadow_samples_per_client',
        type=int,
        **_default_kwargs(defaults.attacker.rl_backdoor_simulator_shadow_samples_per_client, use_defaults),
    )
    parser.add_argument(
        '--rl_checkpoint_interval',
        type=int,
        **_default_kwargs(defaults.attacker.rl_checkpoint_interval, use_defaults),
    )
    parser.add_argument(
        '--rl_save_final_checkpoint',
        action=argparse.BooleanOptionalAction,
        **_default_kwargs(defaults.attacker.rl_save_final_checkpoint, use_defaults),
    )
    parser.add_argument(
        '--rl_strict_reproduction_initial_samples',
        type=int,
        **_default_kwargs(defaults.attacker.rl_strict_reproduction_initial_samples, use_defaults),
    )
    parser.add_argument(
        '--rl_strict_reproduction_samples_per_epoch',
        type=int,
        **_default_kwargs(defaults.attacker.rl_strict_reproduction_samples_per_epoch, use_defaults),
    )
    parser.add_argument('--krum_attackers', type=int, **_default_kwargs(defaults.defender.krum_attackers, use_defaults))
    parser.add_argument(
        '--multi_krum_selected',
        type=int,
        **_default_kwargs(defaults.defender.multi_krum_selected, use_defaults),
    )
    parser.add_argument(
        '--clipped_median_norm',
        type=float,
        **_default_kwargs(defaults.defender.clipped_median_norm, use_defaults),
    )
    parser.add_argument(
        '--trimmed_mean_ratio',
        type=float,
        **_default_kwargs(defaults.defender.trimmed_mean_ratio, use_defaults),
    )
    parser.add_argument(
        '--geometric_median_iters',
        type=int,
        **_default_kwargs(defaults.defender.geometric_median_iters, use_defaults),
    )
    parser.add_argument(
        '--fltrust_root_size',
        type=int,
        **_default_kwargs(defaults.defender.fltrust_root_size, use_defaults),
    )
    parser.add_argument(
        '--rl_distribution_steps',
        type=int,
        **_default_kwargs(defaults.attacker.rl_distribution_steps, use_defaults),
    )
    parser.add_argument(
        '--rl_attack_start_round',
        type=int,
        **_default_kwargs(defaults.attacker.rl_attack_start_round, use_defaults),
    )
    parser.add_argument(
        '--rl_policy_train_end_round',
        type=int,
        **_default_kwargs(defaults.attacker.rl_policy_train_end_round, use_defaults),
    )
    parser.add_argument(
        '--rl_inversion_steps',
        type=int,
        **_default_kwargs(defaults.attacker.rl_inversion_steps, use_defaults),
    )
    parser.add_argument(
        '--rl_reconstruction_batch_size',
        type=int,
        **_default_kwargs(defaults.attacker.rl_reconstruction_batch_size, use_defaults),
    )
    parser.add_argument(
        '--rl_policy_train_episodes_per_round',
        type=int,
        **_default_kwargs(defaults.attacker.rl_policy_train_episodes_per_round, use_defaults),
    )
    parser.add_argument(
        '--rl_simulator_horizon',
        type=int,
        **_default_kwargs(defaults.attacker.rl_simulator_horizon, use_defaults),
    )
    parser.add_argument(
        '--rl_ppo_real_rollout_steps',
        type=int,
        **_default_kwargs(defaults.attacker.rl_ppo_real_rollout_steps, use_defaults),
    )
    parser.add_argument('--output_root', type=str, **_default_kwargs(defaults.output.output_root, use_defaults))
    parser.add_argument('--tb_root', type=str, **_default_kwargs(defaults.output.tb_root, use_defaults))
    return parser


def parse_args(
    argv: Optional[list[str]] = None,
    *,
    description: str = 'Run one configured fl_sandbox experiment',
) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', type=str, default='')
    pre_args, _ = pre_parser.parse_known_args(argv)
    parser = _build_parser(description=description, use_defaults=not bool(pre_args.config))
    args = parser.parse_args(argv)
    if not hasattr(args, 'config'):
        args.config = pre_args.config
    return args


def _prepare_run_config(args: argparse.Namespace) -> tuple[RunConfig, str, str, str]:
    base_config = load_run_config(args.config or None)
    run_config = merge_cli_overrides(base_config, args)
    run_name = build_run_name(
        dataset=run_config.data.dataset,
        attack_type=run_config.attacker.type,
        defense_type=run_config.defender.type,
        split_mode=run_config.data.split_mode,
        noniid_q=run_config.data.noniid_q,
        rounds=run_config.runtime.rounds,
    )
    output_dir = str(Path(run_config.output.output_root) / run_name)
    tb_dir = str(Path(run_config.output.tb_root) / run_name)
    return run_config, run_name, output_dir, tb_dir


def main(
    argv: Optional[list[str]] = None,
    *,
    description: str = 'Run one configured fl_sandbox experiment',
) -> None:
    args = parse_args(argv, description=description)
    run_config, run_name, output_dir, tb_dir = _prepare_run_config(args)
    result = execute_experiment(run_config, progress_desc=run_name, output_dir=output_dir, tb_dir=tb_dir)
    persist_experiment_artifacts(
        result,
        write_client_metrics=True,
        write_tensorboard=False,
        write_round_metrics=True,
    )

    for line in completion_lines(result):
        print(line)
    if run_config.protocol.name != 'none':
        print(f'Protocol: {run_config.protocol.name}')


if __name__ == '__main__':
    main()
