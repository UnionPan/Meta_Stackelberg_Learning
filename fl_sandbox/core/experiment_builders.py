"""Factories and naming helpers for sandbox experiment runs."""

from __future__ import annotations

from fl_sandbox.config.schema import AttackerSection, DataSection, DefenderSection, FLSection, RunConfig
from fl_sandbox.attacks import ATTACK_CHOICES
from fl_sandbox.defenders import build_defender_config_kwargs


def build_run_name(
    *,
    dataset: str,
    attack_type: str,
    defense_type: str,
    split_mode: str,
    noniid_q: float,
    rounds: int,
) -> str:
    suffix = split_suffix(split_mode, noniid_q)
    return f"{dataset}_{attack_type}_{defense_type}_{suffix}_{rounds}r"


def resolve_num_attackers(attacker: AttackerSection, fl: FLSection) -> int:
    if fl.num_attackers is not None:
        return fl.num_attackers
    return 0 if attacker.type == "clean" else 2


def split_suffix(split_mode: str, noniid_q: float) -> str:
    if split_mode == "iid":
        return "iid"
    return f"{split_mode}_q{noniid_q:g}"


def default_output_dir(attacker: AttackerSection, defender: DefenderSection, data: DataSection) -> str:
    suffix = split_suffix(data.split_mode, data.noniid_q)
    if attacker.type == "clean":
        return f"fl_sandbox/outputs/clean_{defender.type}_{suffix}_benchmark"
    return f"fl_sandbox/outputs/{attacker.type}_{defender.type}_{suffix}_demo"


def default_tb_dir(attacker: AttackerSection, defender: DefenderSection, data: DataSection) -> str:
    suffix = split_suffix(data.split_mode, data.noniid_q)
    if attacker.type == "clean":
        return f"fl_sandbox/runs/clean_{defender.type}_{suffix}_benchmark"
    return f"fl_sandbox/runs/{attacker.type}_{defender.type}_{suffix}_demo"


def build_attack(attacker: AttackerSection):
    from fl_sandbox.attacks import create_attack

    return create_attack(attacker)


def build_config(run_config: RunConfig):
    from fl_sandbox.core.fl_runner import SandboxConfig

    return SandboxConfig(
        dataset=run_config.data.dataset,
        data_dir="data",
        device=run_config.runtime.device,
        seed=run_config.runtime.seed,
        init_mode=run_config.init.init_mode,
        init_checkpoint_path=run_config.init.init_checkpoint_path,
        num_clients=run_config.fl.num_clients,
        num_attackers=resolve_num_attackers(run_config.attacker, run_config.fl),
        subsample_rate=run_config.fl.subsample_rate,
        local_epochs=run_config.fl.local_epochs,
        lr=run_config.runtime.lr,
        batch_size=run_config.runtime.batch_size,
        eval_batch_size=run_config.runtime.eval_batch_size,
        max_client_samples_per_client=run_config.runtime.max_client_samples_per_client,
        max_eval_samples=run_config.runtime.max_eval_samples,
        num_workers=run_config.runtime.num_workers,
        parallel_clients=run_config.runtime.parallel_clients,
        base_class=run_config.attacker.base_class,
        target_class=run_config.attacker.target_class,
        pattern_type=run_config.attacker.pattern_type,
        ipm_scaling=run_config.attacker.ipm_scaling,
        lmp_scale=run_config.attacker.lmp_scale,
        bfl_poison_frac=run_config.attacker.bfl_poison_frac,
        dba_poison_frac=run_config.attacker.dba_poison_frac,
        dba_num_sub_triggers=run_config.attacker.dba_num_sub_triggers,
        attacker_action=tuple(run_config.attacker.attacker_action),
        **build_defender_config_kwargs(run_config.defender),
        split_mode=run_config.data.split_mode,
        noniid_q=run_config.data.noniid_q,
        rl_distribution_steps=run_config.attacker.rl_distribution_steps,
        rl_attack_start_round=run_config.attacker.rl_attack_start_round,
        rl_policy_train_end_round=run_config.attacker.rl_policy_train_end_round,
        rl_inversion_steps=run_config.attacker.rl_inversion_steps,
        rl_reconstruction_batch_size=run_config.attacker.rl_reconstruction_batch_size,
        rl_policy_train_episodes_per_round=run_config.attacker.rl_policy_train_episodes_per_round,
        rl_simulator_horizon=run_config.attacker.rl_simulator_horizon,
        rl_ppo_real_rollout_steps=run_config.attacker.rl_ppo_real_rollout_steps,
        rl_attacker_semantics=run_config.attacker.rl_attacker_semantics,
    )
