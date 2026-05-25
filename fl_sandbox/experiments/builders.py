"""Factories and naming helpers for sandbox experiment runs."""

from __future__ import annotations

from fl_sandbox.config.schema import AttackerSection, DataSection, DefenderSection, FLSection
from fl_sandbox.attacks import ATTACK_CHOICES


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
