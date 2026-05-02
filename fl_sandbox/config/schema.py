"""Structured configuration schema for attacker_sandbox runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


PROTOCOL_CHOICES = ('none', 'rlfl')


@dataclass
class FLSection:
    num_clients: int = 20
    num_attackers: int | None = None
    subsample_rate: float = 0.25
    local_epochs: int = 1


@dataclass
class DataSection:
    dataset: str = "mnist"
    split_mode: str = "iid"
    noniid_q: float = 0.5


@dataclass
class ProtocolSection:
    name: str = "none"
    warmup_rounds: int = 0


@dataclass
class RuntimeSection:
    rounds: int = 30
    device: str = "auto"
    lr: float = 0.05
    batch_size: int = 256
    eval_batch_size: int = 1024
    max_client_samples_per_client: int | None = None
    max_eval_samples: int | None = None
    num_workers: int = 0
    parallel_clients: int = 1
    eval_every: int = 1
    seed: int = 42


@dataclass
class InitSection:
    init_mode: str = "seed"
    init_checkpoint_path: str = ""


@dataclass
class AttackerSection:
    type: str = "clean"
    ipm_scaling: float = 2.0
    lmp_scale: float = 5.0
    alie_tau: float = 1.5
    gaussian_sigma: float = 0.01
    base_class: int = 1
    target_class: int = 7
    pattern_type: str = "square"
    bfl_poison_frac: float = 1.0
    dba_poison_frac: float = 0.5
    dba_num_sub_triggers: int = 4
    attacker_action: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rl_distribution_steps: int | None = 10
    rl_attack_start_round: int | None = 10
    rl_policy_train_end_round: int | None = 30
    rl_inversion_steps: int = 50
    rl_reconstruction_batch_size: int = 8
    rl_policy_train_episodes_per_round: int = 1
    rl_simulator_horizon: int = 8


@dataclass
class DefenderSection:
    type: str = "fedavg"
    krum_attackers: int = 1
    multi_krum_selected: int | None = None
    clipped_median_norm: float = 2.0
    trimmed_mean_ratio: float = 0.2
    geometric_median_iters: int = 10
    fltrust_root_size: int = 100


@dataclass
class OutputSection:
    output_root: str = "fl_sandbox/outputs/paper_suite"
    tb_root: str = "fl_sandbox/runs/paper_suite"


@dataclass
class RunConfig:
    data: DataSection = field(default_factory=DataSection)
    fl: FLSection = field(default_factory=FLSection)
    protocol: ProtocolSection = field(default_factory=ProtocolSection)
    runtime: RuntimeSection = field(default_factory=RuntimeSection)
    init: InitSection = field(default_factory=InitSection)
    attacker: AttackerSection = field(default_factory=AttackerSection)
    defender: DefenderSection = field(default_factory=DefenderSection)
    output: OutputSection = field(default_factory=OutputSection)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> 'RunConfig':
        config = cls()
        if not payload:
            return config.normalize()

        _apply_section(config.data, payload.get("data", {}))
        _apply_section(config.fl, payload.get("fl", {}))
        _apply_section(config.protocol, payload.get("protocol", {}))
        _apply_section(config.runtime, payload.get("runtime", {}))
        _apply_section(config.init, payload.get("init", {}))
        _apply_section(config.attacker, payload.get("attacker", {}))
        _apply_section(config.defender, payload.get("defender", {}))
        _apply_section(config.output, payload.get("output", {}))
        return config.normalize()

    @classmethod
    def from_flat_dict(cls, values: Mapping[str, Any]) -> 'RunConfig':
        config = cls()
        for key, value in values.items():
            _set_flat_value(config, key, value)
        return config.normalize()

    def normalize(self) -> 'RunConfig':
        # Sandbox experiments should always emit per-round metrics so TensorBoard
        # and paper-aligned analysis stay comparable across runs.
        self.runtime.eval_every = 1

        if self.protocol.name != 'rlfl':
            return self

        warmup_rounds = int(self.protocol.warmup_rounds or 0)
        if self.attacker.type == 'rl':
            self.attacker.rl_distribution_steps = self.attacker.rl_distribution_steps or warmup_rounds
            self.attacker.rl_policy_train_end_round = self.attacker.rl_policy_train_end_round or warmup_rounds
            self.attacker.rl_attack_start_round = self.attacker.rl_attack_start_round or (warmup_rounds + 1)
            return self

        self.attacker.rl_distribution_steps = self.attacker.rl_distribution_steps or 10
        self.attacker.rl_policy_train_end_round = self.attacker.rl_policy_train_end_round or 30
        self.attacker.rl_attack_start_round = self.attacker.rl_attack_start_round or 10
        return self

    def benchmark_protocol_payload(self) -> dict[str, object] | None:
        if self.protocol.name != 'rlfl':
            return None

        return {
            'name': 'paper_aligned_accuracy_asr',
            'warmup_rounds': int(self.protocol.warmup_rounds or 0),
            'rl_distribution_steps': self.attacker.rl_distribution_steps,
            'rl_policy_train_end_round': self.attacker.rl_policy_train_end_round,
            'rl_attack_start_round': self.attacker.rl_attack_start_round,
        }

    def to_flat_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.data.dataset,
            "attack_type": self.attacker.type,
            "defense_type": self.defender.type,
            "protocol": self.protocol.name,
            "split_mode": self.data.split_mode,
            "noniid_q": self.data.noniid_q,
            "warmup_rounds": self.protocol.warmup_rounds,
            "rounds": self.runtime.rounds,
            "device": self.runtime.device,
            "num_clients": self.fl.num_clients,
            "num_attackers": self.fl.num_attackers,
            "subsample_rate": self.fl.subsample_rate,
            "local_epochs": self.fl.local_epochs,
            "lr": self.runtime.lr,
            "batch_size": self.runtime.batch_size,
            "eval_batch_size": self.runtime.eval_batch_size,
            "max_client_samples_per_client": self.runtime.max_client_samples_per_client,
            "max_eval_samples": self.runtime.max_eval_samples,
            "num_workers": self.runtime.num_workers,
            "parallel_clients": self.runtime.parallel_clients,
            "eval_every": self.runtime.eval_every,
            "seed": self.runtime.seed,
            "init_mode": self.init.init_mode,
            "init_checkpoint_path": self.init.init_checkpoint_path,
            "ipm_scaling": self.attacker.ipm_scaling,
            "lmp_scale": self.attacker.lmp_scale,
            "alie_tau": self.attacker.alie_tau,
            "gaussian_sigma": self.attacker.gaussian_sigma,
            "base_class": self.attacker.base_class,
            "target_class": self.attacker.target_class,
            "pattern_type": self.attacker.pattern_type,
            "bfl_poison_frac": self.attacker.bfl_poison_frac,
            "dba_poison_frac": self.attacker.dba_poison_frac,
            "dba_num_sub_triggers": self.attacker.dba_num_sub_triggers,
            "attacker_action": list(self.attacker.attacker_action),
            "krum_attackers": self.defender.krum_attackers,
            "multi_krum_selected": self.defender.multi_krum_selected,
            "clipped_median_norm": self.defender.clipped_median_norm,
            "trimmed_mean_ratio": self.defender.trimmed_mean_ratio,
            "geometric_median_iters": self.defender.geometric_median_iters,
            "fltrust_root_size": self.defender.fltrust_root_size,
            "rl_distribution_steps": self.attacker.rl_distribution_steps,
            "rl_attack_start_round": self.attacker.rl_attack_start_round,
            "rl_policy_train_end_round": self.attacker.rl_policy_train_end_round,
            "rl_inversion_steps": self.attacker.rl_inversion_steps,
            "rl_reconstruction_batch_size": self.attacker.rl_reconstruction_batch_size,
            "rl_policy_train_episodes_per_round": self.attacker.rl_policy_train_episodes_per_round,
            "rl_simulator_horizon": self.attacker.rl_simulator_horizon,
            "output_root": self.output.output_root,
            "tb_root": self.output.tb_root,
        }


def _apply_section(section: Any, values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        if not hasattr(section, key):
            raise ValueError(f"Unknown config field: {type(section).__name__}.{key}")
        if key == "attacker_action":
            value = tuple(value)
        setattr(section, key, value)


def _set_flat_value(config: RunConfig, key: str, value: Any) -> None:
    mapping = {
        "dataset": (config.data, "dataset"),
        "attack_type": (config.attacker, "type"),
        "defense_type": (config.defender, "type"),
        "protocol": (config.protocol, "name"),
        "split_mode": (config.data, "split_mode"),
        "noniid_q": (config.data, "noniid_q"),
        "warmup_rounds": (config.protocol, "warmup_rounds"),
        "rounds": (config.runtime, "rounds"),
        "device": (config.runtime, "device"),
        "num_clients": (config.fl, "num_clients"),
        "num_attackers": (config.fl, "num_attackers"),
        "subsample_rate": (config.fl, "subsample_rate"),
        "local_epochs": (config.fl, "local_epochs"),
        "lr": (config.runtime, "lr"),
        "batch_size": (config.runtime, "batch_size"),
        "eval_batch_size": (config.runtime, "eval_batch_size"),
        "max_client_samples_per_client": (config.runtime, "max_client_samples_per_client"),
        "max_eval_samples": (config.runtime, "max_eval_samples"),
        "num_workers": (config.runtime, "num_workers"),
        "parallel_clients": (config.runtime, "parallel_clients"),
        "eval_every": (config.runtime, "eval_every"),
        "seed": (config.runtime, "seed"),
        "init_mode": (config.init, "init_mode"),
        "init_checkpoint_path": (config.init, "init_checkpoint_path"),
        "ipm_scaling": (config.attacker, "ipm_scaling"),
        "lmp_scale": (config.attacker, "lmp_scale"),
        "alie_tau": (config.attacker, "alie_tau"),
        "gaussian_sigma": (config.attacker, "gaussian_sigma"),
        "base_class": (config.attacker, "base_class"),
        "target_class": (config.attacker, "target_class"),
        "pattern_type": (config.attacker, "pattern_type"),
        "bfl_poison_frac": (config.attacker, "bfl_poison_frac"),
        "dba_poison_frac": (config.attacker, "dba_poison_frac"),
        "dba_num_sub_triggers": (config.attacker, "dba_num_sub_triggers"),
        "attacker_action": (config.attacker, "attacker_action"),
        "krum_attackers": (config.defender, "krum_attackers"),
        "multi_krum_selected": (config.defender, "multi_krum_selected"),
        "clipped_median_norm": (config.defender, "clipped_median_norm"),
        "trimmed_mean_ratio": (config.defender, "trimmed_mean_ratio"),
        "geometric_median_iters": (config.defender, "geometric_median_iters"),
        "fltrust_root_size": (config.defender, "fltrust_root_size"),
        "rl_distribution_steps": (config.attacker, "rl_distribution_steps"),
        "rl_attack_start_round": (config.attacker, "rl_attack_start_round"),
        "rl_policy_train_end_round": (config.attacker, "rl_policy_train_end_round"),
        "rl_inversion_steps": (config.attacker, "rl_inversion_steps"),
        "rl_reconstruction_batch_size": (config.attacker, "rl_reconstruction_batch_size"),
        "rl_policy_train_episodes_per_round": (config.attacker, "rl_policy_train_episodes_per_round"),
        "rl_simulator_horizon": (config.attacker, "rl_simulator_horizon"),
        "output_root": (config.output, "output_root"),
        "tb_root": (config.output, "tb_root"),
    }
    if key not in mapping:
        raise ValueError(f"Unknown flat config key: {key}")
    target, field_name = mapping[key]
    if field_name == "attacker_action":
        value = tuple(value)
    setattr(target, field_name, value)
