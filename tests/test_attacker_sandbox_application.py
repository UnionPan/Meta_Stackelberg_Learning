import argparse
from pathlib import Path
import random
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np

from fl_sandbox.core.batch_runner import BatchRunRequest, clone_args
from fl_sandbox.attacks import craft_lmp
from fl_sandbox.core.experiment_builders import (
    build_attack,
    build_config,
    build_run_name,
    default_output_dir,
    default_tb_dir,
    resolve_num_attackers,
    split_suffix,
)
from fl_sandbox.core.postprocess import build_postprocess_hint_lines
from fl_sandbox.config import RunConfig, config_to_namespace, load_run_config, merge_cli_overrides
from fl_sandbox.application import execute_experiment as application_execute_experiment
from fl_sandbox.federation import FederatedCoordinator
from fl_sandbox.aggregators import AggregationDefender as NewAggregationDefender
from fl_sandbox.attacks.registry import create_attack as create_benchmark_attack
from fl_sandbox.run.run_experiment import parse_args


def _args(**overrides):
    base = {
        "rounds": 5,
        "dataset": "mnist",
        "protocol": "none",
        "warmup_rounds": 0,
        "device": "cpu",
        "num_clients": 10,
        "num_attackers": None,
        "subsample_rate": 0.5,
        "local_epochs": 1,
        "lr": 0.05,
        "batch_size": 64,
        "eval_batch_size": 256,
        "num_workers": 0,
        "parallel_clients": 1,
        "eval_every": 1,
        "seed": 42,
        "init_mode": "seed",
        "init_checkpoint_path": "",
        "attack_type": "clean",
        "ipm_scaling": 2.0,
        "lmp_scale": 5.0,
        "base_class": 1,
        "target_class": 7,
        "pattern_type": "square",
        "bfl_poison_frac": 1.0,
        "dba_poison_frac": 0.5,
        "dba_num_sub_triggers": 4,
        "attacker_action": (0.0, 0.0, 0.0),
        "rl_algorithm": "td3",
        "defense_type": "fedavg",
        "krum_attackers": 1,
        "multi_krum_selected": None,
        "clipped_median_norm": 2.0,
        "trimmed_mean_ratio": 0.2,
        "geometric_median_iters": 10,
        "fltrust_root_size": 100,
        "split_mode": "iid",
        "noniid_q": 0.5,
        "rl_distribution_steps": 10,
        "rl_attack_start_round": 10,
        "rl_policy_train_end_round": 30,
        "rl_inversion_steps": 50,
        "rl_reconstruction_batch_size": 8,
        "rl_policy_train_episodes_per_round": 1,
        "rl_simulator_horizon": 8,
        "output_dir": "",
        "tb_dir": "",
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _run_config(**overrides):
    return RunConfig.from_flat_dict({key: value for key, value in vars(_args(**overrides)).items() if key not in {"output_dir", "tb_dir"}})


class TestAttackerSandboxApplication(unittest.TestCase):

    def test_split_suffix_formats_modes(self):
        self.assertEqual(split_suffix("iid", 0.5), "iid")
        self.assertEqual(split_suffix("paper_q", 0.1), "paper_q_q0.1")

    def test_default_paths_follow_attack_mode(self):
        clean_config = _run_config(attack_type="clean", defense_type="krum", split_mode="iid")
        attack_config = _run_config(attack_type="ipm", defense_type="fltrust", split_mode="noniid", noniid_q=0.3)

        self.assertEqual(
            default_output_dir(clean_config.attacker, clean_config.defender, clean_config.data),
            "fl_sandbox/outputs/clean_krum_iid_benchmark",
        )
        self.assertEqual(
            default_tb_dir(attack_config.attacker, attack_config.defender, attack_config.data),
            "fl_sandbox/runs/ipm_fltrust_noniid_q0.3_demo",
        )

    def test_resolve_num_attackers_defaults_by_mode(self):
        clean_config = _run_config(attack_type="clean", num_attackers=None)
        attack_default_config = _run_config(attack_type="ipm", num_attackers=None)
        attack_override_config = _run_config(attack_type="ipm", num_attackers=7)

        self.assertEqual(resolve_num_attackers(clean_config.attacker, clean_config.fl), 0)
        self.assertEqual(resolve_num_attackers(attack_default_config.attacker, attack_default_config.fl), 2)
        self.assertEqual(resolve_num_attackers(attack_override_config.attacker, attack_override_config.fl), 7)

    def test_schema_defaults_leave_num_attackers_unset_for_mode_resolution(self):
        config = RunConfig()

        self.assertIsNone(config.fl.num_attackers)
        self.assertEqual(resolve_num_attackers(config.attacker, config.fl), 0)
        self.assertEqual(config.attacker.lmp_scale, 5.0)

    def test_craft_lmp_uses_repo_aligned_median_direction(self):
        random.seed(0)
        old_weights = [np.asarray([0.0], dtype=np.float32)]
        benign_weights = [
            [np.asarray([-10.0], dtype=np.float32)],
            [np.asarray([10.0], dtype=np.float32)],
        ]
        attacker_local_weights = [
            [np.asarray([10.0], dtype=np.float32)],
        ]

        crafted = craft_lmp(
            old_weights,
            benign_weights,
            attacker_local_weights_list=attacker_local_weights,
            scale=5.0,
        )

        self.assertLess(float(crafted[0][0]), -10.0)

    def test_build_run_name_uses_shared_split_suffix(self):
        run_name = build_run_name(
            dataset="mnist",
            attack_type="rl",
            defense_type="clipped_median",
            split_mode="paper_q",
            noniid_q=0.1,
            rounds=30,
        )

        self.assertEqual(run_name, "mnist_rl_clipped_median_paper_q_q0.1_30r")

    def test_clone_args_overrides_selected_fields(self):
        args = _args(dataset="mnist", attack_type="clean")

        cloned = clone_args(args, dataset="cifar10", attack_type="rl")

        self.assertEqual(cloned.dataset, "cifar10")
        self.assertEqual(cloned.attack_type, "rl")
        self.assertEqual(cloned.rounds, args.rounds)

    def test_batch_run_request_keeps_run_metadata(self):
        args = _args()

        request = BatchRunRequest(run_name="demo_run", args=args, progress_desc="demo progress")

        self.assertEqual(request.run_name, "demo_run")
        self.assertEqual(request.progress_desc, "demo progress")
        self.assertIs(request.args, args)

    def test_run_entry_parser_builds_consistent_args(self):
        args = parse_args(['--dataset', 'cifar10', '--attack_type', 'rl', '--defense_type', 'krum'])

        self.assertEqual(args.dataset, "cifar10")
        self.assertEqual(args.attack_type, "rl")
        self.assertEqual(args.defense_type, "krum")
        self.assertEqual(args.split_mode, "iid")
        self.assertEqual(args.noniid_q, 0.5)

    def test_run_entry_parser_suppresses_defaults_when_config_is_set(self):
        args = parse_args(['--config', 'demo.yaml'])

        self.assertEqual(args.config, 'demo.yaml')
        self.assertFalse(hasattr(args, 'dataset'))
        self.assertFalse(hasattr(args, 'rounds'))

    def test_postprocess_hints_use_shared_split_naming(self):
        lines = build_postprocess_hint_lines(
            attack_type="ipm",
            defense_type="krum",
            split_mode="noniid",
            noniid_q=0.3,
            output_dir=Path("attacker_sandbox/outputs/demo"),
            tb_dir=Path("attacker_sandbox/runs/demo"),
        )

        self.assertEqual(len(lines), 1)
        self.assertIn("clean_krum_noniid_q0.3_benchmark", lines[0])
        self.assertIn("core/postprocess/postprocess.py", lines[0])

    def test_new_benchmark_paths_expose_compatibility_entrypoints(self):
        run_config = _run_config(attack_type="ipm", ipm_scaling=3.0)

        self.assertEqual(FederatedCoordinator.__name__, "MinimalFLRunner")
        self.assertTrue(NewAggregationDefender.__module__.startswith("fl_sandbox.aggregators"))
        self.assertEqual(create_benchmark_attack(run_config.attacker).scale, 3.0)
        self.assertTrue(callable(application_execute_experiment))

    def test_schema_round_trip_includes_vector_attack_hyperparameters(self):
        run_config = _run_config(attack_type="alie", alie_tau=2.25, gaussian_sigma=0.2)
        flat = run_config.to_flat_dict()
        restored = RunConfig.from_flat_dict(flat)

        self.assertEqual(restored.attacker.alie_tau, 2.25)
        self.assertEqual(restored.attacker.gaussian_sigma, 0.2)

    def test_rlfl_protocol_adjusts_rl_schedule(self):
        run_config = _run_config(
            protocol="rlfl",
            warmup_rounds=100,
            attack_type="rl",
            rl_distribution_steps=None,
            rl_attack_start_round=None,
            rl_policy_train_end_round=None,
        )

        self.assertEqual(run_config.attacker.rl_distribution_steps, 100)
        self.assertEqual(run_config.attacker.rl_policy_train_end_round, 100)
        self.assertEqual(run_config.attacker.rl_attack_start_round, 101)

    def test_protocol_metadata_is_added_to_payload(self):
        run_config = _run_config(
            protocol="rlfl",
            warmup_rounds=50,
            rl_distribution_steps=50,
            rl_attack_start_round=51,
            rl_policy_train_end_round=50,
        )

        payload = run_config.benchmark_protocol_payload()

        self.assertIsNotNone(payload)
        self.assertEqual(payload["warmup_rounds"], 50)

    def test_load_run_config_reads_nested_yaml(self):
        yaml_text = """
data:
  dataset: cifar10
attacker:
  type: rl
  attacker_action: [0.1, 0.2, 0.3]
protocol:
  name: rlfl
  warmup_rounds: 12
runtime:
  rounds: 12
"""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(yaml_text)
            path = handle.name

        config = load_run_config(path)

        self.assertEqual(config.data.dataset, "cifar10")
        self.assertEqual(config.attacker.type, "rl")
        self.assertEqual(config.protocol.name, "rlfl")
        self.assertEqual(config.protocol.warmup_rounds, 12)
        self.assertEqual(config.runtime.rounds, 12)
        self.assertEqual(config.attacker.attacker_action, (0.1, 0.2, 0.3))

    def test_load_run_config_without_path_uses_schema_defaults(self):
        config = load_run_config()

        self.assertEqual(config, RunConfig())

    def test_merge_cli_overrides_updates_structured_config(self):
        config = RunConfig()
        args = argparse.Namespace(dataset="fmnist", rounds=99, config="demo.yaml")

        merged = merge_cli_overrides(config, args)
        namespace = config_to_namespace(merged)

        self.assertEqual(namespace.dataset, "fmnist")
        self.assertEqual(namespace.rounds, 99)

    def test_build_config_reads_structured_run_config_sections(self):
        run_config = _run_config(
            dataset="cifar10",
            attack_type="rl",
            defense_type="krum",
            num_attackers=None,
            attacker_action=(0.1, 0.2, 0.3),
            rl_distribution_steps=21,
        )

        fake_runner_module = ModuleType("attacker_sandbox.core.fl_runner")

        class FakeSandboxConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        fake_runner_module.SandboxConfig = FakeSandboxConfig

        with patch.dict(sys.modules, {"attacker_sandbox.core.fl_runner": fake_runner_module}):
            sandbox_config = build_config(run_config)

        self.assertEqual(sandbox_config.dataset, "cifar10")
        self.assertEqual(sandbox_config.defense_type, "krum")
        self.assertEqual(sandbox_config.num_attackers, 2)
        self.assertEqual(sandbox_config.attacker_action, (0.1, 0.2, 0.3))
        self.assertEqual(sandbox_config.rl_distribution_steps, 21)

    def test_build_attack_reads_structured_run_config_sections(self):
        run_config = _run_config(
            attack_type="rl",
            attacker_action=(0.2, 0.4, 0.6),
            rl_algorithm="td3",
            rl_distribution_steps=13,
            rl_attack_start_round=14,
            rl_policy_train_end_round=15,
            rl_inversion_steps=16,
            rl_reconstruction_batch_size=17,
            rl_policy_train_episodes_per_round=18,
            rl_simulator_horizon=19,
        )

        attack = build_attack(run_config.attacker)

        self.assertEqual(attack.name, "RL")
        self.assertEqual(tuple(attack.default_action), (0.2, 0.4, 0.6))
        self.assertEqual(attack.config.algorithm, "td3")
        self.assertEqual(attack.config.distribution_steps, 13)
        self.assertEqual(attack.config.attack_start_round, 14)
        self.assertEqual(attack.config.policy_train_end_round, 15)
        self.assertEqual(attack.config.inversion_steps, 16)
        self.assertEqual(attack.config.reconstruction_batch_size, 17)
        self.assertEqual(attack.config.episodes_per_observation, 18)
        self.assertEqual(attack.config.simulator_horizon, 19)

    def test_build_attack_rejects_unknown_attack_type(self):
        run_config = _run_config(attack_type="unknown_attack")

        with self.assertRaisesRegex(ValueError, "Unsupported attack type: unknown_attack"):
            build_attack(run_config.attacker)


if __name__ == "__main__":
    unittest.main()
