from types import SimpleNamespace

import torch
from torch import nn

from fl_sandbox.config import RunConfig
from fl_sandbox.experiments.service import ExperimentCheckpointManager


class _Trainer:
    def __init__(self):
        self.saved = []

    def save(self, path: str) -> None:
        self.saved.append(path)
        torch.save({"algorithm": {"weight": torch.tensor([1.0])}}, path)


def _manager(tmp_path, *, run_config):
    attack = SimpleNamespace(trainer=_Trainer())
    model = nn.Linear(1, 1)
    manager = ExperimentCheckpointManager(
        output_dir=tmp_path,
        run_config=run_config,
        config_payload={},
        attack=attack,
        model=model,
    )
    return manager, attack


def test_frozen_paper_rl_policy_is_saved_once_after_warmup(tmp_path):
    run_config = RunConfig.from_flat_dict(
        {
            "attack_type": "rl",
            "rl_checkpoint_interval": 5,
            "rl_policy_train_steps_per_round": 0,
            "rounds": 20,
            "start_round_idx": 101,
        }
    )
    manager, attack = _manager(tmp_path, run_config=run_config)

    first_paths = manager.maybe_save(round_idx=105)
    second_paths = manager.maybe_save(round_idx=110)

    names = sorted(path.name for path in first_paths + second_paths)
    assert "rl_policy_after_warmup.pt" in names
    assert "rl_policy_latest.pt" in names
    assert "rl_policy_round_000105.pt" not in names
    assert "rl_policy_round_000110.pt" not in names
    assert "global_model_round_000105.pt" in names
    assert "global_model_round_000110.pt" in names
    assert len(attack.trainer.saved) == 2


def test_online_rl_keeps_per_round_policy_checkpoints(tmp_path):
    run_config = RunConfig.from_flat_dict(
        {
            "attack_type": "rl_backdoor",
            "rl_checkpoint_interval": 5,
            "rounds": 20,
            "start_round_idx": 101,
        }
    )
    manager, attack = _manager(tmp_path, run_config=run_config)

    paths = manager.maybe_save(round_idx=105)

    names = sorted(path.name for path in paths)
    assert "rl_policy_latest.pt" in names
    assert "rl_policy_round_000105.pt" in names
    assert "rl_policy_after_warmup.pt" not in names
    assert len(attack.trainer.saved) == 2
