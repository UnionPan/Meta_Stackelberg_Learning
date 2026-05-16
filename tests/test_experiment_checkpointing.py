from pathlib import Path
import tempfile

import torch

from fl_sandbox.config import RunConfig
from fl_sandbox.core.experiment_service import ExperimentCheckpointManager


class _FakeTrainer:
    def __init__(self):
        self.saved_paths = []

    def save(self, path: str) -> None:
        self.saved_paths.append(path)
        torch.save({"algorithm": {"weight": torch.tensor([1.0])}, "diagnostics": {"ok": 1.0}}, path)


class _FakeRLAttack:
    attack_type = "rl"

    def __init__(self):
        self.trainer = _FakeTrainer()


def test_rl_checkpoint_manager_saves_interval_latest_and_final_files():
    run_config = RunConfig.from_flat_dict(
        {
            "attack_type": "rl",
            "rl_checkpoint_interval": 2,
            "rl_save_final_checkpoint": True,
            "rounds": 3,
        }
    )
    model = torch.nn.Linear(2, 1)
    attack = _FakeRLAttack()
    with tempfile.TemporaryDirectory() as tmp:
        manager = ExperimentCheckpointManager(
            output_dir=Path(tmp),
            run_config=run_config,
            config_payload={"attack_type": "rl", "rounds": 3},
            attack=attack,
            model=model,
        )

        assert manager.maybe_save(round_idx=1) == []
        round_two_paths = manager.maybe_save(round_idx=2)
        final_paths = manager.maybe_save(round_idx=3)

        checkpoint_dir = Path(tmp) / "checkpoints"
        assert checkpoint_dir.joinpath("rl_policy_latest.pt").is_file()
        assert checkpoint_dir.joinpath("rl_policy_round_000002.pt").is_file()
        assert checkpoint_dir.joinpath("rl_policy_round_000003.pt").is_file()
        assert checkpoint_dir.joinpath("global_model_latest.pt").is_file()
        assert checkpoint_dir.joinpath("global_model_round_000002.pt").is_file()
        assert checkpoint_dir.joinpath("global_model_round_000003.pt").is_file()
        assert len(round_two_paths) == 4
        assert len(final_paths) == 4

        rl_payload = torch.load(checkpoint_dir / "rl_policy_round_000003.pt", map_location="cpu")
        model_payload = torch.load(checkpoint_dir / "global_model_round_000003.pt", map_location="cpu")
        assert rl_payload["round_idx"] == 3
        assert rl_payload["config"]["rounds"] == 3
        assert model_payload["round_idx"] == 3
        assert "state_dict" in model_payload
