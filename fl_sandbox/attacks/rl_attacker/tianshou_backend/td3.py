"""TD3 trainer entry point backed by Tianshou infrastructure."""

from __future__ import annotations

from fl_sandbox.attacks.rl_attacker.tianshou_backend.common import BaseTianshouTrainer


class TianshouTD3Trainer(BaseTianshouTrainer):
    algorithm_name = "td3"

    def diagnostics(self) -> dict[str, float]:
        data = super().diagnostics()
        data["trainer_exploration_noise"] = float(self.config.exploration_noise)
        return data
