"""SAC trainer entry point backed by Tianshou infrastructure."""

from __future__ import annotations

from fl_sandbox.attacks.rl_attacker.tianshou_backend.common import BaseTianshouTrainer


class TianshouSACTrainer(BaseTianshouTrainer):
    algorithm_name = "sac"

    def diagnostics(self) -> dict[str, float]:
        data = super().diagnostics()
        data["trainer_entropy"] = 0.0
        return data
