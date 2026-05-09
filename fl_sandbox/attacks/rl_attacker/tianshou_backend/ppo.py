"""PPO trainer entry point for the Path-B hybrid action policy."""

from __future__ import annotations

from fl_sandbox.attacks.rl_attacker.tianshou_backend.common import BaseTianshouTrainer


class TianshouPPOTrainer(BaseTianshouTrainer):
    algorithm_name = "ppo"

    def diagnostics(self) -> dict[str, float]:
        data = super().diagnostics()
        data["trainer_entropy_coef"] = float(self.config.ppo_entropy_coef)
        return data
