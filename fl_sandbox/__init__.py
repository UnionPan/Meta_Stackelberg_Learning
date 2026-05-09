"""Standalone attacker validation sandbox."""

__all__ = ["AttackerRLEnv"]


def __getattr__(name):
    if name == "AttackerRLEnv":
        from fl_sandbox.attacks.rl_attacker.env import AttackerRLEnv
        return AttackerRLEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
