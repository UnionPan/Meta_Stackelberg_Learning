"""Lazy Tianshou trainer backend exports."""

__all__ = ["TianshouTD3Trainer"]


def __getattr__(name: str):
    if name == "TianshouTD3Trainer":
        from fl_sandbox.attacks.rl_attacker.tianshou_backend.td3 import TianshouTD3Trainer

        return TianshouTD3Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
