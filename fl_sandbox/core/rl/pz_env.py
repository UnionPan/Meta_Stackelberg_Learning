"""Compatibility shim for PettingZoo-style attacker environments."""

from fl_sandbox.attacks.rl_attacker.pz_env import AttackerPolicyGymEnv, AttackerPolicyParallelEnv

__all__ = ["AttackerPolicyGymEnv", "AttackerPolicyParallelEnv"]
