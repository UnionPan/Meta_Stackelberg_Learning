"""RL backdoor attack package."""

from fl_sandbox.attacks.rl_backdoor.action import BackdoorAction, decode_backdoor_action
from fl_sandbox.attacks.rl_backdoor.attack import RLBackdoorAttack
from fl_sandbox.attacks.rl_backdoor.config import BackdoorRLConfig
from fl_sandbox.attacks.rl_backdoor.env import BackdoorPolicyGymEnv
from fl_sandbox.attacks.rl_backdoor.observation import BackdoorObservationBuilder
from fl_sandbox.attacks.rl_backdoor.policy import EliteBackdoorPolicy, TD3BackdoorPolicy
from fl_sandbox.attacks.rl_backdoor.reward import BackdoorRewardFn, BackdoorRewardInputs

__all__ = [
    "BackdoorAction",
    "BackdoorObservationBuilder",
    "BackdoorPolicyGymEnv",
    "BackdoorRLConfig",
    "BackdoorRewardFn",
    "BackdoorRewardInputs",
    "EliteBackdoorPolicy",
    "RLBackdoorAttack",
    "TD3BackdoorPolicy",
    "decode_backdoor_action",
]
