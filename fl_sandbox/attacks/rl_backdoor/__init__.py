"""RL backdoor attack package."""

from fl_sandbox.attacks.rl_backdoor.action import BackdoorAction, decode_backdoor_action
from fl_sandbox.attacks.rl_backdoor.attack import RLBackdoorAttack
from fl_sandbox.attacks.rl_backdoor.config import BackdoorRLConfig
from fl_sandbox.attacks.rl_backdoor.observation import BackdoorObservationBuilder
from fl_sandbox.attacks.rl_backdoor.policy import TD3BackdoorPolicy
from fl_sandbox.attacks.rl_backdoor.reward import BackdoorRewardFn, BackdoorRewardInputs
from fl_sandbox.attacks.rl_backdoor.simulator import SimulatedBackdoorFLEnv

__all__ = [
    "BackdoorAction",
    "BackdoorObservationBuilder",
    "BackdoorRLConfig",
    "BackdoorRewardFn",
    "BackdoorRewardInputs",
    "RLBackdoorAttack",
    "SimulatedBackdoorFLEnv",
    "TD3BackdoorPolicy",
    "decode_backdoor_action",
]
