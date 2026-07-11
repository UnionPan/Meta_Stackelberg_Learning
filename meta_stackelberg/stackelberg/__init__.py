"""Immutable records and algorithms for Stackelberg experiments."""

from meta_stackelberg.stackelberg.commitment import DefenderCommitment
from meta_stackelberg.stackelberg.best_response import (
    BestResponseResult,
    CandidateIPMBestResponseSolver,
    CandidateSupportRecord,
    SupportFactory,
)
from meta_stackelberg.stackelberg.response_oracle import FixedIPMResponseOracle
from meta_stackelberg.stackelberg.algorithm1 import (
    Algorithm1Event,
    Algorithm1IterationTrace,
    Algorithm1Result,
    Algorithm1TaskTrace,
    MetaSGAlgorithm1,
)
from meta_stackelberg.stackelberg.policy_response import (
    PolicyBestResponseResult,
    PolicyBestResponseTrainer,
)
from meta_stackelberg.stackelberg.algorithm2 import (
    Algorithm2Event,
    Algorithm2IterationTrace,
    Algorithm2Result,
    Algorithm2TaskTrace,
    MetaSGAlgorithm2,
    reptile_update_td3,
)
from meta_stackelberg.stackelberg.policy_leader import (
    PolicyLeaderResult,
    PolicyLeaderTask,
    PolicyLeaderTaskUpdate,
    PolicyLeaderTrainer,
)
from meta_stackelberg.stackelberg.policy_adaptation import (
    PolicyDefenderAdaptationResult,
    PolicyDefenderAdapter,
)
from meta_stackelberg.stackelberg.policy_algorithm1 import (
    PolicyAlgorithm1IterationTrace,
    PolicyAlgorithm1Result,
    PolicyAlgorithm1TaskTrace,
    PolicyMetaSGAlgorithm1,
)

__all__ = [
    'BestResponseResult',
    'CandidateIPMBestResponseSolver',
    'CandidateSupportRecord',
    'DefenderCommitment',
    'FixedIPMResponseOracle',
    'SupportFactory',
    'Algorithm1Event',
    'Algorithm1IterationTrace',
    'Algorithm1Result',
    'Algorithm1TaskTrace',
    'MetaSGAlgorithm1',
    'PolicyBestResponseResult',
    'PolicyBestResponseTrainer',
    'Algorithm2Event',
    'Algorithm2IterationTrace',
    'Algorithm2Result',
    'Algorithm2TaskTrace',
    'MetaSGAlgorithm2',
    'reptile_update_td3',
    'PolicyLeaderResult',
    'PolicyLeaderTask',
    'PolicyLeaderTaskUpdate',
    'PolicyLeaderTrainer',
    'PolicyDefenderAdaptationResult',
    'PolicyDefenderAdapter',
    'PolicyAlgorithm1IterationTrace',
    'PolicyAlgorithm1Result',
    'PolicyAlgorithm1TaskTrace',
    'PolicyMetaSGAlgorithm1',
]
