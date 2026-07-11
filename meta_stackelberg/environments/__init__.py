"""Paper-aligned Markov-game environments and observations."""

from meta_stackelberg.environments.model_tail import ModelTailObservationEncoder
from meta_stackelberg.environments.rewards import (
    PaperAttackerReward,
    PaperDefenderReward,
    evaluate_paper_untargeted_rewards,
)
from meta_stackelberg.environments.paper_bsmg import (
    PaperBSMGEnv,
    PaperRoundStep,
    PendingPaperRound,
)

__all__ = [
    'ModelTailObservationEncoder',
    'PaperAttackerReward',
    'PaperDefenderReward',
    'evaluate_paper_untargeted_rewards',
    'PaperBSMGEnv',
    'PaperRoundStep',
    'PendingPaperRound',
]
