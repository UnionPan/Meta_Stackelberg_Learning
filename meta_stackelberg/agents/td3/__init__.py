"""Paper-aligned TD3 configuration and agents."""

from meta_stackelberg.agents.td3.config import (
    PaperMetaSGConfig,
    ScaledMetaSGConfig,
    ScaledOnlineAdaptationConfig,
)
from meta_stackelberg.agents.td3.agent import (
    TD3Agent,
    TD3FreezeGuard,
    TD3Snapshot,
    TD3UpdateStats,
)
from meta_stackelberg.agents.td3.replay import (
    TD3Batch,
    TD3ReplayBuffer,
    TD3ReplaySnapshot,
    flatten_observation,
)
from meta_stackelberg.agents.td3.checkpoint import (
    TD3TrainingCheckpoint,
    load_td3_training_checkpoint,
    save_td3_training_checkpoint,
)

__all__ = [
    'PaperMetaSGConfig',
    'ScaledMetaSGConfig',
    'ScaledOnlineAdaptationConfig',
    'TD3Agent',
    'TD3Batch',
    'TD3FreezeGuard',
    'TD3ReplayBuffer',
    'TD3ReplaySnapshot',
    'TD3Snapshot',
    'TD3UpdateStats',
    'flatten_observation',
    'TD3TrainingCheckpoint',
    'load_td3_training_checkpoint',
    'save_td3_training_checkpoint',
]
