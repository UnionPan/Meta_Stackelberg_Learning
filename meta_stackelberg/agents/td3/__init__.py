"""Paper-aligned TD3 configuration and agents."""

from meta_stackelberg.agents.td3.config import PaperMetaSGConfig, ScaledMetaSGConfig
from meta_stackelberg.agents.td3.agent import (
    TD3Agent,
    TD3FreezeGuard,
    TD3Snapshot,
    TD3UpdateStats,
)
from meta_stackelberg.agents.td3.replay import TD3Batch, TD3ReplayBuffer, flatten_observation

__all__ = [
    'PaperMetaSGConfig',
    'ScaledMetaSGConfig',
    'TD3Agent',
    'TD3Batch',
    'TD3FreezeGuard',
    'TD3ReplayBuffer',
    'TD3Snapshot',
    'TD3UpdateStats',
    'flatten_observation',
]
