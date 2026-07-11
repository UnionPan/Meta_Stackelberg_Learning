import numpy as np

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.federated.types import ClientUpdate
from meta_stackelberg.experiments.attack_pretraining import (
    AttackPolicyPretrainingConfig,
    fixed_pretraining_aggregator,
)
from meta_stackelberg.agents.td3.config import PaperMetaSGConfig
from meta_stackelberg.security.defenses.clipping import ClippedAggregator
from meta_stackelberg.security.defenses.krum import Krum


def _updates(values: tuple[float, ...]) -> tuple[ClientUpdate, ...]:
    return tuple(
        ClientUpdate(index, ModelState.from_tensors([np.array([value])]), 1)
        for index, value in enumerate(values)
    )


def test_fixed_pretraining_defenses_keep_parameters_explicit() -> None:
    krum = fixed_pretraining_aggregator('krum', byzantine_count=1)
    clipmed = fixed_pretraining_aggregator('clipmed', clip_radius=1.0)

    assert isinstance(krum, Krum)
    assert krum.byzantine_count == 1
    assert isinstance(clipmed, ClippedAggregator)
    assert clipmed.clip_radius == 1.0
    np.testing.assert_allclose(
        clipmed.aggregate(_updates((-10.0, 0.0, 2.0))).vector(),
        np.array([0.0]),
    )


def test_pretraining_config_has_one_paper_parameter_mapping() -> None:
    paper = PaperMetaSGConfig()
    config = AttackPolicyPretrainingConfig.from_paper(paper)

    assert config.fl_rounds == paper.rl_training_rounds == 300
    assert config.batch_size == paper.td3_batch_size == 256
    assert config.learning_starts == paper.learning_starts == 100
    assert config.train_freq == paper.train_freq == 1
    assert config.gradient_steps == paper.gradient_steps == 1
    assert config.replay_capacity == paper.replay_capacity == 1_000_000
    assert config.fixed_defender_raw_action == (0.0, 0.0, 1.0)
    assert not hasattr(config, 'N_A')
