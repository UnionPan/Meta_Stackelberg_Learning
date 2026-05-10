import copy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fl_sandbox.attacks import RLAttack
from fl_sandbox.attacks.rl_attacker.config import RLAttackerConfig
from fl_sandbox.attacks.rl_attacker.trainer import UpdateStats
from fl_sandbox.core.runtime import RoundContext
from fl_sandbox.federation.runner import SandboxConfig
from fl_sandbox.utils.weights import weights_to_vector
from src.models.cnn import MNISTClassifier


def _weights(model):
    return [value.detach().cpu().numpy().copy() for value in model.state_dict().values()]


class FakePPOTrainer:
    algorithm_name = "ppo"

    def __init__(self):
        self.initialized = False
        self.act_calls = []
        self.transitions = []
        self.update_calls = []

    def ensure_initialized(self, obs_space, action_space):
        self.initialized = True
        self.obs_shape = obs_space.shape
        self.action_shape = action_space.shape

    def act(self, obs, *, deterministic=False):
        self.act_calls.append((np.asarray(obs, dtype=np.float32), deterministic))
        return np.zeros(self.action_shape, dtype=np.float32)

    def add_transition(self, obs, act, *, reward, obs_next, terminated=False, truncated=False):
        self.transitions.append(
            SimpleNamespace(
                obs=np.asarray(obs, dtype=np.float32),
                act=np.asarray(act, dtype=np.float32),
                reward=float(reward),
                obs_next=np.asarray(obs_next, dtype=np.float32),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
        )

    def update(self, gradient_steps):
        self.update_calls.append(int(gradient_steps))
        return UpdateStats(gradient_steps=int(gradient_steps), loss=0.25)

    def diagnostics(self):
        return {
            "trainer_algorithm_id": 2.0,
            "trainer_replay_size": float(len(self.transitions)),
            "trainer_loss": 0.25,
        }


def test_ppo_records_real_round_transition_and_skips_simulator_training():
    model = MNISTClassifier()
    weights = _weights(model)
    images = torch.randn(12, 1, 28, 28)
    labels = torch.randint(0, 10, (12,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)
    trainer = FakePPOTrainer()
    attack = RLAttack(
        config=RLAttackerConfig(
            algorithm="ppo",
            attack_start_round=1,
            distribution_steps=0,
            policy_train_end_round=10,
            reconstruction_batch_size=4,
            ppo_real_rollout_steps=2,
            local_search_batch_size=4,
        )
    )
    attack.trainer = trainer

    def fail_if_simulator_training_is_called():
        raise AssertionError("PPO should not train from simulator rollout")

    attack._train_policy = fail_if_simulator_training_is_called
    fl_config = SandboxConfig(num_clients=4, num_attackers=1, batch_size=4)
    ctx = RoundContext(
        round_idx=1,
        old_weights=weights,
        benign_weights=[],
        selected_attacker_ids=[0],
        model=copy.deepcopy(model),
        device=torch.device("cpu"),
        fl_config=fl_config,
        defense_type="krum",
        lr=0.05,
        server_lr=1.0,
        local_epochs=1,
        attacker_train_iter=loader,
        all_attacker_train_iter=loader,
        attacker_action=np.zeros(8, dtype=np.float32),
    )

    with patch(
        "fl_sandbox.attacks.rl_attacker.attack.craft_malicious_update",
        side_effect=lambda **kwargs: [layer + 0.01 for layer in kwargs["old_weights"]],
    ):
        attack.observe_round(ctx)
        malicious = attack.execute(ctx)

    aggregated_weights = malicious[0]
    payload = attack.after_round(
        ctx=ctx,
        all_weights=[aggregated_weights],
        malicious_indices=[0],
        aggregated_weights=aggregated_weights,
        aggregated_delta=weights_to_vector(aggregated_weights) - weights_to_vector(weights),
        clean_loss_before=1.0,
        clean_acc_before=0.8,
        clean_loss=1.2,
        clean_acc=0.7,
        num_byzantine=1,
    )

    assert trainer.initialized
    assert trainer.act_calls
    assert trainer.act_calls[0][1] is False
    assert len(trainer.transitions) == 1
    assert np.isclose(trainer.transitions[0].reward, 0.8)
    assert trainer.transitions[0].obs.shape == trainer.obs_shape
    assert trainer.transitions[0].obs_next.shape == trainer.obs_shape
    assert trainer.update_calls == []
    assert payload["rl_real_ppo_transitions"] == 1.0
    assert payload["rl_real_ppo_buffered_steps"] == 1.0
