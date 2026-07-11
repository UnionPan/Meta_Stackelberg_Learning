"""Sequential per-FL-round paper-aligned Bayesian Stackelberg environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from meta_stackelberg.core.model_state import state_l2_norm
from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.environments.model_tail import ModelTailObservationEncoder
from meta_stackelberg.environments.rewards import (
    PaperAttackerReward,
    PaperDefenderReward,
    evaluate_paper_untargeted_rewards,
)
from meta_stackelberg.federated.engine.round_kernel import (
    finalize_round,
    make_client_state,
    prepare_client_slots,
)
from meta_stackelberg.federated.models.parameters import TorchParameterCodec
from meta_stackelberg.federated.types import (
    ClientUpdate,
    RoundRequest,
    RoundState,
    RoundTransition,
)
from meta_stackelberg.security.attacks.local_search import RLLocalSearchAttack
from meta_stackelberg.security.attacks.rl_action import RLAttackAction, RLAttackActionCodec
from meta_stackelberg.security.defenses.clipped_trimmed_mean import ClippedTrimmedMean
from meta_stackelberg.security.defenses.neuroclip import NeuroClipCopy
from meta_stackelberg.security.defenses.paper_action import (
    PaperDefenderAction,
    PaperDefenderActionCodec,
)
from meta_stackelberg.security.population import FixedMaliciousPopulation
from meta_stackelberg.security.types import RoundAttackContext


@dataclass(frozen=True)
class PendingPaperRound:
    round_index: int
    defender_action: PaperDefenderAction
    attacker_observation: Mapping[str, np.ndarray]


@dataclass(frozen=True, eq=False)
class PaperRoundStep:
    transition: RoundTransition
    defender_raw_action: np.ndarray
    attacker_raw_action: np.ndarray
    defender_action: PaperDefenderAction
    attacker_action: RLAttackAction
    defender_reward: PaperDefenderReward
    attacker_reward: PaperAttackerReward
    post_loss_before: float
    post_loss_after: float
    done: bool

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PaperRoundStep):
            return False
        return (
            self.transition.sampled_clients == other.transition.sampled_clients
            and np.array_equal(
                self.transition.state_after.global_model.vector(),
                other.transition.state_after.global_model.vector(),
            )
            and np.array_equal(self.defender_raw_action, other.defender_raw_action)
            and np.array_equal(self.attacker_raw_action, other.attacker_raw_action)
            and self.defender_action == other.defender_action
            and self.attacker_action == other.attacker_action
            and self.defender_reward == other.defender_reward
            and self.attacker_reward == other.attacker_reward
            and self.post_loss_before == other.post_loss_before
            and self.post_loss_after == other.post_loss_after
            and self.done == other.done
        )


@dataclass
class _PendingExecution:
    public: PendingPaperRound
    defender_raw: np.ndarray
    sampled_clients: tuple[int, ...]
    benign_by_client: dict[int, ClientUpdate]
    malicious_slots: tuple[object, ...]


class PaperBSMGEnv:
    """Expose explicit Defender-then-Attacker phases for every FL round."""

    def __init__(
        self,
        *,
        task_id: str,
        initial_state: RoundState,
        rng: RandomSource,
        horizon: int,
        sample_size: int,
        initial_observed_max_norm: float,
        sampler,
        benign_trainer,
        population: FixedMaliciousPopulation,
        model_factory,
        codec: TorchParameterCodec,
        attacker_dataset: Dataset,
        attacker_num_examples: dict[int, int],
        root_dataset: Dataset,
        observation_encoder: ModelTailObservationEncoder,
        server_optimizer,
        local_search_learning_rate: float,
        local_search_batch_size: int,
        local_search_trajectories: int,
        aggregator_factory=None,
        post_defense_factory=None,
    ) -> None:
        if not task_id:
            raise ValueError('task_id must not be empty')
        if horizon <= 0 or sample_size <= 0:
            raise ValueError('horizon and sample_size must be positive')
        if initial_observed_max_norm <= 1e-6 or not np.isfinite(initial_observed_max_norm):
            raise ValueError('initial_observed_max_norm must exceed alpha_min')
        self.task_id = task_id
        self.state = initial_state
        self.rng = rng
        self.horizon = int(horizon)
        self.sample_size = int(sample_size)
        self.observed_max_norm = float(initial_observed_max_norm)
        self.sampler = sampler
        self.benign_trainer = benign_trainer
        self.population = population
        self.model_factory = model_factory
        self.codec = codec
        self.attacker_dataset = attacker_dataset
        self.attacker_num_examples = dict(attacker_num_examples)
        self.root_dataset = root_dataset
        self.observation_encoder = observation_encoder
        self.server_optimizer = server_optimizer
        self.local_search_learning_rate = float(local_search_learning_rate)
        self.local_search_batch_size = int(local_search_batch_size)
        self.local_search_trajectories = int(local_search_trajectories)
        self.aggregator_factory = (
            aggregator_factory
            if aggregator_factory is not None
            else lambda action: ClippedTrimmedMean(action.alpha, action.beta)
        )
        self.post_defense_factory = (
            post_defense_factory
            if post_defense_factory is not None
            else lambda model, epsilon: NeuroClipCopy(model, epsilon)
        )
        if not callable(self.post_defense_factory):
            raise TypeError('post_defense_factory must be callable')
        self.defender_codec = PaperDefenderActionCodec()
        self.attacker_codec = RLAttackActionCodec()
        self._pending: _PendingExecution | None = None
        self._last_epsilon: float | None = None

    def defender_observation(self) -> Mapping[str, np.ndarray]:
        return self.observation_encoder.encode(
            self.state.global_model,
            round_index=self.state.round_index,
            horizon=self.horizon,
        )

    def begin_round(self, defender_raw_action: np.ndarray) -> PendingPaperRound:
        if self._pending is not None:
            raise RuntimeError('a paper BSMG round is already pending')
        if self.state.round_index >= self.horizon:
            raise RuntimeError('paper BSMG episode is complete')
        raw = _raw_action(defender_raw_action, 'Defender')
        action = self.defender_codec.decode(raw, observed_max_norm=self.observed_max_norm)
        request = RoundRequest(self.task_id, self.state, self.sample_size, 1.0)
        slots = prepare_client_slots(request, self.rng, self.sampler)
        benign_by_client = {}
        malicious_slots = []
        for slot in slots:
            if self.population.contains(slot.client_id):
                malicious_slots.append(slot)
            else:
                update = self.benign_trainer.train(
                    slot.client_id,
                    make_client_state(self.state, slot.rng),
                    slot.rng,
                )
                if update.is_malicious:
                    raise ValueError('benign trainer returned malicious update')
                benign_by_client[slot.client_id] = update
        if not benign_by_client:
            raise ValueError('paper RL round requires at least one sampled benign client')
        attacker_observation = self.observation_encoder.attacker_observation(
            self.defender_observation(),
            malicious_count=len(malicious_slots),
            defender_raw_action=raw,
        )
        public = PendingPaperRound(self.state.round_index, action, attacker_observation)
        self._pending = _PendingExecution(
            public,
            raw,
            tuple(slot.client_id for slot in slots),
            benign_by_client,
            tuple(malicious_slots),
        )
        return public

    def finish_round(self, attacker_raw_action: np.ndarray) -> PaperRoundStep:
        if self._pending is None:
            raise RuntimeError('begin_round must be called before finish_round')
        pending = self._pending
        raw = _raw_action(attacker_raw_action, 'Attacker')
        attacker_action = self.attacker_codec.decode(raw)
        malicious_ids = tuple(slot.client_id for slot in pending.malicious_slots)
        benign_updates = tuple(
            pending.benign_by_client[client_id]
            for client_id in pending.sampled_clients
            if client_id in pending.benign_by_client
        )
        if malicious_ids:
            generator = RLLocalSearchAttack(
                action=attacker_action,
                model_factory=self.model_factory,
                codec=self.codec,
                local_dataset=self.attacker_dataset,
                num_examples_by_client=self.attacker_num_examples,
                learning_rate=self.local_search_learning_rate,
                batch_size=self.local_search_batch_size,
                trajectories=self.local_search_trajectories,
            )
            malicious_updates = generator.craft_round(
                RoundAttackContext(
                    self.state.round_index,
                    self.state.global_model,
                    malicious_ids,
                    benign_updates,
                ),
                tuple(slot.rng for slot in pending.malicious_slots),
            )
        else:
            malicious_updates = ()
        malicious_by_client = {
            update.client_id: update for update in malicious_updates
        }
        ordered = tuple(
            malicious_by_client[client_id]
            if client_id in malicious_by_client
            else pending.benign_by_client[client_id]
            for client_id in pending.sampled_clients
        )
        action = pending.public.defender_action
        post_before = self._post_defense_loss(self.state, action.epsilon)
        transition = finalize_round(
            request=RoundRequest(self.task_id, self.state, self.sample_size, 1.0),
            parent_rng=self.rng,
            sampled_clients=pending.sampled_clients,
            ordered_updates=ordered,
            aggregator=self.aggregator_factory(action),
            server_optimizer=self.server_optimizer,
            private_diagnostics={
                'malicious_client_count': len(malicious_ids),
                'malicious_client_ids': malicious_ids,
            },
        )
        post_after = self._post_defense_loss(transition.state_after, action.epsilon)
        defender_reward, attacker_reward = evaluate_paper_untargeted_rewards(
            post_loss_before=post_before,
            post_loss_after=post_after,
        )
        self.state = transition.state_after
        self.observed_max_norm = max(state_l2_norm(update.delta) for update in ordered)
        self._last_epsilon = action.epsilon
        self._pending = None
        return PaperRoundStep(
            transition=transition,
            defender_raw_action=pending.defender_raw,
            attacker_raw_action=raw,
            defender_action=action,
            attacker_action=attacker_action,
            defender_reward=defender_reward,
            attacker_reward=attacker_reward,
            post_loss_before=post_before,
            post_loss_after=post_after,
            done=self.state.round_index >= self.horizon,
        )

    def final_delivered_model(self) -> torch.nn.Module | None:
        if self._last_epsilon is None:
            return None
        model = self.model_factory()
        self.codec.load(model, self.state.global_model)
        return self.post_defense_factory(model, self._last_epsilon)

    def _post_defense_loss(self, state: RoundState, epsilon: float) -> float:
        model = self.model_factory()
        self.codec.load(model, state.global_model)
        defended = self.post_defense_factory(model, epsilon)
        defended.eval()
        total_loss = 0.0
        total = 0
        with torch.no_grad():
            for inputs, labels in DataLoader(
                self.root_dataset,
                batch_size=max(1, min(128, len(self.root_dataset))),
                shuffle=False,
            ):
                logits = defended(inputs)
                total_loss += float(
                    torch.nn.functional.cross_entropy(
                        logits, labels.long(), reduction='sum',
                    ).item()
                )
                total += int(labels.numel())
        if total <= 0:
            raise ValueError('root_dataset must not be empty')
        return total_loss / total


def _raw_action(value: np.ndarray, role: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (3,) or not np.issubdtype(raw.dtype, np.floating):
        raise ValueError(f'{role} raw action must be floating shape (3,)')
    if not np.all(np.isfinite(raw)) or np.any(raw < -1.0) or np.any(raw > 1.0):
        raise ValueError(f'{role} raw action must be finite within [-1,1]')
    result = raw.astype(np.float32, copy=True)
    result.setflags(write=False)
    return result
