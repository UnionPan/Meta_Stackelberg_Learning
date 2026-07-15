"""Sequential white-box backdoor Bayesian Stackelberg environment."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from meta_stackelberg.core.model_state import state_l2_norm
from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.environments.backdoor_rewards import (
    WhiteBoxAttackerReward,
    WhiteBoxDefenderReward,
    evaluate_whitebox_backdoor_rewards,
)
from meta_stackelberg.environments.model_tail import ModelTailObservationEncoder
from meta_stackelberg.federated.engine.round_kernel import (
    finalize_round,
    make_client_state,
    prepare_client_slots,
)
from meta_stackelberg.federated.models.parameters import TorchParameterCodec
from meta_stackelberg.federated.types import ClientUpdate, RoundRequest, RoundState, RoundTransition
from meta_stackelberg.security.attacks.backdoor_action import BackdoorAction, BackdoorActionCodec
from meta_stackelberg.security.attacks.rl_backdoor import RLBackdoorAttack
from meta_stackelberg.security.data.labels import class_id
from meta_stackelberg.security.data.trigger import ImageTrigger
from meta_stackelberg.security.defenses.clipped_trimmed_mean import ClippedTrimmedMean
from meta_stackelberg.security.defenses.neuroclip import NeuroClipCopy
from meta_stackelberg.security.defenses.paper_action import (
    PaperDefenderAction,
    PaperDefenderActionCodec,
)
from meta_stackelberg.security.population import FixedMaliciousPopulation
from meta_stackelberg.security.types import AttackContext, RoundAttackContext


@dataclass(frozen=True)
class PendingBackdoorRound:
    round_index: int
    defender_action: PaperDefenderAction
    attacker_observation: Mapping[str, np.ndarray]


@dataclass(frozen=True, eq=False)
class PaperBackdoorRoundStep:
    transition: RoundTransition
    defender_raw_action: np.ndarray
    attacker_raw_action: np.ndarray
    defender_action: PaperDefenderAction
    attacker_action: BackdoorAction
    defender_reward: WhiteBoxDefenderReward
    attacker_reward: WhiteBoxAttackerReward
    clean_loss_before: float
    clean_loss_after: float
    safe_loss_after: float
    target_loss_after: float
    done: bool

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PaperBackdoorRoundStep):
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
            and self.clean_loss_before == other.clean_loss_before
            and self.clean_loss_after == other.clean_loss_after
            and self.safe_loss_after == other.safe_loss_after
            and self.target_loss_after == other.target_loss_after
            and self.done == other.done
        )


@dataclass
class _PendingExecution:
    public: PendingBackdoorRound
    defender_raw: np.ndarray
    sampled_clients: tuple[int, ...]
    benign_by_client: dict[int, ClientUpdate]
    malicious_slots: tuple[object, ...]


class PaperBackdoorBSMGEnv:
    """Expose one Defender commitment followed by one BRL action per FL round."""

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
        malicious_client_datasets: Mapping[int, Dataset],
        reward_dataset: Dataset,
        observation_encoder: ModelTailObservationEncoder,
        server_optimizer,
        trigger: ImageTrigger,
        source_class: int,
        target_class: int,
        malicious_batch_size: int,
        defender_lambda: float,
        attacker_lambda: float,
        fixed_attack_generator=None,
        aggregator_factory=None,
        post_defense_factory=None,
        device: str | torch.device = 'cpu',
    ) -> None:
        if not task_id:
            raise ValueError('task_id must not be empty')
        if horizon <= 0 or sample_size <= 0:
            raise ValueError('horizon and sample_size must be positive')
        if initial_observed_max_norm <= 1e-6 or not np.isfinite(initial_observed_max_norm):
            raise ValueError('initial_observed_max_norm must exceed alpha_min')
        if len(reward_dataset) <= 0:
            raise ValueError('reward_dataset must not be empty')
        if malicious_batch_size <= 0:
            raise ValueError('malicious_batch_size must be positive')
        source = class_id(source_class, name='source_class')
        target = class_id(target_class, name='target_class')
        if source == target:
            raise ValueError('source_class and target_class must differ')
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
        self.malicious_client_datasets = dict(malicious_client_datasets)
        self.reward_dataset = reward_dataset
        self.observation_encoder = observation_encoder
        self.server_optimizer = server_optimizer
        self.trigger = trigger
        self.source_class = source
        self.target_class = target
        self.malicious_batch_size = int(malicious_batch_size)
        self.defender_lambda = _weight(defender_lambda, 'defender_lambda')
        self.attacker_lambda = _weight(attacker_lambda, 'attacker_lambda')
        self.fixed_attack_generator = fixed_attack_generator
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError(f'CUDA device {self.device} is not available')
        if fixed_attack_generator is not None and not callable(
            getattr(fixed_attack_generator, 'craft', None),
        ):
            raise TypeError('fixed_attack_generator must provide craft(context, rng)')
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
        if not callable(self.aggregator_factory):
            raise TypeError('aggregator_factory must be callable')
        if not callable(self.post_defense_factory):
            raise TypeError('post_defense_factory must be callable')
        self.defender_codec = PaperDefenderActionCodec()
        self.attacker_codec = BackdoorActionCodec()
        self._pending: _PendingExecution | None = None
        self._last_epsilon: float | None = None

    def defender_observation(self) -> Mapping[str, np.ndarray]:
        return self.observation_encoder.encode(
            self.state.global_model,
            round_index=self.state.round_index,
            horizon=self.horizon,
        )

    def begin_round(self, defender_raw_action: np.ndarray) -> PendingBackdoorRound:
        if self._pending is not None:
            raise RuntimeError('a backdoor BSMG round is already pending')
        if self.state.round_index >= self.horizon:
            raise RuntimeError('backdoor BSMG episode is complete')
        raw = _raw_action(defender_raw_action, 'Defender')
        action = self.defender_codec.decode(raw, observed_max_norm=self.observed_max_norm)
        request = RoundRequest(self.task_id, self.state, self.sample_size, 1.0)
        slots = prepare_client_slots(request, self.rng, self.sampler)
        benign_by_client: dict[int, ClientUpdate] = {}
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
            raise ValueError('paper backdoor round requires at least one sampled benign client')
        attacker_observation = self.observation_encoder.attacker_observation(
            self.defender_observation(),
            malicious_count=len(malicious_slots),
            defender_raw_action=raw,
        )
        public = PendingBackdoorRound(self.state.round_index, action, attacker_observation)
        self._pending = _PendingExecution(
            public=public,
            defender_raw=raw,
            sampled_clients=tuple(slot.client_id for slot in slots),
            benign_by_client=benign_by_client,
            malicious_slots=tuple(malicious_slots),
        )
        return public

    def finish_round(self, attacker_raw_action: np.ndarray) -> PaperBackdoorRoundStep:
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
        if malicious_ids and self.fixed_attack_generator is not None:
            malicious_updates = tuple(
                self.fixed_attack_generator.craft(
                    AttackContext(
                        client_id=slot.client_id,
                        round_index=self.state.round_index,
                        global_model=self.state.global_model,
                    ),
                    slot.rng,
                )
                for slot in pending.malicious_slots
            )
        elif malicious_ids:
            generator = RLBackdoorAttack(
                action=attacker_action,
                model_factory=self.model_factory,
                codec=self.codec,
                client_datasets=self.malicious_client_datasets,
                trigger=self.trigger,
                source_class=self.source_class,
                target_class=self.target_class,
                batch_size=self.malicious_batch_size,
            )
            malicious_updates = generator.craft_round(
                RoundAttackContext(
                    round_index=self.state.round_index,
                    global_model=self.state.global_model,
                    malicious_client_ids=malicious_ids,
                    benign_updates=benign_updates,
                ),
                tuple(slot.rng for slot in pending.malicious_slots),
            )
        else:
            malicious_updates = ()
        malicious_by_client = {update.client_id: update for update in malicious_updates}
        ordered = tuple(
            malicious_by_client[client_id]
            if client_id in malicious_by_client
            else pending.benign_by_client[client_id]
            for client_id in pending.sampled_clients
        )
        action = pending.public.defender_action
        clean_before, _, _ = self._post_defense_losses(self.state, action.epsilon)
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
                'source_class': self.source_class,
                'target_class': self.target_class,
            },
        )
        clean_after, safe_after, target_after = self._post_defense_losses(
            transition.state_after, action.epsilon,
        )
        clean_damage = max(0.0, clean_after - clean_before)
        defender_reward, attacker_reward = evaluate_whitebox_backdoor_rewards(
            clean_loss=clean_after,
            safe_loss=safe_after,
            target_loss=target_after,
            clean_damage=clean_damage,
            defender_lambda=self.defender_lambda,
            attacker_lambda=self.attacker_lambda,
        )
        self.state = transition.state_after
        self.observed_max_norm = max(state_l2_norm(update.delta) for update in ordered)
        self._last_epsilon = action.epsilon
        self._pending = None
        return PaperBackdoorRoundStep(
            transition=transition,
            defender_raw_action=pending.defender_raw,
            attacker_raw_action=raw,
            defender_action=action,
            attacker_action=attacker_action,
            defender_reward=defender_reward,
            attacker_reward=attacker_reward,
            clean_loss_before=clean_before,
            clean_loss_after=clean_after,
            safe_loss_after=safe_after,
            target_loss_after=target_after,
            done=self.state.round_index >= self.horizon,
        )

    def final_delivered_model(self) -> torch.nn.Module | None:
        if self._last_epsilon is None:
            return None
        model = self.model_factory().to(self.device)
        self.codec.load(model, self.state.global_model)
        return self.post_defense_factory(model, self._last_epsilon)

    def _post_defense_losses(
        self,
        state: RoundState,
        epsilon: float,
    ) -> tuple[float, float, float]:
        model = self.model_factory().to(self.device)
        self.codec.load(model, state.global_model)
        defended = self.post_defense_factory(model, epsilon)
        defended.eval()
        clean_sum = 0.0
        clean_count = 0
        safe_sum = 0.0
        target_sum = 0.0
        source_count = 0
        loader = DataLoader(
            self.reward_dataset,
            batch_size=max(1, min(128, len(self.reward_dataset))),
            shuffle=False,
            num_workers=0,
        )
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).long()
                logits = defended(inputs)
                clean_sum += float(torch.nn.functional.cross_entropy(
                    logits, labels, reduction='sum',
                ).item())
                clean_count += int(labels.numel())
                source_mask = labels == self.source_class
                if not bool(source_mask.any()):
                    continue
                triggered = torch.stack([
                    self.trigger.apply(image) for image in inputs[source_mask]
                ])
                triggered_logits = defended(triggered)
                count = int(triggered.shape[0])
                safe_labels = torch.full(
                    (count,), self.source_class, dtype=torch.long,
                    device=triggered_logits.device,
                )
                target_labels = torch.full(
                    (count,), self.target_class, dtype=torch.long,
                    device=triggered_logits.device,
                )
                safe_sum += float(torch.nn.functional.cross_entropy(
                    triggered_logits, safe_labels, reduction='sum',
                ).item())
                target_sum += float(torch.nn.functional.cross_entropy(
                    triggered_logits, target_labels, reduction='sum',
                ).item())
                source_count += count
        if clean_count <= 0:
            raise ValueError('reward_dataset must not be empty')
        if source_count <= 0:
            raise ValueError('reward_dataset has no source-class examples')
        return (
            clean_sum / clean_count,
            safe_sum / source_count,
            target_sum / source_count,
        )


def _raw_action(value: np.ndarray, role: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (3,) or not np.issubdtype(raw.dtype, np.floating):
        raise ValueError(f'{role} raw action must be floating shape (3,)')
    if not np.all(np.isfinite(raw)) or np.any(raw < -1.0) or np.any(raw > 1.0):
        raise ValueError(f'{role} raw action must be finite within [-1,1]')
    result = raw.astype(np.float32, copy=True)
    result.setflags(write=False)
    return result


def _weight(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f'{name} must be a real number')
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError(f'{name} must be finite within [0, 1]')
    return result
