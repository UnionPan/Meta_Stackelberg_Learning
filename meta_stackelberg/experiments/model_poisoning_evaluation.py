"""Final clean/IPM/LMP/RL evaluation for canonical paper MNIST policies."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.agents.td3.replay import flatten_observation
from meta_stackelberg.experiments.paper_meta_sg import (
    ATTACKER_OBSERVATION_KEYS,
    DEFENDER_OBSERVATION_KEYS,
)
from meta_stackelberg.federated.evaluation.classification import (
    ClassificationEvaluator,
)
from meta_stackelberg.security.attacks.ipm import IPMAttack
from meta_stackelberg.security.attacks.lmp import LMPAttack


@dataclass(frozen=True)
class ModelPoisoningScenario:
    name: str
    attack_family: str
    attacker_label: str
    scale: float | None = None

    def __post_init__(self) -> None:
        if self.attack_family not in {'clean', 'ipm', 'lmp', 'rl'}:
            raise ValueError('unknown model-poisoning evaluation family')
        if not self.name or not self.attacker_label:
            raise ValueError('scenario name and attacker label must not be empty')
        if self.attack_family in {'ipm', 'lmp'}:
            if self.scale is None or not math.isfinite(self.scale):
                raise ValueError('fixed attack scenario requires a finite scale')
        elif self.scale is not None:
            raise ValueError('only fixed attack scenarios declare a scale')


def canonical_model_poisoning_scenarios(
    attacker_labels,
    *,
    ipm_scale: float = 2.0,
    lmp_scale: float = 2.0,
) -> tuple[ModelPoisoningScenario, ...]:
    labels = tuple(attacker_labels)
    if not labels:
        raise ValueError('at least one RL attacker label is required')
    return (
        ModelPoisoningScenario('clean', 'clean', labels[0]),
        ModelPoisoningScenario('ipm', 'ipm', labels[0], float(ipm_scale)),
        ModelPoisoningScenario('lmp', 'lmp', labels[0], float(lmp_scale)),
        *(
            ModelPoisoningScenario(label, 'rl', label)
            for label in labels
        ),
    )


def evaluate_model_poisoning_scenario(
    *,
    scenario: ModelPoisoningScenario,
    factory,
    test_dataset: Dataset,
    defender: TD3Agent,
    attacker: TD3Agent,
    seed: int,
    horizon: int,
    device: str = 'cpu',
) -> dict[str, object]:
    """Run one frozen policy pair and retain scalar-only per-round evidence."""
    if defender.role != 'defender' or attacker.role != 'attacker':
        raise ValueError('evaluation policy roles do not match')
    if horizon <= 0:
        raise ValueError('evaluation horizon must be positive')
    malicious_ids = () if scenario.attack_family == 'clean' else factory.malicious_ids
    generator_factory = _fixed_attack_factory(scenario, factory)
    env = factory.make(
        seed=seed,
        horizon=horizon,
        task_id=f'final-evaluation-{scenario.name}',
        malicious_ids=malicious_ids,
        attack_generator_factory=generator_factory,
    )
    evaluator = ClassificationEvaluator(
        model_factory=factory.model_factory,
        dataset=test_dataset,
        codec=factory.codec,
        batch_size=256,
        device=device,
    )
    defender_before = defender.fingerprint()
    attacker_before = attacker.fingerprint()
    rounds = []
    while env.state.round_index < horizon:
        defender_observation = flatten_observation(
            env.defender_observation(), DEFENDER_OBSERVATION_KEYS,
        )
        defender_raw = defender.act(defender_observation, deterministic=True)
        pending = env.begin_round(defender_raw)
        attacker_observation = flatten_observation(
            pending.attacker_observation, ATTACKER_OBSERVATION_KEYS,
        )
        attacker_raw = attacker.act(attacker_observation, deterministic=True)
        step = env.finish_round(attacker_raw)
        clean = evaluator.evaluate(env.state.global_model)
        rounds.append({
            'round': env.state.round_index,
            'clean_loss': _finite(clean.loss, 'clean_loss'),
            'clean_accuracy': _finite(clean.accuracy, 'clean_accuracy'),
            'defender_reward': _finite(
                step.defender_reward.scalar, 'defender_reward',
            ),
            'attacker_reward': _finite(
                step.attacker_reward.scalar, 'attacker_reward',
            ),
            'alpha': _finite(step.defender_action.alpha, 'alpha'),
            'beta': _finite(step.defender_action.beta, 'beta'),
            'epsilon': _finite(step.defender_action.epsilon, 'epsilon'),
            'defender_raw_action': _finite_vector(defender_raw),
            'attacker_raw_action': _finite_vector(attacker_raw),
            'sampled_malicious_clients': int(
                step.transition.private_diagnostics['malicious_client_count'],
            ),
        })
    if defender.fingerprint() != defender_before:
        raise RuntimeError('frozen evaluation Defender mutated')
    if attacker.fingerprint() != attacker_before:
        raise RuntimeError('frozen evaluation Attacker mutated')
    delivered = env.final_delivered_model()
    if delivered is None:
        raise RuntimeError('evaluation did not produce a final delivered model')
    delivered_loss, delivered_accuracy, delivered_examples = _evaluate_model(
        delivered, test_dataset, device,
    )
    return {
        'protocol': 'canonical-model-poisoning-final-evaluation-v1',
        'scenario': scenario.name,
        'attack_family': scenario.attack_family,
        'fixed_attack_scale': scenario.scale,
        'attacker_label': scenario.attacker_label,
        'seed': int(seed),
        'horizon': int(horizon),
        'single_seed': True,
        'defender_fingerprint': defender_before,
        'attacker_fingerprint': attacker_before,
        'round_metrics': rounds,
        'final_raw_clean_loss': rounds[-1]['clean_loss'],
        'final_raw_clean_accuracy': rounds[-1]['clean_accuracy'],
        'final_delivered_clean_loss': delivered_loss,
        'final_delivered_clean_accuracy': delivered_accuracy,
        'test_examples': delivered_examples,
        'mean_defender_reward': _finite(
            np.mean([row['defender_reward'] for row in rounds]),
            'mean_defender_reward',
        ),
        'mean_attacker_reward': _finite(
            np.mean([row['attacker_reward'] for row in rounds]),
            'mean_attacker_reward',
        ),
    }


def summarize_model_poisoning_evaluation(
    records: Mapping[str, Mapping[str, object]],
    *,
    checkpoint: str,
    seed: int,
    horizon: int,
) -> dict[str, object]:
    required = {'clean', 'ipm', 'lmp'}
    if not required.issubset(records):
        raise ValueError('evaluation summary requires clean, IPM and LMP')
    if not any(record['attack_family'] == 'rl' for record in records.values()):
        raise ValueError('evaluation summary requires at least one RL scenario')
    scenarios = {}
    for name, record in records.items():
        rows = record['round_metrics']
        if len(rows) != horizon:
            raise ValueError('evaluation scenario has incomplete round trace')
        accuracies = [float(row['clean_accuracy']) for row in rows]
        losses = [float(row['clean_loss']) for row in rows]
        scenarios[name] = {
            'attack_family': record['attack_family'],
            'rounds': len(rows),
            'final_raw_clean_accuracy': record['final_raw_clean_accuracy'],
            'final_delivered_clean_accuracy': record[
                'final_delivered_clean_accuracy'
            ],
            'mean_clean_accuracy': _finite(
                np.mean(accuracies), 'mean_clean_accuracy',
            ),
            'minimum_clean_accuracy': min(accuracies),
            'maximum_clean_accuracy': max(accuracies),
            'final_raw_clean_loss': record['final_raw_clean_loss'],
            'final_delivered_clean_loss': record['final_delivered_clean_loss'],
            'maximum_clean_loss': max(losses),
            'mean_defender_reward': record['mean_defender_reward'],
            'mean_attacker_reward': record['mean_attacker_reward'],
        }
    rl_records = [
        record for record in records.values() if record['attack_family'] == 'rl'
    ]
    worst_rl = min(
        rl_records, key=lambda item: item['final_delivered_clean_accuracy'],
    )
    return {
        'protocol': 'canonical-model-poisoning-final-summary-v1',
        'checkpoint': checkpoint,
        'evaluation_seed': int(seed),
        'horizon': int(horizon),
        'single_seed': True,
        'confidence_interval': None,
        'in_domain_evaluation': True,
        'scenarios': scenarios,
        'worst_rl_scenario': worst_rl['scenario'],
        'worst_rl_final_delivered_clean_accuracy': worst_rl[
            'final_delivered_clean_accuracy'
        ],
    }


def _fixed_attack_factory(scenario, factory):
    if scenario.attack_family in {'clean', 'rl'}:
        return None
    if scenario.attack_family == 'ipm':
        return lambda action: IPMAttack(
            scale=scenario.scale,
            num_examples_by_client=factory.attacker_num_examples,
        )
    return lambda action: LMPAttack(
        scale=scenario.scale,
        num_examples_by_client=factory.attacker_num_examples,
    )


def _evaluate_model(model, dataset, device):
    target_device = torch.device(device)
    model = model.to(target_device)
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in DataLoader(
            dataset, batch_size=256, shuffle=False, num_workers=0,
        ):
            inputs = inputs.to(target_device, non_blocking=True)
            labels = labels.to(target_device, non_blocking=True).long()
            logits = model(inputs)
            total_loss += float(torch.nn.functional.cross_entropy(
                logits, labels, reduction='sum',
            ).item())
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += int(labels.numel())
    if total <= 0:
        raise ValueError('evaluation dataset must not be empty')
    return total_loss / total, total_correct / total, total


def _finite(value, name):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite')
    return result


def _finite_vector(value):
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError('action vector must be finite and non-empty')
    return [float(item) for item in array]
