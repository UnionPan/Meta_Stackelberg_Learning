from dataclasses import fields, replace

import numpy as np
import pytest

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.core.random_state import RandomSource


CAPABILITY_TO_KNOWLEDGE = {
    'needs_global_model': 'allows_global_model',
    'needs_local_data': 'allows_local_data',
    'observes_benign_updates': 'allows_benign_updates',
    'observes_other_client_data': 'allows_other_client_data',
    'observes_private_diagnostics': 'allows_private_diagnostics',
    'uses_oracle_data': 'allows_oracle_data',
}


def test_capability_validation_accepts_exactly_allowed_information() -> None:
    from meta_stackelberg.security.types import (
        AttackCapabilities,
        AttackKnowledge,
        validate_capabilities,
    )

    capabilities = AttackCapabilities(
        needs_global_model=True,
        needs_local_data=True,
    )
    knowledge = AttackKnowledge(
        allows_global_model=True,
        allows_local_data=True,
    )

    validate_capabilities(capabilities, knowledge)


@pytest.mark.parametrize('capability_name', tuple(CAPABILITY_TO_KNOWLEDGE))
def test_capability_validation_rejects_each_forbidden_information(
    capability_name: str,
) -> None:
    from meta_stackelberg.security.types import (
        AttackCapabilities,
        AttackKnowledge,
        validate_capabilities,
    )

    capabilities = replace(AttackCapabilities(), **{capability_name: True})

    with pytest.raises(ValueError, match=capability_name):
        validate_capabilities(capabilities, AttackKnowledge())


def test_attack_capability_and_knowledge_records_cover_same_six_dimensions() -> None:
    from meta_stackelberg.security.types import AttackCapabilities, AttackKnowledge

    capability_names = {field.name for field in fields(AttackCapabilities)}
    knowledge_names = {field.name for field in fields(AttackKnowledge)}

    assert capability_names == set(CAPABILITY_TO_KNOWLEDGE)
    assert knowledge_names == set(CAPABILITY_TO_KNOWLEDGE.values())


def test_attack_context_is_a_sanitized_client_view() -> None:
    from meta_stackelberg.security.types import AttackContext

    context = AttackContext(
        client_id=2,
        round_index=3,
        global_model=ModelState.from_tensors((np.zeros(2, dtype=np.float32),)),
    )

    assert context.client_id == 2
    assert not hasattr(context, 'component_states')
    assert not hasattr(context, 'benign_updates')
    assert not hasattr(context, 'private_diagnostics')
    assert not hasattr(context, 'task_id')


@pytest.mark.parametrize(
    'kwargs',
    [
        {'client_id': -1},
        {'round_index': -1},
    ],
)
def test_attack_context_rejects_invalid_identity(kwargs) -> None:
    from meta_stackelberg.security.types import AttackContext

    values = {
        'client_id': 0,
        'round_index': 0,
        'global_model': ModelState.from_tensors((np.zeros(1, dtype=np.float32),)),
    }
    values.update(kwargs)

    with pytest.raises(ValueError):
        AttackContext(**values)


def test_fixed_malicious_population_is_immutable_unique_and_validated() -> None:
    from meta_stackelberg.security.population import FixedMaliciousPopulation

    source_ids = [3, 1, 3]
    population = FixedMaliciousPopulation(source_ids)
    source_ids.append(5)

    assert population.client_ids == frozenset({1, 3})
    assert population.contains(1)
    assert not population.contains(2)
    assert not population.contains(5)
    with pytest.raises(ValueError, match='non-negative'):
        FixedMaliciousPopulation({-1, 2})


@pytest.mark.parametrize('invalid_id', [1.9, '1', True])
def test_fixed_malicious_population_rejects_non_integer_ids(invalid_id) -> None:
    from meta_stackelberg.security.population import FixedMaliciousPopulation

    with pytest.raises(TypeError, match='integer'):
        FixedMaliciousPopulation({invalid_id})


def test_attack_plugins_satisfy_structural_protocols() -> None:
    from meta_stackelberg.federated.types import ClientUpdate
    from meta_stackelberg.security.population import FixedMaliciousPopulation
    from meta_stackelberg.security.protocols import (
        MaliciousPopulation,
        MaliciousUpdateGenerator,
    )
    from meta_stackelberg.security.types import AttackCapabilities, AttackContext

    class FixtureGenerator:
        capabilities = AttackCapabilities()

        def craft(self, context: AttackContext, rng: RandomSource) -> ClientUpdate:
            del context, rng
            raise NotImplementedError

    assert isinstance(FixedMaliciousPopulation({1}), MaliciousPopulation)
    assert isinstance(FixtureGenerator(), MaliciousUpdateGenerator)
    assert not isinstance(object(), MaliciousPopulation)
    assert not isinstance(object(), MaliciousUpdateGenerator)


def test_round_attack_context_exposes_only_declared_round_evidence() -> None:
    from meta_stackelberg.federated.types import ClientUpdate
    from meta_stackelberg.security.types import RoundAttackContext

    benign = ClientUpdate(
        client_id=2,
        delta=ModelState.from_tensors((np.array([1.0], dtype=np.float32),)),
        num_examples=4,
    )
    context = RoundAttackContext(
        round_index=3,
        global_model=ModelState.from_tensors((np.zeros(1, dtype=np.float32),)),
        malicious_client_ids=(1, 4),
        benign_updates=(benign,),
    )

    assert context.malicious_client_ids == (1, 4)
    assert context.benign_updates == (benign,)
    assert not hasattr(context, 'oracle_evaluator')
    assert not hasattr(context, 'private_diagnostics')
    assert not hasattr(context, 'task_id')
    with pytest.raises((AttributeError, TypeError)):
        context.malicious_client_ids += (8,)


def test_round_attack_context_validates_ids_and_reference_updates() -> None:
    from meta_stackelberg.federated.types import ClientUpdate
    from meta_stackelberg.security.types import RoundAttackContext

    model = ModelState.from_tensors((np.zeros(1, dtype=np.float32),))
    benign = ClientUpdate(client_id=2, delta=model, num_examples=1)
    with pytest.raises(ValueError, match='duplicate'):
        RoundAttackContext(0, model, (1, 1), (benign,))
    with pytest.raises(ValueError, match='benign'):
        RoundAttackContext(0, model, (1,), (replace(benign, is_malicious=True),))


def test_round_generator_satisfies_structural_protocol() -> None:
    from meta_stackelberg.federated.types import ClientUpdate
    from meta_stackelberg.security.protocols import RoundMaliciousUpdateGenerator
    from meta_stackelberg.security.types import (
        AttackCapabilities,
        RoundAttackContext,
    )

    class FixtureRoundGenerator:
        capabilities = AttackCapabilities(observes_benign_updates=True)

        def craft_round(
            self,
            context: RoundAttackContext,
            rngs: tuple[RandomSource, ...],
        ) -> tuple[ClientUpdate, ...]:
            del context, rngs
            return ()

    assert isinstance(FixtureRoundGenerator(), RoundMaliciousUpdateGenerator)
    assert not isinstance(object(), RoundMaliciousUpdateGenerator)
