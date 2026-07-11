from dataclasses import dataclass

import numpy as np
import pytest
import torch

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.core.random_state import RandomSnapshot, RandomSource
from meta_stackelberg.federated.aggregation.fedavg import FedAvg
from meta_stackelberg.federated.engine.round_engine import RoundEngine
from meta_stackelberg.federated.engine.server_optimizer import ServerSGD
from meta_stackelberg.federated.protocols import RoundExecutor
from meta_stackelberg.federated.types import ClientUpdate, RoundRequest, RoundState
from meta_stackelberg.security.attacks.delta_reversal import DeltaReversalAttack
from meta_stackelberg.security.attacks.identity import IdentityMaliciousUpdateGenerator
from meta_stackelberg.security.population import FixedMaliciousPopulation
from meta_stackelberg.security.training import ScopedLocalTrainer
from meta_stackelberg.security.types import (
    AttackCapabilities,
    AttackContext,
    AttackKnowledge,
    RoundAttackContext,
)


@dataclass
class FixedSampler:
    client_ids: tuple[int, ...]
    calls: int = 0

    def sample(self, request: RoundRequest, rng: RandomSource) -> tuple[int, ...]:
        del request, rng
        self.calls += 1
        return self.client_ids


@dataclass
class ScriptedTrainer:
    calls: int = 0

    def train(self, client_id: int, state: RoundState, rng: RandomSource) -> ClientUpdate:
        del state, rng
        self.calls += 1
        return ClientUpdate(
            client_id=client_id,
            delta=ModelState.from_tensors((
                np.array([float(client_id + 1)], dtype=np.float32),
            )),
            num_examples=1,
            metadata={'source': 'scripted'},
        )


class ComponentSensitiveTrainer(ScriptedTrainer):
    def train(self, client_id: int, state: RoundState, rng: RandomSource) -> ClientUpdate:
        if state.component_states:
            return ClientUpdate(
                client_id=client_id,
                delta=ModelState.from_tensors((np.array([99.0], dtype=np.float32),)),
                num_examples=1,
            )
        return super().train(client_id, state, rng)


class FirstUpdateAggregator:
    def aggregate(self, updates) -> ModelState:
        return tuple(updates)[0].delta.clone()


class FixtureGenerator:
    capabilities = AttackCapabilities(needs_global_model=True, needs_local_data=True)

    def __init__(
        self,
        *,
        client_override: int | None = None,
        malicious: bool = True,
        values: tuple[float, ...] = (1.0,),
        dtype=np.float32,
    ) -> None:
        self.client_override = client_override
        self.malicious = malicious
        self.values = values
        self.dtype = dtype

    def craft(self, context: AttackContext, rng: RandomSource) -> ClientUpdate:
        del rng
        return ClientUpdate(
            client_id=context.client_id if self.client_override is None else self.client_override,
            delta=ModelState.from_tensors((np.asarray(self.values, dtype=self.dtype),)),
            num_examples=1,
            is_malicious=self.malicious,
        )


class RecordingRoundGenerator:
    capabilities = AttackCapabilities(observes_benign_updates=True)

    def __init__(self, *, malicious: bool = True) -> None:
        self.context = None
        self.rngs = None
        self.malicious = malicious

    def craft_round(
        self,
        context: RoundAttackContext,
        rngs: tuple[RandomSource, ...],
    ) -> tuple[ClientUpdate, ...]:
        self.context = context
        self.rngs = rngs
        return tuple(
            ClientUpdate(
                client_id=client_id,
                delta=ModelState.from_tensors((
                    np.array([float(client_id + 1)], dtype=np.float32),
                )),
                num_examples=1,
                is_malicious=self.malicious,
            )
            for client_id in context.malicious_client_ids
        )


def _knowledge() -> AttackKnowledge:
    return AttackKnowledge(allows_global_model=True, allows_local_data=True)


def _state(seed: int = 31, *, with_server_component: bool = False) -> RoundState:
    source = RandomSource(seed)
    return RoundState(
        round_index=0,
        global_model=ModelState.from_tensors((np.zeros(1, dtype=np.float32),)),
        random_snapshot=source.capture(),
        component_states={'server_only': 7} if with_server_component else {},
    )


def _request(state: RoundState, sample_size: int = 3) -> RoundRequest:
    return RoundRequest(
        task_id='attack-round',
        state=state,
        sample_size=sample_size,
        server_lr=1.0,
    )


def _assert_snapshots_equal(left: RandomSnapshot, right: RandomSnapshot) -> None:
    assert left.python_state == right.python_state
    assert left.numpy_state == right.numpy_state
    assert torch.equal(left.torch_cpu_state, right.torch_cpu_state)


def test_identity_attack_matches_clean_model_aggregate_and_parent_rng() -> None:
    from meta_stackelberg.security.engine.attack_round_engine import AttackRoundEngine

    state = _state(with_server_component=True)
    clean_trainer = ComponentSensitiveTrainer()
    clean = RoundEngine(
        sampler=FixedSampler((0, 1, 2)),
        trainer=clean_trainer,
        aggregator=FedAvg(),
        server_optimizer=ServerSGD(),
    ).run_round(_request(state), RandomSource(999))
    attack_trainer = ComponentSensitiveTrainer()
    malicious_trainer = ComponentSensitiveTrainer()
    attacked = AttackRoundEngine(
        sampler=FixedSampler((0, 1, 2)),
        benign_trainer=attack_trainer,
        malicious_generator=IdentityMaliciousUpdateGenerator(
            ScopedLocalTrainer(malicious_trainer, {1})
        ),
        population=FixedMaliciousPopulation({1}),
        knowledge=_knowledge(),
        aggregator=FedAvg(),
        server_optimizer=ServerSGD(),
    ).run_round(_request(state), RandomSource(111))

    assert isinstance(AttackRoundEngine, type)
    np.testing.assert_array_equal(attacked.aggregate_delta.vector(), clean.aggregate_delta.vector())
    np.testing.assert_array_equal(
        attacked.state_after.global_model.vector(),
        clean.state_after.global_model.vector(),
    )
    _assert_snapshots_equal(attacked.state_after.random_snapshot, clean.state_after.random_snapshot)
    assert [update.client_id for update in attacked.benign_updates] == [0, 2]
    assert [update.client_id for update in attacked.malicious_updates] == [1]
    assert 'malicious_client_ids' not in attacked.public_signals
    assert attacked.private_diagnostics == {
        'malicious_client_ids': (1,),
        'malicious_client_count': 1,
    }


def test_attack_engine_preserves_sampled_order_when_aggregating() -> None:
    from meta_stackelberg.security.engine.attack_round_engine import AttackRoundEngine

    trainer = ScriptedTrainer()
    transition = AttackRoundEngine(
        sampler=FixedSampler((1, 0)),
        benign_trainer=trainer,
        malicious_generator=DeltaReversalAttack(
            trainer=ScopedLocalTrainer(trainer, {1}),
            budget=2.0,
        ),
        population=FixedMaliciousPopulation({1}),
        knowledge=_knowledge(),
        aggregator=FirstUpdateAggregator(),
        server_optimizer=ServerSGD(),
    ).run_round(_request(_state(), sample_size=2), RandomSource(0))

    np.testing.assert_array_equal(transition.aggregate_delta.vector(), [-2.0])


def test_replacing_identity_with_delta_reversal_changes_only_attack_operator() -> None:
    from meta_stackelberg.security.engine.attack_round_engine import AttackRoundEngine

    state = _state()

    def run(generator, benign_trainer):
        return AttackRoundEngine(
            sampler=FixedSampler((0, 1, 2)),
            benign_trainer=benign_trainer,
            malicious_generator=generator,
            population=FixedMaliciousPopulation({1}),
            knowledge=_knowledge(),
            aggregator=FedAvg(),
            server_optimizer=ServerSGD(),
        ).run_round(_request(state), RandomSource(5))

    identity = run(
        IdentityMaliciousUpdateGenerator(ScopedLocalTrainer(ScriptedTrainer(), {1})),
        ScriptedTrainer(),
    )
    reversal = run(
        DeltaReversalAttack(
            trainer=ScopedLocalTrainer(ScriptedTrainer(), {1}),
            budget=2.0,
        ),
        ScriptedTrainer(),
    )

    np.testing.assert_allclose(identity.aggregate_delta.vector(), [2.0])
    np.testing.assert_allclose(reversal.aggregate_delta.vector(), [2.0 / 3.0])
    assert identity.sampled_clients == reversal.sampled_clients
    _assert_snapshots_equal(identity.state_after.random_snapshot, reversal.state_after.random_snapshot)


def test_capability_violation_fails_before_sampling_or_training() -> None:
    from meta_stackelberg.security.engine.attack_round_engine import AttackRoundEngine

    sampler = FixedSampler((0,))
    trainer = ScriptedTrainer()
    with pytest.raises(ValueError, match='needs_local_data'):
        AttackRoundEngine(
            sampler=sampler,
            benign_trainer=trainer,
            malicious_generator=FixtureGenerator(),
            population=FixedMaliciousPopulation({0}),
            knowledge=AttackKnowledge(allows_global_model=True),
            aggregator=FedAvg(),
            server_optimizer=ServerSGD(),
        )

    assert sampler.calls == 0
    assert trainer.calls == 0


def test_changed_capability_manifest_fails_before_sampling() -> None:
    from meta_stackelberg.security.engine.attack_round_engine import AttackRoundEngine

    sampler = FixedSampler((0,))
    generator = FixtureGenerator()
    engine = AttackRoundEngine(
        sampler=sampler,
        benign_trainer=ScriptedTrainer(),
        malicious_generator=generator,
        population=FixedMaliciousPopulation({0}),
        knowledge=_knowledge(),
        aggregator=FedAvg(),
        server_optimizer=ServerSGD(),
    )
    generator.capabilities = AttackCapabilities(
        needs_global_model=True,
        needs_local_data=True,
        uses_oracle_data=True,
    )

    with pytest.raises(ValueError, match='manifest changed'):
        engine.run_round(_request(_state(), sample_size=1), RandomSource(0))

    assert sampler.calls == 0


def test_fixed_population_and_scoped_generator_must_align_at_construction() -> None:
    from meta_stackelberg.security.engine.attack_round_engine import AttackRoundEngine

    sampler = FixedSampler((0,))
    with pytest.raises(ValueError, match='scope.*population'):
        AttackRoundEngine(
            sampler=sampler,
            benign_trainer=ScriptedTrainer(),
            malicious_generator=IdentityMaliciousUpdateGenerator(
                ScopedLocalTrainer(ScriptedTrainer(), {1})
            ),
            population=FixedMaliciousPopulation({0}),
            knowledge=_knowledge(),
            aggregator=FedAvg(),
            server_optimizer=ServerSGD(),
        )

    assert sampler.calls == 0


@pytest.mark.parametrize(
    ('generator', 'message'),
    [
        (FixtureGenerator(client_override=9), 'client 9'),
        (FixtureGenerator(malicious=False), 'malicious'),
        (FixtureGenerator(values=(float('nan'),)), 'finite'),
        (FixtureGenerator(values=(1.0, 2.0)), 'structure'),
        (FixtureGenerator(values=(1.0,), dtype=np.float64), 'dtype'),
    ],
)
def test_attack_engine_rejects_invalid_malicious_update(generator, message: str) -> None:
    from meta_stackelberg.security.engine.attack_round_engine import AttackRoundEngine

    with pytest.raises(ValueError, match=message):
        AttackRoundEngine(
            sampler=FixedSampler((0,)),
            benign_trainer=ScriptedTrainer(),
            malicious_generator=generator,
            population=FixedMaliciousPopulation({0}),
            knowledge=_knowledge(),
            aggregator=FedAvg(),
            server_optimizer=ServerSGD(),
        ).run_round(_request(_state(), sample_size=1), RandomSource(0))


def test_attack_round_engine_satisfies_round_executor_protocol() -> None:
    from meta_stackelberg.security.engine.attack_round_engine import AttackRoundEngine

    trainer = ScriptedTrainer()
    engine = AttackRoundEngine(
        sampler=FixedSampler((0,)),
        benign_trainer=trainer,
        malicious_generator=IdentityMaliciousUpdateGenerator(
            ScopedLocalTrainer(ScriptedTrainer(), {0})
        ),
        population=FixedMaliciousPopulation({0}),
        knowledge=_knowledge(),
        aggregator=FedAvg(),
        server_optimizer=ServerSGD(),
    )

    assert isinstance(engine, RoundExecutor)


def test_round_generator_observes_all_benign_updates_and_restores_sampled_order() -> None:
    from meta_stackelberg.security.engine.attack_round_engine import AttackRoundEngine

    generator = RecordingRoundGenerator()
    transition = AttackRoundEngine(
        sampler=FixedSampler((1, 0, 3, 2)),
        benign_trainer=ScriptedTrainer(),
        malicious_generator=generator,
        population=FixedMaliciousPopulation({1, 3}),
        knowledge=AttackKnowledge(allows_benign_updates=True),
        aggregator=FirstUpdateAggregator(),
        server_optimizer=ServerSGD(),
    ).run_round(_request(_state(), sample_size=4), RandomSource(7))

    assert generator.context.malicious_client_ids == (1, 3)
    assert [update.client_id for update in generator.context.benign_updates] == [0, 2]
    assert len(generator.rngs) == 2
    assert transition.sampled_clients == (1, 0, 3, 2)
    np.testing.assert_array_equal(transition.aggregate_delta.vector(), [2.0])
    assert [update.client_id for update in transition.malicious_updates] == [1, 3]
    assert [update.client_id for update in transition.benign_updates] == [0, 2]


def test_round_identity_matches_clean_aggregate_model_and_parent_rng() -> None:
    from meta_stackelberg.security.engine.attack_round_engine import AttackRoundEngine

    state = _state()
    clean = RoundEngine(
        sampler=FixedSampler((1, 0, 3, 2)),
        trainer=ScriptedTrainer(),
        aggregator=FedAvg(),
        server_optimizer=ServerSGD(),
    ).run_round(_request(state, sample_size=4), RandomSource(17))
    attacked = AttackRoundEngine(
        sampler=FixedSampler((1, 0, 3, 2)),
        benign_trainer=ScriptedTrainer(),
        malicious_generator=RecordingRoundGenerator(),
        population=FixedMaliciousPopulation({1, 3}),
        knowledge=AttackKnowledge(allows_benign_updates=True),
        aggregator=FedAvg(),
        server_optimizer=ServerSGD(),
    ).run_round(_request(state, sample_size=4), RandomSource(99))

    np.testing.assert_array_equal(attacked.aggregate_delta.vector(), clean.aggregate_delta.vector())
    np.testing.assert_array_equal(
        attacked.state_after.global_model.vector(), clean.state_after.global_model.vector()
    )
    _assert_snapshots_equal(attacked.state_after.random_snapshot, clean.state_after.random_snapshot)


def test_round_generator_must_return_one_malicious_update_per_declared_client() -> None:
    from meta_stackelberg.security.engine.attack_round_engine import AttackRoundEngine

    generator = RecordingRoundGenerator(malicious=False)
    with pytest.raises(ValueError, match='malicious'):
        AttackRoundEngine(
            sampler=FixedSampler((1, 0)),
            benign_trainer=ScriptedTrainer(),
            malicious_generator=generator,
            population=FixedMaliciousPopulation({1}),
            knowledge=AttackKnowledge(allows_benign_updates=True),
            aggregator=FedAvg(),
            server_optimizer=ServerSGD(),
        ).run_round(_request(_state(), sample_size=2), RandomSource(7))


def test_round_context_hides_evidence_not_declared_by_capabilities() -> None:
    from meta_stackelberg.security.engine.attack_round_engine import AttackRoundEngine

    class MinimalRoundGenerator(RecordingRoundGenerator):
        capabilities = AttackCapabilities()

    generator = MinimalRoundGenerator()
    AttackRoundEngine(
        sampler=FixedSampler((1, 0)),
        benign_trainer=ScriptedTrainer(),
        malicious_generator=generator,
        population=FixedMaliciousPopulation({1}),
        knowledge=AttackKnowledge(),
        aggregator=FedAvg(),
        server_optimizer=ServerSGD(),
    ).run_round(_request(_state(), sample_size=2), RandomSource(7))

    assert generator.context.global_model is None
    assert generator.context.benign_updates == ()
