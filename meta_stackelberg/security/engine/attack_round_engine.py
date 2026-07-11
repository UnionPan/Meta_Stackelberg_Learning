"""Round orchestration that routes malicious slots through an attack generator."""

from __future__ import annotations

from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.engine.round_kernel import (
    finalize_round,
    make_client_state,
    prepare_client_slots,
)
from meta_stackelberg.federated.protocols import Aggregator, ClientSampler, LocalTrainer, ServerOptimizer
from meta_stackelberg.federated.types import ClientUpdate, RoundRequest, RoundTransition
from meta_stackelberg.security.protocols import MaliciousPopulation, MaliciousUpdateGenerator
from meta_stackelberg.security.types import AttackContext, AttackKnowledge, validate_capabilities


class AttackRoundEngine:
    """Execute one synchronous FL round with explicit malicious-client routing."""

    def __init__(
        self,
        *,
        sampler: ClientSampler,
        benign_trainer: LocalTrainer,
        malicious_generator: MaliciousUpdateGenerator,
        population: MaliciousPopulation,
        knowledge: AttackKnowledge,
        aggregator: Aggregator,
        server_optimizer: ServerOptimizer,
    ) -> None:
        validate_capabilities(malicious_generator.capabilities, knowledge)
        generator_scope = getattr(malicious_generator, 'allowed_client_ids', None)
        population_ids = getattr(population, 'client_ids', None)
        if (
            generator_scope is not None
            and population_ids is not None
            and frozenset(generator_scope) != frozenset(population_ids)
        ):
            raise ValueError('malicious generator scope must match fixed population')
        self._capability_manifest = malicious_generator.capabilities
        self._knowledge = knowledge
        self.sampler = sampler
        self.benign_trainer = benign_trainer
        self.malicious_generator = malicious_generator
        self.population = population
        self.aggregator = aggregator
        self.server_optimizer = server_optimizer

    def run_round(self, request: RoundRequest, rng: RandomSource) -> RoundTransition:
        if self.malicious_generator.capabilities != self._capability_manifest:
            raise ValueError('attack capability manifest changed after engine construction')
        validate_capabilities(self._capability_manifest, self._knowledge)
        slots = prepare_client_slots(request, rng, self.sampler)
        ordered_updates: list[ClientUpdate] = []
        malicious_client_ids: list[int] = []
        for slot in slots:
            if self.population.contains(slot.client_id):
                update = self.malicious_generator.craft(
                    AttackContext(
                        client_id=slot.client_id,
                        round_index=request.state.round_index,
                        global_model=request.state.global_model,
                    ),
                    slot.rng,
                )
                if not update.is_malicious:
                    raise ValueError('malicious generator returned a non-malicious update')
                malicious_client_ids.append(slot.client_id)
            else:
                update = self.benign_trainer.train(
                    slot.client_id,
                    make_client_state(request.state, slot.rng),
                    slot.rng,
                )
                if update.is_malicious:
                    raise ValueError('benign trainer returned a malicious update')
            ordered_updates.append(update)

        private_diagnostics = {}
        if malicious_client_ids:
            private_diagnostics = {
                'malicious_client_ids': tuple(malicious_client_ids),
                'malicious_client_count': len(malicious_client_ids),
            }
        return finalize_round(
            request=request,
            parent_rng=rng,
            sampled_clients=tuple(slot.client_id for slot in slots),
            ordered_updates=ordered_updates,
            aggregator=self.aggregator,
            server_optimizer=self.server_optimizer,
            private_diagnostics=private_diagnostics,
        )
