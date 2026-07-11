"""Fixed follower response compatible with the best-response interface."""

from meta_stackelberg.agents import IPMScalePolicy, IPMScalePolicySnapshot
from meta_stackelberg.stackelberg.best_response import BestResponseResult, SupportFactory
from meta_stackelberg.stackelberg.commitment import DefenderCommitment


class FixedIPMResponseOracle:
    def __init__(self, scale: float) -> None:
        self.snapshot = IPMScalePolicy(scale).snapshot()

    def solve(
        self,
        commitment: DefenderCommitment,
        follower_initialization: IPMScalePolicy,
        support_seeds: tuple[int, ...],
        support_factory: SupportFactory,
    ) -> BestResponseResult:
        del follower_initialization, support_seeds, support_factory
        commitment.verify()
        fingerprint = commitment.policy_fingerprint
        fixed = IPMScalePolicySnapshot(self.snapshot.scale)
        return BestResponseResult(
            fixed,
            fixed,
            (),
            0,
            fingerprint,
            commitment.policy_fingerprint,
            'fixed-ipm-response-v1',
        )
