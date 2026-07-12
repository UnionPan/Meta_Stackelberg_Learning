"""Pinned global MNIST trigger used by the white-box backdoor task."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from meta_stackelberg.security.data.trigger import CompositeTrigger, PatchTrigger


@dataclass(frozen=True)
class MNISTGlobalTriggerFixture:
    """Immutable source/target task and normalized trigger provenance."""

    identifier: str
    source_class: int
    target_class: int
    pixels: tuple[tuple[int, int, float], ...]
    sha256: str

    @property
    def normalized_white(self) -> float:
        return (1.0 - 0.1307) / 0.3081

    @property
    def trigger(self) -> CompositeTrigger:
        return CompositeTrigger(tuple(
            PatchTrigger(row, column, 1, 1, value)
            for row, column, value in self.pixels
        ))

    def compute_sha256(self) -> str:
        payload = json.dumps(self.pixels, separators=(',', ':')).encode('utf-8')
        return hashlib.sha256(payload).hexdigest()


_NORMALIZED_WHITE = (1.0 - 0.1307) / 0.3081
_PIXELS = tuple(
    (row, column, _NORMALIZED_WHITE)
    for row in range(5, 7)
    for column in range(6, 11)
)
_FIXTURE = MNISTGlobalTriggerFixture(
    identifier='mnist-global-1-to-7-v1',
    source_class=1,
    target_class=7,
    pixels=_PIXELS,
    sha256='c5226726b8b70efa2f667d59784a81e6f3f7e1a359a53d067b2db086d570c93b',
)


def mnist_global_trigger() -> MNISTGlobalTriggerFixture:
    return _FIXTURE
