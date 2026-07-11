import numpy as np
import pytest

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.federated.types import ClientUpdate
from meta_stackelberg.security.defenses.krum import Krum


def _update(client_id, values):
    return ClientUpdate(
        client_id,
        ModelState.from_tensors((np.asarray(values, dtype=np.float32),)),
        1,
    )


def test_krum_selects_deterministic_cluster_update_not_outlier() -> None:
    updates = (
        _update(0, [0.0, 0.0]),
        _update(1, [0.1, 0.0]),
        _update(2, [0.0, 0.1]),
        _update(3, [0.1, 0.1]),
        _update(4, [20.0, -20.0]),
    )
    result = Krum(byzantine_count=1).aggregate(updates)
    np.testing.assert_array_equal(result.vector(), updates[0].delta.vector())


def test_krum_rejects_population_too_small_for_byzantine_bound() -> None:
    with pytest.raises(ValueError, match='2f'):
        Krum(byzantine_count=2).aggregate(tuple(
            _update(index, [float(index)]) for index in range(6)
        ))
