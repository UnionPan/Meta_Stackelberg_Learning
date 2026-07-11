import numpy as np
import torch

from meta_stackelberg.federated.models.parameters import TorchParameterCodec
from meta_stackelberg.federated.models.tiny_cnn import TinyImageCNN
from meta_stackelberg.environments.model_tail import ModelTailObservationEncoder


def test_encoder_uses_final_two_parameter_tensors_in_stable_order() -> None:
    model = TinyImageCNN()
    codec = TorchParameterCodec()
    state = codec.capture(model)
    encoder = ModelTailObservationEncoder.from_model(model)
    observation = encoder.encode(state, round_index=2, horizon=8)

    parameters = tuple(model.named_parameters())
    assert encoder.parameter_names == tuple(name for name, _ in parameters[-2:])
    expected_size = sum(parameter.numel() for _, parameter in parameters[-2:])
    assert observation['model_tail'].shape == (expected_size,)
    assert observation['round_progress'].tolist() == [0.25]
    assert observation['model_tail'].dtype == np.float32
    assert np.all(np.isfinite(observation['model_tail']))
    assert not observation['model_tail'].flags.writeable


def test_attacker_observation_adds_count_and_current_3d_defender_action() -> None:
    model = TinyImageCNN()
    encoder = ModelTailObservationEncoder.from_model(model)
    base = encoder.encode(TorchParameterCodec().capture(model), round_index=0, horizon=8)
    attacker = encoder.attacker_observation(
        base,
        malicious_count=3,
        defender_raw_action=np.array([0.1, -0.2, 0.3], dtype=np.float32),
    )
    assert set(attacker) == {'model_tail', 'round_progress', 'malicious_count', 'defender_action'}
    assert attacker['malicious_count'].tolist() == [3.0]
    np.testing.assert_array_equal(
        attacker['defender_action'],
        np.array([0.1, -0.2, 0.3], dtype=np.float32),
    )


def test_encoder_rejects_invalid_horizon_count_action_and_model_structure() -> None:
    model = TinyImageCNN()
    encoder = ModelTailObservationEncoder.from_model(model)
    state = TorchParameterCodec().capture(model)
    for kwargs in ({'round_index': -1, 'horizon': 8}, {'round_index': 0, 'horizon': 0}):
        try:
            encoder.encode(state, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError('accepted invalid round coordinates')
    base = encoder.encode(state, round_index=0, horizon=8)
    try:
        encoder.attacker_observation(base, malicious_count=-1, defender_raw_action=np.zeros(3))
    except ValueError:
        pass
    else:
        raise AssertionError('accepted invalid malicious count')
