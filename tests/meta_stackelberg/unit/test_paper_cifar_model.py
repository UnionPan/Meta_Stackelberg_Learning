import torch

from meta_stackelberg.environments.model_tail import ModelTailObservationEncoder
from meta_stackelberg.federated.models.paper_cifar import PaperCIFARResNet18
from meta_stackelberg.federated.models.parameters import TorchModelStateCodec


def test_paper_cifar_resnet18_outputs_ten_logits_and_has_5130_tail() -> None:
    model = PaperCIFARResNet18()
    output = model(torch.zeros(2, 3, 32, 32))
    assert output.shape == (2, 10)
    assert model.linear.in_features == 512
    encoder = ModelTailObservationEncoder.from_model(model)
    observation = encoder.encode(
        TorchModelStateCodec().capture(model), round_index=0, horizon=500,
    )
    assert encoder.parameter_names == ('linear.weight', 'linear.bias')
    assert observation['model_tail'].shape == (5130,)


def test_paper_cifar_resnet18_contains_and_roundtrips_batchnorm_buffers() -> None:
    model = PaperCIFARResNet18()
    codec = TorchModelStateCodec()
    state = codec.capture(model)
    parameter_count = codec.parameter_tensor_count(model)
    assert len(state.tensors) > parameter_count
    target = PaperCIFARResNet18()
    codec.load(target, state)
    for left, right in zip(model.buffers(), target.buffers()):
        if torch.is_floating_point(left):
            torch.testing.assert_close(left, right)
