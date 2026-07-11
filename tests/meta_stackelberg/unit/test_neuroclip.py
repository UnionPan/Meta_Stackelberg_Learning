import copy
import math

import torch

from meta_stackelberg.security.defenses.neuroclip import NeuroClipCopy


class ProbeCNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(2.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv(inputs)).flatten(start_dim=1)


def test_neuroclip_clamps_conv_activations_on_independent_model_copy() -> None:
    model = ProbeCNN()
    original_state = copy.deepcopy(model.state_dict())
    defended = NeuroClipCopy(model, epsilon=1.0)
    inputs = torch.ones(2, 1, 1, 1)

    torch.testing.assert_close(model(inputs), torch.full((2, 1), 2.0))
    torch.testing.assert_close(defended(inputs), torch.ones(2, 1))
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original_state[name])
    assert not model.conv._forward_hooks


def test_neuroclip_copy_is_deterministic_and_does_not_share_parameters() -> None:
    model = ProbeCNN()
    first = NeuroClipCopy(model, epsilon=0.5)
    second = NeuroClipCopy(model, epsilon=0.5)
    with torch.no_grad():
        next(first.parameters()).zero_()
    assert not torch.equal(next(first.parameters()), next(second.parameters()))
    torch.testing.assert_close(second(torch.ones(1, 1, 1, 1)), torch.tensor([[0.5]]))


def test_neuroclip_rejects_invalid_epsilon_and_model() -> None:
    for value in (0.0, -1.0, math.nan, math.inf, True):
        try:
            NeuroClipCopy(ProbeCNN(), epsilon=value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f'accepted invalid epsilon {value!r}')
    try:
        NeuroClipCopy(object(), epsilon=1.0)  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError('accepted non-module model')
