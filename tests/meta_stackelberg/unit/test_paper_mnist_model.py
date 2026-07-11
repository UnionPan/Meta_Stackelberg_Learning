import torch

from meta_stackelberg.federated.models.paper_mnist import PaperMNISTCNN


def test_paper_mnist_cnn_uses_declared_8_6_5_kernels_and_ten_logits() -> None:
    model = PaperMNISTCNN()
    assert (
        model.conv1.kernel_size,
        model.conv2.kernel_size,
        model.conv3.kernel_size,
    ) == ((8, 8), (6, 6), (5, 5))
    output = model(torch.zeros(3, 1, 28, 28))
    assert output.shape == (3, 10)
    assert model.fc1.in_features == 128


def test_paper_mnist_factory_initialization_is_seed_replayable() -> None:
    with torch.random.fork_rng():
        torch.manual_seed(9)
        first = PaperMNISTCNN()
    with torch.random.fork_rng():
        torch.manual_seed(9)
        second = PaperMNISTCNN()
    assert all(torch.equal(left, right) for left, right in zip(
        first.parameters(), second.parameters(),
    ))
