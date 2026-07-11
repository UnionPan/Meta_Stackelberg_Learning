"""Model factories and canonical parameter conversion."""

from meta_stackelberg.federated.models.paper_mnist import PaperMNISTCNN
from meta_stackelberg.federated.models.paper_cifar import PaperCIFARResNet18
from meta_stackelberg.federated.models.parameters import TorchModelStateCodec

__all__ = ['PaperCIFARResNet18', 'PaperMNISTCNN', 'TorchModelStateCodec']
