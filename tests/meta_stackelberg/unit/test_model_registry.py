import pytest
import torch


def test_model_registry_creates_fresh_named_models() -> None:
    from meta_stackelberg.federated.models.registry import ModelRegistry

    registry = ModelRegistry()
    registry.register('linear', lambda: torch.nn.Linear(2, 2))

    first = registry.create('linear')
    second = registry.create('linear')

    assert isinstance(first, torch.nn.Linear)
    assert first is not second
    with pytest.raises(KeyError, match='already registered'):
        registry.register('linear', lambda: torch.nn.Linear(2, 2))
    with pytest.raises(KeyError, match='unknown model'):
        registry.create('missing')


def test_model_registry_does_not_mask_factory_key_error() -> None:
    from meta_stackelberg.federated.models.registry import ModelRegistry

    def broken_factory() -> torch.nn.Module:
        raise KeyError('factory configuration is missing')

    registry = ModelRegistry()
    registry.register('broken', broken_factory)

    with pytest.raises(KeyError, match='factory configuration is missing'):
        registry.create('broken')


def test_model_registry_rejects_reused_live_instance() -> None:
    from meta_stackelberg.federated.models.registry import ModelRegistry

    shared = torch.nn.Linear(2, 2)
    registry = ModelRegistry()
    registry.register('shared', lambda: shared)

    assert registry.create('shared') is shared
    with pytest.raises(RuntimeError, match='fresh model instance'):
        registry.create('shared')


def test_tiny_image_cnn_has_expected_shape_and_no_persistent_buffers() -> None:
    from meta_stackelberg.federated.models.tiny_cnn import TinyImageCNN

    model = TinyImageCNN(num_classes=3)

    assert model(torch.zeros(5, 1, 8, 8)).shape == (5, 3)
    assert tuple(model.named_buffers()) == ()


def test_tiny_image_cnn_rejects_invalid_class_count() -> None:
    from meta_stackelberg.federated.models.tiny_cnn import TinyImageCNN

    with pytest.raises(ValueError, match='num_classes'):
        TinyImageCNN(num_classes=1)
