"""Paper RL simulator and Phase 1 distribution-learning assets."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "AttackerRLEnv",
    "ConvDenoisingAutoencoder",
    "KerasMnistAutoencoder",
    "PaperAttackerPolicyGymEnv",
    "PaperFLSimulator",
    "PaperGradientReconstructor",
    "ReconstructorConfig",
    "build_noisy_images",
    "craft_paper_malicious_update",
    "decode_paper_action",
    "estimate_aggregate_gradient",
    "load_keras_mnist_autoencoder",
    "load_torch_denoiser",
    "save_torch_denoiser",
    "train_denoising_autoencoder",
    "write_distribution_artifacts",
]

_EXPORTS = {
    "AttackerRLEnv": ("fl_sandbox.attacks.rl_attacker.simulator.paper_env", "PaperAttackerPolicyGymEnv"),
    "PaperAttackerPolicyGymEnv": ("fl_sandbox.attacks.rl_attacker.simulator.paper_env", "PaperAttackerPolicyGymEnv"),
    "PaperFLSimulator": ("fl_sandbox.attacks.rl_attacker.simulator.paper_env", "PaperFLSimulator"),
    "craft_paper_malicious_update": (
        "fl_sandbox.attacks.rl_attacker.simulator.paper_env",
        "craft_paper_malicious_update",
    ),
    "decode_paper_action": ("fl_sandbox.attacks.rl_attacker.simulator.paper_env", "decode_paper_action"),
    "ConvDenoisingAutoencoder": (
        "fl_sandbox.attacks.rl_attacker.simulator.distribution_learning.denoiser",
        "ConvDenoisingAutoencoder",
    ),
    "KerasMnistAutoencoder": (
        "fl_sandbox.attacks.rl_attacker.simulator.distribution_learning.denoiser",
        "KerasMnistAutoencoder",
    ),
    "build_noisy_images": (
        "fl_sandbox.attacks.rl_attacker.simulator.distribution_learning.denoiser",
        "build_noisy_images",
    ),
    "load_keras_mnist_autoencoder": (
        "fl_sandbox.attacks.rl_attacker.simulator.distribution_learning.denoiser",
        "load_keras_mnist_autoencoder",
    ),
    "load_torch_denoiser": (
        "fl_sandbox.attacks.rl_attacker.simulator.distribution_learning.denoiser",
        "load_torch_denoiser",
    ),
    "save_torch_denoiser": (
        "fl_sandbox.attacks.rl_attacker.simulator.distribution_learning.denoiser",
        "save_torch_denoiser",
    ),
    "train_denoising_autoencoder": (
        "fl_sandbox.attacks.rl_attacker.simulator.distribution_learning.denoiser",
        "train_denoising_autoencoder",
    ),
    "PaperGradientReconstructor": (
        "fl_sandbox.attacks.rl_attacker.simulator.distribution_learning.core",
        "PaperGradientReconstructor",
    ),
    "ReconstructorConfig": (
        "fl_sandbox.attacks.rl_attacker.simulator.distribution_learning.core",
        "ReconstructorConfig",
    ),
    "estimate_aggregate_gradient": (
        "fl_sandbox.attacks.rl_attacker.simulator.distribution_learning.core",
        "estimate_aggregate_gradient",
    ),
    "write_distribution_artifacts": (
        "fl_sandbox.attacks.rl_attacker.simulator.distribution_learning.core",
        "write_distribution_artifacts",
    ),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

