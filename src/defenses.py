"""
Post-training defense mechanisms for Federated Learning.

- NeuroClip: clamps activations at early conv layers to suppress backdoor triggers
- Pruning: zeros out low-activation neurons to remove backdoor pathways
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

from .utils.fl_utils import get_parameters, set_parameters


class NeuroClipDefense(nn.Module):
    """
    NeuroClip post-training defense.
    Wraps a CNN model and clamps activations at early layers.
    The clip range epsilon controls how tight the clamping is.
    Tighter clamping = more backdoor suppression but potential accuracy loss.

    For MNISTClassifier: clamps after conv1, conv2, conv3.
    """
    def __init__(self, base_model, clip_range=7.0):
        super().__init__()
        self.base_model = base_model
        self.clip_range = clip_range

    def forward(self, x):
        # MNISTClassifier forward with activation clamping
        x = F.relu(self.base_model.conv1(x))
        x = torch.clamp(x, max=self.clip_range)
        x = F.relu(self.base_model.conv2(x))
        x = torch.clamp(x, max=self.clip_range)
        x = F.relu(self.base_model.conv3(x))
        x = torch.clamp(x, max=self.clip_range)
        x = x.view(x.size(0), -1)
        x = self.base_model.fc1(x)
        return x


class PruningDefense(nn.Module):
    """
    Neural Pruning post-training defense.
    Identifies neurons with lowest average activation and zeros them out.
    The mask_rate sigma controls what fraction of neurons to prune.
    Higher mask_rate = more pruning = more backdoor suppression.

    For MNISTClassifier: prunes neurons in the last conv layer (conv3).
    """
    def __init__(self, base_model, mask_rate=0.1, eval_loader=None, device=None):
        super().__init__()
        self.base_model = base_model
        self.mask_rate = mask_rate
        self.device = device or torch.device("cpu")
        self.prune_mask = None

        if eval_loader is not None:
            self._compute_mask(eval_loader)

    def _compute_mask(self, eval_loader):
        """Compute pruning mask based on average activation magnitudes."""
        self.base_model.eval()
        activation_sums = None
        count = 0

        with torch.no_grad():
            for images, _ in eval_loader:
                images = images.to(self.device)
                x = F.relu(self.base_model.conv1(images))
                x = F.relu(self.base_model.conv2(x))
                x = self.base_model.conv3(x)  # before ReLU to see full activations

                # Sum activations per channel
                batch_sum = x.sum(dim=(0, 2, 3))  # sum over batch, H, W
                if activation_sums is None:
                    activation_sums = batch_sum
                else:
                    activation_sums += batch_sum
                count += images.size(0)

        activation_sums /= count

        # Create mask: zero out channels with lowest activations
        num_channels = activation_sums.shape[0]
        num_prune = max(1, int(num_channels * self.mask_rate))
        _, indices = torch.sort(activation_sums)
        prune_indices = indices[:num_prune]

        self.prune_mask = torch.ones(num_channels, 1, 1, device=self.device)
        self.prune_mask[prune_indices] = 0.0

    def forward(self, x):
        x = F.relu(self.base_model.conv1(x))
        x = F.relu(self.base_model.conv2(x))
        x = self.base_model.conv3(x)

        # Apply pruning mask
        if self.prune_mask is not None:
            x = x * self.prune_mask

        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.base_model.fc1(x)
        return x


def apply_post_defense(model, defense_type, param, eval_loader=None, device=None):
    """
    Apply a post-training defense to a model copy for evaluation.

    Args:
        model: the base nn.Module
        defense_type: "neuroclip" or "pruning"
        param: defense parameter (epsilon for neuroclip, sigma for pruning)
        eval_loader: data loader for computing pruning mask
        device: torch device

    Returns:
        defended model (nn.Module) — a wrapper, does NOT modify original
    """
    if defense_type == "neuroclip":
        return NeuroClipDefense(model, clip_range=max(0.1, float(param)))
    elif defense_type == "pruning":
        mask_rate = max(0.0, min(1.0, float(param)))
        defense = PruningDefense(model, mask_rate=mask_rate,
                                 eval_loader=eval_loader, device=device)
        return defense
    else:
        raise ValueError(f"Unknown defense type: {defense_type}")
