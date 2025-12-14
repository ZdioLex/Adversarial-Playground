"""
Unified ImageNet Normalization Utilities

This module provides centralized normalization functions for ImageNet-pretrained models.
All attack modules should import from here instead of defining their own.
"""

import torch
from torchvision import transforms

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Normalize transform for converting [0, 1] pixel space to ImageNet normalized space
normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

# Denormalize transform for converting ImageNet normalized space back to [0, 1] pixel space
denormalize = transforms.Normalize(
    mean=[-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
    std=[1 / s for s in IMAGENET_STD]
)


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize a tensor from [0, 1] pixel space to ImageNet normalized space.

    Args:
        tensor: Image tensor with shape (N, C, H, W) or (C, H, W) in range [0, 1]

    Returns:
        Normalized tensor in ImageNet normalized space
    """
    if tensor.dim() == 3:
        return normalize(tensor)
    elif tensor.dim() == 4:
        return torch.stack([normalize(t) for t in tensor])
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {tensor.dim()}D")


def denormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize a tensor from ImageNet normalized space to [0, 1] pixel space.

    Args:
        tensor: Normalized image tensor with shape (N, C, H, W) or (C, H, W)

    Returns:
        Denormalized tensor in range [0, 1]
    """
    if tensor.dim() == 3:
        return denormalize(tensor)
    elif tensor.dim() == 4:
        return torch.stack([denormalize(t) for t in tensor])
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {tensor.dim()}D")


def get_mean_std_tensors(device: torch.device = None):
    """
    Get mean and std as tensors for manual normalization operations.

    Args:
        device: Target device for tensors

    Returns:
        Tuple of (mean, std) tensors with shape (1, 3, 1, 1)
    """
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    if device is not None:
        mean = mean.to(device)
        std = std.to(device)

    return mean, std
