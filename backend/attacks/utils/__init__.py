"""
Attack utility modules.

Provides base classes for attack specifications and re-exports normalization utilities.
"""

from .base import ParamSpec, AttackSpec

# Re-export normalization from main utils for convenience
# Attacks can import from here: from ..utils import normalize, denormalize
try:
    from utils.normalization import (
        IMAGENET_MEAN,
        IMAGENET_STD,
        normalize,
        denormalize,
        normalize_tensor,
        denormalize_tensor
    )
except ImportError:
    # Fallback if import fails (e.g., during isolated testing)
    from torchvision import transforms
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    denormalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1 / s for s in IMAGENET_STD]
    )
    normalize_tensor = None
    denormalize_tensor = None

__all__ = [
    'ParamSpec',
    'AttackSpec',
    'IMAGENET_MEAN',
    'IMAGENET_STD',
    'normalize',
    'denormalize',
    'normalize_tensor',
    'denormalize_tensor'
]
