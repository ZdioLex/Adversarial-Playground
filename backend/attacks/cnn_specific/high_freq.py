"""
High-Frequency Perturbation Attack (CNN-Specific)

CNNs are particularly vulnerable to high-frequency perturbations due to their
local receptive fields and the way convolutional filters process spatial information.

This attack uses FFT to target only high-frequency components of the image.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict

from ..utils import ParamSpec, AttackSpec, normalize

ATTACK_SPEC = AttackSpec(
    id="high_freq",
    name="High-Frequency",
    description="Targets high-frequency components - CNNs are sensitive to these",
    category="cnn",
    params=[
        ParamSpec(
            name="epsilon",
            type="float",
            default=0.03,
            min=0.001,
            max=0.3,
            step=0.001,
            description="Maximum perturbation magnitude"
        ),
        ParamSpec(
            name="freq_threshold",
            type="int",
            default=30,
            min=5,
            max=200,
            step=5,
            description="Frequency threshold (higher = more high-freq)"
        ),
        ParamSpec(
            name="steps",
            type="int",
            default=10,
            min=1,
            max=100,
            step=1,
            description="Number of attack iterations"
        )
    ]
)


def high_freq_attack(
    model: nn.Module,
    image: torch.Tensor,
    label: torch.Tensor,
    epsilon: float,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    High-frequency perturbation attack targeting CNN vulnerabilities.

    Args:
        model: Target CNN model
        image: Input image tensor (1, 3, 224, 224) in range [0, 1]
        label: True label tensor
        epsilon: Perturbation magnitude
        **kwargs: Additional arguments:
            - freq_threshold: Threshold for high-frequency components (default: 30)
            - steps: Number of iterations (default: 10)
            - device: Computation device

    Returns:
        Dict with keys:
            - adv_image: Final adversarial image
            - perturbation: Final perturbation
            - first_adv_image: First successful misclassification
            - first_perturbation: Perturbation at first misclassification
            - last_adv_image: Last successful misclassification
            - last_perturbation: Perturbation at last misclassification
    """
    device = kwargs.get('device', image.device)
    freq_threshold = kwargs.get('freq_threshold', 30)
    steps = kwargs.get('steps', 10)

    # Clone original image
    original_image = image.clone().detach().to(device)
    label = label.to(device)

    # Create high-frequency mask
    h, w = image.shape[2], image.shape[3]
    mask = _create_high_freq_mask(h, w, freq_threshold).to(device)
    true_label_val = label.item()

    # Initialize perturbation
    perturbation = torch.zeros_like(original_image)

    # Track first and last successful misclassification
    first_adv = None
    last_adv = None

    for step in range(steps):
        adv_image = original_image + perturbation
        adv_image.requires_grad = True

        # Forward pass (normalize for ImageNet models)
        adv_image_norm = normalize(adv_image)
        output = model(adv_image_norm)

        # Check attack success & track first/last
        with torch.no_grad():
            pred_class = output.argmax(dim=1).item()

            if pred_class != true_label_val:
                if first_adv is None:
                    first_adv = adv_image.clone().detach()
                last_adv = adv_image.clone().detach()

        loss = nn.CrossEntropyLoss()(output, label)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Get gradient and transform to frequency domain
        grad = adv_image.grad.data

        with torch.no_grad():
            # Apply FFT to gradient for each channel
            grad_freq = _apply_fft_filter(grad, mask)

            # Update perturbation using filtered gradient
            perturbation = perturbation + (epsilon / steps) * grad_freq.sign()

            # Project to epsilon ball (L-infinity)
            perturbation = torch.clamp(perturbation, -epsilon, epsilon)

            # Enforce frequency constraint on perturbation itself
            perturbation = _apply_fft_filter(perturbation, mask)

            # Ensure adversarial image stays in valid range
            adv_image = torch.clamp(original_image + perturbation, 0, 1)
            perturbation = adv_image - original_image

    # Final check
    with torch.no_grad():
        adv_image = original_image + perturbation
        adv_image_norm = normalize(adv_image)
        output = model(adv_image_norm)
        pred_class = output.argmax(dim=1).item()
        if pred_class != true_label_val:
            if first_adv is None:
                first_adv = adv_image.clone().detach()
            last_adv = adv_image.clone().detach()

    # Prepare final result
    final_adv = last_adv if last_adv is not None else adv_image
    final_perturbation = final_adv - original_image

    # Prepare first/last results - only if misclassification actually occurred
    if first_adv is not None:
        first_perturbation = first_adv - original_image
    else:
        first_perturbation = None

    if last_adv is not None:
        last_perturbation = last_adv - original_image
    else:
        last_perturbation = None

    return {
        'adv_image': final_adv.detach(),
        'perturbation': final_perturbation.detach(),
        'first_adv_image': first_adv.detach() if first_adv is not None else None,
        'first_perturbation': first_perturbation.detach() if first_perturbation is not None else None,
        'last_adv_image': last_adv.detach() if last_adv is not None else None,
        'last_perturbation': last_perturbation.detach() if last_perturbation is not None else None,
    }


def _create_high_freq_mask(h: int, w: int, threshold: int) -> torch.Tensor:
    """
    Create a high-frequency mask in frequency domain.

    Args:
        h: Image height
        w: Image width
        threshold: Distance threshold from center (DC component)

    Returns:
        Binary mask where 1 indicates high-frequency regions
    """
    cy, cx = h // 2, w // 2
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')

    # Calculate distance from center
    dist = torch.sqrt((y - cy).float() ** 2 + (x - cx).float() ** 2)

    # High-frequency: regions far from center
    mask = (dist > threshold).float()

    return mask


def _apply_fft_filter(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply FFT filtering to keep only high-frequency components.

    Args:
        tensor: Input tensor (B, C, H, W)
        mask: High-frequency mask

    Returns:
        Filtered tensor with only high-frequency components
    """
    batch_size, channels, h, w = tensor.shape
    result = torch.zeros_like(tensor)

    for b in range(batch_size):
        for c in range(channels):
            # Apply FFT
            freq = torch.fft.fft2(tensor[b, c])
            freq_shifted = torch.fft.fftshift(freq)

            # Apply high-frequency mask
            freq_filtered = freq_shifted * mask

            # Inverse FFT
            freq_unshifted = torch.fft.ifftshift(freq_filtered)
            result[b, c] = torch.fft.ifft2(freq_unshifted).real

    return result
