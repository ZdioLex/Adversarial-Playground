"""
Low-Frequency Attack (ViT-Specific)

ViT models are more vulnerable to low-frequency perturbations compared to CNNs.
This is because ViTs process images as patches and are less sensitive to
high-frequency local patterns.

Reference: "On the Adversarial Robustness of Vision Transformers"
"""

import torch
import torch.nn as nn
from typing import Dict

from ..utils import ParamSpec, AttackSpec, normalize

ATTACK_SPEC = AttackSpec(
    id="low_freq",
    name="Low-Frequency",
    description="Exploits ViT vulnerability to low-frequency perturbations",
    category="vit",
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
            default=20,
            min=5,
            max=100,
            step=5,
            description="Low-frequency threshold (lower = more low-freq)"
        ),
        ParamSpec(
            name="steps",
            type="int",
            default=20,
            min=1,
            max=100,
            step=1,
            description="Number of attack iterations"
        ),
        ParamSpec(
            name="alpha",
            type="float",
            default=0.003,
            min=0.001,
            max=0.01,
            step=0.001,
            description="Step size per iteration (default: epsilon/10)"
        )
    ]
)


def low_freq_attack(
    model: nn.Module,
    image: torch.Tensor,
    label: torch.Tensor,
    epsilon: float,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Low-frequency adversarial attack exploiting ViT's vulnerability
    to low-frequency perturbations.

    Args:
        model: Target ViT model
        image: Input image tensor (1, 3, 224, 224) in range [0, 1]
        label: True label tensor
        epsilon: Perturbation magnitude
        **kwargs: Additional arguments:
            - freq_threshold: Threshold for low-frequency components (default: 20)
            - steps: Number of iterations (default: 20)
            - alpha: Step size (default: epsilon/10)
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
    freq_threshold = kwargs.get('freq_threshold', 20)
    steps = kwargs.get('steps', 20)
    alpha = kwargs.get('alpha', epsilon / 10)

    original_image = image.clone().detach().to(device)
    label = label.to(device)

    h, w = image.shape[2], image.shape[3]

    # Create low-frequency mask
    low_freq_mask = _create_low_freq_mask(h, w, freq_threshold).to(device)
    true_label_val = label.item()

    adv_image = original_image.clone().detach()

    # Track first and last successful misclassification
    first_adv = None
    last_adv = None

    for step in range(steps):
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

        with torch.no_grad():
            grad = adv_image.grad.data

            # Filter gradient to keep only low-frequency components
            filtered_grad = _apply_low_freq_filter(grad, low_freq_mask)

            # Update adversarial image
            adv_image = adv_image + alpha * filtered_grad.sign()

            # Project to epsilon ball
            perturbation = torch.clamp(adv_image - original_image, -epsilon, epsilon)
            adv_image = original_image + perturbation

            # Clamp to valid range
            adv_image = torch.clamp(adv_image, 0, 1).detach()

    # Final check
    with torch.no_grad():
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

    # Prepare first/last results
    if first_adv is not None:
        first_perturbation = first_adv - original_image
    else:
        first_adv = final_adv
        first_perturbation = final_perturbation

    if last_adv is not None:
        last_perturbation = last_adv - original_image
    else:
        last_adv = final_adv
        last_perturbation = final_perturbation

    return {
        'adv_image': final_adv.detach(),
        'perturbation': final_perturbation.detach(),
        'first_adv_image': first_adv.detach(),
        'first_perturbation': first_perturbation.detach(),
        'last_adv_image': last_adv.detach(),
        'last_perturbation': last_perturbation.detach(),
    }


def _create_low_freq_mask(h: int, w: int, threshold: int) -> torch.Tensor:
    """
    Create a low-frequency mask in frequency domain.

    Args:
        h: Image height
        w: Image width
        threshold: Distance threshold from center (DC component)
                   Smaller values = more aggressive low-pass filter

    Returns:
        Mask where 1 indicates low-frequency regions
    """
    cy, cx = h // 2, w // 2
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')

    # Calculate distance from center
    dist = torch.sqrt((y - cy).float() ** 2 + (x - cx).float() ** 2)

    # Low-frequency: regions close to center
    mask = (dist <= threshold).float()

    # Apply Gaussian smoothing to avoid sharp cutoff artifacts
    sigma = threshold / 3
    gaussian = torch.exp(-dist ** 2 / (2 * sigma ** 2))
    mask = mask * gaussian

    # Normalize
    mask = mask / (mask.max() + 1e-8)

    return mask


def _apply_low_freq_filter(
    tensor: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Apply low-frequency filtering using FFT.

    Args:
        tensor: Input tensor (B, C, H, W)
        mask: Low-frequency mask

    Returns:
        Filtered tensor with only low-frequency components
    """
    batch_size, channels, h, w = tensor.shape
    result = torch.zeros_like(tensor)

    for b in range(batch_size):
        for c in range(channels):
            # Apply FFT
            freq = torch.fft.fft2(tensor[b, c])
            freq_shifted = torch.fft.fftshift(freq)

            # Apply low-frequency mask
            freq_filtered = freq_shifted * mask

            # Inverse FFT
            freq_unshifted = torch.fft.ifftshift(freq_filtered)
            result[b, c] = torch.fft.ifft2(freq_unshifted).real

    return result
