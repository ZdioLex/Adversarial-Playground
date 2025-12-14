"""
Texture-Based Attack (CNN-Specific)

CNNs heavily rely on texture information for classification.
This attack generates texture-like perturbations that exploit this bias.

Reference: Geirhos et al., "ImageNet-trained CNNs are biased towards textures" (2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from ..utils import ParamSpec, AttackSpec, normalize

ATTACK_SPEC = AttackSpec(
    id="texture",
    name="Texture",
    description="Exploits CNN texture bias with pattern-based perturbations",
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
            name="patch_size",
            type="int",
            default=8,
            min=4,
            max=32,
            step=1,
            description="Texture patch size"
        ),
        ParamSpec(
            name="steps",
            type="int",
            default=20,
            min=1,
            max=100,
            step=1,
            description="Number of attack iterations"
        )
    ]
)


def texture_attack(
    model: nn.Module,
    image: torch.Tensor,
    label: torch.Tensor,
    epsilon: float,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Texture-based adversarial attack exploiting CNN texture bias.

    Args:
        model: Target CNN model
        image: Input image tensor (1, 3, 224, 224) in range [0, 1]
        label: True label tensor
        epsilon: Perturbation magnitude
        **kwargs: Additional arguments:
            - patch_size: Size of texture patches (default: 8)
            - steps: Number of iterations (default: 20)
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
    patch_size = kwargs.get('patch_size', 8)
    steps = kwargs.get('steps', 20)

    original_image = image.clone().detach().to(device)
    label = label.to(device)

    h, w = image.shape[2], image.shape[3]
    true_label_val = label.item()

    # Initialize texture pattern perturbation
    # Use smaller resolution and upsample for texture-like patterns
    small_h, small_w = h // patch_size, w // patch_size
    texture_pattern = torch.zeros(1, 3, small_h, small_w, device=device)

    # Track first and last successful misclassification
    first_adv = None
    last_adv = None

    for step in range(steps):
        # Upsample texture pattern to full resolution
        perturbation = F.interpolate(
            texture_pattern,
            size=(h, w),
            mode='nearest'
        )

        # Scale perturbation
        perturbation = epsilon * torch.tanh(perturbation)

        # Apply perturbation
        adv_image = torch.clamp(original_image + perturbation, 0, 1)
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

        # Get gradient and downsample to texture pattern resolution
        grad = adv_image.grad.data
        grad_small = F.avg_pool2d(grad, kernel_size=patch_size)

        # Update texture pattern
        with torch.no_grad():
            texture_pattern = texture_pattern + (1.0 / steps) * grad_small.sign()
            # Keep pattern bounded
            texture_pattern = torch.clamp(texture_pattern, -3, 3)

    # Generate final perturbation
    with torch.no_grad():
        perturbation = F.interpolate(
            texture_pattern,
            size=(h, w),
            mode='nearest'
        )
        perturbation = epsilon * torch.tanh(perturbation)

        # Ensure perturbation is within epsilon ball
        perturbation = torch.clamp(perturbation, -epsilon, epsilon)

        adv_image = torch.clamp(original_image + perturbation, 0, 1)

        # Final check
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
