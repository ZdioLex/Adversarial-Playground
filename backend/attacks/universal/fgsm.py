"""
Fast Gradient Sign Method (FGSM) Attack

FGSM is a single-step attack that uses the sign of the loss gradient with respect to
the input image to generate adversarial perturbations. This method exploits the
linearity of neural networks in high-dimensional spaces - even tiny per-pixel changes
accumulate into a significant shift in the model's output.

Mathematical Formula:
    x_adv = x + ε · sign(∇_x J(θ, x, y))

Where:
    - x: Original input image
    - x_adv: Adversarial example
    - ε (epsilon): Perturbation magnitude, controls the maximum strength of perturbation
    - sign(): Sign function, extracts the direction (positive/negative) of the gradient
    - ∇_x J(θ, x, y): Gradient of the loss function with respect to input x
    - θ: Model parameters
    - y: True label

Applicable Architectures: CNN and Vision Transformer (Universal Attack)

Reference: Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (2015)
"""

import torch
import torch.nn as nn
from typing import Tuple

from ..utils import ParamSpec, AttackSpec, normalize, denormalize

ATTACK_SPEC = AttackSpec(
    id="fgsm",
    name="FGSM",
    description="Fast Gradient Sign Method - Single-step attack using gradient sign",
    category="universal",
    params=[
        ParamSpec(
            name="epsilon",
            type="float",
            default=0.031,  # 8/255 ≈ 0.031, standard ImageNet benchmark (Goodfellow 2015)
            min=0.001,
            max=0.1,        # Beyond 0.1 perturbations become very visible
            step=0.001,
            description="Maximum perturbation magnitude (L∞ bound, 8/255 recommended)"
        )
    ]
)


def fgsm_attack(
    model: nn.Module,
    image: torch.Tensor,
    label: torch.Tensor,
    epsilon: float,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FGSM attack (ResNet / ViT compatible, ImageNet normalized space).

    Args:
        model: Target model (ResNet / ViT)
        image: Input image tensor in [0, 1], shape (1, 3, H, W)
        label: True label tensor
        epsilon: Perturbation magnitude (L∞, normalized space)
        **kwargs:
            - device: torch.device
            - targeted: bool
            - target_label: int

    Returns:
        adv_image_pixel: adversarial image in pixel space [0, 1]
        perturbation_norm: perturbation in normalized space
    """

    # 1. Parse kwargs for device and attack mode settings
    device = kwargs.get("device", image.device)
    targeted = kwargs.get("targeted", False)
    target_label = kwargs.get("target_label", None)

    # 2. Prepare model and data
    model.eval()                                                # Set model to eval mode， freeze batchnorm/dropout
    image = image.clone().detach().to(device)                   # Clone image and enable gradient computation
    image_norm = normalize(image)                               # Normalize to ImageNet space   
    image_norm.requires_grad = True                             # Enable gradient computation      
    label = label.to(device)                                    # Move label to device


    # 3. Zero gradients
    if image_norm.grad is not None:
        image_norm.grad.zero_()
    model.zero_grad()

    # 4. Forward and backward pass
    with torch.enable_grad():
        output = model(image_norm)

        if targeted and target_label is not None:
            target = torch.tensor([target_label], device=device)
            loss = nn.CrossEntropyLoss()(output, target)
            loss = -loss  # targeted FGSM
        else:
            loss = nn.CrossEntropyLoss()(output, label)

        loss.backward()

    # 5. FGSM perturbation (normalized space)
    perturbation_norm = epsilon * image_norm.grad.sign()
    adv_image_norm = image_norm + perturbation_norm

    # 6. Convert back to pixel space
    adv_image_pixel = denormalize(adv_image_norm)
    adv_image_pixel = torch.clamp(adv_image_pixel, 0.0, 1.0)        # Clamp to valid pixel range [0, 1]

    return adv_image_pixel.detach(), perturbation_norm.detach()