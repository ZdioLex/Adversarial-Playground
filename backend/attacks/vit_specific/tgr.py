"""
TGR - Token Gradient Regularization (ViT-Specific)

Reduces variance of back-propagated gradients in intermediate blocks of ViTs
by eliminating extreme token gradients, improving transferability.

This simplified implementation applies gradient regularization directly on
the input gradient rather than using backward hooks, which avoids compatibility
issues with different ViT architectures.

Reference: "Transferable Adversarial Attack for Both Vision Transformers and
Convolutional Networks via Token Gradient Regularization" (Wei et al., ICCV 2023)
"""

import torch
import torch.nn as nn
from typing import Dict

from ..utils import ParamSpec, AttackSpec, normalize

ATTACK_SPEC = AttackSpec(
    id="tgr",
    name="TGR",
    description="Token Gradient Regularization - Reduces gradient variance for better transferability",
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
            name="steps",
            type="int",
            default=10,
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
            description="Step size per iteration (default: epsilon/steps)"
        ),
        ParamSpec(
            name="top_k",
            type="int",
            default=1,
            min=1,
            max=10,
            step=1,
            description="Number of extreme gradient regions to suppress"
        ),
        ParamSpec(
            name="attn_scale",
            type="float",
            default=0.25,
            min=0.0,
            max=1.0,
            step=0.05,
            description="Scale factor for attention gradient"
        ),
        ParamSpec(
            name="mlp_scale",
            type="float",
            default=0.25,
            min=0.0,
            max=1.0,
            step=0.05,
            description="Scale factor for MLP gradient"
        ),
        ParamSpec(
            name="qkv_scale",
            type="float",
            default=0.75,
            min=0.0,
            max=1.0,
            step=0.05,
            description="Scale factor for QKV gradient"
        )
    ]
)


def _regularize_gradient(grad: torch.Tensor, top_k: int = 1) -> torch.Tensor:
    """
    Regularize gradient by suppressing extreme values.

    This simulates the effect of TGR by identifying and suppressing
    the most extreme gradient values in the spatial dimensions.

    Args:
        grad: Gradient tensor of shape (B, C, H, W)
        top_k: Number of extreme regions to suppress

    Returns:
        Regularized gradient
    """
    if grad is None:
        return grad

    grad = grad.clone()
    B, C, H, W = grad.shape

    # Compute gradient magnitude per spatial location
    # Shape: (B, H, W)
    grad_magnitude = grad.abs().mean(dim=1)

    # Flatten spatial dimensions for topk
    grad_flat = grad_magnitude.view(B, -1)  # (B, H*W)

    # Find top-k extreme locations (highest magnitude)
    k = min(top_k, grad_flat.shape[1] // 2)
    if k > 0:
        _, top_indices = torch.topk(grad_flat, k, dim=1, largest=True)
        _, bottom_indices = torch.topk(grad_flat, k, dim=1, largest=False)

        # Create mask to zero out extreme gradients
        for b in range(B):
            for idx in top_indices[b]:
                h_idx = idx // W
                w_idx = idx % W
                grad[b, :, h_idx, w_idx] *= 0.1  # Suppress rather than zero
            for idx in bottom_indices[b]:
                h_idx = idx // W
                w_idx = idx % W
                grad[b, :, h_idx, w_idx] *= 0.1

    return grad


def tgr_attack(
    model: nn.Module,
    image: torch.Tensor,
    label: torch.Tensor,
    epsilon: float,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    TGR (Token Gradient Regularization) attack.

    Reduces variance of back-propagated gradients by eliminating extreme
    gradient values for improved transferability.

    Args:
        model: Target ViT model
        image: Input image tensor (1, 3, 224, 224) in range [0, 1]
        label: True label tensor
        epsilon: Perturbation magnitude
        **kwargs: Additional arguments:
            - steps: Number of iterations (default: 10)
            - alpha: Step size (default: epsilon/steps)
            - top_k: Number of extreme regions to suppress (default: 1)
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
    steps = kwargs.get('steps', 10)
    alpha = kwargs.get('alpha', epsilon / steps)
    top_k = kwargs.get('top_k', 1)

    original_image = image.clone().detach().to(device)
    label = label.to(device)
    true_label_val = label.item()

    # Initialize adversarial image
    adv_image = original_image.clone().detach()

    # Momentum for MI-FGSM style update
    momentum = torch.zeros_like(original_image)
    decay = 1.0

    # Track first and last successful misclassification
    first_adv = None
    last_adv = None

    model.eval()

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

        # Compute loss
        loss = nn.CrossEntropyLoss()(output, label)

        # Backward pass
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad = adv_image.grad.data

            # Apply TGR-style gradient regularization
            grad = _regularize_gradient(grad, top_k)

            # Normalize gradient (L1 norm as in MI-FGSM)
            grad = grad / (grad.abs().mean() + 1e-12)

            # Update momentum
            momentum = decay * momentum + grad

            # Update adversarial image using sign of momentum
            adv_image = adv_image + alpha * momentum.sign()

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
