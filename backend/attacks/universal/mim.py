"""
Momentum Iterative Method (MIM) Attack

MIM enhances iterative gradient-based attacks by introducing momentum, which
accumulates normalized gradients across iterations. This stabilizes the update
direction, reduces oscillation, and significantly improves adversarial transferability
across different model architectures.

Mathematical Formula:
    g_{t+1} = μ · g_t + ∇_x J(θ, x^t, y) / ||∇_x J(θ, x^t, y)||_1
    x^{t+1} = x^t + α · sign(g_{t+1})

Where:
    - g_t: Momentum term (accumulated gradients)
    - μ (decay): Momentum decay factor
    - ||·||_1: L1 norm used for gradient normalization
    - α (alpha): Step size
    - ε (epsilon): L∞ perturbation budget

Applicable Architectures:
    CNNs and Vision Transformers (Universal Attack)

Reference:
    Dong et al., "Boosting Adversarial Attacks with Momentum" (CVPR 2018)
"""

import torch
import torch.nn as nn
from typing import Dict

from ..utils import ParamSpec, AttackSpec, normalize, denormalize

ATTACK_SPEC = AttackSpec(
    id="mim",
    name="MIM",
    description="Momentum Iterative Method - Iterative attack with momentum for transferability",
    category="universal",
    params=[
        ParamSpec(
            name="epsilon",
            type="float",
            default=0.031,  # 8/255 for untargeted, 16/255 for targeted (Dong 2018)
            min=0.001,
            max=0.1,
            step=0.001,
            description="Maximum perturbation magnitude (L∞ bound, 8/255 recommended)"
        ),
        ParamSpec(
            name="alpha",
            type="float",
            default=0.003,  # epsilon/steps ≈ 0.031/10 (Dong 2018)
            min=0.001,
            max=0.05,
            step=0.001,
            description="Step size per iteration (epsilon/steps recommended)"
        ),
        ParamSpec(
            name="steps",
            type="int",
            default=10,     # 10 iterations standard in original paper (Dong 2018)
            min=5,
            max=50,
            step=5,
            description="Number of attack iterations (10 recommended)"
        ),
        ParamSpec(
            name="decay",
            type="float",
            default=1.0,    # μ=1.0 optimal for transferability (Dong 2018)
            min=0.0,
            max=1.5,
            step=0.1,
            description="Momentum decay factor μ (1.0 optimal for transfer)"
        )
    ]
)

def mim_attack(
    model: nn.Module,
    image: torch.Tensor,
    label: torch.Tensor,
    epsilon: float,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    MIM attack (ResNet / ViT compatible, ImageNet normalized space).

    Args:
        model: Target model (ResNet / ViT)
        image: Input image tensor in pixel space [0, 1], shape (1, 3, H, W)
        label: True label tensor
        epsilon: Maximum L∞ perturbation magnitude (normalized space)

        **kwargs:
            - alpha: Step size per iteration (default: epsilon / 10)
            - steps: Number of iterations (default: 10)
            - decay: Momentum decay factor μ (default: 1.0)
            - device: torch.device
            - targeted: Whether to perform targeted attack (default: False)
            - target_label: Target class for targeted attack

    Returns:
        Dict with keys:
            - adv_image: Final adversarial image
            - perturbation: Final perturbation
            - first_adv_image: First successful misclassification
            - first_perturbation: Perturbation at first misclassification
            - last_adv_image: Last successful misclassification
            - last_perturbation: Perturbation at last misclassification
    """

    # 1. Parse attack hyperparameters
    device = kwargs.get("device", image.device)
    alpha = kwargs.get("alpha", epsilon / 10)
    steps = kwargs.get("steps", 10)
    decay = kwargs.get("decay", 1.0)
    targeted = kwargs.get("targeted", False)
    target_label = kwargs.get("target_label", None)

    # 2. Prepare model and original input
    model.eval()                                                # Freeze BatchNorm / Dropout
    image = image.clone().detach().to(device)                   # Clean image (pixel space)
    label = label.to(device)
    true_label_val = label.item()

    # Normalize original image (attack happens in model input space)
    original_norm = normalize(image).detach()

    # Initialize adversarial example (start from clean image)
    adv_norm = original_norm.clone().detach()

    # Initialize momentum term g_0 = 0
    momentum = torch.zeros_like(adv_norm)

    # Track first and last successful misclassification
    first_adv_norm = None
    last_adv_norm = None

    # Helper: enforce pixel-space box constraint via denorm → clamp → norm
    def project_to_pixel_bounds(x_norm: torch.Tensor) -> torch.Tensor:
        x_pix = denormalize(x_norm)
        x_pix = torch.clamp(x_pix, 0.0, 1.0)
        return normalize(x_pix).detach()

    # 3. Iterative momentum-based attack
    for _ in range(steps):
        adv_norm.requires_grad = True

        # Forward and backward pass
        with torch.enable_grad():
            output = model(adv_norm)

            # Check for misclassification and track results
            with torch.no_grad():
                pred_class = output.argmax(dim=1).item()
                if pred_class != true_label_val:
                    if first_adv_norm is None:
                        first_adv_norm = adv_norm.clone().detach()
                    last_adv_norm = adv_norm.clone().detach()

            if targeted and target_label is not None:
                target = torch.tensor([target_label], device=device)
                loss = nn.CrossEntropyLoss()(output, target)
                loss = -loss                                      # Targeted MIM
            else:
                loss = nn.CrossEntropyLoss()(output, label)       # Untargeted MIM

        # Clear stale gradients
        if adv_norm.grad is not None:
            adv_norm.grad.zero_()
        model.zero_grad()

        loss.backward()

        # 4. Normalize gradient and update momentum
        with torch.no_grad():
            grad = adv_norm.grad
            grad_norm = torch.norm(grad, p=1)

            if grad_norm > 0:
                grad = grad / grad_norm

            # g_{t+1} = μ · g_t + normalized_gradient
            momentum = decay * momentum + grad

            # 5. Signed momentum update
            adv_norm = adv_norm + alpha * momentum.sign()

            # Project back onto ε-ball around original input
            delta = torch.clamp(adv_norm - original_norm, -epsilon, epsilon)
            adv_norm = original_norm + delta

            # Enforce valid pixel range [0, 1]
            adv_norm = project_to_pixel_bounds(adv_norm)

    # Final check after last iteration
    with torch.no_grad():
        output = model(adv_norm)
        pred_class = output.argmax(dim=1).item()
        if pred_class != true_label_val:
            if first_adv_norm is None:
                first_adv_norm = adv_norm.clone().detach()
            last_adv_norm = adv_norm.clone().detach()

    # 6. Prepare return values
    if last_adv_norm is not None:
        final_adv_norm = last_adv_norm
    else:
        final_adv_norm = adv_norm

    adv_image_pixel = torch.clamp(denormalize(final_adv_norm), 0.0, 1.0)
    perturbation_pixel = adv_image_pixel - image

    # Prepare first/last results - only if misclassification actually occurred
    if first_adv_norm is not None:
        first_adv_pixel = torch.clamp(denormalize(first_adv_norm), 0.0, 1.0)
        first_perturbation = first_adv_pixel - image
    else:
        first_adv_pixel = None
        first_perturbation = None

    if last_adv_norm is not None:
        last_adv_pixel = torch.clamp(denormalize(last_adv_norm), 0.0, 1.0)
        last_perturbation = last_adv_pixel - image
    else:
        last_adv_pixel = None
        last_perturbation = None

    return {
        'adv_image': adv_image_pixel.detach(),
        'perturbation': perturbation_pixel.detach(),
        'first_adv_image': first_adv_pixel.detach() if first_adv_pixel is not None else None,
        'first_perturbation': first_perturbation.detach() if first_perturbation is not None else None,
        'last_adv_image': last_adv_pixel.detach() if last_adv_pixel is not None else None,
        'last_perturbation': last_perturbation.detach() if last_perturbation is not None else None,
    }