"""
Projected Gradient Descent (PGD) Attack

PGD is an iterative first-order adversarial attack that extends FGSM by repeatedly
applying small gradient-based updates and projecting the perturbed input back onto
a constrained threat region (ε-ball). This iterative refinement enables PGD to find
stronger adversarial examples and makes it a standard robustness benchmark.

Mathematical Formula:
    x^{t+1} = Π_{B_ε(x)} ( x^t + α · sign(∇_x J(θ, x^t, y)) )

Where:
    - x^t: Adversarial example at iteration t
    - Π_{B_ε(x)}: Projection onto the L∞ ε-ball centered at original input x
    - α (alpha): Step size
    - ∇_x J(θ, x^t, y): Gradient of loss w.r.t. current adversarial input
    - ε (epsilon): Maximum allowed L∞ perturbation

Key Characteristics:
    1. Iterative refinement (stronger than FGSM)
    2. Projection ensures perturbation constraint
    3. Optional random initialization within ε-ball

Applicable Architectures: CNN and Vision Transformer (Universal Attack)

Reference:
    Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (2018)
"""

import torch
import torch.nn as nn
from typing import Dict

from ..utils import ParamSpec, AttackSpec, normalize, denormalize

ATTACK_SPEC = AttackSpec(
    id="pgd",
    name="PGD",
    description="Projected Gradient Descent - Iterative attack with projection",
    category="universal",
    params=[
        ParamSpec(
            name="epsilon",
            type="float",
            default=0.031,  # 8/255 ≈ 0.031, standard ImageNet benchmark (Madry 2018)
            min=0.001,
            max=0.1,        # Beyond 0.1 perturbations become very visible
            step=0.001,
            description="Maximum perturbation magnitude (L∞ bound, 8/255 recommended)"
        ),
        ParamSpec(
            name="alpha",
            type="float",
            default=0.008,  # 2/255 ≈ 0.008, commonly epsilon/4 (Madry 2018)
            min=0.001,
            max=0.05,
            step=0.001,
            description="Step size per iteration (2/255 recommended)"
        ),
        ParamSpec(
            name="steps",
            type="int",
            default=20,     # 7-50 steps typical, 20 is good balance (Madry 2018)
            min=5,
            max=100,
            step=5,
            description="Number of attack iterations (20-40 recommended)"
        ),
        ParamSpec(
            name="random_start",
            type="bool",
            default=True,
            description="Initialize with random perturbation within ε-ball"
        )
    ]
)

def pgd_attack(
    model: nn.Module,
    image: torch.Tensor,
    label: torch.Tensor,
    epsilon: float,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    PGD attack (ResNet / ViT compatible, ImageNet normalized space).

    Args:
        model: Target model (ResNet / ViT)
        image: Input image tensor in pixel space [0, 1], shape (1, 3, H, W)
        label: True label tensor
        epsilon: Maximum L∞ perturbation magnitude (normalized space)

        **kwargs:
            - alpha: Step size per iteration (default: epsilon / 4)
            - steps: Number of PGD iterations (default: 10)
            - device: torch.device
            - random_start: Whether to initialize randomly within ε-ball (default: True)
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
    alpha = kwargs.get("alpha", epsilon / 4)
    steps = kwargs.get("steps", 10)
    random_start = kwargs.get("random_start", True)
    targeted = kwargs.get("targeted", False)
    target_label = kwargs.get("target_label", None)

    # 2. Prepare model and original input
    model.eval()                                                # Freeze BatchNorm / Dropout
    image = image.clone().detach().to(device)                   # Original clean image (pixel space)
    label = label.to(device)
    true_label_val = label.item()

    # Normalize original image (attack is performed in model input space)
    original_norm = normalize(image).detach()

    # 3. Initialize adversarial example
    if random_start:
        # Random initialization within L∞ ε-ball (normalized space)
        delta = torch.empty_like(original_norm).uniform_(-epsilon, epsilon)
        adv_norm = original_norm + delta
    else:
        adv_norm = original_norm.clone()

    # Enforce valid pixel range after initialization
    adv_norm = normalize(
        torch.clamp(denormalize(adv_norm), 0.0, 1.0)
    ).detach()

    # Track first and last successful misclassification
    first_adv_norm = None
    last_adv_norm = None

    # 4. Iterative PGD refinement
    for _ in range(steps):
        adv_norm.requires_grad = True

        # Forward and backward pass
        with torch.enable_grad():
            output = model(adv_norm)

            # Check for misclassification and track results
            with torch.no_grad():
                pred_class = output.argmax(dim=1).item()
                if pred_class != true_label_val:
                    # Save first misclassification
                    if first_adv_norm is None:
                        first_adv_norm = adv_norm.clone().detach()
                    # Always update last misclassification
                    last_adv_norm = adv_norm.clone().detach()

            if targeted and target_label is not None:
                target = torch.tensor([target_label], device=device)
                loss = nn.CrossEntropyLoss()(output, target)
                loss = -loss                                      # Targeted PGD
            else:
                loss = nn.CrossEntropyLoss()(output, label)       # Untargeted PGD

        # Clear stale gradients
        if adv_norm.grad is not None:
            adv_norm.grad.zero_()
        model.zero_grad()

        loss.backward()

        # 5. Gradient update + projection
        with torch.no_grad():
            # Gradient ascent step in L∞ direction
            adv_norm = adv_norm + alpha * adv_norm.grad.sign()

            # Project back onto ε-ball around original input
            delta = torch.clamp(adv_norm - original_norm, -epsilon, epsilon)
            adv_norm = original_norm + delta

            # Enforce valid pixel range [0, 1]
            adv_norm = normalize(
                torch.clamp(denormalize(adv_norm), 0.0, 1.0)
            ).detach()

    # Final check after last iteration
    with torch.no_grad():
        output = model(adv_norm)
        pred_class = output.argmax(dim=1).item()
        if pred_class != true_label_val:
            if first_adv_norm is None:
                first_adv_norm = adv_norm.clone().detach()
            last_adv_norm = adv_norm.clone().detach()

    # 6. Prepare return values
    # Use last successful result if available, otherwise use final iteration
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