"""
Carlini & Wagner (C&W) L2 Attack

This implementation closely follows the original formulation proposed by
Carlini & Wagner (2017). The attack explicitly optimizes for minimal L2
perturbation while enforcing misclassification via a logit-based objective.

Key characteristics:
    1. L2 distortion minimization (no explicit L∞ constraint)
    2. Logit-based adversarial loss (more stable than cross-entropy)
    3. Tanh-space change of variables to enforce pixel bounds
    4. Binary search over trade-off parameter c
    5. Targeted attack (as in the original paper)

Reference:
    Carlini & Wagner, "Towards Evaluating the Robustness of Neural Networks", 2017
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict

from ..utils import ParamSpec, AttackSpec, normalize, denormalize

ATTACK_SPEC = AttackSpec(
    id="cw",
    name="C&W",
    description="Carlini & Wagner L2 - Targeted attack to misclassify as specified class",
    category="universal",
    params=[
        ParamSpec(
            name="target_label",
            type="int",
            default=0,
            min=0,
            max=999,
            step=1,
            description="Target class index (0-999 for ImageNet)"
        ),
        ParamSpec(
            name="steps",
            type="int",
            default=1000,   # 1000 iterations per binary search step (Carlini 2017)
            min=100,
            max=10000,
            step=100,
            description="Optimization steps per binary search iteration (1000 recommended)"
        ),
        ParamSpec(
            name="kappa",
            type="float",
            default=0.0,    # κ=0 minimal, κ=40 high confidence (Carlini 2017)
            min=0.0,
            max=40.0,
            step=1.0,
            description="Confidence margin κ (0=minimal, 40=high confidence)"
        ),
        ParamSpec(
            name="lr",
            type="float",
            default=0.01,   # Standard Adam lr (Carlini 2017)
            min=0.001,
            max=0.1,
            step=0.001,
            description="Adam optimizer learning rate (0.01 recommended)"
        ),
        ParamSpec(
            name="c_init",
            type="float",
            default=0.001,  # Start small, binary search will find optimal (Carlini 2017)
            min=0.0001,
            max=1.0,
            step=0.0001,
            description="Initial trade-off constant c (start small)"
        ),
        ParamSpec(
            name="c_steps",
            type="int",
            default=9,      # 9 binary search steps standard (Carlini 2017)
            min=1,
            max=20,
            step=1,
            description="Binary search steps for c (9 recommended)"
        )
    ]
)

def cw_attack(
    model: nn.Module,
    image: torch.Tensor,
    target_label: int,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Carlini & Wagner targeted attack

    Args:
        model: Target model (ResNet / ViT)
        image: Input image tensor in pixel space [0,1], shape (1,3,H,W)
        target_label: Target class t for targeted attack

        **kwargs:
            - device: torch.device
            - kappa: Confidence margin κ (default: 0)
            - steps: Optimization steps per c (default: 1000)
            - lr: Adam learning rate (default: 0.01)
            - c_init: Initial c for binary search (default: 1e-3)
            - c_steps: Number of binary search steps (default: 9)

    Returns:
        Dict with keys:
            - adv_image: Final adversarial image
            - perturbation: Final perturbation
            - first_adv_image: First successful misclassification
            - first_perturbation: Perturbation at first misclassification
            - last_adv_image: Last successful misclassification
            - last_perturbation: Perturbation at last misclassification
    """

    # 1. Parse hyperparameters
    device = kwargs.get("device", image.device)
    kappa = kwargs.get("kappa", 0.0)
    steps = kwargs.get("steps", 1000)
    lr = kwargs.get("lr", 0.01)
    c_init = kwargs.get("c_init", 1e-3)
    c_steps = kwargs.get("c_steps", 9)

    # 2. Prepare model and input
    model.eval()                                                # Deterministic BN / Dropout
    image = image.clone().detach().to(device)

    # Normalize image (model input space)
    image_norm = normalize(image).detach()

    # Convert image to tanh-space (optimization variable)
    w_init = _inverse_tanh(image_norm)
    best_adv = None
    best_l2 = float("inf")

    # Track first and last successful misclassification
    first_adv = None
    last_adv = None

    # Binary search bounds for c
    c_lower = 0.0
    c_upper = float("inf")
    c = c_init

    # 3. Binary search over c
    for _ in range(c_steps):
        w = w_init.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([w], lr=lr)

        found_adv = False

        # 4. Inner optimization loop
        for _ in range(steps):
            with torch.enable_grad():
                adv_norm = _tanh(w)
                logits = model(adv_norm)

                # ----- Logit-based adversarial loss f(x') -----
                target = torch.tensor([target_label], device=device)
                target_logit = logits[0, target_label]
                other_logit = torch.max(
                    torch.cat([logits[0, :target_label], logits[0, target_label+1:]])
                )

                # Targeted C&W loss
                f_loss = torch.clamp(other_logit - target_logit + kappa, min=0)

                # L2 distortion
                l2_loss = torch.norm(adv_norm - image_norm)

                loss = l2_loss + c * f_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 5. Track best adversarial example
            with torch.no_grad():
                adv_norm_eval = _tanh(w)
                logits_eval = model(adv_norm_eval)
                pred = logits_eval.argmax(dim=1).item()

                if pred == target_label:
                    found_adv = True
                    l2 = torch.norm(adv_norm_eval - image_norm).item()

                    # Track first successful
                    if first_adv is None:
                        first_adv = adv_norm_eval.clone().detach()
                    # Always update last successful
                    last_adv = adv_norm_eval.clone().detach()

                    if l2 < best_l2:
                        best_l2 = l2
                        best_adv = adv_norm_eval.clone().detach()

        # 6. Update binary search bounds
        if found_adv:
            c_upper = min(c_upper, c)
            if c_upper < float("inf"):
                c = (c_lower + c_upper) / 2
        else:
            c_lower = max(c_lower, c)
            c = (c_lower + c_upper) / 2 if c_upper < float("inf") else c * 10

    # Convert final adversarial example back to pixel space
    if best_adv is None:
        best_adv = _tanh(w)

    adv_image_pixel = torch.clamp(denormalize(best_adv), 0.0, 1.0)
    perturbation = adv_image_pixel - image

    # Prepare first/last results - only if misclassification actually occurred
    if first_adv is not None:
        first_adv_pixel = torch.clamp(denormalize(first_adv), 0.0, 1.0)
        first_perturbation = first_adv_pixel - image
    else:
        first_adv_pixel = None
        first_perturbation = None

    if last_adv is not None:
        last_adv_pixel = torch.clamp(denormalize(last_adv), 0.0, 1.0)
        last_perturbation = last_adv_pixel - image
    else:
        last_adv_pixel = None
        last_perturbation = None

    return {
        'adv_image': adv_image_pixel.detach(),
        'perturbation': perturbation.detach(),
        'first_adv_image': first_adv_pixel.detach() if first_adv_pixel is not None else None,
        'first_perturbation': first_perturbation.detach() if first_perturbation is not None else None,
        'last_adv_image': last_adv_pixel.detach() if last_adv_pixel is not None else None,
        'last_perturbation': last_perturbation.detach() if last_perturbation is not None else None,
    }


# ==================== Tanh-space utilities ====================

def _tanh(w: torch.Tensor) -> torch.Tensor:
    """
    Map from tanh-space to normalized input space.

    Formula:
        x = (tanh(w) + 1) / 2
    """
    return (torch.tanh(w) + 1) / 2


def _inverse_tanh(x: torch.Tensor) -> torch.Tensor:
    """
    Map from normalized input space to tanh-space.

    Formula:
        w = arctanh(2x - 1)
    """
    x = torch.clamp(x, 1e-6, 1 - 1e-6)
    return torch.atanh(2 * x - 1)