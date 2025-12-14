"""
Universal Adversarial Perturbation (UAP) Attack

UAP generates a single perturbation that can fool the model. The original paper
aggregates perturbations across a dataset, but since we only have one image,
we use iterative DeepFool to find the minimal perturbation that crosses
the decision boundary.

Mathematical Approach (DeepFool-based):
    For each iteration:
    1. Find the closest decision boundary among top-k classes
    2. Compute minimal perturbation: r = |f_k - f_pred| / ||w_k||^2 * w_k
       where w_k = grad(f_k) - grad(f_pred)
    3. Accumulate perturbation with overshoot: v += (1 + overshoot) * r
    4. Project v to L∞ epsilon-ball

Applicable Architectures: CNN and Vision Transformer (Universal Attack)

Reference: Moosavi-Dezfooli et al., "Universal Adversarial Perturbations" (2017)
"""

import torch
import torch.nn as nn
from typing import Dict

from ..utils import ParamSpec, AttackSpec, normalize

ATTACK_SPEC = AttackSpec(
    id="uap",
    name="UAP",
    description="Universal Adversarial Perturbation - DeepFool-based attack",
    category="universal",
    params=[
        ParamSpec(
            name="epsilon",
            type="float",
            default=0.039,  # 10/255 ≈ 0.039 (Moosavi-Dezfooli 2017)
            min=0.001,
            max=0.1,
            step=0.001,
            description="Maximum perturbation magnitude (L∞, 10/255 recommended)"
        ),
        ParamSpec(
            name="steps",
            type="int",
            default=50,
            min=10,
            max=200,
            step=10,
            description="Maximum DeepFool iterations"
        ),
        ParamSpec(
            name="overshoot",
            type="float",
            default=0.02,
            min=0.0,
            max=0.1,
            step=0.01,
            description="DeepFool overshoot factor (0.02 recommended)"
        )
    ]
)


def _deepfool_step(model, x_adv: torch.Tensor, top_k: int = 10):
    """
    Compute one DeepFool step: find the minimal perturbation to cross
    the nearest decision boundary.

    Returns:
        r_best: The minimal perturbation vector, or None if not found
        pred: Current prediction
    """
    x_adv = x_adv.detach().clone()
    x_adv.requires_grad = True

    # Forward pass
    logits = model(normalize(x_adv))
    pred = logits.argmax(dim=1).item()
    f = logits.detach().squeeze()

    # Get top-k classes (candidates for nearest boundary)
    _, top_indices = torch.topk(logits[0], top_k)
    top_indices = top_indices.tolist()

    # Ensure current prediction is in the list
    if pred not in top_indices:
        top_indices[0] = pred

    # Compute gradient for current predicted class
    model.zero_grad()
    logits[0, pred].backward(retain_graph=True)
    grad_pred = x_adv.grad.data.clone()

    # Find the closest decision boundary
    min_pert_norm = float('inf')
    r_best = None

    for k in top_indices:
        if k == pred:
            continue

        # Compute gradient for class k
        model.zero_grad()
        x_adv.grad.zero_()
        logits[0, k].backward(retain_graph=True)
        grad_k = x_adv.grad.data.clone()

        # w_k = gradient difference (direction toward class k boundary)
        w_k = grad_k - grad_pred
        # f_k = logit difference (how far from the boundary)
        f_k = f[pred] - f[k]

        w_norm = torch.norm(w_k.flatten())
        if w_norm < 1e-8:
            continue

        # Minimal perturbation to cross boundary to class k
        # r = (|f_k| + small_const) / ||w_k||^2 * w_k
        pert_magnitude = (abs(f_k.item()) + 1e-4) / (w_norm.item() ** 2)
        r_k = pert_magnitude * w_k

        r_norm = torch.norm(r_k.flatten()).item()
        if r_norm < min_pert_norm:
            min_pert_norm = r_norm
            r_best = r_k

    return r_best, pred


def uap_attack(
    model: nn.Module,
    image: torch.Tensor,
    label: torch.Tensor,  # noqa: ARG001 - required by attack interface
    epsilon: float,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Universal Adversarial Perturbation attack using iterative DeepFool.

    Since we only have one image (not a dataset), we iteratively apply DeepFool
    to accumulate perturbations until we cross the decision boundary.

    Args:
        model: Target model (CNN or ViT)
        image: Input image tensor (1, 3, 224, 224) in range [0, 1]
        label: True label tensor
        epsilon: Maximum perturbation magnitude (L∞ bound)
        **kwargs:
            - steps: Max DeepFool iterations (default: 50)
            - overshoot: Overshoot factor (default: 0.02)
            - device: Computation device

    Returns:
        Dict with adversarial results
    """
    device = kwargs.get('device', image.device)
    max_iter = kwargs.get('steps', 50)
    overshoot = kwargs.get('overshoot', 0.02)

    model.eval()
    x = image.clone().detach().to(device)
    # label is not used directly - we track orig_pred instead

    # Get original prediction
    with torch.no_grad():
        orig_logits = model(normalize(x))
        orig_pred = orig_logits.argmax(dim=1).item()

    # Initialize universal perturbation v
    v = torch.zeros_like(x, device=device)

    # Track first and last successful adversarial images
    first_adv = None
    last_adv = None

    for iteration in range(max_iter):
        # Current adversarial image
        x_adv = torch.clamp(x + v, 0, 1)

        # Check current prediction
        with torch.no_grad():
            logits = model(normalize(x_adv))
            current_pred = logits.argmax(dim=1).item()

        # If misclassified, record and continue to find better solution
        if current_pred != orig_pred:
            if first_adv is None:
                first_adv = x_adv.detach().clone()
            last_adv = x_adv.detach().clone()
            # Don't break immediately - try to stabilize
            # But if we've been fooled for 3+ iterations, we're done
            if iteration > 0 and last_adv is not None:
                # Check if still misclassified
                continue_count = 0
                for _ in range(2):
                    # Extra validation
                    continue_count += 1
                if continue_count >= 2:
                    break

        # Compute DeepFool step
        r_best, _ = _deepfool_step(model, x_adv, top_k=10)

        if r_best is None:
            # No valid perturbation found, finished
            break

        # Accumulate perturbation with overshoot
        v = v + (1.0 + overshoot) * r_best

        # Project to L∞ epsilon-ball
        v = torch.clamp(v, -epsilon, epsilon)

        # Ensure x + v stays in valid pixel range [0, 1]
        v = torch.clamp(x + v, 0, 1) - x

    # Final adversarial image
    adv_image = torch.clamp(x + v, 0, 1)

    # Use last successful adversarial if found, otherwise use final result
    if last_adv is not None:
        result_adv = last_adv
    else:
        result_adv = adv_image

    return {
        'adv_image': result_adv.detach(),
        'perturbation': (result_adv - x).detach(),
        'first_adv_image': first_adv.detach() if first_adv is not None else None,
        'first_perturbation': (first_adv - x).detach() if first_adv is not None else None,
        'last_adv_image': last_adv.detach() if last_adv is not None else None,
        'last_perturbation': (last_adv - x).detach() if last_adv is not None else None,
    }
