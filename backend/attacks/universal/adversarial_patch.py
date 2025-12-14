"""
Adversarial Patch Attack (Enhanced Version)

Adversarial patches are localized, physically-deployable perturbations that can cause
misclassification when overlaid on any input image. This implementation includes:

1. EOT (Expectation over Transformations): Random rotation, scale, brightness
2. Random location per iteration: Patch learns to work anywhere
3. Margin-based loss: Maximize confidence gap, not just correct prediction
4. Circular patch support: More natural, rotation-invariant shape
5. Multi-objective loss: Push target while suppressing original class

Mathematical Formula (from Brown et al., 2017):
    p̂ = arg max_p E_{x,t,l}[log Pr(ŷ|A(p, x, l, t))]

Applicable Architectures: CNN and Vision Transformer (Universal Attack)

Reference: Brown et al., "Adversarial Patch" (2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Tuple, Optional
import math

from ..utils import ParamSpec, AttackSpec, normalize

ATTACK_SPEC = AttackSpec(
    id="adversarial_patch",
    name="Adversarial Patch",
    description="Targeted patch attack with EOT - physically realizable perturbation",
    category="universal",
    params=[
        ParamSpec(
            name="target_label",
            type="int",
            default=859,    # toaster - classic adversarial patch target (Brown 2017)
            min=0,
            max=999,
            step=1,
            description="Target class index (0-999 for ImageNet, 859=toaster)"
        ),
        ParamSpec(
            name="patch_size",
            type="int",
            default=64,     # ~28% of 224x224, effective size (Brown 2017)
            min=32,
            max=112,        # Max 50% of image
            step=8,
            description="Size of the adversarial patch (pixels, 64 recommended)"
        ),
        ParamSpec(
            name="steps",
            type="int",
            default=500,    # 250-1000 typical (Brown 2017)
            min=100,
            max=2000,
            step=100,
            description="Optimization steps (500 recommended)"
        ),
        ParamSpec(
            name="lr",
            type="float",
            default=0.1,    # 0.1-5.0 typical, start higher (Brown 2017)
            min=0.01,
            max=1.0,
            step=0.01,
            description="Learning rate for patch optimization (0.1 recommended)"
        ),
        ParamSpec(
            name="use_eot",
            type="bool",
            default=True,
            description="Use Expectation over Transformations (rotation ±30°, scale 0.8-1.2)"
        ),
        ParamSpec(
            name="circular",
            type="bool",
            default=True,
            description="Use circular patch shape (more rotation-invariant)"
        )
    ]
)


def adversarial_patch_attack(
    model: nn.Module,
    image: torch.Tensor,
    label: torch.Tensor,
    epsilon: float,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Enhanced Adversarial Patch attack with EOT and margin-based loss.

    Args:
        model: Target neural network model (CNN or ViT)
        image: Input image tensor, shape (1, 3, 224, 224), pixel values in range [0, 1]
        label: True label tensor
        epsilon: Not used directly (patch_size controls patch area)
        **kwargs: Additional arguments:
            - target_label: Target class for targeted attack (required)
            - patch_size: Size of the patch in pixels (default: 50)
            - steps: Number of optimization steps (default: 300)
            - lr: Learning rate for patch optimization (default: 0.05)
            - use_eot: Enable EOT transformations (default: True)
            - circular: Use circular patch (default: True)
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
    target_label = kwargs.get('target_label', 859)  # Default: toaster
    patch_size = kwargs.get('patch_size', 64)
    steps = kwargs.get('steps', 500)
    lr = kwargs.get('lr', 0.1)
    use_eot = kwargs.get('use_eot', True)
    circular = kwargs.get('circular', True)

    # Image dimensions
    _, _, H, W = image.shape
    # Ensure minimum patch size for effectiveness
    patch_size = max(patch_size, 32)
    patch_size = min(patch_size, min(H, W) - 4)

    # Prepare tensors
    original_image = image.clone().detach().to(device)
    label = label.to(device)
    true_label_val = label.item()
    target = torch.tensor([target_label], device=device)

    # Initialize patch with random values
    patch = torch.rand(1, 3, patch_size, patch_size, device=device, requires_grad=True)

    # Create circular mask if enabled
    if circular:
        mask = _create_circular_mask(patch_size, patch_size, device)
    else:
        mask = torch.ones(1, 1, patch_size, patch_size, device=device)

    # Optimizer
    optimizer = optim.Adam([patch], lr=lr)

    # Best result tracking
    best_patch = None
    best_confidence = 0.0
    best_location = (W // 2 - patch_size // 2, H // 2 - patch_size // 2)

    # Track first and last successful attack
    first_adv_image = None
    first_location = None
    first_patch = None
    last_adv_image = None
    last_location = None
    last_patch = None

    model.eval()

    for step in range(steps):
        optimizer.zero_grad()

        # Random location each iteration (key for robustness)
        max_x = W - patch_size
        max_y = H - patch_size
        patch_x = torch.randint(0, max_x + 1, (1,)).item()
        patch_y = torch.randint(0, max_y + 1, (1,)).item()

        # Apply EOT transformations
        if use_eot:
            transformed_patch = _apply_eot(patch, device)
        else:
            transformed_patch = patch

        # Clamp and apply mask
        clamped_patch = torch.clamp(transformed_patch, 0, 1)
        masked_patch = clamped_patch * mask

        # Apply patch to image
        adv_image = original_image.clone()
        # Blend patch with image using mask
        patch_region = adv_image[:, :, patch_y:patch_y + patch_size, patch_x:patch_x + patch_size]
        adv_image[:, :, patch_y:patch_y + patch_size, patch_x:patch_x + patch_size] = \
            masked_patch + patch_region * (1 - mask)

        # Forward pass (normalize for ImageNet models)
        adv_image_norm = normalize(adv_image)
        output = model(adv_image_norm)
        probs = torch.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()

        # Track best result and first/last successful
        target_conf = probs[0, target_label].item()
        if pred_class == target_label:
            # Track first successful
            if first_adv_image is None:
                first_patch = torch.clamp(patch.detach().clone(), 0, 1)
                first_location = (patch_x, patch_y)
                # Build first adv image
                first_adv_image = original_image.clone()
                first_masked = first_patch * mask
                first_region = first_adv_image[:, :, patch_y:patch_y + patch_size, patch_x:patch_x + patch_size]
                first_adv_image[:, :, patch_y:patch_y + patch_size, patch_x:patch_x + patch_size] = \
                    first_masked + first_region * (1 - mask)

            # Always update last successful
            last_patch = torch.clamp(patch.detach().clone(), 0, 1)
            last_location = (patch_x, patch_y)
            last_adv_image = adv_image.clone().detach()

            # Track best by confidence
            if target_conf > best_confidence:
                best_confidence = target_conf
                best_patch = torch.clamp(patch.detach().clone(), 0, 1)
                best_location = (patch_x, patch_y)

        # Early stopping
        if best_confidence > 0.95:
            break

        # Margin-based loss: maximize (target_logit - max_other_logit)
        logits = output[0]
        target_logit = logits[target_label]

        # Get max logit excluding target class
        mask_logits = logits.clone()
        mask_logits[target_label] = float('-inf')
        max_other_logit = mask_logits.max()

        # Multi-objective loss:
        # 1. Maximize target logit
        # 2. Suppress original class
        # 3. Maximize margin
        margin_loss = -(target_logit - max_other_logit)
        suppress_loss = logits[true_label_val]  # Minimize original class logit

        # Combined loss
        loss = margin_loss + 0.3 * suppress_loss

        loss.backward()
        optimizer.step()

        # Project patch to valid range
        with torch.no_grad():
            patch.data = torch.clamp(patch.data, 0, 1)

    # Generate final adversarial image using best location
    with torch.no_grad():
        if best_patch is not None:
            final_patch = best_patch * mask
            patch_x, patch_y = best_location
        else:
            # No successful attack found, use last patch at center
            final_patch = torch.clamp(patch, 0, 1) * mask
            patch_x = (W - patch_size) // 2
            patch_y = (H - patch_size) // 2

        adv_image = original_image.clone()
        patch_region = adv_image[:, :, patch_y:patch_y + patch_size, patch_x:patch_x + patch_size]
        adv_image[:, :, patch_y:patch_y + patch_size, patch_x:patch_x + patch_size] = \
            final_patch + patch_region * (1 - mask)

    perturbation = adv_image - original_image

    # Prepare first/last results - only if misclassification actually occurred
    if first_adv_image is not None:
        first_perturbation = first_adv_image - original_image
    else:
        first_perturbation = None

    if last_adv_image is not None:
        last_perturbation = last_adv_image - original_image
    else:
        last_perturbation = None

    return {
        'adv_image': adv_image.detach(),
        'perturbation': perturbation.detach(),
        'first_adv_image': first_adv_image.detach() if first_adv_image is not None else None,
        'first_perturbation': first_perturbation.detach() if first_perturbation is not None else None,
        'last_adv_image': last_adv_image.detach() if last_adv_image is not None else None,
        'last_perturbation': last_perturbation.detach() if last_perturbation is not None else None,
    }


def _apply_eot(patch: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Apply Expectation over Transformations (EOT) to the patch.

    Transformations:
    - Rotation: ±30 degrees
    - Scale: 0.8 - 1.2
    - Brightness: 0.8 - 1.2
    - Contrast: 0.9 - 1.1
    """
    B, C, H, W = patch.shape

    # Random rotation angle (-30 to +30 degrees)
    angle = (torch.rand(1).item() - 0.5) * 60  # degrees
    angle_rad = angle * math.pi / 180

    # Random scale (0.8 to 1.2)
    scale = 0.8 + torch.rand(1).item() * 0.4

    # Create affine transformation matrix
    cos_a = math.cos(angle_rad) * scale
    sin_a = math.sin(angle_rad) * scale

    theta = torch.tensor([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0]
    ], dtype=torch.float32, device=device).unsqueeze(0)

    # Apply affine transformation
    grid = F.affine_grid(theta, patch.size(), align_corners=False)
    transformed = F.grid_sample(patch, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    # Random brightness adjustment (0.8 to 1.2)
    brightness = 0.8 + torch.rand(1, device=device).item() * 0.4
    transformed = transformed * brightness

    # Random contrast adjustment (0.9 to 1.1)
    contrast = 0.9 + torch.rand(1, device=device).item() * 0.2
    mean = transformed.mean()
    transformed = (transformed - mean) * contrast + mean

    return transformed


def _create_circular_mask(h: int, w: int, device: torch.device) -> torch.Tensor:
    """
    Create a soft circular mask for more natural patch appearance.

    Uses a smooth edge to avoid sharp boundaries.
    """
    center_y, center_x = h // 2, w // 2
    radius = min(h, w) // 2 - 1

    Y, X = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing='ij'
    )

    dist = torch.sqrt((X - center_x).float() ** 2 + (Y - center_y).float() ** 2)

    # Soft edge (smooth transition over 3 pixels)
    mask = torch.clamp(1 - (dist - radius + 2) / 4, 0, 1)

    return mask.unsqueeze(0).unsqueeze(0)
