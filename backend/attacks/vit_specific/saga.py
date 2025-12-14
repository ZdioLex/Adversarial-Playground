"""
Self-Attention Gradient Attack (SAGA)

SAGA is a white-box attack designed to break ensemble defenses containing
both Vision Transformers and CNNs. It blends gradients from multiple models
and uses self-attention rollout to weight the ViT gradients.

Mathematical Formula:
    x_adv^(i+1) = x_adv^(i) + s * sign(G_blend(x_adv^(i)))

    G_blend(x) = Σ(k∈K) α_k * ∂L_k/∂x + Σ(v∈V) α_v * φ_v ⊙ ∂L_v/∂x

Where:
    - K: Set of CNN models
    - V: Set of Vision Transformer models
    - α_k, α_v: Weighting coefficients for each model
    - φ_v: Self-attention rollout map for ViT model v
    - ⊙: Element-wise Hadamard product

Self-Attention Rollout (φ_v):
    φ_v = ∏(l=1 to n_l)[Σ(i=1 to n_h)(0.5*W_att_l,i + 0.5*I)]

    This recursively multiplies attention weights across layers,
    accounting for skip connections with the identity matrix.

Reference:
    Mahmood et al., "On the Robustness of Vision Transformers to Adversarial Examples" (2021)
    https://arxiv.org/abs/2104.02610
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from ..utils import ParamSpec, AttackSpec, normalize

ATTACK_SPEC = AttackSpec(
    id="saga",
    name="SAGA",
    description="Self-Attention Gradient Attack - Blends gradients with attention rollout for ViT",
    category="vit",
    params=[
        ParamSpec(
            name="epsilon",
            type="float",
            default=0.031,
            min=0.001,
            max=0.1,
            step=0.001,
            description="Maximum perturbation magnitude (L∞ bound)"
        ),
        ParamSpec(
            name="alpha",
            type="float",
            default=0.008,
            min=0.001,
            max=0.05,
            step=0.001,
            description="Step size per iteration"
        ),
        ParamSpec(
            name="steps",
            type="int",
            default=20,
            min=5,
            max=100,
            step=5,
            description="Number of attack iterations"
        ),
        ParamSpec(
            name="attention_weight",
            type="float",
            default=0.5,
            min=0.0,
            max=1.0,
            step=0.1,
            description="Weight for attention-blended gradient (0=pure gradient, 1=full attention)"
        )
    ]
)


def saga_attack(
    model: nn.Module,
    image: torch.Tensor,
    label: torch.Tensor,
    epsilon: float,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Self-Attention Gradient Attack (SAGA) for Vision Transformers.

    This attack uses attention rollout to weight the gradient, making it
    more effective against ViT models by focusing perturbations on
    regions the model attends to.

    Args:
        model: Target ViT model
        image: Input image tensor in pixel space [0, 1], shape (1, 3, H, W)
        label: True label tensor
        epsilon: Maximum L∞ perturbation magnitude

        **kwargs:
            - alpha: Step size per iteration (default: epsilon / 4)
            - steps: Number of iterations (default: 20)
            - attention_weight: Weight for attention blending (default: 0.5)
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
    alpha = kwargs.get('alpha', epsilon / 4)
    steps = kwargs.get('steps', 20)
    attention_weight = kwargs.get('attention_weight', 0.5)

    model.eval()
    original_image = image.clone().detach().to(device)
    label = label.to(device)
    true_label_val = label.item()

    # Initialize adversarial image
    adv_image = original_image.clone().detach()

    # Track first and last successful misclassification
    first_adv = None
    last_adv = None

    # Compute attention rollout map once (it's based on clean image attention patterns)
    attention_map = _compute_attention_rollout(model, original_image, device)

    for step in range(steps):
        adv_image.requires_grad = True

        # Forward pass
        adv_image_norm = normalize(adv_image)
        output = model(adv_image_norm)

        # Check for misclassification
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

            # Apply SAGA: blend gradient with attention map
            # G_blend = (1 - attention_weight) * grad + attention_weight * (φ ⊙ grad)
            blended_grad = _blend_gradient_with_attention(
                grad, attention_map, attention_weight
            )

            # Update adversarial image using sign of blended gradient
            adv_image = adv_image + alpha * blended_grad.sign()

            # Project to epsilon ball (L∞)
            perturbation = torch.clamp(adv_image - original_image, -epsilon, epsilon)
            adv_image = original_image + perturbation

            # Clamp to valid pixel range
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


def _compute_attention_rollout(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Compute attention rollout map following Equation (6) from the SAGA paper.

    Attention Rollout:
        φ = ∏(l=1 to n_l)[Σ(i=1 to n_h)(0.5*W_att_l,i + 0.5*I)]

    This recursively multiplies attention weights across layers,
    accounting for skip connections by adding 0.5*I (identity).

    Args:
        model: ViT model
        image: Input image tensor
        device: Computation device

    Returns:
        Attention map tensor resized to image dimensions (1, 3, H, W)
    """
    attention_weights = []

    def attention_hook(module, input, output):
        """Hook to capture attention weights from ViT attention layers."""
        # Try to get attention weights from different ViT implementations
        if hasattr(module, 'attn_weights') and module.attn_weights is not None:
            attention_weights.append(module.attn_weights.detach())
        elif isinstance(output, tuple) and len(output) > 1:
            # Some implementations return (output, attention_weights)
            if output[1] is not None:
                attention_weights.append(output[1].detach())

    # For timm ViTs with fused attention, we need to modify the forward
    # to capture attention weights
    hooks = []
    attention_modules = []

    for name, module in model.named_modules():
        # Match attention modules in timm ViT
        if 'attn' in name.lower() and hasattr(module, 'forward'):
            if 'drop' not in name and 'proj' not in name:
                attention_modules.append((name, module))

    # Try to capture attention by hooking into softmax or attention computation
    captured_attentions = []

    def qkv_hook(module, input, output):
        """Hook for capturing QKV output to compute attention manually."""
        captured_attentions.append(('qkv', output.detach()))

    def softmax_hook(module, input, output):
        """Hook for capturing softmax output (attention weights)."""
        if output.dim() >= 3:  # Likely attention weights
            captured_attentions.append(('softmax', output.detach()))

    # Register hooks on relevant modules
    for name, module in model.named_modules():
        if 'attn' in name.lower():
            if hasattr(module, 'qkv'):
                hooks.append(module.qkv.register_forward_hook(qkv_hook))
            if 'softmax' in name.lower() or isinstance(module, nn.Softmax):
                hooks.append(module.register_forward_hook(softmax_hook))

    # Forward pass to capture attention
    with torch.no_grad():
        image_norm = normalize(image)
        _ = model(image_norm)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Process captured attention weights
    if len(captured_attentions) > 0:
        # Try to extract attention weights from captured data
        attn_list = []
        for name, data in captured_attentions:
            if name == 'softmax' and data.dim() >= 3:
                attn_list.append(data)

        if len(attn_list) > 0:
            # Compute attention rollout
            rollout = _compute_rollout_from_attention_list(attn_list, device)
            if rollout is not None:
                return _resize_attention_to_image(rollout, image.shape[2:], device)

    # Fallback: compute gradient-based saliency as attention proxy
    return _compute_gradient_saliency_map(model, image, device)


def _compute_rollout_from_attention_list(
    attention_list: list,
    device: torch.device
) -> Optional[torch.Tensor]:
    """
    Compute attention rollout from a list of attention weight matrices.

    Following Equation (6):
        φ = ∏(l=1 to n_l)[Σ(i=1 to n_h)(0.5*W_att_l,i + 0.5*I)]

    Args:
        attention_list: List of attention weight tensors [B, H, N, N]
        device: Computation device

    Returns:
        Rollout attention map or None if computation fails
    """
    if len(attention_list) == 0:
        return None

    try:
        # Initialize rollout with identity
        first_attn = attention_list[0]

        # Determine dimensions
        if first_attn.dim() == 4:  # [B, H, N, N]
            num_tokens = first_attn.shape[-1]
        elif first_attn.dim() == 3:  # [H, N, N] or [B, N, N]
            num_tokens = first_attn.shape[-1]
        else:
            return None

        rollout = torch.eye(num_tokens, device=device)

        for attn in attention_list:
            # Average over heads: Σ(i=1 to n_h)(W_att_l,i) / n_h
            if attn.dim() == 4:  # [B, H, N, N]
                attn_avg = attn.mean(dim=1).squeeze(0)  # [N, N]
            elif attn.dim() == 3:  # [H, N, N]
                attn_avg = attn.mean(dim=0)  # [N, N]
            else:
                attn_avg = attn

            # Add identity for skip connection: 0.5*W + 0.5*I
            identity = torch.eye(attn_avg.shape[-1], device=device)
            attn_with_skip = 0.5 * attn_avg + 0.5 * identity

            # Multiply into rollout
            rollout = torch.matmul(attn_with_skip, rollout)

        # Extract attention from CLS token to all patches
        # CLS token is usually at position 0
        cls_attention = rollout[0, 1:]  # Skip CLS token itself

        return cls_attention

    except Exception:
        return None


def _compute_gradient_saliency_map(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Compute gradient-based saliency map as fallback attention proxy.

    This serves as a reasonable approximation when attention weights
    cannot be directly extracted from the model.

    Args:
        model: Target model
        image: Input image tensor
        device: Computation device

    Returns:
        Saliency map tensor (1, 3, H, W)
    """
    image_var = image.clone().detach().requires_grad_(True)

    # Forward pass
    image_norm = normalize(image_var)
    output = model(image_norm)

    # Backward from max logit
    score = output.max()
    model.zero_grad()
    score.backward()

    # Compute saliency
    saliency = image_var.grad.abs()

    # Normalize to [0, 1]
    saliency_min = saliency.min()
    saliency_max = saliency.max()
    if saliency_max > saliency_min:
        saliency = (saliency - saliency_min) / (saliency_max - saliency_min)
    else:
        saliency = torch.ones_like(saliency)

    return saliency.detach()


def _resize_attention_to_image(
    attention: torch.Tensor,
    image_size: tuple,
    device: torch.device
) -> torch.Tensor:
    """
    Resize attention map to match image dimensions.

    Args:
        attention: Attention vector for patches
        image_size: Target (H, W) size
        device: Computation device

    Returns:
        Resized attention map (1, 3, H, W)
    """
    num_patches = attention.shape[0]
    patch_grid_size = int(num_patches ** 0.5)

    if patch_grid_size * patch_grid_size != num_patches:
        # Cannot form square grid, return uniform map
        return torch.ones(1, 3, image_size[0], image_size[1], device=device)

    # Reshape to 2D grid
    attention_2d = attention.reshape(patch_grid_size, patch_grid_size)

    # Normalize
    attn_min = attention_2d.min()
    attn_max = attention_2d.max()
    if attn_max > attn_min:
        attention_2d = (attention_2d - attn_min) / (attn_max - attn_min)

    # Resize to image dimensions
    attention_resized = F.interpolate(
        attention_2d.unsqueeze(0).unsqueeze(0),
        size=image_size,
        mode='bilinear',
        align_corners=False
    )

    # Expand to 3 channels
    attention_resized = attention_resized.expand(-1, 3, -1, -1)

    return attention_resized


def _blend_gradient_with_attention(
    gradient: torch.Tensor,
    attention_map: torch.Tensor,
    weight: float
) -> torch.Tensor:
    """
    Blend gradient with attention map following SAGA formulation.

    G_blend = (1 - weight) * grad + weight * (φ ⊙ grad)

    This focuses the gradient on high-attention regions while
    maintaining some gradient flow in low-attention areas.

    Args:
        gradient: Raw gradient tensor (1, 3, H, W)
        attention_map: Attention map tensor (1, 3, H, W)
        weight: Blending weight [0, 1]

    Returns:
        Blended gradient tensor
    """
    # Ensure attention map matches gradient size
    if attention_map.shape != gradient.shape:
        attention_map = F.interpolate(
            attention_map,
            size=gradient.shape[2:],
            mode='bilinear',
            align_corners=False
        )

    # Blend: combine pure gradient with attention-weighted gradient
    # This follows the spirit of Equation (5) for a single ViT model
    blended = (1 - weight) * gradient + weight * (attention_map * gradient)

    return blended
