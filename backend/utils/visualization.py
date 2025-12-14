"""
Visualization Utilities

Provides functions for generating heatmaps, difference images,
and other visualizations for adversarial attack analysis.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import base64
import io
from PIL import Image


def generate_perturbation_heatmap(
    original: torch.Tensor,
    adversarial: torch.Tensor,
    colormap: str = 'hot'
) -> str:
    """
    Generate a heatmap visualization of the perturbation.

    Args:
        original: Original image tensor (1, 3, H, W)
        adversarial: Adversarial image tensor (1, 3, H, W)
        colormap: Matplotlib colormap name

    Returns:
        Base64 encoded PNG image
    """
    # Calculate perturbation magnitude
    perturbation = (adversarial - original).abs()

    # Sum across channels to get overall magnitude
    magnitude = perturbation.sum(dim=1).squeeze().cpu().numpy()

    # Normalize to [0, 1]
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max()

    # Apply colormap
    heatmap = _apply_colormap(magnitude, colormap)

    # Convert to base64
    return _numpy_to_base64(heatmap)


def generate_difference_image(
    original: torch.Tensor,
    adversarial: torch.Tensor,
    amplification: float = 10.0
) -> str:
    """
    Generate an amplified difference image between original and adversarial.

    Args:
        original: Original image tensor (1, 3, H, W)
        adversarial: Adversarial image tensor (1, 3, H, W)
        amplification: Factor to amplify the difference for visibility

    Returns:
        Base64 encoded PNG image
    """
    # Calculate difference
    diff = adversarial - original

    # Shift to [0, 1] range (original diff is in [-1, 1])
    # 0.5 is the neutral point
    diff_shifted = diff * amplification + 0.5

    # Clamp to valid range
    diff_shifted = torch.clamp(diff_shifted, 0, 1)

    # Convert to numpy
    diff_np = diff_shifted.squeeze(0).permute(1, 2, 0).cpu().numpy()
    diff_np = (diff_np * 255).astype(np.uint8)

    return _numpy_to_base64(diff_np)


def generate_attention_map(
    attention_weights: torch.Tensor,
    image_size: Tuple[int, int] = (224, 224),
    colormap: str = 'viridis'
) -> str:
    """
    Generate attention map visualization for ViT models.

    Args:
        attention_weights: Attention weights from ViT
        image_size: Target image size
        colormap: Matplotlib colormap name

    Returns:
        Base64 encoded PNG image
    """
    # Process attention weights
    attn = attention_weights.detach()

    # Average over heads if present
    while attn.dim() > 2:
        attn = attn.mean(dim=0)

    # Get CLS token attention to patches
    if attn.shape[0] > 1:
        cls_attn = attn[0, 1:]  # Skip CLS token itself
    else:
        cls_attn = attn.flatten()

    # Reshape to 2D grid
    num_patches = cls_attn.shape[0]
    patch_size = int(num_patches ** 0.5)

    if patch_size * patch_size == num_patches:
        attn_map = cls_attn.reshape(patch_size, patch_size)
    else:
        # Fallback
        attn_map = cls_attn[:196].reshape(14, 14)

    # Normalize
    attn_map = attn_map.cpu().numpy()
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    # Resize to image size
    attn_map = np.array(Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
        image_size, Image.BILINEAR
    )) / 255.0

    # Apply colormap
    heatmap = _apply_colormap(attn_map, colormap)

    return _numpy_to_base64(heatmap)


def generate_frequency_spectrum(
    image: torch.Tensor,
    log_scale: bool = True
) -> str:
    """
    Generate frequency spectrum visualization using FFT.

    Args:
        image: Image tensor (1, 3, H, W)
        log_scale: Whether to apply log scaling

    Returns:
        Base64 encoded PNG image
    """
    # Convert to grayscale
    gray = image.mean(dim=1).squeeze().cpu()

    # Apply FFT
    freq = torch.fft.fft2(gray)
    freq_shifted = torch.fft.fftshift(freq)

    # Get magnitude
    magnitude = torch.abs(freq_shifted).numpy()

    if log_scale:
        magnitude = np.log1p(magnitude)

    # Normalize
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)

    # Apply colormap
    spectrum = _apply_colormap(magnitude, 'viridis')

    return _numpy_to_base64(spectrum)


def generate_saliency_map(
    model: torch.nn.Module,
    image: torch.Tensor,
    target_class: Optional[int] = None
) -> str:
    """
    Generate gradient-based saliency map.

    Args:
        model: Neural network model
        image: Input image tensor
        target_class: Target class for saliency (None = predicted class)

    Returns:
        Base64 encoded PNG image
    """
    image = image.clone().requires_grad_(True)

    output = model(image)

    if target_class is None:
        target_class = output.argmax(dim=1).item()

    score = output[0, target_class]

    model.zero_grad()
    score.backward()

    saliency = image.grad.abs().sum(dim=1).squeeze().cpu().numpy()

    # Normalize
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    # Apply colormap
    saliency_map = _apply_colormap(saliency, 'hot')

    return _numpy_to_base64(saliency_map)


def overlay_heatmap_on_image(
    image: torch.Tensor,
    heatmap: np.ndarray,
    alpha: float = 0.5
) -> str:
    """
    Overlay a heatmap on the original image.

    Args:
        image: Original image tensor (1, 3, H, W)
        heatmap: Heatmap array (H, W) or (H, W, 3)
        alpha: Blending factor

    Returns:
        Base64 encoded PNG image
    """
    # Convert image to numpy
    img = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)

    # Ensure heatmap is RGB
    if heatmap.ndim == 2:
        heatmap = _apply_colormap(heatmap, 'jet')

    # Resize heatmap if needed
    if heatmap.shape[:2] != img.shape[:2]:
        heatmap = np.array(Image.fromarray(heatmap).resize(
            (img.shape[1], img.shape[0]), Image.BILINEAR
        ))

    # Blend
    blended = (alpha * heatmap + (1 - alpha) * img).astype(np.uint8)

    return _numpy_to_base64(blended)


def _apply_colormap(data: np.ndarray, colormap: str = 'viridis') -> np.ndarray:
    """
    Apply a colormap to grayscale data.

    Args:
        data: 2D numpy array normalized to [0, 1]
        colormap: Name of colormap

    Returns:
        RGB numpy array
    """
    # Simple colormap implementations (avoiding matplotlib dependency)
    colormaps = {
        'hot': _hot_colormap,
        'viridis': _viridis_colormap,
        'jet': _jet_colormap,
        'gray': lambda x: np.stack([x, x, x], axis=-1)
    }

    cmap_func = colormaps.get(colormap, _viridis_colormap)
    return (cmap_func(data) * 255).astype(np.uint8)


def _hot_colormap(x: np.ndarray) -> np.ndarray:
    """Hot colormap (black -> red -> yellow -> white)."""
    r = np.clip(3 * x, 0, 1)
    g = np.clip(3 * x - 1, 0, 1)
    b = np.clip(3 * x - 2, 0, 1)
    return np.stack([r, g, b], axis=-1)


def _viridis_colormap(x: np.ndarray) -> np.ndarray:
    """Simplified viridis-like colormap."""
    r = np.clip(0.267004 + 0.993248 * x, 0, 1)
    g = np.clip(0.004874 + 0.906157 * x, 0, 1)
    b = np.clip(0.329415 - 0.158961 * x + 0.8 * (1 - x), 0, 1)
    return np.stack([r, g, b], axis=-1)


def _jet_colormap(x: np.ndarray) -> np.ndarray:
    """Jet colormap."""
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def _numpy_to_base64(image: np.ndarray) -> str:
    """Convert numpy array to base64 encoded PNG."""
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    pil_image = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)

    return base64.b64encode(buffer.getvalue()).decode('utf-8')
