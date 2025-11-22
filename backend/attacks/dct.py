import torch
import torch.nn as nn
import numpy as np
from scipy.fftpack import dct, idct


def dct_attack(
    model: nn.Module,
    image: torch.Tensor,
    label: torch.Tensor,
    epsilon: float,
    freq_threshold: int,
    device: torch.device
) -> torch.Tensor:
    """
    DCT-based frequency domain adversarial attack.

    This attack perturbs high-frequency components in the DCT domain,
    which are often less perceptible to humans but can fool neural networks.

    Args:
        model: Target neural network model
        image: Input image tensor (1, 3, 224, 224)
        label: True label tensor
        epsilon: Perturbation magnitude
        freq_threshold: Threshold for high-frequency components (higher = more high-freq)
        device: Computation device (CPU/CUDA)

    Returns:
        Adversarial image tensor
    """
    # Clone image and enable gradient computation
    image = image.clone().detach().to(device)
    image.requires_grad = True
    label = label.to(device)

    # Forward pass
    output = model(image)
    loss = nn.CrossEntropyLoss()(output, label)

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Get gradient
    grad = image.grad.data.cpu().numpy()

    # Apply DCT-based perturbation
    adv_image_np = _apply_dct_perturbation(
        image.detach().cpu().numpy(),
        grad,
        epsilon,
        freq_threshold
    )

    # Convert back to tensor
    adv_image = torch.from_numpy(adv_image_np).float().to(device)

    # Clamp to valid image range [0, 1]
    adv_image = torch.clamp(adv_image, 0, 1)

    return adv_image


def _apply_dct_perturbation(
    image: np.ndarray,
    grad: np.ndarray,
    epsilon: float,
    freq_threshold: int
) -> np.ndarray:
    """
    Apply perturbation in DCT frequency domain.

    Args:
        image: Original image (1, 3, H, W)
        grad: Gradient of loss w.r.t. image
        epsilon: Perturbation magnitude
        freq_threshold: Threshold for high-frequency mask

    Returns:
        Perturbed image in spatial domain
    """
    batch_size, channels, height, width = image.shape
    adv_image = np.copy(image)

    for b in range(batch_size):
        for c in range(channels):
            # Get image channel and gradient
            img_channel = image[b, c]
            grad_channel = grad[b, c]

            # Transform to DCT domain
            img_dct = dct(dct(img_channel.T, norm='ortho').T, norm='ortho')
            grad_dct = dct(dct(grad_channel.T, norm='ortho').T, norm='ortho')

            # Create high-frequency mask
            mask = _create_high_freq_mask(height, width, freq_threshold)

            # Apply perturbation only to high-frequency components
            perturbation_dct = epsilon * np.sign(grad_dct) * mask

            # Add perturbation in DCT domain
            adv_dct = img_dct + perturbation_dct

            # Transform back to spatial domain
            adv_channel = idct(idct(adv_dct.T, norm='ortho').T, norm='ortho')

            adv_image[b, c] = adv_channel

    return adv_image


def _create_high_freq_mask(height: int, width: int, threshold: int) -> np.ndarray:
    """
    Create a mask for high-frequency DCT components.

    Args:
        height: Image height
        width: Image width
        threshold: Distance threshold from top-left (DC component)

    Returns:
        Binary mask where 1 indicates high-frequency regions
    """
    mask = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            # Manhattan distance from DC component (top-left)
            if i + j >= threshold:
                mask[i, j] = 1.0

    return mask
