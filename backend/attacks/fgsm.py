import torch
import torch.nn as nn


def fgsm_attack(
    model: nn.Module,
    image: torch.Tensor,
    label: torch.Tensor,
    epsilon: float,
    device: torch.device
) -> torch.Tensor:
    """
    Fast Gradient Sign Method (FGSM) attack.

    Args:
        model: Target neural network model
        image: Input image tensor (1, 3, 224, 224)
        label: True label tensor
        epsilon: Perturbation magnitude (0-1 range for normalized images)
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

    # Generate perturbation using sign of gradient
    perturbation = epsilon * image.grad.sign()

    # Create adversarial image
    adv_image = image + perturbation

    # Clamp to valid image range [0, 1]
    adv_image = torch.clamp(adv_image, 0, 1)

    return adv_image.detach()
