import torch
import torch.nn as nn


def pgd_attack(
    model: nn.Module,
    image: torch.Tensor,
    label: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    device: torch.device
) -> torch.Tensor:
    """
    Projected Gradient Descent (PGD) attack.

    Args:
        model: Target neural network model
        image: Input image tensor (1, 3, 224, 224)
        label: True label tensor
        epsilon: Maximum perturbation magnitude
        alpha: Step size for each iteration
        steps: Number of attack iterations
        device: Computation device (CPU/CUDA)

    Returns:
        Adversarial image tensor
    """
    # Clone original image
    original_image = image.clone().detach().to(device)
    label = label.to(device)

    # Initialize adversarial image with random perturbation within epsilon ball
    adv_image = original_image.clone().detach()
    adv_image = adv_image + torch.empty_like(adv_image).uniform_(-epsilon, epsilon)
    adv_image = torch.clamp(adv_image, 0, 1).detach()

    for _ in range(steps):
        adv_image.requires_grad = True

        # Forward pass
        output = model(adv_image)
        loss = nn.CrossEntropyLoss()(output, label)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Update adversarial image
        with torch.no_grad():
            adv_image = adv_image + alpha * adv_image.grad.sign()

            # Project back to epsilon ball around original image
            perturbation = torch.clamp(adv_image - original_image, -epsilon, epsilon)
            adv_image = original_image + perturbation

            # Clamp to valid image range [0, 1]
            adv_image = torch.clamp(adv_image, 0, 1).detach()

    return adv_image
