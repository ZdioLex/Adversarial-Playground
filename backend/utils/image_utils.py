import base64
import io
import torch
import numpy as np
from PIL import Image
from torchvision import transforms


# ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess PIL image for ResNet18 inference.

    Args:
        image: PIL Image object

    Returns:
        Preprocessed tensor (1, 3, 224, 224) in range [0, 1]
    """
    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Define preprocessing pipeline
    # Note: We don't normalize here to keep image in [0, 1] for attack algorithms
    # Normalization will be applied during inference
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # Converts to [0, 1] range
    ])

    # Apply preprocessing
    tensor = preprocess(image)

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    return tensor


def normalize_for_model(tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply ImageNet normalization for model inference.

    Args:
        tensor: Image tensor in [0, 1] range

    Returns:
        Normalized tensor for ResNet18
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return normalize(tensor.squeeze(0)).unsqueeze(0)


def denormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reverse ImageNet normalization.

    Args:
        tensor: Normalized tensor

    Returns:
        Tensor in [0, 1] range
    """
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(tensor.device)
    return tensor * std + mean


def tensor_to_base64(tensor: torch.Tensor) -> str:
    """
    Convert tensor to base64 encoded PNG image.

    Args:
        tensor: Image tensor (1, 3, H, W) in range [0, 1]

    Returns:
        Base64 encoded PNG string
    """
    # Remove batch dimension and move to CPU
    tensor = tensor.squeeze(0).cpu()

    # Clamp to valid range
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to PIL Image
    to_pil = transforms.ToPILImage()
    image = to_pil(tensor)

    # Encode to base64
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return base64_string


def base64_to_image(base64_string: str) -> Image.Image:
    """
    Convert base64 string to PIL Image.

    Args:
        base64_string: Base64 encoded image string

    Returns:
        PIL Image object
    """
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]

    # Decode base64
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))

    return image


def calculate_perturbation_stats(
    original: torch.Tensor,
    adversarial: torch.Tensor
) -> dict:
    """
    Calculate statistics about the perturbation.

    Args:
        original: Original image tensor
        adversarial: Adversarial image tensor

    Returns:
        Dictionary with perturbation statistics
    """
    perturbation = adversarial - original

    return {
        "l2_norm": torch.norm(perturbation).item(),
        "linf_norm": torch.max(torch.abs(perturbation)).item(),
        "mean_abs": torch.mean(torch.abs(perturbation)).item()
    }
