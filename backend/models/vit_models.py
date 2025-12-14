"""
ViT Model Manager

Handles loading and inference for Vision Transformer models using timm library.
Uses singleton pattern to avoid loading models multiple times.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, List
from torchvision.models import ResNet50_Weights

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not installed. ViT models will not be available.")


class ViTModelManager:
    """Manages ViT model loading and inference with GPU support."""

    _instance = None
    _models: Dict[str, nn.Module] = {}
    _device: Optional[torch.device] = None
    _imagenet_classes: Optional[List[str]] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._imagenet_classes = self._load_imagenet_classes()
            self._initialized = True
            print(f"ViTModelManager initialized on device: {self._device}")

    def load_model(self, model_name: str = "vit_base_patch16_224") -> nn.Module:
        """
        Load a ViT model by name.

        Args:
            model_name: Name of the model from timm library

        Returns:
            Loaded model
        """
        if not TIMM_AVAILABLE:
            raise ImportError("timm library is required for ViT models. Install with: pip install timm")

        if model_name in self._models:
            return self._models[model_name]

        print(f"Loading {model_name} on device: {self._device}")

        # Load pretrained model from timm
        model = timm.create_model(model_name, pretrained=True)
        model = model.to(self._device)
        model.eval()

        self._models[model_name] = model
        print(f"{model_name} loaded successfully!")

        return model

    def get_model(self, model_name: str = "vit_base_patch16_224") -> nn.Module:
        """Get a model, loading it if necessary."""
        if model_name not in self._models:
            self.load_model(model_name)
        return self._models[model_name]

    @property
    def device(self) -> torch.device:
        return self._device

    def predict(
        self,
        image_tensor: torch.Tensor,
        model_name: str = "vit_base_patch16_224"
    ) -> Tuple[int, float, str]:
        """
        Make prediction on input image tensor.

        Args:
            image_tensor: Preprocessed image tensor (1, 3, 224, 224)
            model_name: Name of the model to use

        Returns:
            Tuple of (class_index, confidence, class_name)
        """
        model = self.get_model(model_name)

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        class_idx = predicted.item()
        conf = confidence.item()
        class_name = self.get_class_name(class_idx)

        return class_idx, conf, class_name

    def predict_top5(
        self,
        image_tensor: torch.Tensor,
        model_name: str = "vit_base_patch16_224"
    ) -> List[Tuple[int, float, str]]:
        """
        Get top-5 predictions for input image tensor.

        Args:
            image_tensor: Preprocessed image tensor (1, 3, 224, 224)
            model_name: Name of the model to use

        Returns:
            List of 5 tuples: [(class_index, confidence, class_name), ...]
        """
        model = self.get_model(model_name)

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            top5_probs, top5_indices = torch.topk(probabilities, 5, dim=1)

        results = []
        for i in range(5):
            class_idx = top5_indices[0, i].item()
            conf = top5_probs[0, i].item()
            class_name = self.get_class_name(class_idx)
            results.append((class_idx, conf, class_name))

        return results

    def get_class_name(self, class_idx: int) -> str:
        """Get ImageNet class name from index."""
        if self._imagenet_classes and 0 <= class_idx < len(self._imagenet_classes):
            return self._imagenet_classes[class_idx]
        return f"class_{class_idx}"

    def get_attention_weights(
        self,
        image_tensor: torch.Tensor,
        model_name: str = "vit_base_patch16_224"
    ) -> Optional[torch.Tensor]:
        """
        Extract attention weights from ViT model.

        Args:
            image_tensor: Input image tensor
            model_name: Name of the model

        Returns:
            Attention weights tensor or None
        """
        model = self.get_model(model_name)

        attention_weights = []

        def attention_hook(module, input, output):
            # For timm ViT, capture attention from Attention modules
            if hasattr(output, 'shape') and len(output.shape) >= 2:
                attention_weights.append(output.detach())

        hooks = []
        for name, module in model.named_modules():
            if 'attn' in name.lower() and 'drop' not in name:
                try:
                    hook = module.register_forward_hook(attention_hook)
                    hooks.append(hook)
                except Exception:
                    pass

        with torch.no_grad():
            _ = model(image_tensor)

        for hook in hooks:
            hook.remove()

        if attention_weights:
            return attention_weights[-1]
        return None

    def _load_imagenet_classes(self) -> List[str]:
        """Load ImageNet class names from torchvision weights metadata."""
        # Get the complete 1000 class names from torchvision
        return ResNet50_Weights.IMAGENET1K_V1.meta["categories"]

    def get_available_models(self) -> list:
        """Return list of available ViT models."""
        return [
            "vit_base_patch16_224",
            "vit_small_patch16_224",
            "vit_tiny_patch16_224",
            "deit_base_patch16_224",
            "deit_small_patch16_224"
        ]

    @staticmethod
    def is_available() -> bool:
        """Check if timm is available."""
        return TIMM_AVAILABLE


# Singleton instance
vit_manager = ViTModelManager()
