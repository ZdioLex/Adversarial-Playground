"""
Attack Service

Handles the core logic for executing adversarial attacks.
"""

import torch
from PIL import Image
from typing import Dict, Any, Optional, Callable
from fastapi import HTTPException

from schemas import Top5Prediction, AdversarialResult, AttackResponse
from utils.image_utils import (
    preprocess_image,
    tensor_to_base64,
    calculate_perturbation_stats
)
from utils.visualization import (
    generate_perturbation_heatmap,
    generate_difference_image
)


class AttackService:
    """Service for executing adversarial attacks."""

    def __init__(self, cnn_manager, vit_manager):
        self.cnn_manager = cnn_manager
        self.vit_manager = vit_manager

    async def execute_attack(
        self,
        attack_func: Callable,
        method: str,
        image: Image.Image,
        model_type: str,
        model_name: str,
        epsilon: float,
        **kwargs
    ) -> AttackResponse:
        """
        Execute an adversarial attack and return the response.

        Args:
            attack_func: The attack function to execute
            method: Attack method name
            image: PIL Image object
            model_type: 'cnn' or 'vit'
            model_name: Specific model name
            epsilon: Perturbation magnitude
            **kwargs: Additional attack parameters

        Returns:
            AttackResponse with attack results
        """
        # Validate epsilon (not used by C&W)
        if method != "cw" and not 0 < epsilon <= 1:
            raise HTTPException(status_code=400, detail="Epsilon must be between 0 and 1")

        try:
            # Get model and manager
            manager, model, device = self._get_model_and_device(model_type, model_name)

            # Preprocess image
            image_tensor = preprocess_image(image).to(device)

            # Get original prediction
            orig_idx, orig_conf, orig_name = manager.predict(image_tensor, model_name)
            orig_top5 = self._get_top5_predictions(manager, image_tensor, model_name)

            # Execute attack
            attack_result = self._run_attack(
                attack_func, method, model, image_tensor,
                orig_idx, device, epsilon, **kwargs
            )

            # Process attack results
            return self._build_response(
                attack_result, method, model_type, model_name,
                manager, image_tensor, orig_idx, orig_conf, orig_name, orig_top5
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Attack failed: {str(e)}")

    def _get_model_and_device(self, model_type: str, model_name: str):
        """Get the appropriate model manager, model, and device."""
        if model_type == "vit":
            if not self.vit_manager or not self.vit_manager.is_available():
                raise HTTPException(status_code=503, detail="ViT models not available")
            manager = self.vit_manager
        else:
            manager = self.cnn_manager

        model = manager.get_model(model_name)
        device = manager.device
        return manager, model, device

    def _get_top5_predictions(self, manager, image_tensor, model_name):
        """Get top-5 predictions as list of Top5Prediction objects."""
        top5_raw = manager.predict_top5(image_tensor, model_name)
        return [
            Top5Prediction(
                class_idx=idx,
                confidence=round(conf * 100, 2),
                class_name=name
            )
            for idx, conf, name in top5_raw
        ]

    def _run_attack(
        self,
        attack_func: Callable,
        method: str,
        model,
        image_tensor: torch.Tensor,
        orig_idx: int,
        device: torch.device,
        epsilon: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute the attack function with appropriate parameters."""
        # Prepare attack kwargs
        attack_kwargs = {
            'device': device,
            **{k: v for k, v in kwargs.items() if v is not None and k != 'target_label'}
        }

        # C&W attack has different signature (targeted attack)
        if method == "cw":
            target_label = kwargs.get('target_label', 0)
            return attack_func(
                model=model,
                image=image_tensor,
                target_label=target_label,
                **attack_kwargs
            )
        else:
            return attack_func(
                model=model,
                image=image_tensor,
                label=torch.tensor([orig_idx]).to(device),
                epsilon=epsilon,
                **attack_kwargs
            )

    def _build_response(
        self,
        attack_result,
        method: str,
        model_type: str,
        model_name: str,
        manager,
        image_tensor: torch.Tensor,
        orig_idx: int,
        orig_conf: float,
        orig_name: str,
        orig_top5
    ) -> AttackResponse:
        """Build the attack response from results."""
        # Handle both Dict (iterative) and Tuple (FGSM) return formats
        if isinstance(attack_result, dict):
            adv_tensor = attack_result['adv_image']
            first_adv_tensor = attack_result.get('first_adv_image')
            last_adv_tensor = attack_result.get('last_adv_image')
        else:
            adv_tensor, _ = attack_result
            first_adv_tensor = None
            last_adv_tensor = None

        # Get adversarial prediction
        adv_idx, adv_conf, adv_name = manager.predict(adv_tensor, model_name)
        adv_top5 = self._get_top5_predictions(manager, adv_tensor, model_name)

        # Calculate perturbation stats
        stats = calculate_perturbation_stats(image_tensor, adv_tensor)

        # Generate visualizations
        heatmap, diff_img = self._generate_visualizations(image_tensor, adv_tensor)

        # Determine if attack is iterative
        is_iterative = method != "fgsm"

        # Build first/last results for iterative attacks
        first_result = self._build_iterative_result(
            first_adv_tensor, image_tensor, manager, model_name
        ) if is_iterative and first_adv_tensor is not None else None

        last_result = self._build_iterative_result(
            last_adv_tensor, image_tensor, manager, model_name
        ) if is_iterative and last_adv_tensor is not None else None

        return AttackResponse(
            original_prediction=orig_name,
            original_confidence=round(orig_conf * 100, 2),
            original_class_idx=orig_idx,
            original_top5=orig_top5,
            adversarial_prediction=adv_name,
            adversarial_confidence=round(adv_conf * 100, 2),
            adversarial_class_idx=adv_idx,
            adversarial_top5=adv_top5,
            attack_success=orig_idx != adv_idx,
            adversarial_image_base64=tensor_to_base64(adv_tensor),
            original_image_base64=tensor_to_base64(image_tensor),
            perturbation_heatmap_base64=heatmap,
            perturbation_diff_base64=diff_img,
            perturbation_stats=stats,
            model_type=model_type,
            attack_method=method,
            first_result=first_result,
            last_result=last_result,
            is_iterative=is_iterative
        )

    def _generate_visualizations(self, original: torch.Tensor, adversarial: torch.Tensor):
        """Generate heatmap and difference visualizations."""
        try:
            heatmap = generate_perturbation_heatmap(original, adversarial)
            diff_img = generate_difference_image(original, adversarial)
            return heatmap, diff_img
        except Exception:
            return None, None

    def _build_iterative_result(
        self,
        adv_tensor: Optional[torch.Tensor],
        original_tensor: torch.Tensor,
        manager,
        model_name: str
    ) -> Optional[AdversarialResult]:
        """Build AdversarialResult for first/last misclassification."""
        if adv_tensor is None:
            return None

        idx, conf, name = manager.predict(adv_tensor, model_name)
        top5 = self._get_top5_predictions(manager, adv_tensor, model_name)
        stats = calculate_perturbation_stats(original_tensor, adv_tensor)
        heatmap, diff = self._generate_visualizations(original_tensor, adv_tensor)

        return AdversarialResult(
            prediction=name,
            confidence=round(conf * 100, 2),
            class_idx=idx,
            top5=top5,
            image_base64=tensor_to_base64(adv_tensor),
            perturbation_heatmap_base64=heatmap,
            perturbation_diff_base64=diff,
            perturbation_stats=stats
        )
