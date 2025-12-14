"""Pydantic response models for the API."""

from pydantic import BaseModel
from typing import Optional, List


class Top5Prediction(BaseModel):
    """Single prediction with class index, confidence, and name."""
    class_idx: int
    confidence: float
    class_name: str


class AdversarialResult(BaseModel):
    """Result for a single adversarial example (first or last misclassification)."""
    prediction: str
    confidence: float
    class_idx: int
    top5: List[Top5Prediction]
    image_base64: str
    perturbation_heatmap_base64: Optional[str] = None
    perturbation_diff_base64: Optional[str] = None
    perturbation_stats: dict


class AttackResponse(BaseModel):
    """Complete response for an attack request."""
    original_prediction: str
    original_confidence: float
    original_class_idx: int
    original_top5: List[Top5Prediction]
    adversarial_prediction: str
    adversarial_confidence: float
    adversarial_class_idx: int
    adversarial_top5: List[Top5Prediction]
    attack_success: bool
    adversarial_image_base64: str
    original_image_base64: str
    perturbation_heatmap_base64: Optional[str] = None
    perturbation_diff_base64: Optional[str] = None
    perturbation_stats: dict
    model_type: str
    attack_method: str
    first_result: Optional[AdversarialResult] = None
    last_result: Optional[AdversarialResult] = None
    is_iterative: bool = False


class HealthResponse(BaseModel):
    """Health check response with model and device status."""
    status: str
    cnn_device: str
    vit_device: str
    cnn_loaded: bool
    vit_available: bool
    gpu_name: Optional[str] = None
    gpu_memory_total: Optional[str] = None
    gpu_memory_used: Optional[str] = None
    cuda_version: Optional[str] = None
    gpu_available: bool = False


class DeviceSwitchRequest(BaseModel):
    """Request to switch computation device."""
    device: str  # "cpu" or "gpu"


class ModelsInfoResponse(BaseModel):
    """Response with available models and attacks."""
    cnn_models: List[str]
    vit_models: List[str]
    universal_attacks: List[str]
    cnn_attacks: List[str]
    vit_attacks: List[str]
