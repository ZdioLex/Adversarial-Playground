"""
Model information and ImageNet class routes.
"""

from fastapi import APIRouter, HTTPException

from schemas import ModelsInfoResponse
from attacks import UNIVERSAL_ATTACKS, CNN_ATTACKS, VIT_ATTACKS
from attacks.utils.attack_params import get_all_attack_specs, get_attacks_by_category

router = APIRouter(tags=["Models"])

# These will be set by main.py
cnn_manager = None
vit_manager = None


def init_managers(cnn, vit):
    """Initialize model managers from main app."""
    global cnn_manager, vit_manager
    cnn_manager = cnn
    vit_manager = vit


@router.get("/models/info", response_model=ModelsInfoResponse)
async def get_models_info():
    """Return available models and attacks."""
    return ModelsInfoResponse(
        cnn_models=cnn_manager.get_available_models() if cnn_manager else [],
        vit_models=vit_manager.get_available_models() if vit_manager and vit_manager.is_available() else [],
        universal_attacks=list(UNIVERSAL_ATTACKS.keys()),
        cnn_attacks=list(CNN_ATTACKS.keys()),
        vit_attacks=list(VIT_ATTACKS.keys())
    )


@router.get("/attacks/specs")
async def get_attack_specs():
    """
    Return specifications for all attacks including configurable parameters.

    Each attack includes:
    - id, name, description, category
    - params: list of parameter specs with name, type, default, min, max, step, description
    """
    return get_all_attack_specs()


@router.get("/attacks/specs/{category}")
async def get_attack_specs_by_category(category: str):
    """
    Return attack specifications for a specific category.

    Categories: universal, cnn, vit
    """
    if category not in ["universal", "cnn", "vit"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category: {category}. Must be one of: universal, cnn, vit"
        )
    return get_attacks_by_category(category)


@router.get("/imagenet/classes")
async def get_imagenet_classes():
    """
    Return all ImageNet class names with their indices.

    Returns:
        List of {index, name} objects for all 1000 ImageNet classes
    """
    if not cnn_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    class_names = cnn_manager.get_all_class_names()
    return [{"index": i, "name": name} for i, name in enumerate(class_names)]


@router.get("/imagenet/classes/{class_idx}")
async def get_imagenet_class_name(class_idx: int):
    """Return the class name for a specific ImageNet class index."""
    if not cnn_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    if not 0 <= class_idx <= 999:
        raise HTTPException(status_code=400, detail="Class index must be between 0 and 999")

    class_name = cnn_manager.get_class_name(class_idx)
    return {"index": class_idx, "name": class_name}
