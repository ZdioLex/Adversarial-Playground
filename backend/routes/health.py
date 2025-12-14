"""
Health and device management routes.
"""

import torch
from fastapi import APIRouter, HTTPException

from schemas import HealthResponse, DeviceSwitchRequest

router = APIRouter(tags=["Health"])

# These will be set by main.py
cnn_manager = None
vit_manager = None


def init_managers(cnn, vit):
    """Initialize model managers from main app."""
    global cnn_manager, vit_manager
    cnn_manager = cnn
    vit_manager = vit


def get_device_display_name(device: torch.device) -> str:
    """Convert device to user-friendly display name."""
    if device.type == "cuda":
        return "GPU"
    return "CPU"


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    gpu_name = None
    gpu_memory_total = None
    gpu_memory_used = None
    cuda_version = None
    gpu_available = torch.cuda.is_available()

    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated(0)
        gpu_memory_total = f"{total_memory / 1024**3:.1f} GB"
        gpu_memory_used = f"{allocated_memory / 1024**3:.2f} GB"

    return HealthResponse(
        status="healthy",
        cnn_device=get_device_display_name(cnn_manager.device) if cnn_manager else "not loaded",
        vit_device=get_device_display_name(vit_manager.device) if vit_manager else "not loaded",
        cnn_loaded=cnn_manager is not None,
        vit_available=vit_manager is not None and vit_manager.is_available(),
        gpu_name=gpu_name,
        gpu_memory_total=gpu_memory_total,
        gpu_memory_used=gpu_memory_used,
        cuda_version=cuda_version,
        gpu_available=gpu_available
    )


@router.post("/device/switch")
async def switch_device(request: DeviceSwitchRequest):
    """Switch computation device between CPU and GPU."""
    target_device = request.device.lower()
    if target_device not in ["cpu", "gpu"]:
        raise HTTPException(status_code=400, detail="Device must be 'cpu' or 'gpu'")

    if target_device == "gpu" and not torch.cuda.is_available():
        raise HTTPException(status_code=400, detail="GPU not available on this system")

    new_device = torch.device("cuda" if target_device == "gpu" else "cpu")

    try:
        if cnn_manager:
            cnn_manager._device = new_device
            for name, model in cnn_manager._models.items():
                cnn_manager._models[name] = model.to(new_device)
            print(f"CNN models moved to {target_device.upper()}")

        if vit_manager and vit_manager.is_available():
            vit_manager._device = new_device
            for name, model in vit_manager._models.items():
                vit_manager._models[name] = model.to(new_device)
            print(f"ViT models moved to {target_device.upper()}")

        if target_device == "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "success": True,
            "device": target_device.upper(),
            "message": f"Successfully switched to {target_device.upper()}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to switch device: {str(e)}")
