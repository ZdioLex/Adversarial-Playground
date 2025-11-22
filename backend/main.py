import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import io

from models import ModelManager
from attacks import fgsm_attack, pgd_attack, dct_attack
from utils.image_utils import (
    preprocess_image,
    tensor_to_base64,
    calculate_perturbation_stats
)

# Initialize FastAPI app
app = FastAPI(
    title="Adversarial Playground API",
    description="Interactive platform for demonstrating adversarial attacks on ResNet18",
    version="1.0.0"
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model manager (singleton - loads model once)
model_manager: Optional[ModelManager] = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup to keep it in VRAM."""
    global model_manager
    print("Starting up Adversarial Playground API...")
    model_manager = ModelManager()
    print(f"Model loaded on device: {model_manager.device}")


# Response models
class AttackResponse(BaseModel):
    original_prediction: str
    original_confidence: float
    original_class_idx: int
    adversarial_prediction: str
    adversarial_confidence: float
    adversarial_class_idx: int
    attack_success: bool
    adversarial_image_base64: str
    original_image_base64: str
    perturbation_stats: dict


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        device=str(model_manager.device) if model_manager else "not loaded",
        model_loaded=model_manager is not None
    )


@app.post("/attack/fgsm", response_model=AttackResponse)
async def attack_fgsm(
    file: UploadFile = File(...),
    epsilon: float = Form(0.03)
):
    """
    Perform FGSM attack on uploaded image.

    Parameters:
    - file: Image file (jpg/png)
    - epsilon: Perturbation magnitude (default: 0.03)
    """
    # Validate epsilon
    if not 0 < epsilon <= 1:
        raise HTTPException(status_code=400, detail="Epsilon must be between 0 and 1")

    # Validate file type
    if not file.content_type in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPG/PNG allowed")

    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_tensor = preprocess_image(image).to(model_manager.device)

        # Get original prediction
        orig_idx, orig_conf, orig_name = model_manager.predict(image_tensor)

        # Perform FGSM attack
        adv_tensor = fgsm_attack(
            model_manager.model,
            image_tensor,
            torch.tensor([orig_idx]).to(model_manager.device),
            epsilon,
            model_manager.device
        )

        # Get adversarial prediction
        adv_idx, adv_conf, adv_name = model_manager.predict(adv_tensor)

        # Calculate perturbation stats
        stats = calculate_perturbation_stats(image_tensor, adv_tensor)

        return AttackResponse(
            original_prediction=orig_name,
            original_confidence=round(orig_conf * 100, 2),
            original_class_idx=orig_idx,
            adversarial_prediction=adv_name,
            adversarial_confidence=round(adv_conf * 100, 2),
            adversarial_class_idx=adv_idx,
            attack_success=orig_idx != adv_idx,
            adversarial_image_base64=tensor_to_base64(adv_tensor),
            original_image_base64=tensor_to_base64(image_tensor),
            perturbation_stats=stats
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Attack failed: {str(e)}")


@app.post("/attack/pgd", response_model=AttackResponse)
async def attack_pgd(
    file: UploadFile = File(...),
    epsilon: float = Form(0.03),
    alpha: float = Form(0.01),
    steps: int = Form(10)
):
    """
    Perform PGD attack on uploaded image.

    Parameters:
    - file: Image file (jpg/png)
    - epsilon: Maximum perturbation magnitude (default: 0.03)
    - alpha: Step size (default: 0.01)
    - steps: Number of iterations (default: 10)
    """
    # Validate parameters
    if not 0 < epsilon <= 1:
        raise HTTPException(status_code=400, detail="Epsilon must be between 0 and 1")
    if not 0 < alpha <= epsilon:
        raise HTTPException(status_code=400, detail="Alpha must be between 0 and epsilon")
    if not 1 <= steps <= 100:
        raise HTTPException(status_code=400, detail="Steps must be between 1 and 100")

    # Validate file type
    if not file.content_type in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPG/PNG allowed")

    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_tensor = preprocess_image(image).to(model_manager.device)

        # Get original prediction
        orig_idx, orig_conf, orig_name = model_manager.predict(image_tensor)

        # Perform PGD attack
        adv_tensor = pgd_attack(
            model_manager.model,
            image_tensor,
            torch.tensor([orig_idx]).to(model_manager.device),
            epsilon,
            alpha,
            steps,
            model_manager.device
        )

        # Get adversarial prediction
        adv_idx, adv_conf, adv_name = model_manager.predict(adv_tensor)

        # Calculate perturbation stats
        stats = calculate_perturbation_stats(image_tensor, adv_tensor)

        return AttackResponse(
            original_prediction=orig_name,
            original_confidence=round(orig_conf * 100, 2),
            original_class_idx=orig_idx,
            adversarial_prediction=adv_name,
            adversarial_confidence=round(adv_conf * 100, 2),
            adversarial_class_idx=adv_idx,
            attack_success=orig_idx != adv_idx,
            adversarial_image_base64=tensor_to_base64(adv_tensor),
            original_image_base64=tensor_to_base64(image_tensor),
            perturbation_stats=stats
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Attack failed: {str(e)}")


@app.post("/attack/dct", response_model=AttackResponse)
async def attack_dct(
    file: UploadFile = File(...),
    epsilon: float = Form(0.05),
    freq_threshold: int = Form(50)
):
    """
    Perform DCT frequency-based attack on uploaded image.

    Parameters:
    - file: Image file (jpg/png)
    - epsilon: Perturbation magnitude (default: 0.05)
    - freq_threshold: High-frequency threshold (default: 50)
    """
    # Validate parameters
    if not 0 < epsilon <= 1:
        raise HTTPException(status_code=400, detail="Epsilon must be between 0 and 1")
    if not 1 <= freq_threshold <= 448:
        raise HTTPException(status_code=400, detail="Frequency threshold must be between 1 and 448")

    # Validate file type
    if not file.content_type in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPG/PNG allowed")

    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_tensor = preprocess_image(image).to(model_manager.device)

        # Get original prediction
        orig_idx, orig_conf, orig_name = model_manager.predict(image_tensor)

        # Perform DCT attack
        adv_tensor = dct_attack(
            model_manager.model,
            image_tensor,
            torch.tensor([orig_idx]).to(model_manager.device),
            epsilon,
            freq_threshold,
            model_manager.device
        )

        # Get adversarial prediction
        adv_idx, adv_conf, adv_name = model_manager.predict(adv_tensor)

        # Calculate perturbation stats
        stats = calculate_perturbation_stats(image_tensor, adv_tensor)

        return AttackResponse(
            original_prediction=orig_name,
            original_confidence=round(orig_conf * 100, 2),
            original_class_idx=orig_idx,
            adversarial_prediction=adv_name,
            adversarial_confidence=round(adv_conf * 100, 2),
            adversarial_class_idx=adv_idx,
            attack_success=orig_idx != adv_idx,
            adversarial_image_base64=tensor_to_base64(adv_tensor),
            original_image_base64=tensor_to_base64(image_tensor),
            perturbation_stats=stats
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Attack failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
