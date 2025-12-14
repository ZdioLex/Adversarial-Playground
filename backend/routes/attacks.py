"""
Attack endpoint routes.
"""

import io
from PIL import Image
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from typing import Optional

from schemas import AttackResponse
from attacks import UNIVERSAL_ATTACKS, CNN_ATTACKS, VIT_ATTACKS
from services import AttackService

router = APIRouter(tags=["Attacks"])

# These will be set by main.py
attack_service: Optional[AttackService] = None


def init_service(service: AttackService):
    """Initialize attack service from main app."""
    global attack_service
    attack_service = service


async def _validate_and_load_image(file: UploadFile) -> Image.Image:
    """Validate file type and load as PIL Image."""
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPG/PNG allowed")

    contents = await file.read()
    return Image.open(io.BytesIO(contents))


# ============== Universal Attack Endpoints ==============

@router.post("/attack/universal/{method}", response_model=AttackResponse)
async def universal_attack(
    method: str,
    file: UploadFile = File(...),
    model_type: str = Form("cnn"),
    model_name: str = Form("resnet50"),
    epsilon: float = Form(0.03),
    alpha: Optional[float] = Form(None),
    steps: Optional[int] = Form(None),
    decay: Optional[float] = Form(None),
    random_start: Optional[bool] = Form(None),
    # C&W specific parameters
    target_label: Optional[int] = Form(None),
    kappa: Optional[float] = Form(None),
    lr: Optional[float] = Form(None),
    c_init: Optional[float] = Form(None),
    c_steps: Optional[int] = Form(None),
    # Adversarial Patch parameters
    patch_size: Optional[int] = Form(None),
    random_location: Optional[bool] = Form(None),
    use_eot: Optional[bool] = Form(None),
    circular: Optional[bool] = Form(None),
    # UAP specific parameters
    overshoot: Optional[float] = Form(None)
):
    """
    Perform a universal adversarial attack.

    Supported methods: fgsm, pgd, mim, cw, adversarial_patch, uap
    """
    if method not in UNIVERSAL_ATTACKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown attack method: {method}. Available: {list(UNIVERSAL_ATTACKS.keys())}"
        )

    image = await _validate_and_load_image(file)

    return await attack_service.execute_attack(
        attack_func=UNIVERSAL_ATTACKS[method],
        method=method,
        image=image,
        model_type=model_type,
        model_name=model_name,
        epsilon=epsilon,
        alpha=alpha,
        steps=steps,
        decay=decay,
        random_start=random_start,
        target_label=target_label,
        kappa=kappa,
        lr=lr,
        c_init=c_init,
        c_steps=c_steps,
        patch_size=patch_size,
        random_location=random_location,
        use_eot=use_eot,
        circular=circular,
        overshoot=overshoot
    )


# ============== CNN-Specific Attack Endpoints ==============

@router.post("/attack/cnn/{method}", response_model=AttackResponse)
async def cnn_attack(
    method: str,
    file: UploadFile = File(...),
    model_name: str = Form("resnet50"),
    epsilon: float = Form(0.03),
    freq_threshold: Optional[int] = Form(None),
    steps: Optional[int] = Form(None),
    patch_size: Optional[int] = Form(None),
    use_cached: Optional[bool] = Form(True),
    compute_steps: Optional[int] = Form(None)
):
    """
    Perform a CNN-specific adversarial attack.

    Supported methods: high_freq, texture
    """
    if method not in CNN_ATTACKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown CNN attack: {method}. Available: {list(CNN_ATTACKS.keys())}"
        )

    image = await _validate_and_load_image(file)

    return await attack_service.execute_attack(
        attack_func=CNN_ATTACKS[method],
        method=method,
        image=image,
        model_type="cnn",
        model_name=model_name,
        epsilon=epsilon,
        freq_threshold=freq_threshold,
        steps=steps,
        patch_size=patch_size,
        use_cached=use_cached,
        compute_steps=compute_steps
    )


# ============== ViT-Specific Attack Endpoints ==============

@router.post("/attack/vit/{method}", response_model=AttackResponse)
async def vit_attack(
    method: str,
    file: UploadFile = File(...),
    model_name: str = Form("vit_base_patch16_224"),
    epsilon: float = Form(0.03),
    steps: Optional[int] = Form(None),
    alpha: Optional[float] = Form(None),
    num_tokens: Optional[int] = Form(None),
    freq_threshold: Optional[int] = Form(None),
    entropy_weight: Optional[float] = Form(None),
    attention_threshold: Optional[float] = Form(None),
    attention_weight: Optional[float] = Form(None),
    patch_size: Optional[int] = Form(None),
    # TGR specific
    top_k: Optional[int] = Form(None),
    attn_scale: Optional[float] = Form(None),
    mlp_scale: Optional[float] = Form(None),
    qkv_scale: Optional[float] = Form(None)
):
    """
    Perform a ViT-specific adversarial attack.

    Supported methods: tgr, low_freq, saga
    """
    if method not in VIT_ATTACKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown ViT attack: {method}. Available: {list(VIT_ATTACKS.keys())}"
        )

    image = await _validate_and_load_image(file)

    return await attack_service.execute_attack(
        attack_func=VIT_ATTACKS[method],
        method=method,
        image=image,
        model_type="vit",
        model_name=model_name,
        epsilon=epsilon,
        steps=steps,
        alpha=alpha,
        num_tokens=num_tokens,
        freq_threshold=freq_threshold,
        entropy_weight=entropy_weight,
        attention_threshold=attention_threshold,
        attention_weight=attention_weight,
        patch_size=patch_size,
        top_k=top_k,
        attn_scale=attn_scale,
        mlp_scale=mlp_scale,
        qkv_scale=qkv_scale
    )
