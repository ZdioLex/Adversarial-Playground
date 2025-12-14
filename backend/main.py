"""
Adversarial Playground API

FastAPI backend for demonstrating adversarial attacks on CNN and ViT models.
Supports universal attacks (FGSM, PGD, MIM, CW) and architecture-specific attacks.

This is the modular version that uses separate route, schema, and service modules.
"""

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import model managers
from models import CNNModelManager, ViTModelManager

# Import routes
from routes import health_router, models_router, attacks_router
from routes.health import init_managers as init_health_managers
from routes.models import init_managers as init_models_managers
from routes.attacks import init_service as init_attack_service

# Import services
from services import AttackService


# Initialize FastAPI app
app = FastAPI(
    title="Adversarial Playground API",
    description="Interactive platform for demonstrating adversarial attacks on CNN and ViT models",
    version="2.0.0"
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router)
app.include_router(models_router)
app.include_router(attacks_router)


@app.on_event("startup")
async def startup_event():
    """Load models on startup and initialize services."""
    print("=" * 60)
    print("Starting up Adversarial Playground API v2.0")
    print("=" * 60)

    # Initialize CNN manager
    cnn_manager = CNNModelManager()
    cnn_manager.load_model("resnet50")
    print(f"[OK] CNN models loaded on device: {cnn_manager.device}")

    # Initialize ViT manager
    vit_manager = ViTModelManager()
    if vit_manager.is_available():
        try:
            vit_manager.load_model("vit_base_patch16_224")
            print(f"[OK] ViT models loaded on device: {vit_manager.device}")
        except Exception as e:
            print(f"[WARN] Could not load ViT model: {e}")
    else:
        print("[WARN] timm not available, ViT models disabled")

    # Initialize managers in route modules
    init_health_managers(cnn_manager, vit_manager)
    init_models_managers(cnn_manager, vit_manager)

    # Initialize attack service with managers
    attack_service = AttackService(cnn_manager, vit_manager)
    init_attack_service(attack_service)

    print("=" * 60)
    print("Adversarial Playground API ready!")
    print("Endpoints: /health, /models/info, /attacks/specs")
    print("Attack routes: /attack/universal/{method}, /attack/cnn/{method}, /attack/vit/{method}")
    print("=" * 60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
