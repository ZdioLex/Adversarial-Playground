"""API route modules."""

from .health import router as health_router
from .models import router as models_router
from .attacks import router as attacks_router

__all__ = ['health_router', 'models_router', 'attacks_router']
