"""
Attack Parameter Registry

Collects all attack specs from individual attack files
and provides utility functions to access them.
"""

from typing import Dict, Any
from dataclasses import asdict

from .base import ParamSpec, AttackSpec


def _spec_to_dict(spec: AttackSpec) -> Dict[str, Any]:
    """Convert AttackSpec to dictionary for JSON serialization."""
    return {
        "id": spec.id,
        "name": spec.name,
        "description": spec.description,
        "category": spec.category,
        "params": [{k: v for k, v in asdict(p).items() if v is not None} for p in spec.params]
    }


# Import attack specs from each attack file
from ..universal.fgsm import ATTACK_SPEC as FGSM_SPEC
from ..universal.pgd import ATTACK_SPEC as PGD_SPEC
from ..universal.mim import ATTACK_SPEC as MIM_SPEC
from ..universal.cw import ATTACK_SPEC as CW_SPEC
from ..universal.adversarial_patch import ATTACK_SPEC as ADVERSARIAL_PATCH_SPEC

from ..cnn_specific.high_freq import ATTACK_SPEC as HIGH_FREQ_SPEC
from ..universal.uap import ATTACK_SPEC as UAP_SPEC
from ..cnn_specific.texture import ATTACK_SPEC as TEXTURE_SPEC

from ..vit_specific.tgr import ATTACK_SPEC as TGR_SPEC
from ..vit_specific.low_freq import ATTACK_SPEC as LOW_FREQ_SPEC
from ..vit_specific.saga import ATTACK_SPEC as SAGA_SPEC


# ==================== Attack Registry ====================

ATTACK_SPECS: Dict[str, AttackSpec] = {
    # Universal (works on both CNN and ViT)
    "fgsm": FGSM_SPEC,
    "pgd": PGD_SPEC,
    "mim": MIM_SPEC,
    "cw": CW_SPEC,
    "adversarial_patch": ADVERSARIAL_PATCH_SPEC,
    "uap": UAP_SPEC,
    # CNN-specific
    "high_freq": HIGH_FREQ_SPEC,
    "texture": TEXTURE_SPEC,
    # ViT-specific
    "tgr": TGR_SPEC,
    "low_freq": LOW_FREQ_SPEC,
    "saga": SAGA_SPEC,
}


def get_attack_spec(attack_id: str) -> AttackSpec:
    """Get attack specification by ID."""
    return ATTACK_SPECS.get(attack_id)


def get_all_attack_specs() -> Dict[str, Dict[str, Any]]:
    """Get all attack specifications as dictionaries."""
    return {k: v.to_dict() for k, v in ATTACK_SPECS.items()}


def get_attacks_by_category(category: str) -> Dict[str, Dict[str, Any]]:
    """Get attack specifications filtered by category."""
    return {
        k: v.to_dict()
        for k, v in ATTACK_SPECS.items()
        if v.category == category
    }


def get_default_params(attack_id: str) -> Dict[str, Any]:
    """Get default parameter values for an attack."""
    spec = ATTACK_SPECS.get(attack_id)
    if not spec:
        return {}
    return {p.name: p.default for p in spec.params}
